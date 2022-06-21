# -*- coding: utf-8 -*-
import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator, DatasetEvaluators
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog

# additional 
from detectron2.structures import pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals


from ubteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
    build_detection_semisup_train_loader_three_crops
)
from ubteacher.data.dataset_mapper import (
    DatasetMapperTwoCropSeparate,
    DatasetMapperThreeCropSeparate
)
from ubteacher.engine.hooks import LossEvalHook, BestCheckpointer
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from ubteacher.solver.build import build_lr_scheduler

from ubteacher.engine.trainer import UBTeacherTrainer

# backbone과 roi head 모두 teacher
class MoCoTrainer(UBTeacherTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg, two_strong_aug_unlabel=True)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher
        del self.model_teacher.roi_heads.feat_predictor

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        self.ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        # contrastive learning parameters 
        self.burn_up_with_contrastive = cfg.SEMISUPNET.BURN_UP_WITH_CONTRASTIVE 
        self.queue_size = cfg.MOCO.QUEUE_SIZE
        self.contrastive_feature_dim = cfg.MOCO.CONTRASTIVE_FEATURE_DIM
        self.labeled_contrastive_iou_thres = cfg.MOCO.LABELED_CONTRASTIVE_IOU_THRES
        self.unlabeled_contrastive_iou_thres = cfg.MOCO.UNLABELED_CONTRASTIVE_IOU_THRES
        self.box_in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.temperature = cfg.MOCO.TEMPERATURE
        self.classwise_queue = cfg.MOCO.CLASSWISE_QUEUE
        self.ensem_ts_model._init_queue(self.cfg, self.classwise_queue)

        self.checkpointer = DetectionTSCheckpointer(
            self.ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )

        self.register_hooks(self.build_hooks())
        self.cont_iou_threshold = cfg.MOCO.CONT_IOU_THRESHOLD
        self.cont_class_threshold = cfg.MOCO.CONT_CLASS_THRESHOLD

        self.warmup_ema = cfg.MOCO.WARMUP_EMA # If true, EMA parameter 0.5 -> 0.9996

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[MoCov3_with_two_strong_aug] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_q_2, unlabel_data_k = data # 각각 list
        data_time = time.perf_counter() - start

        # remove unlabeled data labels
        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_q_2 = self.remove_label(unlabel_data_q_2)
        unlabel_data_k = self.remove_label(unlabel_data_k)

        # burn-in stage (supervised training with labeled data)
        # train model with weak augmented input + strong augmented input
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)

            # contrastive learning at the burn_up_step
            # TODO: how to implementation?? 
            if self.burn_up_with_contrastive:
                raise NotImplementedError
            else:    
                record_dict, _, _, _ = self.model(
                    label_data_q, branch="supervised")
            
            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:
            # init
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)
            # update
            elif (self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                alpha = self.cfg.SEMISUPNET.EMA_KEEP_RATE # 0.9996
                if self.warmup_ema:
                    global_step = self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
                    alpha = min(1 - 1 / (global_step + 1), alpha) 
                self._update_teacher_model(keep_rate=alpha)# hyperparameter.. should be tuned

            record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well

            # Generate pseudo label for unlabeled data
            # update contrastive queue 
            with torch.no_grad():
                if self.first_forward == "pseudo_labeling":
                    (
                        _,
                        proposals_rpn_unsup_k,
                        proposals_roih_unsup_k,
                        _,
                    ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")
                elif self.first_forward == "pseudo_labeling_with_image_features"

                elif self.first_forward == "pseudo_labeling_with_rpn_projection_features"

                else:
                    raise NotImplementedError
                (
                    proposals_rpn,
                    proposals_roih,
                    projection_features
                ) = self.model_teacher(unlabel_data_k + unlabel_data_q_2, branch="unsup_data_weak_with_projection")
                # Need to generate target proposal features
                len_unlabel_data_k = len(unlabel_data_k)
                joint_proposal_dict = {}
                #  Pseudo-labeling
                cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD #default: 0.7
                # Unlabeled data
                # Pseudo_labeling for ROI head (bbox location/objectness)
                # gt_boxes, gt_classes, scores
                proposals_roih_unsup_k = proposals_roih[:len_unlabel_data_k]
                pseudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                    proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
                )
                joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_roih_unsup_k

                # count pseudo label
                num_pseudo_label = 0
                for pseudo_label in pseudo_proposals_roih_unsup_k:
                    num_pseudo_label += pseudo_label.gt_classes.shape[0]
                num_pseudo_label /= len(pseudo_proposals_roih_unsup_k)
                self.storage.put_scalar("num_pseudo_label", num_pseudo_label)

                # queue update with unlabel_data_q_2 and model_teacher
                projection_features_unsup_q_2 = projection_features[len_unlabel_data_k:]
                proposals_rpn_unsup_q_2 = proposals_rpn[len_unlabel_data_k:]
                num_queue_update = 0
                for proposals_per_image, targets_per_image, projection_features_per_image in zip(proposals_rpn_unsup_q_2, pseudo_proposals_roih_unsup_k, projection_features_unsup_q_2):
                    if len(targets_per_image) == 0:
                        continue
                    select = (targets_per_image.scores > self.cont_class_threshold)

                    if (len(targets_per_image[select]) == 0) or (len(proposals_per_image) == 0):
                        continue
                    match_quality_matrix = pairwise_iou(
                        targets_per_image[select].gt_boxes, proposals_per_image.proposal_boxes
                    )
                    matched_vals, matched_idx = match_quality_matrix.max(dim=0)
                    gt_classes = targets_per_image.gt_classes[matched_idx]
                    scores = targets_per_image.scores[matched_idx]

                    select = (matched_vals > self.cont_iou_threshold)

                    selected_projection_features = projection_features_per_image[select]
                    selected_gt_classes = gt_classes[select]
                    selected_scores = scores[select]
                    num_queue_update += scores[select].shape[0]
                    self.ensem_ts_model.feat_queue._dequeue_and_enqueue_score(
                        selected_projection_features,
                        selected_gt_classes,
                        selected_scores
                    )
                self.storage.put_scalar("num_queue_update", num_queue_update)
                print(self.ensem_ts_model.feat_queue.queue_ptr)
                print(self.ensem_ts_model.feat_queue.cycles)
                del unlabel_data_q_2
                #  add pseudo-label to unlabeled data
                unlabel_data_q = self.add_label(
                    unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
                )
                unlabel_data_k = self.add_label(
                    unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
                )
            
            #################################################################
            # labeled image에 weak와 strong aug가 적용된 이미지들을 이용해서 student 학습
            all_label_data = label_data_q + label_data_k 
            # unlabeled image에 weak aug을 적용하여 pseudo label을 만들고 strong aug가 적용된 이미지들을 이용해서 student 학습
            all_unlabel_data = unlabel_data_q 

            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)
            del label_data_q, label_data_k
            
            # pseudo label learning이랑 
            record_all_unlabel_data, _, _, _ = self.model(
                all_unlabel_data, 
                branch="contrastive_unlabel",
                queue_obj=self.ensem_ts_model.feat_queue,
                temperature=self.temperature,
                noise=False
            )
            
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]
            record_dict.update(new_record_all_unlabel_data)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        if key == "loss_cont_pseudo":
                            loss_dict[key] = (
                                record_dict[key] *
                                self.cfg.MOCO.CONTRASTIVE_LOSS_WEIGHT
                            )
                        else:
                            loss_dict[key] = (
                                record_dict[key] *
                                self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                            )
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()