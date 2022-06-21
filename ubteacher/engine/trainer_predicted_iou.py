# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
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
)
from ubteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from ubteacher.engine.hooks import LossEvalHook, BestCheckpointer
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from ubteacher.solver.build import build_lr_scheduler

from ubteacher.engine.trainer import UBTeacherTrainer

# Unbiased Teacher Trainer
class Trainer_predicted_IoU(UBTeacherTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
                find_unused_parameters=True
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

        self.iou_threshold = self.cfg.IOUNET.IOU_THRESHOLD

        # For training iou branch
        self.training_with_jittering = self.cfg.IOUNET.TRAINING_WITH_JITTERING
        self.jittering_times = self.cfg.IOUNET.JITTERING_TIMES
        self.jittering_frac = self.cfg.IOUNET.JITTERING_FRAC 
    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[valid_map]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
            # predicted iou is used as localization confidence 
            new_proposal_inst.localization_confidences = proposal_bbox_inst.predict_ious[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output
    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[Trainer_predicted_IoU] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        # remove unlabeled data labels
        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_k = self.remove_label(unlabel_data_k)

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            record_dict, _, _, _ = self.model(
                label_data_q, 
                branch="supervised",
                training_with_jittering=self.training_with_jittering,
                jittering_times=self.jittering_times,
                jittering_frac=self.jittering_frac 
            )
            
            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key == "loss_iou":
                    loss_dict[key] = record_dict[key] * self.cfg.IOUNET.IOULOSS_WEIGHT
                elif key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

            metrics_dict = record_dict
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)

            elif (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    pseudo_label_with_iou,
                    _,
                ) = self.model_teacher(
                    unlabel_data_k, 
                    branch="unsup_data_weak_with_iou"
                )

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD
            pseudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                pseudo_label_with_iou, cur_threshold, "roih", "thresholding"
            )
            joint_proposal_dict = {}
            joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_roih_unsup_k
            
            num_pseudo_label = 0
            for pseudo_label in pseudo_proposals_roih_unsup_k:
                num_pseudo_label += pseudo_label.gt_classes.shape[0]
            num_pseudo_label /= len(pseudo_proposals_roih_unsup_k)
            self.storage.put_scalar("num_pseudo_label", num_pseudo_label)
            
            #  add pseudo-label to unlabeled data
            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )

            # initialize
            metrics_dict = {}
            self.optimizer.zero_grad()
            #############################
            #                           #
            #    supervised learning    #
            #                           #
            #############################
            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q
            # labeled data
            record_all_label_data, _, _, _ = self.model(
                all_label_data, 
                branch="supervised",
                training_with_jittering=self.training_with_jittering,
                jittering_times=self.jittering_times,
                jittering_frac=self.jittering_frac 
            )
            # weight losses
            loss_dict = {}
            for key in record_all_label_data.keys():
                if key == "loss_iou":
                    loss_dict[key] = record_all_label_data[key] * self.cfg.IOUNET.IOULOSS_WEIGHT
                    metrics_dict[key] = record_all_label_data[key].item()
                elif key[:4] == "loss":
                    loss_dict[key] = record_all_label_data[key] * 1
                    metrics_dict[key] = record_all_label_data[key].item()
            losses = sum(loss_dict.values())
            losses.backward()
            del losses, record_all_label_data
            # #############################
            #                           #
            #  unsupervised learning    #
            #                           #
            #############################
            # prepare data 
            # preparing image data
            # unlabeled data 
            record_all_unlabel_data, _, _, _ = self.model(
                all_unlabel_data, 
                branch="supervised_pseudo",
                uncertainty_threshold=self.iou_threshold 
            )
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]
 
            # weight losses
            loss_dict = {}
            for key in new_record_all_unlabel_data.keys():
                if key[:4] == "loss":
                    if key == "loss_rpn_loc_pseudo":
                        # pseudo bbox regression <- 0
                        metrics_dict[key] = new_record_all_unlabel_data[key].item()
                        loss_dict[key] = new_record_all_unlabel_data[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                    elif key == "loss_box_reg_pseudo":
                        metrics_dict[key] = new_record_all_unlabel_data[key].item()
                        loss_dict[key] = new_record_all_unlabel_data[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        metrics_dict[key] = new_record_all_unlabel_data[key].item()
                        loss_dict[key] = new_record_all_unlabel_data[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                    else:  # supervised loss
                        raise NotImplementedError

            losses = sum(loss_dict.values())
            losses.backward()

            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
            self.optimizer.step()
