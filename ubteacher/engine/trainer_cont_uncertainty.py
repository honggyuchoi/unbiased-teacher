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
from ubteacher.engine.trainer_class_aware_cont import Trainer_class_aware_cont
from ubteacher.utils import box_jittering
class Trainer_cont_uncertainty(Trainer_class_aware_cont):
    def __init__(self, cfg):
        super().__init__()
        # parameter for uncertainty regression
        self.uncertainty_threshold = cfg.UNCERTAINTY.THRESHOLD
        self.unlabel_reg_loss_type = cfg.UNCERTAINTY.UNLABEL_REG_LOSS_TYPE
        # "uncertainty_threshold", "weighted_smoothl1_loss", "uncertainty_threshold_with_NLL", "uncertainty_threshold_with_bhattacharyya_loss"

    # Add uncertainty in pseudo label
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

            # add uncertainty to instances
            new_proposal_inst.set("gt_uncertainties", proposal_bbox_inst.uncertainties[valid_map,:])

        return new_proposal_inst
    # =====================================================
    # =================== Training Flow ===================
    # =====================================================
    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[class-aware_cont_uncertainty] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_q_2, unlabel_data_k = data # 각각 list
        data_time = time.perf_counter() - start

        # get real gt for statistic
        gt_instances = [x["instances"].to(self.device) for x in unlabel_data_q]
        # remove unlabeled data labels
        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_q_2 = self.remove_label(unlabel_data_q_2)
        unlabel_data_k = self.remove_label(unlabel_data_k)

        # burn-in stage (supervised training with labeled data)
        # train model with weak augmented input + strong augmented input
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            metrics_dict = {}
            del unlabel_data_q_2
            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)

            record_dict, _, _, _ = self.model(
                label_data_q, branch="supervised")
            
            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key == "loss_box_reg_first_term" or key == "loss_box_reg_second_term":
                    metrics_dict[key] = record_dict[key].item()
                elif key[:4] == "loss":
                    metrics_dict[key] = record_dict[key].item()
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
            
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

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

            #############################
            #                           #
            #   Generate Pseudo label   #
            #                           #
            #############################
            with torch.no_grad():
                (   
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")
                # Need to generate target proposal features
                joint_proposal_dict = {}
                #  Pseudo-labeling
                cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD #default: 0.7
                # Unlabeled data
                # Pseudo_labeling for ROI head (bbox location/objectness)
                # gt_boxes, gt_classes, scores, gt_uncertainties
                pseudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                    proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
                )
                joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_roih_unsup_k

                ###############################################
                #            Get pseudo label info            #
                ###############################################
                num_pseudo_label = 0
                for pseudo_label in pseudo_proposals_roih_unsup_k:
                    num_pseudo_label += pseudo_label.gt_classes.shape[0]
                num_pseudo_label /= len(pseudo_proposals_roih_unsup_k)
                self.storage.put_scalar("num_pseudo_label", num_pseudo_label)
                # pseudo label과 실제 gt
                # roi와 gt를 비교해보면 좋을것 같은데(classification threshold에 따라서 변화하는 정확도를 보면 좋을것)
                
                precisions, recalls, f1_scores, score_threshold = self.get_info_per_score_thereshold(gt_instances, proposals_roih_unsup_k)
                for i, threshold in enumerate(score_threshold):
                    self.storage.put_scalar(f"precision_{threshold}", precisions[i])
                    self.storage.put_scalar(f"recall_{threshold}", recalls[i])
                    self.storage.put_scalar(f"f1_score_{threshold}", f1_scores[i])

                del gt_instances, precisions, recalls, f1_scores, score_threshold

                #  add pseudo-label to unlabeled data
                unlabel_data_q = self.add_label(
                    unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
                )
                unlabel_data_k = self.add_label(
                    unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
                )

                ###############################################
                #       contrastive target generation         #
                ###############################################
                # detections per image(max) = 100
                unlabel_data_strong_aug = unlabel_data_q + unlabel_data_q_2
                
                if self.target_generate_model == 'teacher':
                    targets = self.generate_target_features(unlabel_data_strong_aug, self.model_teacher, proposals_roih_unsup_k, box_jitter=self.box_jitter)
                elif self.target_generate_model == "student":
                    raise NotImplementedError
                    # if comm.get_world_size > 1:
                    #     targets = self.generate_target_features(unlabel_data_strong_aug, self.model, proposals_roih_unsup_k)
                    # else:
                    #     targets = self.generate_target_features(unlabel_data_strong_aug, self.model.module, proposals_roih_unsup_k)
                else:
                    raise NotImplementedError
                
                del unlabel_data_strong_aug
            #################################################################
            # initialize
            metrics_dict = {}
            self.optimizer.zero_grad()
            #############################
            #                           #
            #    supervised learning    #
            #                           #
            #############################
            # prepare data 
            all_label_data = label_data_q + label_data_k 
            # supervised learning with label data
            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )

            loss_dict = {}
            for key in record_all_label_data.keys():
                if key == "loss_box_reg_first_term" or key == "loss_box_reg_second_term":
                    metrics_dict[key] = record_all_label_data[key].item()
                elif key[:4] == "loss":
                    metrics_dict[key] = record_all_label_data[key].item()
                    loss_dict[key] = record_all_label_data[key] * 1.0
            label_loss = sum(loss_dict.values())
            label_loss.backward()
            del label_loss, record_all_label_data, label_data_q, label_data_k, all_label_data

            #############################
            #                           #
            #  unsupervised learning    #
            #                           #
            #############################
            # prepare data 
            # preparing image data
            record_all_unlabel_data, sources = self.model(
                [unlabel_data_q, unlabel_data_q_2],
                branch="unsup_forward", 
                proposals_roih_unsup_k=proposals_roih_unsup_k,
                prediction=self.cont_prediction_head,
                box_jitter=self.box_jitter,
                jitter_times=self.jitter_times,
                jitter_frac=self.jitter_frac
            )
            len_unlabel_data_q = len(unlabel_data_q)
            sym_cont_loss = self.get_sym_cont_loss(sources, targets, batch_size=len_unlabel_data_q)
            record_all_unlabel_data.update({
                'loss_cont': sym_cont_loss 
            })

            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]

            loss_dict = {}              
            for key in new_record_all_unlabel_data.keys():
                if key[:4] == "loss":
                    # rpn reg loss
                    if key == "loss_rpn_loc_pseudo": # smooth l1 loss로 학습하면 될듯 ??
                        metrics_dict[key] = new_record_all_unlabel_data[key].item()
                        loss_dict[key] = new_record_all_unlabel_data[key] * 0 
                    # box reg loss
                    elif key == "loss_box_reg_pseudo":
                        metrics_dict[key] = new_record_all_unlabel_data[key].item()
                        loss_dict[key] = new_record_all_unlabel_data[key] * 0
                    # uncertainty info
                    elif key == "loss_box_reg_first_term_pseudo" or key == "loss_box_reg_second_term_pseudo":
                        metrics_dict[key] = new_record_all_unlabel_data[key].item()
                    # class-aware contrastive loss
                    elif key == "loss_cont_pseudo":
                        metrics_dict[key] = new_record_all_unlabel_data[key].item()
                        if self.contrastive_loss_weight_decay and (self.storage.iter > self.contrastive_loss_weight_decay_step):
                            loss_dict[key] = new_record_all_unlabel_data[key] * self.cfg.MOCO.CONTRASTIVE_LOSS_WEIGHT * 0.1
                        else:
                            loss_dict[key] = new_record_all_unlabel_data[key] * self.cfg.MOCO.CONTRASTIVE_LOSS_WEIGHT
                    # rpn cls loss, box cls loss    
                    elif key[-6:] == "pseudo":
                        # other unsup loss
                        metrics_dict[key] = new_record_all_unlabel_data[key].item()
                        loss_dict[key] = new_record_all_unlabel_data[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                    else:
                        raise NotImplementedError
            unlabel_loss = sum(loss_dict.values())
            unlabel_loss.backward()
            del unlabel_loss, loss_dict, record_all_unlabel_data, new_record_all_unlabel_data

            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
            self.optimizer.step()

@torch.no_grad()
def concat_all_gather(tensor):
    # batch_size * 100
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    rank = torch.distributed.get_rank()
    # first block is related to corrent rank
    tensors_gather[0], tensors_gather[rank] = tensors_gather[rank], tensors_gather[0]

    output = torch.cat(tensors_gather, dim=0)
    return output

