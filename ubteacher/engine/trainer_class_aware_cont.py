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

from ubteacher.utils import box_jittering
class Trainer_class_aware_cont(UBTeacherTrainer):
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
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=True,
                find_unused_parameters=True
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
        self.box_in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.temperature = cfg.MOCO.TEMPERATURE
        self.target_generate_model = cfg.MOCO.TARGET_GENERATE_MODEL
        self.cont_score_threshold = cfg.MOCO.CONT_SCORE_THRESHOLD
        self.contrastive_loss_weight_decay  = self.cfg.MOCO.CONTRASTIVE_LOSS_WEIGHT_DECAY
        self.contrastive_loss_weight_decay_step = self.cfg.MOCO.CONTRASTIVE_LOSS_WEIGHT_DECAY_STEP

        self.checkpointer = DetectionTSCheckpointer(
            self.ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.register_hooks(self.build_hooks())
        self.warmup_ema = cfg.MOCO.WARMUP_EMA # If true, EMA parameter 0.5 -> 0.9996
        self.cont_loss_type = self.cfg.MOCO.CONTRASTIVE_LOSS_TYPE 

        # applying box jittering => kind of geometry augmentation
        # this can make training more hard, but could learn more sementic information
        self.box_jitter = self.cfg.MOCO.BOX_JITTERING
        self.jitter_frac = self.cfg.MOCO.JITTER_FRAC
        self.jitter_times = self.cfg.MOCO.JITTER_TIMES

        self.cont_prediction_head = self.cfg.MOCO.CONT_PREDICTION_HEAD
        self.feature_noise = False

        # If True, apply contrastive learning
        self.contrastive_learning = self.cfg.MOCO.CONTRASTIVE_LEARNING 
        # If true, apply class-aware. If not True, apply self contrastive
        self.class_aware_cont = self.cfg.MOCO.CLASS_AWARE_CONTRASTIVE

        self.pseudo_label_reg = self.cfg.UNCERTAINTY.PSEUDO_LABEL_REG
        self.uncertainty_threshold = self.cfg.UNCERTAINTY.UNCERTAINTY_THRESHOLD

        # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            #valid_map = proposal_bbox_inst.objectness_logits > thres

            # objectness_logits is not score!!
            objectness_scores = torch.sigmoid(proposal_bbox_inst.objectness_logits)
            valid_map = (objectness_scores > thres)

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
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
            if proposal_bbox_inst.has('uncertainties'):
                new_proposal_inst.gt_uncertainties = proposal_bbox_inst.uncertainties[valid_map]

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

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    @property
    def device(self):
        return self.model.device

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if self.cfg.MODEL.WEIGHTS and self.cfg.MOCO.RESUME_AFTER_BURNUP:
            self.start_iter = checkpoint['scheduler'].get('last_epoch', -1) + 1
            checkpoint = self.checkpointer._load_optimizer_scheduler(checkpoint)
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[class_aware_cont] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_q_2, unlabel_data_k = data # 각각 list
        data_time = time.perf_counter() - start

        # get real gt for statistic
        # gt_instances = [x["instances"].to(self.device) for x in unlabel_data_q]
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
                if key[:4] == "loss":
                    if (key == "loss_box_reg_first_term") or (key =="loss_box_reg_second_term"):
                        metrics_dict[key] = record_dict[key].item()
                    else:
                        metrics_dict[key] = record_dict[key].item()
                        loss_dict[key] = record_dict[key] * 1.0
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
                print('teacher init!!')
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
            #  Generate pseudo label    #
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
                # gt_boxes, gt_classes, scores
                
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
                
                # precisions, recalls, f1_scores, score_threshold = self.get_info_per_score_thereshold(gt_instances, proposals_roih_unsup_k)
                
                # for i, threshold in enumerate(score_threshold):
                #     self.storage.put_scalar(f"precision_{threshold}", precisions[i])
                #     self.storage.put_scalar(f"recall_{threshold}", recalls[i])
                #     self.storage.put_scalar(f"f1_score_{threshold}", f1_scores[i])
                
                # del gt_instances, precisions, recalls, f1_scores, score_threshold

                #  add pseudo-label to unlabeled data
                unlabel_data_q = self.add_label(
                    unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
                )
                unlabel_data_k = self.add_label(
                    unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
                )
                if self.contrastive_learning == True:
                    # target generate
                    # detections per image = 100
                    unlabel_data_strong_aug = unlabel_data_q + unlabel_data_q_2
                    
                    if self.target_generate_model == 'teacher':
                        targets = self.generate_target_features(
                            unlabel_data_strong_aug, 
                            self.model_teacher, 
                            proposals_roih_unsup_k, 
                            box_jitter=self.box_jitter, 
                            jitter_times=self.jitter_times, 
                            jitter_frac=self.jitter_frac
                        )
                    elif self.target_generate_model == "student":
                        raise NotImplementedError
                        # if comm.get_world_size > 1:
                        #     targets = self.generate_target_features(unlabel_data_strong_aug, self.model.module, proposals_roih_unsup_k, box_jitter=self.box_jitter)
                        # else:
                        #     targets = self.generate_target_features(unlabel_data_strong_aug, self.model, proposals_roih_unsup_k, box_jitter=self.box_jitter)
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
                if key[:4] == "loss":
                    if (key == "loss_box_reg_first_term") or (key =="loss_box_reg_second_term"):
                        metrics_dict[key] = record_all_label_data[key].item()
                    else:
                        loss_dict[key] = record_all_label_data[key] * 1.0
                        metrics_dict[key] = record_all_label_data[key].item()
            label_loss = sum(loss_dict.values())
            label_loss.backward()
            del label_loss, record_all_label_data, label_data_q, label_data_k, all_label_data
            # #############################
            #                           #
            #  unsupervised learning    #
            #                           #
            #############################
            # prepare data 
            # preparing image data
            if self.contrastive_learning == True:
                record_all_unlabel_data, _, _, _ = self.model(
                    [unlabel_data_q, unlabel_data_q_2],
                    branch="unsup_forward", 
                    pseudo_label_reg=self.pseudo_label_reg,
                    uncertainty_threshold=self.uncertainty_threshold,
                    proposals_roih_unsup_k=proposals_roih_unsup_k,
                    prediction=self.cont_prediction_head,
                    box_jitter=self.box_jitter,
                    jitter_times=self.jitter_times,
                    jitter_frac=self.jitter_frac,
                    class_aware_cont=self.class_aware_cont,
                    cont_score_threshold=self.cont_score_threshold,
                    cont_loss_type=self.cont_loss_type,
                    temperature=self.temperature,
                    targets=targets,
                )
            else:
                record_all_unlabel_data, _, _, _ = self.model(
                    unlabel_data_q,
                    branch="supervised_smoothL1", 
                    pseudo_label_reg=self.pseudo_label_reg,
                    uncertainty_threshold=self.uncertainty_threshold,
                )

            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]
            
            loss_dict = {}
            for key in new_record_all_unlabel_data.keys():
                if key[:4] == "loss":
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo": # Now, it is smoothL1 loss
                        metrics_dict[key] = new_record_all_unlabel_data[key]
                        if self.pseudo_label_reg == True:
                            loss_dict[key] = new_record_all_unlabel_data[key] * self.cfg.SEMISUPNET.UNSUP_REG_LOSS_WEIGHT
                        else:
                            loss_dict[key] = new_record_all_unlabel_data[key] * 0.0
                    elif (key == "loss_box_reg_first_term_pseudo") or (key =="loss_box_reg_second_term_pseudo"):
                        metrics_dict[key] = new_record_all_unlabel_data[key]
                    elif key == "loss_cont_pseudo":
                        metrics_dict[key] = new_record_all_unlabel_data[key].item()
                        if self.contrastive_loss_weight_decay and (self.storage.iter > self.contrastive_loss_weight_decay_step):
                            loss_dict[key] = new_record_all_unlabel_data[key] * self.cfg.MOCO.CONTRASTIVE_LOSS_WEIGHT * 0.1
                        else:
                            loss_dict[key] = new_record_all_unlabel_data[key] * self.cfg.MOCO.CONTRASTIVE_LOSS_WEIGHT
                    elif key[-6:] == "pseudo":
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
    def generate_target_features(self, data, model, proposals_roih_unsup_k, box_jitter=False, jitter_times=1, jitter_frac=0.06):
        images = model.preprocess_image(data)
        features = model.backbone(images.tensor)
        
        features = [features[f] for f in self.box_in_features]
        
        targets = model.extract_projection_features(
            features, \
            proposals_roih_unsup_k, \
            prediction=False,
            box_jitter=box_jitter,
            jitter_times=jitter_times,
            jitter_frac=jitter_frac
        )
        del features, images.tensor, images

        return targets

    def get_info_per_score_thereshold(self, gt_instances, rois, iou_threshold=0.5):
        '''
            score_threshold 0.6, 0.7, 0.8, 0.9
            precisions, recalls, f1_scores, score_threshold
        '''
        score_threshold = [0.6, 0.7, 0.8, 0.9]
        total_prediction_num = torch.zeros(4).float().to(self.device)
        total_gt_num = 0.0
        
        matched_predictions = torch.zeros(4).float().to(self.device)
        matched_gt = torch.zeros(4).float().to(self.device)
        precisions = torch.zeros(4).float().to(self.device)
        recalls = torch.zeros(4).float().to(self.device)
        f1_scores = torch.zeros(4).float().to(self.device)
        
        for predictions_per_image, targets_per_image in zip(rois, gt_instances):
            if len(targets_per_image) == 0:
                continue
            pred_boxes = predictions_per_image.pred_boxes
            pred_classes = predictions_per_image.pred_classes

            gt_boxes = targets_per_image.gt_boxes
            gt_classes = targets_per_image.gt_classes
            total_gt_num += len(targets_per_image)

            matched_quality_matrix = pairwise_iou(gt_boxes, pred_boxes)
            class_match_matrix = torch.eq(gt_classes.view(-1,1), pred_classes.view(1,-1))

            matched_quality_matrix = matched_quality_matrix * class_match_matrix
            matched_vals, matched_idx = matched_quality_matrix.max(dim=0)

            valid_iou_map = (matched_vals > iou_threshold)

            for i in range(4):
                # total thresholded number
                valid_score_map = (predictions_per_image.scores > score_threshold[i])
                total_prediction_num[i] += pred_classes[valid_score_map].numel()

                valid_map = (valid_score_map & valid_iou_map)
                
                matched_predictions[i] += valid_map.sum()
                matched_gt[i] += torch.unique(matched_idx[valid_map]).numel()

        precisions = matched_predictions / (total_prediction_num + 1e-6) 
        recalls = matched_gt / (total_gt_num + 1e-6)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
        return precisions, recalls, f1_scores, score_threshold