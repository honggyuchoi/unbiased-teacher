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
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
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
        if self.box_jitter is True:
            self.jitter_frac = 0.06
            self.jitter_times = 1
        self.cont_prediction_head = cfg.MOCO.CONT_PREDICTION_HEAD
        self.feature_noise = False

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
                metrics_dict[key] = record_dict[key].item()
                if key[:4] == "loss":
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

            # initialize
            metrics_dict = {}
            label_loss = 0.0
            unlabel_loss = 0.0
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
            for key in record_all_label_data.keys():
                metrics_dict[key] = record_all_label_data[key].item()
                label_loss += record_all_label_data[key]
            label_loss.backward()
            del label_loss, record_all_label_data, label_data_q, label_data_k, all_label_data
            #torch.cuda.empty_cache()
            
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

                # target generate
                # detections per image = 100
                unlabel_data_strong_aug = unlabel_data_q + unlabel_data_q_2
                
                if self.target_generate_model == 'teacher':
                    targets = self.generate_target_features(unlabel_data_strong_aug, self.model_teacher, proposals_roih_unsup_k)
                elif self.target_generate_model == "student":
                    if comm.get_world_size > 1:
                        targets = self.generate_target_features(unlabel_data_strong_aug, self.model, proposals_roih_unsup_k)
                    else:
                        targets = self.generate_target_features(unlabel_data_strong_aug, self.model.module, proposals_roih_unsup_k)
                else:
                    raise NotImplementedError
                
                del unlabel_data_strong_aug
            #################################################################

            #############################
            #                           #
            #  unsupervised learning    #
            #                           #
            #############################
            # prepare data 
            # preparing image data
            len_unlabel_data_q = len(unlabel_data_q)
            
            if comm.get_world_size() > 1:
                record_all_unlabel_data, sources = self.unsup_forward(self.model.module, unlabel_data_q, unlabel_data_q_2, proposals_roih_unsup_k)
                images_q = self.model.module.preprocess_image(unlabel_data_q)
                images_q_2 = self.model.module.preprocess_image(unlabel_data_q_2)
                gt_instances = [x["instances"].to(self.device) for x in unlabel_data_q]
                # generate image features
                features = self.model.module.backbone(torch.cat([images_q.tensor, images_q_2.tensor], dim=0))
                features_q = {}
                for key in features.keys():
                    features_q[key] = features[key][:len_unlabel_data_q]
                # generate proposals_rpn of unlabel_data_q
                proposals_rpn, proposal_losses = self.model.module.proposal_generator(
                    images_q, features_q, gt_instances
                )
                del images_q.tensor, images_q_2.tensor
                proposals_rpn = self.model.module.roi_heads.label_and_sample_proposals(
                    proposals_rpn, gt_instances
                )
                del gt_instances
                features_q = [features_q[f] for f in self.box_in_features]
                # detection loss of unlabel_data_q
                box_features_q = self.model.module.roi_heads.box_pooler(features_q, [x.proposal_boxes for x in proposals_rpn])
                box_features_q = self.model.module.roi_heads.box_head(box_features_q)
                predictions = self.model.module.roi_heads.box_predictor(box_features_q)
                detector_losses = self.model.module.roi_heads.box_predictor.losses(predictions, proposals_rpn)
                record_all_unlabel_data.update(proposal_losses)
                record_all_unlabel_data.update(detector_losses)
                # compute cont loss
                features = [features[f] for f in self.box_in_features]
                sources = self.extract_projection_features(self.model.module, features, proposals_roih_unsup_k, prediction=self.cont_prediction_head, box_jitter=self.box_jitter)
            
            else:
                record_all_unlabel_data, sources = self.unsup_forward(self.model, unlabel_data_q, unlabel_data_q_2, proposals_roih_unsup_k)
                images_q = self.model.preprocess_image(unlabel_data_q)
                images_q_2 = self.model.preprocess_image(unlabel_data_q_2)
                gt_instances = [x["instances"].to(self.device) for x in unlabel_data_q]
                # generate image features
                features = self.model.backbone(torch.cat([images_q.tensor, images_q_2.tensor], dim=0))
                features_q = {}
                for key in features.keys():
                    features_q[key] = features[key][:len_unlabel_data_q]
                # generate proposals_rpn of unlabel_data_q
                proposals_rpn, proposal_losses = self.model.proposal_generator(
                    images_q, features_q, gt_instances
                )
                del images_q.tensor, images_q_2.tensor
                proposals_rpn = self.model.roi_heads.label_and_sample_proposals(
                    proposals_rpn, gt_instances
                )
                del gt_instances
                features_q = [features_q[f] for f in self.box_in_features]
                # detection loss of unlabel_data_q
                box_features_q = self.model.roi_heads.box_pooler(features_q, [x.proposal_boxes for x in proposals_rpn])
                box_features_q = self.model.roi_heads.box_head(box_features_q)
                predictions = self.model.roi_heads.box_predictor(box_features_q)
                detector_losses = self.model.roi_heads.box_predictor.losses(predictions, proposals_rpn)
                record_all_unlabel_data.update(proposal_losses)
                record_all_unlabel_data.update(detector_losses)
                # compute cont loss
                features = [features[f] for f in self.box_in_features]
                sources = self.extract_projection_features(self.model, features, proposals_roih_unsup_k, prediction=self.cont_prediction_head, box_jitter=self.box_jitter)

            sym_cont_loss = self.get_sym_cont_loss(sources, targets, batch_size=len_unlabel_data_q)
            record_all_unlabel_data.update({
                'loss_cont': sym_cont_loss 
            })

            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]
              
            for key in new_record_all_unlabel_data.keys():
                if key[:4] == "loss":
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        metrics_dict[key] = new_record_all_unlabel_data[key].item()
                        unlabel_loss += new_record_all_unlabel_data[key] * 0
                    elif key[-6:] == "pseudo":
                        # class-aware cont loss
                        if key == "loss_cont_pseudo":
                            metrics_dict[key] = new_record_all_unlabel_data[key].item()
                            if self.contrastive_loss_weight_decay and (self.storage.iter > self.contrastive_loss_weight_decay_step):
                                unlabel_loss += new_record_all_unlabel_data[key] * self.cfg.MOCO.CONTRASTIVE_LOSS_WEIGHT * 0.1
                            else:
                                unlabel_loss += new_record_all_unlabel_data[key] * self.cfg.MOCO.CONTRASTIVE_LOSS_WEIGHT
                        # other unsup loss
                        else:
                            metrics_dict[key] = new_record_all_unlabel_data[key].item()
                            unlabel_loss += new_record_all_unlabel_data[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                    else:
                        raise NotImplementedError

            unlabel_loss.backward()
            del unlabel_loss, record_all_unlabel_data, new_record_all_unlabel_data

            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
            self.optimizer.step()

    def extract_projection_features(self, model, image_features, rois, prediction=False, box_jitter=False):
        total_rois = rois + rois
        
        # TODO: Remove bad prediction 0.1?
        # i think lower than 0.1 has no important information
        predicted_classes = [x.pred_classes[x.scores > 0.1] for x in total_rois]
        predicted_boxes = [x.pred_boxes[x.scores > 0.1] for x in total_rois]
        predicted_scores = [x.scores[x.scores > 0.1] for x in total_rois]

        image_size = [x.image_size for x in total_rois]

        assert len(image_size) == len(predicted_classes)
        assert len(predicted_classes) == len(predicted_scores)
        assert len(predicted_scores) == len(predicted_boxes)

        if box_jitter == True:
            predicted_boxes, predicted_classes, predicted_scores = self.box_jittering(predicted_boxes, predicted_classes, predicted_scores, image_size, times= self.jitter_times, frac=self.jitter_frac)

        predicted_scores = torch.cat(predicted_scores, dim=0)
        predicted_classes = torch.cat(predicted_classes, dim=0)
                
        box_features = model.roi_heads.box_pooler(image_features, predicted_boxes) # 7x7xC feature map
        if self.feature_noise:
            raise NotImplementedError
        box_features = model.roi_heads.box_head(box_features)
        box_features = model.roi_heads.box_projector(box_features)
        if prediction:
            box_features = model.roi_heads.feat_predictor(box_features)
        
        box_features = F.normalize(box_features, dim=1)
        assert predicted_classes.shape[0] == predicted_scores.shape[0]
        assert predicted_classes.shape[0] == box_features.shape[0]

        targets = {
            'features': box_features,
            'gt_classes': predicted_classes.detach(),
            'gt_scores': predicted_scores.detach() 
        }
        return targets

    def get_sym_cont_loss(self, source, target, batch_size):
        # from student 
        source_features_q, source_features_q_2 = source['features'].chunk(2)
        source_gt_classes_q, source_gt_classes_q_2 = source['gt_classes'].chunk(2)
        source_gt_scores = source['gt_scores']
        source_gt_scores[(source_gt_scores < self.cont_score_threshold)] = 0.0
        source_gt_scores_q, source_gt_scores_q_2 = source_gt_scores.chunk(2)

        # from teacher 
        with torch.no_grad():
            target_features_q = torch.zeros(100 * batch_size, 128).to(self.device)
            target_features_q_2 = torch.zeros(100 * batch_size, 128).to(self.device)

            target_gt_classes_q = torch.zeros(100 * batch_size).to(self.device).fill_(-1.0)
            target_gt_classes_q_2 = torch.zeros(100 * batch_size).to(self.device).fill_(-1.0)
            
            target_gt_scores_q = torch.zeros(100 * batch_size).to(self.device)
            target_gt_scores_q_2 = torch.zeros(100 * batch_size).to(self.device)

            num_of_targets = int(target['features'].shape[0] / 2)
            target_features_q[:num_of_targets], target_features_q_2[:num_of_targets] = target['features'].chunk(2) # (N,128)
            target_gt_classes_q[:num_of_targets], target_gt_classes_q_2[:num_of_targets] = target['gt_classes'].chunk(2) # (N)
            target_gt_scores = target['gt_scores']
            target_gt_scores[(target_gt_scores <self.cont_score_threshold)] = 0.0
            target_gt_scores_q[:num_of_targets], target_gt_scores_q_2[:num_of_targets] = target_gt_scores.chunk(2) #(N)

            target_features_q_2 = concat_all_gather(target_features_q_2)
            target_gt_classes_q_2 = concat_all_gather(target_gt_classes_q_2)
            target_gt_scores_q_2 = concat_all_gather(target_gt_scores_q_2)

            valid = (target_gt_classes_q_2 != -1.0)
            target_features_q_2 = target_features_q_2[valid]
            target_gt_classes_q_2 = target_gt_classes_q_2[valid]
            target_gt_scores_q_2 = target_gt_scores_q_2[valid]

        if self.cont_loss_type == 'infoNCE':
            loss_first = self.cont_loss(source_features_q, source_gt_classes_q, source_gt_scores_q, target_features_q_2, target_gt_classes_q_2, target_gt_scores_q_2)
        elif self.cont_loss_type == 'byol':
            loss_first = self.byol_loss(source_features_q, source_gt_classes_q, source_gt_scores_q, target_features_q_2, target_gt_classes_q_2, target_gt_scores_q_2)
        else:
            raise NotImplementedError

        del source_features_q, source_gt_classes_q, source_gt_scores_q, target_features_q_2, target_gt_classes_q_2, target_gt_scores_q_2

        with torch.no_grad():
            target_features_q = concat_all_gather(target_features_q)
            target_gt_classes_q = concat_all_gather(target_gt_classes_q)
            target_gt_scores_q = concat_all_gather(target_gt_scores_q)

            valid = (target_gt_classes_q != -1.0)
            target_features_q = target_features_q[valid]
            target_gt_classes_q = target_gt_classes_q[valid]
            target_gt_scores_q = target_gt_scores_q[valid]

        if self.cont_loss_type == 'infoNCE':
            loss_second = self.cont_loss(source_features_q_2, source_gt_classes_q_2, source_gt_scores_q_2, target_features_q, target_gt_classes_q, target_gt_scores_q)
        elif self.cont_loss_type == 'byol':
            loss_second = self.byol_loss(source_features_q_2, source_gt_classes_q_2, source_gt_scores_q_2, target_features_q, target_gt_classes_q, target_gt_scores_q)
        else:
            raise NotImplementedError

        del source_features_q_2, source_gt_classes_q_2, source_gt_scores_q_2, target_features_q, target_gt_classes_q, target_gt_scores_q

        loss = 0.5 * (loss_first + loss_second) # symmetric loss function

        return loss.mean()
    # if train model with multi gpus, matrix is not NxN
    def cont_loss(self, source_features, source_gt_classes, source_gt_scores, target_features, target_gt_classes, target_gt_scores):
        
        self_mask = torch.zeros(source_features.shape[0], target_features.shape[0]).float().fill_diagonal_(1.0).to(self.device)
        
        class_match_mask = torch.eq(source_gt_classes.view(-1,1), target_gt_classes.view(1,-1)).float()

        score_weight = torch.mm(source_gt_scores.view(-1,1), target_gt_scores.view(1,-1))

        weighted_matched_mask = (score_weight * class_match_mask).fill_diagonal_(1.0)
        #weighted_matched_mask = (score_weight * class_match_mask).fill_diagonal_(1.0)

        cos_sim = torch.mm(source_features, target_features.T.detach())
        del target_features, target_gt_classes, target_gt_scores 
        cos_sim /= self.temperature

        logits_row_max, _ = torch.max(cos_sim, dim=1, keepdim=True)
        cos_sim = cos_sim - logits_row_max.detach()

        log_prob = cos_sim - torch.log((torch.exp(cos_sim)*(1.0-self_mask)).sum(dim=1, keepdim=True))
        
        loss = -(log_prob * weighted_matched_mask).sum(dim=1) / torch.count_nonzero(weighted_matched_mask,dim=1)

        return loss.mean()

    # L = 2 - 2 * || z_i * z_j || 
    # cont_loss에서 같은 클래스 이미지인데 negative로 학습될수있는 문제가 있다(score가 낮아서 classwise positive pair가 되지 않은 녀석들)
    # class aware byol loss를 적용하면 negative pair가 적용되지 않기때문에 이런 negative한 경우가 발생하지 않을 것이다. 
    def byol_loss(self, source_features, source_gt_classes, source_gt_scores, target_features, target_gt_classes, target_gt_scores):
        self_mask = torch.zeros(source_features.shape[0], source_features.shape[0]).float().fill_diagonal_(1.0).cuda()
        
        class_match_mask = torch.eq(source_gt_classes.view(-1,1), target_gt_classes.view(1,-1)).float()

        score_weight = torch.mm(source_gt_scores.view(-1,1), target_gt_scores.view(1,-1))

        weighted_matched_mask = (score_weight * class_match_mask).fill_diagonal_(1.0)

        cos_sim = torch.mm(source_features, target_features.T.detach())
        
        loss =  - 2 * (cos_sim * weighted_matched_mask).sum(dim=1) / torch.count_nonzero(weighted_matched_mask, dim=1)

        return loss.mean()

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

    # From soft teacher 
    # https://github.com/microsoft/SoftTeacher/blob/main/configs/soft_teacher/base.py
    def box_jittering(self, boxes, boxes_class, boxes_score, images_size, times=4, frac=0.06):
        def _aug_single(box, image_size): # image_size (height, width)
            # random translate and resizing

            box_scale = box.tensor[:, 2:4] - box.tensor[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.tensor.shape[0], 4, device=box.tensor.device) # normal distribution N(0,1)
                * aug_scale[None, ...]
            )
            new_box = box.tensor.clone()[None, ...].expand(times, box.tensor.shape[0], -1)
            new_box = new_box[:,:,:4] + offset

            # (x1,y1,x2,y2)
            new_box[:,:,0] = new_box[:,:,0].clamp(min=0.0)
            new_box[:,:,1] = new_box[:,:,1].clamp(min=0.0)

            new_box[:,:,2] = new_box[:,:,2].clamp(max=image_size[1]) # image width
            new_box[:,:,3] = new_box[:,:,3].clamp(max=image_size[0]) # image height

            return Boxes(new_box.reshape(-1,4))

        def _aug_single_class(box_class):
            new_class = box_class.clone()[None, ...].expand(times, box_class.shape[0]).reshape(-1)
            return new_class 

        def _aug_single_score(box_score):
            new_score = box_score.clone()[None, ...].expand(times, box_score.shape[0]).reshape(-1)
            return new_score

        jittered_boxes = [_aug_single(box, image_size) for box, image_size in zip(boxes,images_size)]
        jittered_classes = [_aug_single_class(box_class) for box_class in boxes_class]
        jittered_scores = [_aug_single_score(box_score) for box_score in boxes_score]

        return jittered_boxes, jittered_classes, jittered_scores

    @torch.no_grad()
    def generate_target_features(self, data, model, proposals_roih_unsup_k):
        images = model.preprocess_image(data)
        features = model.backbone(images.tensor)
        
        features = [features[f] for f in self.box_in_features]
        
        targets = self.extract_projection_features(
            model, \
            features, \
            proposals_roih_unsup_k, \
            prediction=False,
            box_jitter=self.box_jitter
        )
        del features, images.tensor, data

        return targets

    def unsup_forward(self, model, unlabel_data_q, unlabel_data_q_2, proposals_roih_unsup_k):
        record_all_unlabel_data = {}
        len_unlabel_data_q = len(unlabel_data_q)

        images_q = model.preprocess_image(unlabel_data_q)
        images_q_2 = model.preprocess_image(unlabel_data_q_2)
        gt_instances = [x["instances"].to(self.device) for x in unlabel_data_q]
        # generate image features
        # unlabel_data_q -> proposal_losses and detector_losses
        # unlabel_data_q, unlabel_data_q_2 -> cont loss 
        features = model.backbone(torch.cat([images_q.tensor, images_q_2.tensor], dim=0))
        features_q = {}
        for key in features.keys():
            features_q[key] = features[key][:len_unlabel_data_q]
        # generate proposals_rpn of unlabel_data_q
        proposals_rpn, proposal_losses = model.proposal_generator(
            images_q, features_q, gt_instances
        )
        del images_q.tensor, images_q_2.tensor
        # sampling for roi training
        proposals_rpn = model.roi_heads.label_and_sample_proposals(
            proposals_rpn, gt_instances
        )
        del gt_instances
        features_q = [features_q[f] for f in self.box_in_features]
        # detection loss of unlabel_data_q
        box_features_q = model.roi_heads.box_pooler(features_q, [x.proposal_boxes for x in proposals_rpn])
        box_features_q = model.roi_heads.box_head(box_features_q)
        predictions = model.roi_heads.box_predictor(box_features_q)
        detector_losses = model.roi_heads.box_predictor.losses(predictions, proposals_rpn)
        record_all_unlabel_data.update(proposal_losses)
        record_all_unlabel_data.update(detector_losses)
        # compute cont loss
        features = [features[f] for f in self.box_in_features]
        sources = self.extract_projection_features(model, features, proposals_roih_unsup_k, prediction=self.cont_prediction_head, box_jitter=self.box_jitter)
        
        return record_all_unlabel_data, sources
        
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

