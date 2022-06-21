# -*- coding: utf-8 -*-
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
# MoCov2
# backbone과 roi head 모두 teacher
class MoCov2Trainer(UBTeacherTrainer):
    def __init__(self, cfg):
        """
        1. labeled data
            weak augmentation, strong augmentation -> training 
        2. unlabeled data
            weak augmentation -> generate pseudo label 
            strong augmentation -> training

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


        for param in self.model_teacher.parameters():
            param.requires_grad = False
        
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
        self.temperature = cfg.MOCO.TEMPERATURE
        self.labeled_contrastive_iou_thres = cfg.MOCO.LABELED_CONTRASTIVE_IOU_THRES
        self.unlabeled_contrastive_iou_thres = cfg.MOCO.UNLABELED_CONTRASTIVE_IOU_THRES
        self.debug_deque_and_enque = False 
        self.box_in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.pseudo_label_jittering = cfg.MOCO.PSEUDO_LABEL_JITTERING
        self.classwise_queue = cfg.MOCO.CLASSWISE_QUEUE
        self.queue_update_label_with_background = cfg.MOCO.QUEUE_UPDATE_LABEL_WITH_BACKGROUND
        
        self.enabled_unlabeled_contrastive_loss = cfg.MOCO.ENABLED_UNLABELED_CONTRASTIVE_LOSS 
        self.enabled_unlabeled_queue_update = cfg.MOCO.ENABLED_UNLABELED_QUEUE_UPDATE

        self.insert_noise_to_features = cfg.MOCO.INSERT_NOISE_TO_FEATURES
        
        self.ensem_ts_model._init_queue(self.cfg, self.classwise_queue)

        self.checkpointer = DetectionTSCheckpointer(
            self.ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )

        self.register_hooks(self.build_hooks())

        self.warmup_ema = cfg.MOCO.WARMUP_EMA # If true, EMA parameter 0.5 -> 0.9996

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[MoCov2] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data # 각각 list
        data_time = time.perf_counter() - start

        # remove unlabeled data labels
        unlabel_data_q = self.remove_label(unlabel_data_q)
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
                if self.warmup_ema:
                    alpha = self.cfg.SEMISUPNET.EMA_KEEP_RATE # 0.9996
                    global_step = self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
                    alpha = min(1 - 1 / (global_step + 1), alpha) 
                self._update_teacher_model(keep_rate=alpha)# hyperparameter.. should be tuned

            record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well

            """
                2022.3.25
                    구현한 사항
                    1. label_data_k, unlabel_data_k -> model_teacher = contrastive queue update
                        label_data_k: gt와 0.8 이상 겹치는 proposal box를 foreground sample로, 0.4이하는 background sample로 
                        Unlabel_data_k: pseudo_label + jittered_pseudo_label(x10)을 이용(threshold 0.7)
                            -> score값을 이용해서 weighting을 적용하면 좋을듯(TODO)
                    2. label_data_k, label_data_q, unlabel_data_q ->model_student 
                        label_data: loss_cls + loss_reg + loss_cont
                        unlabel_data: loss_cls + loss_cont
            """
            # Generate pseudo label for unlabeled data
            # update contrastive queue 
            # class imbalance를 해결하기 위해서는 class별로 queue를 유지하는게 어떨까?? -> 일단은 하나의 queue유지 
            with torch.no_grad():
                data_k = unlabel_data_k + label_data_k
                # shuffle bn's problem??(issued in moco)
                unlabel_data_k_batch_size = len(unlabel_data_k)
                (
                    _,
                    proposals_rpn_k, # len(proposals_rpn_unsup_k) = 16
                    proposals_roih_k, 
                    _,
                    features_k # features_unsup_k['p5'].shape = [number of images , feature_dim, feature_h, feature_w]
                ) = self.model_teacher(data_k, branch="unsup_data_weak")
                # Need to generate target proposal features

                # seperate features
                features_unlabel_k = [features_k[f][:unlabel_data_k_batch_size] for f in self.box_in_features]
                features_label_k = [features_k[f][unlabel_data_k_batch_size:] for f in self.box_in_features]
                del features_k
                # seperate rpn_proposals
                proposals_rpn_unsup_k = proposals_rpn_k[:unlabel_data_k_batch_size]
                proposals_rpn_sup_k = proposals_rpn_k[unlabel_data_k_batch_size:]
                del proposals_rpn_k
                # seperate roih_proposals
                proposals_roih_unsup_k = proposals_roih_k[:unlabel_data_k_batch_size]
                proposals_roih_sup_k = proposals_roih_k[unlabel_data_k_batch_size:]
                del proposals_roih_k

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

                # count pseudo label
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
                ############# contrastive target 만들기 #########################
                self.contrastive_queue_update_v1(
                    label_data_k=label_data_k, # gt
                    proposals_rpn_sup_k=proposals_rpn_sup_k, # proposal 
                    features_label_k=features_label_k, # features
                    joint_proposal_dict=joint_proposal_dict,  # pseudo label
                    features_unlabel_k=features_unlabel_k # features
                )

            #################################################################
            # labeled image에 weak와 strong aug가 적용된 이미지들을 이용해서 student 학습
            all_label_data = label_data_q + label_data_k 
            # unlabeled image에 weak aug을 적용하여 pseudo label을 만들고 strong aug가 적용된 이미지들을 이용해서 student 학습
            # weak aug를 쓰지 않는 이유는 pseudo label에서 weak aug를 사용하지 않는 이유는 
            # pseudo label과 proposal 사이의 gap이 줄어들수있고 심지어 student가 더 좋은 bounding box를 생성할수도 있기 때문
            # contrastive loss를 적용할때는 상관없을듯?
            all_unlabel_data = unlabel_data_q

         
            # queue에 label_data_k에서 teacher모델을 통해 만들어낸 projected feature가 존재하는데
            # 동일한 image에 대해서 student모델을 통해 만들어낸 projected feature와 contrastive loss를 만든다.. 
            # feature단에 dropout이나 dropblock을 적용해야하지않을까??
            # TODO:strong_aug가 적용된 이미지에도 feature noise를 넣어주어도 될까?
            record_all_label_data, _, _, _ = self.model(
                all_label_data, 
                branch="contrastive_label", 
                queue=self.ensem_ts_model.feat_queue.queue, 
                queue_label=self.ensem_ts_model.feat_queue.queue_label,
                temperature=self.temperature,
                noise=self.insert_noise_to_features
            )
            record_dict.update(record_all_label_data)
            
            if self.enabled_unlabeled_contrastive_loss == True:
                # general loss(cls loss, reg loss) -> unlabel_data_q
                # contrastive loss -> unlabel_data_q, unlabel_data_k
                record_all_unlabel_data, _, _, _ = self.model(
                    all_unlabel_data + unlabel_data_k, 
                    branch="contrastive_unlabel",
                    queue=self.ensem_ts_model.feat_queue.queue,
                    queue_label=self.ensem_ts_model.feat_queue.queue_label,
                    temperature=self.temperature,
                    noise=self.insert_noise_to_features,
                    split=len(all_unlabel_data)
                )
            else:
                record_all_unlabel_data, _, _, _ = self.model(
                all_unlabel_data, branch="supervised"
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
                        if key == "loss_cont":
                            loss_dict[key] = record_dict[key] * self.cfg.MOCO.CONTRASTIVE_LOSS_WEIGHT
                        else:
                            loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
            # save best performance model
            ret.append(
                BestCheckpointer(
                    cfg.TEST.EVAL_PERIOD, 
                    self.checkpointer, 
                    'bbox/AP', # teacher model
                    'max', 
                    'model_best'
                )
            )

        return ret

    # TODO: labeled와 unlabeled에 대해서 iou thres를 다르게 설정할건데 
    # 구현 어떻게 할까??
    # @torch.no_grad()
    # def _dequeue_and_enqueue(self, key, proposals):
    #     label = torch.cat([p.gt_classes for p in proposals], dim=0)
    #     iou = torch.cat([p.iou for p in proposals], dim=0)
       
    #     select = ((iou >= self.contrastive_iou_thres) | (iou < 0.4))# select iou over 0.7 (background also need)
    #     if self.debug_deque_and_enque:
    #         keys = key[select]
    #         labels = label[select]
    #         ious = iou[select]
    #     else:
    #         try:
    #             keys = concat_all_gathered(key[select])
    #             labels = concat_all_gathered(label[select])
    #             ious = concat_all_gathered(iou[select])
    #         except:
    #             keys = key[select]
    #             labels = label[select]
    #             ious = iou[select]

    #     batch_size = keys.shape[0]
    #     if self.queue_size % batch_size != 0:
    #         print()
    #         print(self.ensem_ts_model.queue_ptr, self.ensem_ts_model.cycles, batch_size, self.ensem_ts_model.queue.shape)
    #         print()

    #     ptr = int(self.ensem_ts_model.queue_ptr)
    #     cycles = int(self.ensem_ts_model.cycles)
    #     if ptr + batch_size <= self.ensem_ts_model.queue.shape[0]:
    #         self.ensem_ts_model.queue[ptr:ptr + batch_size, :] = keys
    #         self.ensem_ts_model.queue_label[ptr:ptr + batch_size] = labels
    #     else:
    #         rem = self.ensem_ts_model.queue.shape[0] - ptr
    #         self.ensem_ts_model.queue[ptr:ptr + rem, :] = keys[:rem, :]
    #         self.ensem_ts_model.queue_label[ptr:ptr + rem] = labels[:rem]

    #     ptr += batch_size
    #     if ptr >= self.ensem_ts_model.queue.shape[0]:
    #         ptr = 0
    #         cycles += 1
    #     self.ensem_ts_model.cycles[0] = cycles
    #     self.ensem_ts_model.queue_ptr[0] = ptr
    #     return cycles

    # roi로 들어온 proposal중에 gt와의 iou에 따라 foreground sample과 background를 저장 
    # background sample이 있는게 더 좋은 성능이 나옴
    # 0.1 < <0.3 만 queue에 넣어도 background가 굉장히 많음// 
    # 0.3 < <0.4 로 hard negaitve만들고 foreground sample이랑 같은 갯수로 sampling 하는것??
    @torch.no_grad()
    def _dequeue_and_enqueue_label(self, key, proposals, background=False):
        label = torch.cat([p.gt_classes for p in proposals], dim=0)
        iou = torch.cat([p.iou for p in proposals], dim=0)
    
        if background == True:
            select_foreground = torch.nonzero((iou > self.labeled_contrastive_iou_thres)).view(-1)
            num_foreground = select_foreground.shape[0]

            select_background = torch.nonzero(((0.3 < iou) & (iou < 0.4))).view(-1)
            num_background = select_background.shape[0]

            if num_foreground < num_background:
                indices = torch.randperm(num_background)[:num_foreground] # without replacement
                select_background = select_background[indices]
                num_background = select_background.shape[0]

            num_select = num_foreground + num_background
            select = torch.cat([select_foreground, select_background], dim = 0) 
            
        else:
            select = torch.nonzero((iou > self.labeled_contrastive_iou_thres)).view(-1)
            num_select = select.shape[0]# select iou over 0.8 (No background)

        print(f'selected projected feature(label data): {num_select}')
        if self.debug_deque_and_enque:
            keys = key[select]
            labels = label[select]
            ious = iou[select]
        else:
            try:
                keys = concat_all_gathered(key[select])
                labels = concat_all_gathered(label[select])
                ious = concat_all_gathered(iou[select])
            except:
                keys = key[select]
                labels = label[select]
                ious = iou[select]

        batch_size = keys.shape[0]
        if batch_size == 0:
            return 0
        if self.queue_size % batch_size != 0:
            print()
            print('update by labeled_k')
            print(self.ensem_ts_model.queue_ptr, self.ensem_ts_model.cycles, batch_size, self.ensem_ts_model.queue.shape)
            print()

        ptr = int(self.ensem_ts_model.queue_ptr)
        cycles = int(self.ensem_ts_model.cycles)
        if ptr + batch_size <= self.ensem_ts_model.queue.shape[0]:
            self.ensem_ts_model.queue[ptr:ptr + batch_size, :] = keys
            self.ensem_ts_model.queue_label[ptr:ptr + batch_size] = labels
        else:
            rem = self.ensem_ts_model.queue.shape[0] - ptr
            self.ensem_ts_model.queue[ptr:ptr + rem, :] = keys[:rem, :]
            self.ensem_ts_model.queue_label[ptr:ptr + rem] = labels[:rem]

        ptr += batch_size
        if ptr >= self.ensem_ts_model.queue.shape[0]:
            ptr = 0
            cycles += 1
        self.ensem_ts_model.cycles[0] = cycles
        self.ensem_ts_model.queue_ptr[0] = ptr
        return cycles

    # background도 포함될듯? roi head에서 class별로 score가 나올텐데 거기에 background도 포함되어있고
    # background score가 높으면 pseudo label로 background도 값이 나올 수 있다.(class 80이 background)
    @torch.no_grad()
    def _dequeue_and_enqueue_unlabel(self, key, classes):
        label = torch.cat([c for c in classes], dim=0)
       
        try:
            keys = concat_all_gathered(key)
            labels = concat_all_gathered(label)
        except:
            keys = key
            labels = label

        batch_size = keys.shape[0]
        print(f'selected projected feature(unlabel data): {batch_size}')
        if batch_size == 0:
            return 0 

        if self.queue_size % batch_size != 0:
            print()
            print('update by unlabeled_k')
            print(self.ensem_ts_model.queue_ptr, self.ensem_ts_model.cycles, batch_size, self.ensem_ts_model.queue.shape)
            print()

        ptr = int(self.ensem_ts_model.queue_ptr)
        cycles = int(self.ensem_ts_model.cycles)
        if ptr + batch_size <= self.ensem_ts_model.queue.shape[0]:
            self.ensem_ts_model.queue[ptr:ptr + batch_size, :] = keys
            self.ensem_ts_model.queue_label[ptr:ptr + batch_size] = labels
        else:
            rem = self.ensem_ts_model.queue.shape[0] - ptr
            self.ensem_ts_model.queue[ptr:ptr + rem, :] = keys[:rem, :]
            self.ensem_ts_model.queue_label[ptr:ptr + rem] = labels[:rem]

        ptr += batch_size
        if ptr >= self.ensem_ts_model.queue.shape[0]:
            ptr = 0
            cycles += 1
        self.ensem_ts_model.cycles[0] = cycles
        self.ensem_ts_model.queue_ptr[0] = ptr
        return cycles

    def generate_projected_box_features(self, model, boxes, features):
        """
            features: List(dim 0: image index)
            boxes: List(dim 0: image index)
        """
        box_features = model.roi_heads.box_pooler(features, boxes)
        box_features = model.roi_heads.box_head(box_features)
        box_projected_features = model.roi_heads.box_projector(box_features)

        return box_projected_features

    # From soft teacher 
    # https://github.com/microsoft/SoftTeacher/blob/main/configs/soft_teacher/base.py
    def box_jittering(self, boxes, boxes_class, times=4, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
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

            return Boxes(new_box.reshape(-1,4))

        def _aug_single_class(box_class):
            new_class = box_class.clone()[None, ...].expand(times, box_class.shape[0]).reshape(-1)
            return new_class 

        jittered_pseudo_boxes = [_aug_single(box) for box in boxes]
        jittered_pseudo_classes = [_aug_single_class(box_class) for box_class in boxes_class]

        return jittered_pseudo_boxes, jittered_pseudo_classes
    
    # 인접 feature끼리도 contrastive 하면?? 
    # queue에 들어가야하는 feature은?? 
    # queue에 background 부분도 들어가야할까?? 
    #labeled data에서 background까지 queue의 input으로 쓰기위해서 rpn과 gt를 이용하여 sampling
    # pseudo label에서는 jittering만 이용
    @torch.no_grad()
    def contrastive_queue_update_v1(self, label_data_k, proposals_rpn_sup_k, features_label_k, joint_proposal_dict, features_unlabel_k):
        label_data_k_targets = [x["instances"].to(self.model_teacher.device) for x in label_data_k]
        # sampling
        # sampling 하는게 좀 많은듯함 512*8 = 4072
        # 128 * 8 =1024
        # 64 * 8 = 512 
        # sample foreground proposals and background proposals
        # RPN에서 gt와 iou가 0.7이상은 foreground, 0.7~0.3 ignore, 0.3 이하는 background
        # ROI로는 foreground, background sample이 sampling되어 넘어감.
        proposals_rpn_sup_k = self.model_teacher.roi_heads.label_and_sample_proposals(proposals_rpn_sup_k, label_data_k_targets, event_storage=False)
        # get projected features
        boxes_per_image = [x.proposal_boxes for x in proposals_rpn_sup_k]
        projected_features = self.generate_projected_box_features(self.model_teacher, boxes_per_image, features_label_k)
        # label_data_k를 이용해서 queue update
        _ = self.ensem_ts_model.feat_queue._dequeue_and_enqueue_label(projected_features, proposals_rpn_sup_k, self.queue_update_label_with_background)
        del projected_features, boxes_per_image, proposals_rpn_sup_k, label_data_k_targets

        # 2. Unlabeled data 
        # 이웃 feature에서도 pooling해서 positive로 사용하면 좋을듯?  -> 어떻게 구현?? 
        # features_unsup_k: features
        # joint_proposal_dict["proposals_pseudo_roih"]: pseudo label
        if self.enabled_unlabeled_queue_update is True:
            pseudo_boxes = [x.gt_boxes for x in joint_proposal_dict["proposals_pseudo_roih"]]
            pseudo_classes = [x.gt_classes for x in joint_proposal_dict["proposals_pseudo_roih"]]

            # apply jittering
            if self.pseudo_label_jittering: 
                cont_pseudo_boxes, cont_pseudo_classes = self.box_jittering(pseudo_boxes, pseudo_classes, times=10, frac=0.06)
            else:
                cont_pseudo_boxes, cont_pseudo_classes = (pseudo_boxes, pseudo_classes)
            
            projected_features = self.generate_projected_box_features(self.model_teacher, cont_pseudo_boxes, features_unlabel_k)

            _ = self.ensem_ts_model.feat_queue._dequeue_and_enqueue_unlabel(projected_features, cont_pseudo_classes)
    
    # label과 pseudo label을 jittering을 적용하여 queue update
    # @torch.no_grad()
    # def contrastive_queue_update_v2(self, label_data_k, proposals_rpn_sup_k, features_label_k, joint_proposal_dict, features_unlabel_k):
    #     label_data_k_targets = [x["instances"].to(self.model_teacher.device) for x in label_data_k]
    #     # sampling
    #     # sampling 하는게 좀 많은듯함 512*8 = 4072
    #     # 128 * 8 =1024
    #     # 64 * 8 = 512 
    #     # sample foreground proposals and background proposals
    #     proposals_rpn_sup_k = self.model_teacher.roi_heads.label_and_sample_proposals(proposals_rpn_sup_k, label_data_k_targets, event_storage=False)
    #     # get projected features
    #     boxes_per_image = [x.proposal_boxes for x in proposals_rpn_sup_k]
    #     projected_features = self.generate_projected_box_features(self.model_teacher, boxes_per_image, features_label_k)
    #     # label_data_k를 이용해서 queue update
    #     _ = self._dequeue_and_enqueue_label(projected_features, proposals_rpn_sup_k)
    #     del projected_features, boxes_per_image, proposals_rpn_sup_k, label_data_k_targets

    #     # 2. Unlabeled data 
    #     # 이웃 feature에서도 pooling해서 positive로 사용하면 좋을듯?  -> 어떻게 구현?? 
    #     # features_unsup_k: features
    #     # joint_proposal_dict["proposals_pseudo_roih"]: pseudo label
    #     pseudo_boxes = [x.gt_boxes for x in joint_proposal_dict["proposals_pseudo_roih"]]
    #     pseudo_classes = [x.gt_classes for x in joint_proposal_dict["proposals_pseudo_roih"]]
    #     pseudo_label_scores =  [x.scores for x in joint_proposal_dict["proposals_pseudo_roih"]]

    #     # apply jittering
    #     if self.pseudo_label_jittering: 
    #         cont_pseudo_boxes, cont_pseudo_classes = self.box_jittering(pseudo_boxes, pseudo_classes, times=5, frac=0.06)
    #     else:
    #         cont_pseudo_boxes, cont_pseudo_classes = (pseudo_boxes, pseudo_classes)
        
    #     projected_features = self.generate_projected_box_features(self.model_teacher, cont_pseudo_boxes, features_unlabel_k)

    #     _ = self._dequeue_and_enqueue_unlabel(projected_features, cont_pseudo_classes)