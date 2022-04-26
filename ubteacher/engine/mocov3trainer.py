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
)
from ubteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from ubteacher.engine.hooks import LossEvalHook, BestCheckpointer
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from ubteacher.solver.build import build_lr_scheduler

# MoCov3
# backbone과 roi head 모두 teacher
class MoCov3Trainer(DefaultTrainer):
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
        self.pseudo_label_jittering = cfg.MOCO.PSEUDO_LABEL_JITTERING
        self.queue_update_label_with_background = cfg.MOCO.QUEUE_UPDATE_LABEL_WITH_BACKGROUND
        
        self.enabled_unlabeled_contrastive_loss = cfg.MOCO.ENABLED_UNLABELED_CONTRASTIVE_LOSS 
        self.enabled_unlabeled_queue_update = cfg.MOCO.ENABLED_UNLABELED_QUEUE_UPDATE
        self.enabled_labeled_queue_update = cfg.MOCO.ENABLED_LABELED_QUEUE_UPDATE
        self.enabled_labeled_contrastive_loss = cfg.MOCO.ENABLED_LABELED_CONTRASTIVE_LOSS

        self.insert_noise_to_features = cfg.MOCO.INSERT_NOISE_TO_FEATURES
        
        self.classwise_queue = cfg.MOCO.CLASSWISE_QUEUE
        self.ensem_ts_model._init_queue(self.cfg, self.classwise_queue)

        self.checkpointer = DetectionTSCheckpointer(
            self.ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )

        self.register_hooks(self.build_hooks())

        self.warmup_ema = cfg.MOCO.WARMUP_EMA # If true, EMA parameter 0.5 -> 0.9996


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
            self.ensem_ts_model._init_queue(self.cfg, self.classwise_queue)

        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        
        if cfg.TEST.EVALUATOR == "coco" or cfg.TEST.EVALUATOR == "cocoEVAL":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
            return evaluator_list[0]
        elif cfg.TEST.EVALUATOR == "voc":
            return PascalVOCDetectionEvaluator(dataset_name)
  
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    # save model end of burnup stage
                    if self.iter + 1 == self.cfg.SEMISUPNET.BURN_UP_STEP:
                        self.checkpointer.save('model_burnup')
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            # filtered by objectness score
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
            # filtered by objectness? classsification score?
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
                raise ValueError("Unknown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k) # per image
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

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[MoCov3] model was changed to eval mode!"
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

            # Generate pseudo label for unlabeled data
            # update contrastive queue 
            with torch.no_grad():
                data_k = unlabel_data_k + label_data_k
                # shuffle bn's problem??(issued in moco)
                unlabel_data_k_batch_size = len(unlabel_data_k)
                (
                    _,
                    proposals_rpn_k, # len(proposals_rpn_unsup_k) = 16
                    proposals_roih_k, 
                    _,
                    box_features_k # [batch_size, num_box_per_image, feature_dim],  from box_head
                ) = self.model_teacher(data_k, branch="unsup_data_weak_box_features")
                # Need to generate target proposal features

                # seperate features
                box_features_unlabel_k = box_features_k[:unlabel_data_k_batch_size] 
                box_features_label_k =  box_features_k[unlabel_data_k_batch_size:] 
                del box_features_k
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
                if self.enabled_labeled_queue_update:
                    self.contrastive_queue_update(label_data_k, proposals_rpn_sup_k, box_features_label_k, self.queue_update_label_with_background)
                if self.enabled_unlabeled_queue_update:
                    self.contrastive_queue_update(unlabel_data_k, proposals_rpn_unsup_k, box_features_unlabel_k)
            #################################################################
            # labeled image에 weak와 strong aug가 적용된 이미지들을 이용해서 student 학습
            all_label_data = label_data_q + label_data_k 
            # unlabeled image에 weak aug을 적용하여 pseudo label을 만들고 strong aug가 적용된 이미지들을 이용해서 student 학습
            all_unlabel_data = unlabel_data_q 

            if self.enabled_labeled_contrastive_loss == True:
                record_all_label_data, _, _, _ = self.model(
                    all_label_data, 
                    branch="contrastive_label", 
                    queue=self.ensem_ts_model.feat_queue.get_queue(), 
                    queue_label=self.ensem_ts_model.feat_queue.get_queue_label(),
                    temperature=None,
                    noise=self.insert_noise_to_features
                )
            else:
                 record_all_label_data, _, _, _ = self.model(
                    all_label_data, branch="supervised"
                 )
            record_dict.update(record_all_label_data)

            if self.enabled_unlabeled_contrastive_loss == True:
                record_all_unlabel_data, _, _, _ = self.model(
                    all_unlabel_data, 
                    branch="contrastive_unlabel",
                    queue=self.ensem_ts_model.feat_queue.get_queue(),
                    queue_label=self.ensem_ts_model.feat_queue.get_queue_label(),
                    temperature=None,
                    noise=self.insert_noise_to_features
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

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

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
        box_projected_features = F.normalize(box_projected_features, dim=1)

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

    @torch.no_grad()
    def contrastive_queue_update(self, data, proposals_rpn, box_features, background=False):
        data_targets = [x["instances"].to(self.model_teacher.device) for x in data]
        # proposal-gt matching
        proposals_rpn, sampled_idxs_list = self.model_teacher.roi_heads.label_and_sample_proposals_with_sampled_idx(proposals_rpn, data_targets)
        
        sampled_box_features = []
        for i in range(len(sampled_idxs_list)):
            sampled_box_features.append(box_features[i][sampled_idxs_list[i]])
        sampled_box_features = torch.cat(sampled_box_features, dim=0)
        sampled_box_features = sampled_box_features.view(-1, sampled_box_features.shape[-1])
        
        box_projected_features = self.model_teacher.roi_heads.box_projector(sampled_box_features)
        box_projected_features = F.normalize(box_projected_features, dim=1)
        # label_data_k를 이용해서 queue update
        _ = self.ensem_ts_model.feat_queue._dequeue_and_enqueue_label(box_projected_features, proposals_rpn, background)
        del box_projected_features, proposals_rpn, data_targets
