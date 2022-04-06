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
from detectron2.modeling.roi_heads import build_roi_heads

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
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel, MoCov1TSModel
from ubteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from ubteacher.solver.build import build_lr_scheduler

# for consistency loss
from fvcore.nn import smooth_l1_loss
import torch.nn.KLDivLoss as KLDivLoss

# from dropblock import DropBlock2D
from dropblock import DropBlock2D

# proposal learning 구현 
class PLTrainer(DefaultTrainer):
    def __init__(self, cfg):
        
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        self.model = self.build_model(cfg)
        self.optimizer = self.build_optimizer(cfg, model)

        self.sspl = SSPL_network()

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            self.model = DistributedDataParallel(
                self.model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            self.model, data_loader, self.optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        self.checkpointer = DetectionTSCheckpointer(
            self.model,
            cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.feature_noise_layer = nn.Sequential(
            nn.Dropout2d(p=1/64)
            DropBlock2D(block_size=2, drop_prob=1/64)
        )

        self.kl_loss = KLDivLoss(reduction="batchmean")

        self.register_hooks(self.build_hooks())

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
        
        if cfg.TEST.EVALUATOR == "coco":
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
            #valid_map = proposal_bbox_inst.objectness_logits > thres
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

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data, unlabel_data = data
        data_time = time.perf_counter() - start

        # remove unlabeled data labels
        unlabel_data = self.remove_label(unlabel_data)

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            record_dict, _, _, _ = self.model(
                label_data, branch="supervised")
            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:
            record_dict = {}
            
            data = unlabel_data + label_data
            num_unlabel_data = unlabel_data.shape[0]
            unlabel_proposals_threshold = 0.5
            # generate feature map and proposals 
            _, proposals_rpn, _, _, features = self.model(data, branch="generate_rpn_proposals")
            
            # TODO: filtering proposal bboxes
            proposals_rpn_unsup = proposals_rpn[:num_unlabel_data]
            proposals_rpn_sup = proposals_rpn[num_unlabel_data:]

            list_instances = []
            for proposal_bbox_inst in proposals_rpn_unsup:
                # objectness_logits is not score!!
                objectness_scores = torch.sigmoid(proposal_bbox_inst.objectness_logits)
                valid_map = (objectness_scores > unlabel_proposals_threshold)

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
                list_instances.append(new_proposal_inst)

            targets = [x["instances"].to('cuda') for x in label_data]
            gt_boxes = [x.gt_boxes for x in targets]
            proposals_rpn_sup = add_ground_truth_to_proposals(gt_boxes, proposals_rpn_sup)

            for proposals_per_image, targets_per_image in zip(proposals_rpn_sup, targets):
                # TODO: only select foreground
                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                )
                # iou값 저장하자
                if match_quality_matrix.numel() == 0:
                    matched_vals = torch.zeros(match_quality_matrix.shape[1]).to(match_quality_matrix.device)
                else:
                    matched_vals, _ = match_quality_matrix.max(dim=0) # max iou, _

                matched_vals > 0.7



            
            


            # box_features: after box_pooler
            proposals_roih, box_pooled_features, box_feature_vectors = self.model.roi_heads(
                images=None,
                features=features, 
                proposals=proposals_rpn, 
                branch="forward"
            )
            

            # generate noise feature 
            # box_features shape should be (batch_size, channel, height, width)
            noise_box_feature_vectors_list = self.generate_noise_box_feature_vectors(self.model.roi_heads.box_head, box_pooled_features, k=4)

            # Compute loss
            # positive_proposals(label data) + over 0.5 proposals(unlabel data)
            # 1. supervised learning(with label data)
            record_label_data, _, _, _ = self.model(
                label_data, branch="supervised"
            )
            record_dict.update(record_label_data)

            # 2. unsupervised learning(with label + unlabel data)
            #   1. consistency loss( cls_consistency, reg_consistency)
            loss_consistency = self.consistency_losses(proposals_roih, noise_expanded_box_features)
            #   2. SSPL loss(contrastive, proposal_prediction)
            loss_sspl = self.sspl(box_feature_vectors, noise_box_feature_vectors_list)

            # weight losses
            loss_dict = {}
            # for supervised loss
            for key in record_dict.keys():
                loss_dict[key] = record_dict[key] * 1

            # consistency loss
            for key in loss_consistency.keys():
                if key == "loss_cons_cls":
                    loss_dict[key] = loss_consistency[key] * self.loss_cons_cls_weight # 1
                elif key == "loss_cons_reg":
                    loss_dict[key] = loss_consistency[key] * self.loss_cons_reg_weight # 0.5
                else:
                    raise NotImplementedError

            # sspl loss
            for key in loss_sspl.keys():
                if key == "loss_contrastive":
                    loss_dict[key] = loss_sspl[key] * self.loss_contrastive_weight #1
                elif key == "loss_proposal_predict"
                    loss_dict[key] = loss_sspl[key] * self.loss_proposal_predict_weight # 0.25
                else:
                    raise NotImplementedError


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
            ret.append(
                BestCheckpointer(
                    cfg.SOLVER.CHECKPOINT_PERIOD, 
                    self.checkpointer, 
                    'bbox/AP', 
                    'max', 
                    'model_best'
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return _last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    # generate 1024 feature vectors
    def generate_noise_box_feature_vectors(self, box_head, box_pooled_features, k=4):
        #(batch_size, channel, height, width) - > # (batch_size*4, channel, height, width)
        # -> (batch_size, feature_dim)
        noise_box_feature_vectors_list = []
        
        for i in range(k):
            noise_box_pooled_features = self.feature_noise_layer(box_pooled_features.clone())
            noise_box_feature_vectors = box_head(noise_box_pooled_features)
            noise_box_feature_vectors_list.append(noise_box_feature_vectors)

        return noise_box_feature_vectors_list

    # for unsupervised learning
    # original feature map generate target
    def consistency_losses(self, proposals_roih, noise_box_features_list):
        # target
        # i번째 target은  - (i, i+batch_size, i + batch_size*2, i + batch_size*3) 
        def compute_loss_cons_cls(inputs, targets):
            log_softmax_inputs = F.log_softmax(inputs)
            softmax_targets = F.softmax(targets)
            return self.kl_loss(log_softmax_inputs, softmax_targets)

        def compute_loss_cons_reg(inputs, targets):
            loss_box_reg = smooth_l1_loss(
                inputs,
                targets,
                beta=1.0,
                reduction="none",
            )
            return loss_box_reg
            
        target_scores, target_proposal_deltas = proposals_roih.clone().detach()
        
        total_loss_cons_cls = 0
        loss_cons_reg_list = []

        for noise_box_features in noise_box_features_list:
            scores, proposal_deltas = self.model.roi_heads.box_predictor(
                noise_box_feature
            )    
            loss_cons_cls = compute_loss_cons_cls(scores, target_scores)
            loss_cons_reg = compute_loss_cons_reg(proposal_deltas, target_proposal_deltas)

            total_loss_cons_cls += loss_cons_cls
            loss_cons_reg_list.appned(loss_cons_reg)
        
        loss_cons_reg = torch.cat(loss_cons_reg, dim=1)

        loss_cons_reg = loss_cons_reg.min(dim=1)

        return {
            'loss_cons_cls': total_loss_cons_cls / len(noise_box_features_list)
            'loss_cons_reg': loss_cons_reg.mean()
        }





    