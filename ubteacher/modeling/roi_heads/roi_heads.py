# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures.boxes import matched_boxlist_iou
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

from detectron2.modeling.poolers import ROIPooler
from detectron2.config import configurable

from ubteacher.modeling.roi_heads.contrastive_head import (
    ContrastiveHead,
    Projector,
    Projector_bn,
    Predictor,
    Predictor_bn
)
from ubteacher.modeling.roi_heads.fast_rcnn import (
    FastRCNNFocaltLossOutputLayers, 
    FastRCNNUncertaintyOutputLayers,
    MoCoOutputLayers,
    FastRCNNIoUOutputLayers
)
from dropblock import DropBlock2D
import copy


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLab(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape)
        else:
            raise ValueError("Unknown ROI head loss.")

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
        cur_threshold=None,
        jittering_times=None,
        jittering_frac=None,
        pseudo_label_reg=False,
        uncertainty_threshold=None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            # proposals = self.label_and_sample_proposals(
            #     proposals, targets, branch=branch
            # )
            if pseudo_label_reg is True:
                # pseudo label matching for classification
                proposals_for_classification = self.label_and_sample_proposals(
                    copy.deepcopy(proposals), targets
                )
                # pseudo label matching for regression
                # remove high uncertain pseudo label
                proposals_for_regression = self.label_and_sample_proposals_uncertainty(
                    copy.deepcopy(proposals), targets, uncertainty_threshold
                )
            else:
                proposals = self.label_and_sample_proposals(
                    proposals, targets
                )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            # proposals = self.label_and_sample_proposals(
            #     proposals, targets, branch=branch
            # )  # do not apply target on proposals
            proposals = self.label_and_sample_proposals(
                proposals, targets, event_storage=False
            )
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
            if pseudo_label_reg is True:
                losses_for_classification, _ = self._forward_box(
                    features, proposals_for_classification, compute_loss, compute_val_loss, branch
                )
                losses_for_regression, _ = self._forward_box(
                    features, proposals_for_regression, compute_loss, compute_val_loss, branch
                )
                losses = {
                    'loss_cls': losses_for_classification['loss_cls'] * 1.0 + losses_for_regression['loss_cls'] * 0.0,
                    'loss_box_reg': losses_for_classification['loss_box_reg'] * 0.0 + losses_for_regression['loss_box_reg'] * 1.0
                }
                return _, losses
            else:
                losses, _ = self._forward_box(
                    features, proposals, compute_loss, compute_val_loss, branch
                )
                return proposals, losses
        # follow the soft teacher method
        elif branch == "unsup_data_weak_with_uncertainty":
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )
            # threshold with classification scores 
            # and compute uncertainty 
            pred_instances_with_uncertainty = self.compute_uncertainty_with_aug(
                features, 
                pred_instances, 
                proposals,
                cur_threshold=cur_threshold,
                times=jittering_times,
                frac=jittering_frac
            ) 
            return pred_instances_with_uncertainty, None
        # inference and generate pseudo label
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )

            return pred_instances, predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if (
            self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss
            losses = self.box_predictor.losses(predictions, proposals)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:

            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    # pseudo label 만들떄는 sampling 안함
    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], event_storage:bool = True
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # iou값 저장하자
            if match_quality_matrix.numel() == 0:
                matched_vals = torch.zeros(match_quality_matrix.shape[1]).to(match_quality_matrix.device)
            else:
                matched_vals, _ = match_quality_matrix.max(dim=0) # max iou, _

            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # if not proposals_per_image.has('iou'):
            #     proposals_per_image.set('iou', matched_vals[sampled_idxs])

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        if event_storage == True:
            storage = get_event_storage()
            storage.put_scalar(
                "roi_head/num_target_fg_samples", np.mean(num_fg_samples)
            )
            storage.put_scalar(
                "roi_head/num_target_bg_samples", np.mean(num_bg_samples)
            )

        return proposals_with_gt

    @torch.no_grad()
    def compute_epistemic_uncertainty(self, features, proposals, times=10):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        total_proposal_deltas = torch.zeros((times, box_features.shape[0], 4)) 
        for i in range(times):
            box_features = self.box_head(box_features)
            predictions = self.box_predictor(box_features)
            scores, proposal_deltas = predictions
            total_proposal_deltas[i] = proposal_deltas
        uncertainties = total_proposal_deltas.std(dim=0)
        return uncertainties

    # TODO: Soft-teacher, jittering based uncertainty
    #  https://github.com/microsoft/SoftTeacher/blob/150db3b4be07483868e94430b7d2bb832255bf7b/ssod/models/soft_teacher.py
    # compute uncertainty via jittering
    # compute uncertainty only for reliable boxes(over the classification scores)
    # In soft-teacher, only calculate the reliability for the boxes with a foreground score greater than 0.5 
    @torch.no_grad()
    def compute_uncertainty_with_aug(self, features, pred_instances, proposals, cur_threshold=None, times=None, frac=None):
        features = [features[f] for f in self.box_in_features]

        # remove low classification samples for computation efficiency
        scores = [x.scores[x.scores > cur_threshold] for x in pred_instances]
        image_size = [x.image_size for x in pred_instances]
        pred_boxes = [x.pred_boxes[x.scores > cur_threshold] for x in pred_instances]
        pred_classes = [x.pred_classes[x.scores > cur_threshold] for x in pred_instances]
        num_prop_per_image = [len(p) for p in pred_classes]

        auged_pred_boxes, auged_pred_classes = self.aug_box(pred_boxes, pred_classes, image_size, times=times, frac=frac)
        auged_pred_per_image = [len(p) for p in auged_pred_classes]
        # compute refined boxes
        box_features = self.box_pooler(features, auged_pred_boxes)
        box_features = self.box_head(box_features)
        _, proposal_deltas = self.box_predictor(box_features) # [N, num_classes * 4]

        auged_pred_boxes = torch.cat([p.tensor for p in auged_pred_boxes], dim=0)
        refined_boxes = self.box_predictor.box2box_transform.apply_deltas(
            proposal_deltas,
            auged_pred_boxes,
        )  # Nx(KxB)
        auged_pred_classes = torch.cat(auged_pred_classes, dim=0)
        num_bbox_reg_classes = refined_boxes.shape[1] // 4
        refined_boxes = refined_boxes.reshape(-1, num_bbox_reg_classes, 4) # [N,320] -> [N, 80, 4]
        select = auged_pred_classes.unsqueeze(1).unsqueeze(2).expand(-1,-1,4)
        
        refined_boxes = refined_boxes.gather(1, select)
        refined_boxes = refined_boxes.split(auged_pred_per_image)
        
        bboxes = [boxes.view(times, -1, 4).mean(dim=0) for boxes in refined_boxes]
        bboxes_unc = [boxes.view(times, -1, 4).std(dim=0) for boxes in refined_boxes]
        
        # Implemented based on paper 
        # bboxes_shape = [(boxes.tensor[:,2:4] - boxes.tensor[:,:2]).clamp(min=1.0) for boxes in pred_boxes]
        # bboxes_unc = [2 * box_unc / (box_shape[:,0] + box_shape[:,1]).unsqueeze(1) for box_unc, box_shape in zip(bboxes_unc, bboxes_shape)]
        
        # Implemented based on released code
        bboxes_shape = [(boxes[:,2:4] - boxes[:,:2]).clamp(min=1.0) for boxes in bboxes]
        bboxes_unc = [box_unc / box_shape[:, None, :].expand(-1,2,2).reshape(-1,4) for box_unc, box_shape in zip(bboxes_unc, bboxes_shape)]
        
        pred_instances_with_uncertainty = []
        for image_size_per_image, pred_boxes_per_image, pred_classes_per_image, scores_per_image, box_unc_per_image in \
            zip(image_size, pred_boxes, pred_classes, scores, bboxes_unc):
            result = Instances(image_size_per_image)
            result.pred_boxes = pred_boxes_per_image
            result.pred_classes = pred_classes_per_image
            result.scores = scores_per_image
            result.pred_uncertainties = box_unc_per_image

            pred_instances_with_uncertainty.append(result)

        return pred_instances_with_uncertainty 

    def aug_box(self, boxes, boxes_class, images_size, times=1, frac=0.06):
        def _aug_single(box, image_size):
            # random translate
            box_scale = box.tensor[:, 2:4] - box.tensor[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.tensor.shape[0], 4, device=box.tensor.device)
                * aug_scale[None, ...]
            )
            new_box = box.tensor.clone()[None, ...].expand(times, box.tensor.shape[0], -1).clone()
            new_box = new_box[:,:,:4] + offset

            new_box[:,:,0] = new_box[:,:,0].clamp(min=0.0)
            new_box[:,:,1] = new_box[:,:,1].clamp(min=0.0)

            new_box[:,:,2] = new_box[:,:,2].clamp(max=image_size[1]) # image width
            new_box[:,:,3] = new_box[:,:,3].clamp(max=image_size[0]) # image height
            
            return Boxes(new_box.reshape(-1,4))

        def _aug_single_class(box_class):
            new_class = box_class.clone()[None, ...].expand(times, box_class.shape[0]).reshape(-1)
            return new_class

        jittered_boxes = [_aug_single(box, image_size) for box, image_size in zip(boxes, images_size)]
        jittered_classes = [_aug_single_class(box_class) for box_class in boxes_class]
        
        return jittered_boxes, jittered_classes

    @torch.no_grad()
    def label_and_sample_proposals_uncertainty(
        self, proposals: List[Instances], targets: List[Instances], uncertainty_threshold=None, event_storage:bool = True
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes[x.gt_uncertainties.mean(dim=1) < uncertainty_threshold] for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []
        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            valid = (targets_per_image.gt_uncertainties.mean(dim=1) < uncertainty_threshold)
            low_uncertainty_targets_per_image = targets_per_image[valid]

            has_gt = len(low_uncertainty_targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                low_uncertainty_targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )

            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, low_uncertainty_targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in low_uncertainty_targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    low_uncertainty_targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
                gt_uncertainties = low_uncertainty_targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                proposals_per_image.set("gt_uncertainties", gt_uncertainties)

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)
        if event_storage == True:
            storage = get_event_storage()
            storage.put_scalar("roi_head/num_reg_target_fg_samples", np.mean(num_fg_samples))
            storage.put_scalar("roi_head/num_reg_target_bg_samples", np.mean(num_bg_samples))
        return proposals_with_gt


class MoCoROIHeadsPseudoLab(StandardROIHeads):
    @configurable
    def __init__(self,*,contrastive_feature_dim,**kwargs):
        super().__init__(**kwargs)
        self.contrastive_feature_dim = contrastive_feature_dim
        self.box_projector = ContrastiveHead(dim_in=self.box_head.output_shape.channels, feat_dim=self.contrastive_feature_dim)
        self.feature_noise_layer = nn.Sequential(
            nn.Dropout2d(p=1/64),
            DropBlock2D(block_size=2, drop_prob=1/64)
        )
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["contrastive_feature_dim"] = cfg.MOCO.CONTRASTIVE_FEATURE_DIM
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = MoCoOutputLayers(cfg, box_head.output_shape)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = MoCoOutputLayers(cfg, box_head.output_shape, "FocalLoss")
        else:
            raise ValueError("Unknown ROI head loss.")

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances],  event_storage:bool = True
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )

            # iou값 저장하자
            if match_quality_matrix.numel() == 0:
                matched_vals = torch.zeros(match_quality_matrix.shape[1]).to(match_quality_matrix.device)
            else:
                matched_vals, _ = match_quality_matrix.max(dim=0) # max iou, _

            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if not proposals_per_image.has('iou'):
                proposals_per_image.set('iou', matched_vals[sampled_idxs])

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
                    if (trg_name == 'scores') and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
                if not proposals_per_image.has('scores'):
                    proposals_per_image.set('scores', targets_per_image.gt_boxes.tensor.new_zeros(len(sampled_idxs)))

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        if event_storage == True:
            storage = get_event_storage()
            storage.put_scalar(
                "roi_head/num_target_fg_samples", np.mean(num_fg_samples)
            )
            storage.put_scalar(
                "roi_head/num_target_bg_samples", np.mean(num_bg_samples)
            )

        return proposals_with_gt

##########################
# For MoCOv1 head   #
##########################
# Add contrastive branch to standardroihead module
@ROI_HEADS_REGISTRY.register()
class MoCov1ROIHeadsPseudoLab(MoCoROIHeadsPseudoLab):
    """
        Add contrastive head
    """
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
        split=None, # split output if the input is aggregated 
        noise=False, # insert feature noise to feature map 
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            proposals = self.label_and_sample_proposals(
                proposals, targets
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
            if branch == "contrastive_label" or branch == "contrastive_unlabel":
                projected_features, target_box_features, sampled_proposals, losses = self._forward_box(
                    features=features, 
                    proposals=proposals, 
                    compute_loss=compute_loss, 
                    compute_val_loss=compute_val_loss, 
                    branch=branch,
                    noise=noise,
                    split=split
                )
                return projected_features, target_box_features, sampled_proposals, losses

            else:
                losses, _ = self._forward_box(
                    features=features, 
                    proposals=proposals, 
                    compute_loss=compute_loss, 
                    compute_val_loss=compute_val_loss, 
                    branch=branch
                )
                return proposals, losses
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )
            return pred_instances, predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
        noise=False,
        split=None
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        batch_size = features[0].shape[0]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals]) 

        # generate target cont feature via teacher box head
        target_box_features = box_features.clone().detach()

        # insert feature noise to sources
        # insert noise to pooled feature map
        # To follow the proposal learning paper
        # dropout and dropblock is implemented
        if noise is True:
            box_features = self.feature_noise_layer(box_features)
        
        box_features = self.box_head(box_features)
    
        if (self.training and compute_loss) or compute_val_loss:  # apply if training loss or val loss
            # compute detection loss only for strong aug
            if split is not None:
                split_box_features = box_features.view(batch_size, -1, box_features.shape[-1])[:split] # only strong aug
                split_box_features = split_box_features.view(-1, box_features.shape[-1])

                predictions = self.box_predictor(split_box_features)
                losses = self.box_predictor.losses(predictions, proposals[:split])
            else: 
                predictions = self.box_predictor(box_features)
                losses = self.box_predictor.losses(predictions, proposals)

            # for contrastive loss
            if branch == "contrastive_label" or branch == "contrastive_unlabel":
                box_projected_features = self.box_projector(box_features)
                del box_features
                # label_data_k -> cont features, noise cont features
                # label_data_q -> cont features, noise cont features, cls loss, reg loss

                return box_projected_features, target_box_features, proposals, losses
            
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        
        else:
            predictions = self.box_predictor(box_features) # predict reg, cls
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions
##########################
# For Mocov2   #
##########################

# Add contrastive branch to standardroihead module
@ROI_HEADS_REGISTRY.register()
class MoCov2ROIHeadsPseudoLab(MoCoROIHeadsPseudoLab):
    """
        Add contrastive head
    """
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="", # contrastive일때만 contrastive loss 계산하도록 할 수 있나?
        compute_val_loss=False,
        queue=None,
        queue_label=None,
        temperature=None,
        noise=False, # insert feature noise to feature map 
        split=None
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            proposals = self.label_and_sample_proposals(
                proposals, targets
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        # compute loss(for student)
        if (self.training and compute_loss) or compute_val_loss:
            losses, _ = self._forward_box(
                features, 
                proposals, 
                compute_loss, 
                compute_val_loss, 
                branch,
                queue,
                queue_label,
                temperature,
                noise,
                split
            )
            return proposals, losses

        # generate pseudo label (for teacher) or inference
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )

            return pred_instances, predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
        queue=None,
        queue_label=None,
        temperature=None,
        noise=False,
        split=None
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        # M, C, output_size, output_size 
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # insert noise to pooled feature map
        # To follow the proposal learning paper
        # dropout and dropblock is implemented
        if noise is True:
            box_features = self.feature_noise_layer(box_features)

        box_features = self.box_head(box_features) # list
        # compute regular reg loss, cls loss for query data 
        if split is not None:
            num_box_per_image = proposals[0].proposal_boxes.tensor.shape[0]
            num_box_q = int(num_box_per_image * split)
            box_features_q = box_features[:num_box_q]
            predictions = self.box_predictor(box_features_q) # predict reg, cls
        else:
            predictions = self.box_predictor(box_features)
         
        if (self.training and compute_loss) or compute_val_loss:  # apply if training loss or val loss

            if split is not None:
                proposals_q = proposals[:split]
                losses = self.box_predictor.losses(predictions, proposals_q)
            else:
                losses = self.box_predictor.losses(predictions, proposals)

            # compute contrastive loss 
            if branch == "contrastive_label" or branch == "contrastive_unlabel":
                box_projected_features = self.box_projector(box_features)
                del box_features

                contrastive_loss = self.box_predictor.contrastive_loss(
                    box_projected_features,
                    proposals,
                    queue,
                    queue_label,
                    temperature,
                    branch
                )                
                losses.update({
                    "loss_cont": contrastive_loss,
                })

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions


##########################
#####    For Mocov3   ####
##########################
'''
    teacher network: encoder -> box_head -> projection
    student network: encoder -> box_head -> projection -> prediction 
'''

# Add contrastive branch to standardroihead module
@ROI_HEADS_REGISTRY.register()
class MoCov3ROIHeadsPseudoLab(MoCoROIHeadsPseudoLab):
    """
        Add contrastive head
    """
    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.box_projector = Projector_bn(dim_in=self.box_head.output_shape.channels, feat_dim=self.contrastive_feature_dim)
        self.feat_predictor = Predictor_bn(dim_in=self.contrastive_feature_dim,feat_dim=self.contrastive_feature_dim)
        # follow the proposal learning 
        # applying both? or random select??
        self.feature_noise_layer = nn.Sequential(
            nn.Dropout2d(p=1/64),
            DropBlock2D(block_size=2, drop_prob=1/64)
        )
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="", # contrastive일때만 contrastive loss 계산하도록 할 수 있나?
        compute_val_loss=False,
        queue_obj=None,
        temperature=None,
        noise=False, # insert feature noise to feature map 
        split=None
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            proposals = self.label_and_sample_proposals(
                proposals, targets
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets
        
        # compute loss(for student)
        if (self.training and compute_loss) or compute_val_loss:
            losses, _ = self._forward_box(
                features, 
                proposals, 
                compute_loss, 
                compute_val_loss, 
                branch,
                queue_obj,
                temperature,
                noise,
                split
            )
            return proposals, losses

        # generate pseudo label (for teacher) or inference
        elif branch == "unsup_data_weak_box_features":
            pred_instances, predictions, box_features = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )

            return pred_instances, predictions, box_features

        elif branch == "unsup_data_weak_with_projection":
            pred_instances, predictions, projection_features = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )

            return pred_instances, predictions, projection_features
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )

            return pred_instances, predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
        queue_obj=None,
        temperature=None,
        noise=False,
        split=None # split weak and strong aug data
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        # M, C, output_size, output_size 
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # insert noise to pooled feature map
        # To follow the proposal learning paper
        # dropout and dropblock is implemented
        if noise is True:
            box_features = self.feature_noise_layer(box_features)

        box_features = self.box_head(box_features) # list
        # compute regular reg loss, cls loss for query data 
        if split is not None:
            num_box_per_image = proposals[0].proposal_boxes.tensor.shape[0]
            num_box_q = int(num_box_per_image * split)
            box_features_q = box_features[:num_box_q]
            predictions = self.box_predictor(box_features_q) # predict reg, cls
        else:
            predictions = self.box_predictor(box_features)
         
        if (self.training and compute_loss) or compute_val_loss:  # apply if training loss or val loss

            if split is not None:
                proposals_q = proposals[:split]
                losses = self.box_predictor.losses(predictions, proposals_q)
            else:
                losses = self.box_predictor.losses(predictions, proposals)

            # compute contrastive loss 
            if branch == "contrastive_label" or branch == "contrastive_unlabel":
                box_projected_features = self.box_projector(box_features)
                box_predicted_features = self.feat_predictor(box_projected_features)
                box_predicted_features = F.normalize(box_predicted_features, dim=1)
                del box_features

                contrastive_loss = self.box_predictor.contrastive_loss(
                    box_predicted_features,
                    proposals,
                    queue_obj,
                    temperature,
                    branch
                )                
                losses.update({
                    "loss_cont": contrastive_loss,
                })

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions

        elif branch == "unsup_data_weak_box_features":
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            num_box_per_image = proposals[0].proposal_boxes.tensor.shape[0]
            return pred_instances, predictions, box_features.view(-1, num_box_per_image, box_features.shape[-1])
        
        elif branch == "unsup_data_weak_with_projection":
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            projection_features = self.box_projector(box_features)
            projection_features = F.normalize(projection_features, dim=1)
            num_box_per_image = proposals[0].proposal_boxes.tensor.shape[0]
            return pred_instances, predictions, projection_features.view(-1, num_box_per_image, projection_features.shape[-1])
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals_with_sampled_idx(
        self, proposals: List[Instances], targets: List[Instances]) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        
        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        sampled_idxs_list = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )

            # iou값 저장하자
            if match_quality_matrix.numel() == 0:
                matched_vals = torch.zeros(match_quality_matrix.shape[1]).to(match_quality_matrix.device)
            else:
                matched_vals, _ = match_quality_matrix.max(dim=0) # max iou, _
            
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            sampled_idxs_list.append(sampled_idxs)
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if not proposals_per_image.has('iou'):
                proposals_per_image.set('iou', matched_vals[sampled_idxs])

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt, sampled_idxs_list

@ROI_HEADS_REGISTRY.register()
class ContROIHeadsPseudoLab(StandardROIHeadsPseudoLab):
    """
        Add contrastive head
    """
    @configurable
    def __init__(self, contrastive_feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_feature_dim = contrastive_feature_dim
        self.box_projector = Projector_bn(dim_in=self.box_head.output_shape.channels, feat_dim=self.contrastive_feature_dim)
        self.feat_predictor = Predictor_bn(dim_in=self.contrastive_feature_dim,feat_dim=self.contrastive_feature_dim)
            
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["contrastive_feature_dim"] = cfg.MOCO.CONTRASTIVE_FEATURE_DIM
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        box_encode_type = cfg.MODEL.ROI_BOX_HEAD.BOX_ENCODE_TYPE 
        box_head = ret['box_head']

        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape, box_encode_type)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape, box_encode_type)
        else:
            raise ValueError("Unknown ROI head loss.")
        
        ret['box_predictor'] = box_predictor

        return ret 

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="", # contrastive일때만 contrastive loss 계산하도록 할 수 있나?
        compute_val_loss=False,
        pseudo_label_reg=False,
        uncertainty_threshold=None
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images.tensor, images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            proposals = self.label_and_sample_proposals(
                proposals, targets
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, event_storage=False 
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets
        
        # compute loss(for student)
        if (self.training and compute_loss) or compute_val_loss:
            losses, _ = self._forward_box(
                features, 
                proposals, 
                compute_loss, 
                compute_val_loss, 
                branch,
            )
            return proposals, losses
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )

            return pred_instances, predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        # M, C, output_size, output_size 
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features) # list
        predictions = self.box_predictor(box_features)
        del box_features 
        if (self.training and compute_loss) or compute_val_loss:  # apply if training loss or val loss
            losses = self.box_predictor.losses(predictions, proposals, compute_val_loss=compute_val_loss)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions



##########################
# For uncertainty head   #
##########################

@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLabUncertainty(StandardROIHeadsPseudoLab):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        box_encode_type  = cfg.MODEL.ROI_BOX_HEAD.BOX_ENCODE_TYPE
        assert box_encode_type == "xyxy", 'uncertainty regression should be xyxy'
        box_head = ret['box_head']

        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNUncertaintyOutputLayers(cfg, box_head.output_shape, 'CrossEntropy', box_encode_type)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNUncertaintyOutputLayers(cfg, box_head.output_shape, 'FocalLoss', box_encode_type)
        else:
            raise ValueError("Unknown ROI head loss.")
        
        ret['box_predictor'] = box_predictor

        return ret 

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images.tensor, images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            proposals = self.label_and_sample_proposals(
                proposals, targets
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, event_storage=False
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
            if branch == "val_loss_with_uncertainty":
                losses, pred_instances = self._forward_box(
                    features, proposals, compute_loss, compute_val_loss, branch
                )
                return pred_instances, losses
            else:
                losses, _ = self._forward_box(
                    features, proposals, compute_loss, compute_val_loss, branch
                )
                return proposals, losses
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )

            return pred_instances, predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if (self.training and compute_loss) or compute_val_loss:  # apply if training loss or val loss

            # reg_loss -> uncertainty loss
            losses = self.box_predictor.losses(predictions, proposals, branch, compute_val_loss=compute_val_loss)
            if branch == "val_loss_with_uncertainty":
                pred_instances, _ = self.box_predictor.inference(predictions, proposals)
                return losses, pred_instances

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:

            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], event_storage:bool = True
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )

            # iou값 저장하자
            if match_quality_matrix.numel() == 0:
                matched_vals = torch.zeros(match_quality_matrix.shape[1]).to(match_quality_matrix.device)
            else:
                matched_vals, _ = match_quality_matrix.max(dim=0) # max iou, _

            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if not proposals_per_image.has('iou'):
                proposals_per_image.set('iou', matched_vals[sampled_idxs])

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
                gt_uncertainties = targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                proposals_per_image.set("gt_uncertainties", gt_uncertainties)

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        if event_storage == True:
            storage = get_event_storage()
            storage.put_scalar(
                "roi_head/num_target_fg_samples", np.mean(num_fg_samples)
            )
            storage.put_scalar(
                "roi_head/num_target_bg_samples" , np.mean(num_bg_samples)
            )

        return proposals_with_gt


##########################
#     uncertainty head   #
#           +            #
#     projection head    #
#           +            #
#     prediction head    #
##########################

@ROI_HEADS_REGISTRY.register()
class ContROIHeadsPseudoLabUncertainty(StandardROIHeadsPseudoLab):
    '''
        box_projector: dimension reduction branch
        feat_predictor: predict projected feature
        box_predictor: predict class and bounding box coordinates 
    '''
    @configurable
    def __init__(self, contrastive_feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_feature_dim = contrastive_feature_dim
        self.box_projector = Projector_bn(dim_in=self.box_head.output_shape.channels, feat_dim=self.contrastive_feature_dim)
        self.feat_predictor = Predictor_bn(dim_in=self.contrastive_feature_dim,feat_dim=self.contrastive_feature_dim)
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["contrastive_feature_dim"] = cfg.MOCO.CONTRASTIVE_FEATURE_DIM
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        
        return ret
 
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        box_encode_type = cfg.MODEL.ROI_BOX_HEAD.BOX_ENCODE_TYPE 
        assert box_encode_type == "xyxy", 'uncertainty regression should be xyxy'
        box_head = ret['box_head']

        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNUncertaintyOutputLayers(cfg, box_head.output_shape, 'CrossEntropy', box_encode_type)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNUncertaintyOutputLayers(cfg, box_head.output_shape, 'FocalLoss', box_encode_type)
        else:
            raise ValueError("Unknown ROI head loss.")
        
        ret['box_predictor'] = box_predictor

        return ret 

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
        pseudo_label_reg=False,
        uncertainty_threshold=None
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images.tensor, images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            if pseudo_label_reg:
                # pseudo label matching for classification
                proposals_for_classification = self.label_and_sample_proposals(
                    copy.deepcopy(proposals), targets
                )
                # pseudo label matching for regression
                # remove high uncertain pseudo label
                proposals_for_regression = self.label_and_sample_proposals_uncertainty(
                    copy.deepcopy(proposals), targets, uncertainty_threshold
                )

            else:
                proposals = self.label_and_sample_proposals(
                    proposals, targets
                )

        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, event_storage=False
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
            if (branch == "val_loss") or (branch == "val_loss_without_uncertainty") or (branch == "val_loss_with_uncertainty"):
                losses, pred_instances = self._forward_box(
                    features, proposals, compute_loss, compute_val_loss, branch
                )
                return pred_instances, losses
            # twice forward(for classification and regression)
            # TODO: How to change two forward -> one forward 
            else:
                # pseudo label for classification and regression are different 
                # therefore, compute loss twice
                # if branch is unsup_forward, it is smooth L1
                # if branch is supervised, it is uncertainty loss 
                if pseudo_label_reg:
                    losses_for_classification, _ = self._forward_box(
                        features, proposals_for_classification, compute_loss, compute_val_loss, branch
                    )
                    losses_for_regression, _ = self._forward_box(
                        features, proposals_for_regression, compute_loss, compute_val_loss, branch
                    )
                    losses = {
                        'loss_cls': losses_for_classification['loss_cls'] * 1.0 + losses_for_regression['loss_cls'] * 0.0,
                        'loss_box_reg': losses_for_classification['loss_box_reg'] * 0.0 + losses_for_regression['loss_box_reg'] * 1.0,
                        'loss_box_reg_first_term': losses_for_classification['loss_box_reg_first_term'] * 0.0 + losses_for_regression['loss_box_reg_first_term'] * 1.0,
                        'loss_box_reg_second_term': losses_for_classification['loss_box_reg_second_term'] * 0.0 + losses_for_regression['loss_box_reg_second_term'] * 1.0
                    }
                    return _, losses
                else:
                    losses, _ = self._forward_box(
                        features, proposals, compute_loss, compute_val_loss, branch
                    )
                    return proposals, losses
        # Inference
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )

            return pred_instances, predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if (self.training and compute_loss) or compute_val_loss:  # apply if training loss or val loss
            # reg_loss -> uncertainty loss
            losses = self.box_predictor.losses(predictions, proposals, branch, compute_val_loss=compute_val_loss)
            if branch == "val_loss_with_uncertainty":
                pred_instances, _ = self.box_predictor.inference(predictions, proposals)
                return losses, pred_instances

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], event_storage:bool = True
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )

            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
                gt_uncertainties = targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                proposals_per_image.set("gt_uncertainties", gt_uncertainties)

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        if event_storage == True:
            storage = get_event_storage()
            storage.put_scalar(
                "roi_head/num_target_fg_samples", np.mean(num_fg_samples)
            )
            storage.put_scalar(
                "roi_head/num_target_bg_samples" , np.mean(num_bg_samples)
            )

        return proposals_with_gt

    @torch.no_grad()
    def label_and_sample_proposals_uncertainty(
        self, proposals: List[Instances], targets: List[Instances], 
        uncertainty_threshold, event_storage:bool = True
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes[x.gt_uncertainties.mean(dim=1) < uncertainty_threshold] for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []
        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            valid = (targets_per_image.gt_uncertainties.mean(dim=1) < uncertainty_threshold)
            low_uncertainty_targets_per_image = targets_per_image[valid]

            has_gt = len(low_uncertainty_targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                low_uncertainty_targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # iou값 저장하자
            if match_quality_matrix.numel() == 0:
                matched_vals = torch.zeros(match_quality_matrix.shape[1]).to(match_quality_matrix.device)
            else:
                matched_vals, _ = match_quality_matrix.max(dim=0) # max iou, _
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, low_uncertainty_targets_per_image.gt_classes
            )
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in low_uncertainty_targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(low_uncertainty_targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4)))
                proposals_per_image.gt_boxes = gt_boxes
                gt_uncertainties = low_uncertainty_targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                proposals_per_image.set("gt_uncertainties", gt_uncertainties)

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)
        if event_storage == True:
            storage = get_event_storage()
            storage.put_scalar("roi_head/num_reg_target_fg_samples_", np.mean(num_fg_samples))
            storage.put_scalar("roi_head/num_reg_target_bg_samples" , np.mean(num_bg_samples))
        return proposals_with_gt


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLab_IoU(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        box_encode_type = cfg.MODEL.ROI_BOX_HEAD.BOX_ENCODE_TYPE
        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNIoUOutputLayers(cfg, box_head.output_shape, 'CrossEntropy', box_encode_type)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNIoUOutputLayers(cfg, box_head.output_shape, 'FocalLoss', box_encode_type)
        else:
            raise ValueError("Unknown ROI head loss.")

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
        pseudo_label_reg=False,
        uncertainty_threshold=None,
        training_with_jittering=False,
        jittering_times=None,
        jittering_frac=None
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            if pseudo_label_reg is True:
                # loss for classification
                proposals_for_classification = self.label_and_sample_proposals(
                    copy.deepcopy(proposals), targets
                )
                # Remove pseudo labels that have low iou confidence
                proposals_for_regression = self.label_and_sample_proposals_iou(
                    copy.deepcopy(proposals), targets, iou_threshold=uncertainty_threshold
                )
            # 'supervised learning'
            else:
                proposals = self.label_and_sample_proposals(
                    proposals, targets
                )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, event_storage=False
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        

        if (self.training and compute_loss) or compute_val_loss:
            # training with pseudo label
            if pseudo_label_reg is True:
                del targets
                losses_for_classification, _ = self._forward_box(
                    features, proposals_for_classification, compute_loss, compute_val_loss, branch
                )
                losses_for_regression, _ = self._forward_box(
                    features, proposals_for_regression, compute_loss, compute_val_loss, branch
                )
                losses = {
                    'loss_cls': losses_for_classification['loss_cls'] * 1.0 + losses_for_regression['loss_cls'] * 0.0,
                    'loss_box_reg': losses_for_classification['loss_box_reg'] * 0.0 + losses_for_regression['loss_box_reg'] * 1.0,
                }
                return _, losses
            # Training with gt label(iou branch)
            else:
                losses, _ = self._forward_box(
                    features, 
                    proposals, 
                    compute_loss, 
                    compute_val_loss, 
                    branch, 
                    training_with_jittering=training_with_jittering,
                    jittering_times=jittering_times,
                    jittering_frac=jittering_frac,
                    targets=targets 
                )
                return _, losses
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )

            return pred_instances, predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
        training_with_jittering=False,
        jittering_times=None,
        jittering_frac=None,
        targets=None
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]

        # forward for compute classification branch and regression branch
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        
        # compute IoU-loss for training iou branch
        # Follow the training strategy of IoU-net
        # appling jittering to gt_boxes
        if training_with_jittering and self.training:
            
            gt_boxes = [x.gt_boxes for x in targets]
            gt_classes = [x.gt_classes for x in targets]
            del targets
            image_size = [x.image_size for x in proposals]
            batch_size = len(image_size)

            jittered_gt_boxes, expanded_gt_classes, expanded_gt_boxes = self.box_jittering(
                gt_classes, 
                gt_boxes, 
                image_size, 
                times=jittering_times, 
                frac=jittering_frac
            )
            
            box_features = self.box_pooler(features, jittered_gt_boxes)
            box_features = self.box_head(box_features)
            _, _, pred_iou = self.box_predictor(box_features)
            
            gt_iou = self.compute_iou_labels(jittered_gt_boxes, expanded_gt_boxes)
            expanded_gt_classes = torch.cat(expanded_gt_classes, dim=0)
            loss_iou = self.box_predictor.compute_iou_loss(pred_iou, gt_iou, expanded_gt_classes)

        del box_features
        # training or validation loss
        if (self.training and compute_loss) or compute_val_loss:  # apply if training loss or val loss
            # reg_loss -> uncertainty loss
            losses = self.box_predictor.losses(predictions, proposals, branch, compute_val_loss=compute_val_loss)
            if training_with_jittering:
                losses.update({
                    'loss_iou': loss_iou
                })

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        # Inference 
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], event_storage:bool = True
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4)))
                proposals_per_image.gt_boxes = gt_boxes
                
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        if event_storage == True:
            storage = get_event_storage()
            storage.put_scalar("roi_head/num_target_fg_samples", np.mean(num_fg_samples))
            storage.put_scalar("roi_head/num_target_bg_samples" , np.mean(num_bg_samples))
        return proposals_with_gt

    @torch.no_grad()
    def label_and_sample_proposals_iou(
        self, proposals: List[Instances], targets: List[Instances], iou_threshold, event_storage:bool = True
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes[x.localization_confidences > iou_threshold] for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)
        
        proposals_with_gt = []
        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            valid = (targets_per_image.localization_confidences > iou_threshold)
            high_iou_targets_per_image = targets_per_image[valid]

            has_gt = len(high_iou_targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                high_iou_targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, high_iou_targets_per_image.gt_classes
            )
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in high_iou_targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(high_iou_targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4)))
                proposals_per_image.gt_boxes = gt_boxes
            
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)
        if event_storage == True:
            storage = get_event_storage()
            storage.put_scalar("roi_head/num_reg_target_fg_samples_", np.mean(num_fg_samples))
            storage.put_scalar("roi_head/num_reg_target_bg_samples" , np.mean(num_bg_samples))
        return proposals_with_gt
   
    @torch.no_grad()
    def compute_iou_labels(self, proposals, gt_boxes, mean=0.5, std=0.5):
        gt_ious = []
        for proposals_per_image, targets_per_image in zip(proposals, gt_boxes):
            # already 1:1 matched 
            matched_vals = matched_boxlist_iou(
                targets_per_image, proposals_per_image
            ) 
            gt_ious.append(matched_vals)
            
        gt_ious = torch.cat(gt_ious, dim=0)
        gt_ious = (gt_ious - mean) / std # normalize gt_ious [0:1]
        return gt_ious

    # Based soft teacher 
    # https://github.com/microsoft/SoftTeacher/blob/main/configs/soft_teacher/base.py
    # translation and rescaling
    # Follow the 3DIoUMatch 
    def box_jittering(self, gt_classes, gt_boxes, images_size, times=4, frac=0.01):
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
            new_box = box.tensor.clone()[None, ...].expand(times + 1, box.tensor.shape[0], -1).clone()
            new_box[1:,:,:4] = new_box[1:,:,:4] + offset

            # (x1,y1,x2,y2)
            new_box[:,:,0] = torch.clamp(new_box[:,:,0], min=0.0)
            new_box[:,:,1] = torch.clamp(new_box[:,:,1], min=0.0)
            new_box[:,:,2] = torch.clamp(new_box[:,:,2], max=image_size[1]) # image width
            new_box[:,:,3] = torch.clamp(new_box[:,:,3], max=image_size[0]) # image height

            return Boxes(new_box.reshape(-1,4))

        def _aug_single_class(box_class):
            new_class = box_class.clone()[None, ...].expand(times + 1, box_class.shape[0]).reshape(-1)
            return new_class 

        def _expand_box(box):
            new_box = box.tensor.clone()[None, ...].expand(times + 1, box.tensor.shape[0], -1)
            return Boxes(new_box.reshape(-1,4))

        jittered_gt_boxes = [_aug_single(box, image_size) for box, image_size in zip(gt_boxes, images_size)]
        expanded_gt_classes = [_aug_single_class(box_class) for box_class in gt_classes]
        expanded_gt_boxes = [_expand_box(gt_box) for gt_box in gt_boxes]

        return jittered_gt_boxes, expanded_gt_classes, expanded_gt_boxes