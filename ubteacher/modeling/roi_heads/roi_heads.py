# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
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
    MoCoOutputLayers
)
from dropblock import DropBlock2D


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
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            # proposals = self.label_and_sample_proposals(
            #     proposals, targets, branch=branch
            # )
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
                proposals, targets
            )
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
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
# For uncertainty head   #
##########################

@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLabUncertainty(StandardROIHeads):
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
            box_predictor = FastRCNNUncertaintyOutputLayers(cfg, box_head.output_shape)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            raise NotImplementedError
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
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
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

        if (
            self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss

            # reg_loss -> uncertainty loss
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
        queue=None,
        queue_label=None,
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
                queue,
                queue_label,
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

        elif branch == "unsup_data_weak_box_features":
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            num_box_per_image = proposals[0].proposal_boxes.tensor.shape[0]
            return pred_instances, predictions, box_features.view(-1, num_box_per_image, box_features.shape[-1])
        
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