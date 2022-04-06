# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Union
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from fvcore.nn import smooth_l1_loss
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    FastRCNNOutputs
)

from detectron2.utils.events import EventStorage


##########################
###### OutputLayers#######
##########################

class MoCoOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape,cls_loss_type="CrossEntropy"):
        super().__init__(cfg, input_shape)
        self.contrastive_loss_version = cfg.MOCO.CONTRASTIVE_LOSS_VERSION
        self.class_score_weight = cfg.MOCO.CLASS_SCORE_WEIGHT # "none", "proposal", "queue", "both"
        self.cls_loss_type = cls_loss_type 
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions

        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)
        
        if self.cls_loss_type == "CrossEntropy":
            # parse box regression outputs
            if len(proposals):
                proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
                assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
                # If "gt_boxes" does not exist, the proposals must be all negative and
                # should not be included in regression loss computation.
                # Here we just use proposal_boxes as an arbitrary placeholder because its
                # value won't be used in self.box_reg_loss().
                gt_boxes = cat(
                    [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                    dim=0,
                )
            else:
                proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

            losses = {
                "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
                "loss_box_reg": self.box_reg_loss(
                    proposal_boxes, gt_boxes, proposal_deltas, gt_classes
                ),
            }
            return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

        elif self.cls_loss_type == "FocalLoss":

            losses = FastRCNNFocalLoss(
                self.box2box_transform,
                scores,
                proposal_deltas,
                proposals,
                self.smooth_l1_beta,
                self.box_reg_loss_type,
                num_classes=self.num_classes,
            ).losses()

            return losses
        
        else:
            raise NotImplementedError

    def contrastive_loss(self, projection_features, projection_labels, queue_features, queue_labels, temperature, branch):
        if self.contrastive_loss_version == 'v1':
            return self.moco_contrastive_loss_v1(projection_features, projection_labels, queue_features, queue_labels, temperature, branch)
        elif self.contrastive_loss_version == 'v2':
            return self.moco_contrastive_loss_v2(projection_features, projection_labels, queue_features, queue_labels, temperature, branch)
        elif self.contrastive_loss_version == 'v2_2':
            return self.moco_contrastive_loss_v2_2(projection_features, projection_labels, queue_features, queue_labels, temperature, branch)
    # 같은 class object끼리도 negative pair로 사용됨 
    def moco_contrastive_loss_v1(self, projection_features, proposals, queue_features, queue_labels, temperature, branch):
        '''
        projection_features: from student model = [number of features, feature_dim]
        proposals: [number of features, 1]
        queue_features: [number of queue instances, feature_dim]
        queue_labels: [number of queue instances]
        '''

        if torch.any(queue_labels == -1):
            return 0
        
        gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        iou = torch.cat([p.iou for p in proposals], dim=0) if len(proposals) else torch.empty(0)

        # gt_bbox와 proposalbox가 0.7이상 겹칠때 contrastive learning에 사용.
        if branch == "contrastive_label":
            select = (iou > 0.7)
        elif branch == "contrastive_unlabel":
            select = (iou > 0.8)
        else:
            raise NotImplementedError

        num_proposals = gt_classes.shape[0]

        gt_classes = gt_classes[select]
        iou = iou[select]
        projection_features = projection_features[select]

        if select.sum() == 0:
            return 0

        match_mask = torch.eq(gt_classes.view(-1,1), queue_labels.view(1,-1).cuda()) # [number of features, number of queue instances]
        num_matched = match_mask.sum(dim=1)

        keep = num_matched != 0
        match_mask = match_mask[keep]

        if keep.sum() == 0:
            return 0

        moco_logits = torch.mm(projection_features[keep], queue_features.cuda().T.clone().detach()) # [number of features, number of queue instances]
        moco_logits /= temperature

        logits_row_max, _ = torch.max(moco_logits, dim=1, keepdim=True)

        moco_logits = moco_logits - logits_row_max.detach()

        # 같은 class인 경우 0을 곱해줘서 분모에서 영향을 미치지 못하도록 함
        # exp전에 해야하나 exp 후에 해야하나
        # exp 후에 하는게 맞는듯
        log_prob = moco_logits - torch.log(torch.exp(moco_logits).sum(dim=1, keepdim=True))

        # 같은 class끼리 positive pair
        loss = -(log_prob * match_mask).sum(dim=1) / match_mask.sum(dim=1) 

        # reweight with iou  
        if self.class_score_weight == "none":
            loss = loss.view(-1)
        elif self.class_score_weight == "proposal":
            raise NotImplementedError 
        elif self.class_score_weight == "queue":
            raise NotImplementedError 
        elif self.class_score_weight == "both":
            raise NotImplementedError 
        else:
            raise NotImplementedError 
        
        return loss.sum() / num_proposals

    # 같은 class object에 대해서 negative pair로 사용하지 않음
    def moco_contrastive_loss_v2(self, projection_features, proposals, queue_features, queue_labels, temperature, branch):
        '''
        projection_features: from student model = [number of features, feature_dim]
        
        queue_features: [number of queue instances, feature_dim]
        queue_labels: [number of queue instances]

        foreground: gt_label과 0.8이상 맞춰져야 contrastive loss
        background: gt_label과 0.4이하로 맞춰지면 backgroun로 
        '''

        if torch.any(queue_labels == -1):
            return 0
        
        gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        iou = torch.cat([p.iou for p in proposals], dim=0) if len(proposals) else torch.empty(0)

        # gt_bbox와 proposalbox가 0.7이상 겹칠때 contrastive learning에 사용.
        if branch == "contrastive_label":
            select = (iou > 0.7)
        elif branch == "contrastive_unlabel":
            select = (iou > 0.8)
        else:
            raise NotImplementedError
        num_proposals = gt_classes.shape[0]

        gt_classes = gt_classes[select]
        iou = iou[select]
        projection_features = projection_features[select]

        if select.sum() == 0:
            return 0

        match_mask = torch.eq(gt_classes.view(-1,1), queue_labels.view(1,-1).cuda()) # [number of features, number of queue instances]
        num_matched = match_mask.sum(dim=1)

        keep = num_matched != 0
        match_mask = match_mask[keep]

        if keep.sum() == 0:
            return 0

        moco_logits = torch.mm(projection_features[keep], queue_features.cuda().T.clone().detach()) # [number of features, number of queue instances]
        moco_logits /= temperature

        logits_row_max, _ = torch.max(moco_logits, dim=1, keepdim=True)

        moco_logits = moco_logits - logits_row_max.detach()

        # 같은 class인 경우 0을 곱해줘서 분모에서 영향을 미치지 못하도록 함
        # exp전에 해야하나 exp 후에 해야하나
        # exp 후에 하는게 맞는듯
        log_prob = moco_logits - torch.log((torch.exp(moco_logits) * (1 - match_mask.float())).sum(dim=1, keepdim=True))

        # 같은 class끼리 positive pair
        loss = -(log_prob * match_mask).sum(dim=1) / match_mask.sum(dim=1) 

        # reweight with iou  
        if self.class_score_weight == "none":
            loss = loss.view(-1)
        elif self.class_score_weight == "proposal":
            raise NotImplementedError 
        elif self.class_score_weight == "queue":
            raise NotImplementedError 
        elif self.class_score_weight == "both":
            raise NotImplementedError 
        else:
            raise NotImplementedError 
        
        return loss.sum() / num_proposals

    # 같은 class object에 대해서 negative pair로 사용하지 않음
    # normalization 다르게 
    # FSCE에서 분모에서 자기 자신만 뺌
    def moco_contrastive_loss_v2_2(self, projection_features, proposals, queue_features, queue_labels, temperature, branch):
        '''
        projection_features: from student model = [number of features, feature_dim]
        
        queue_features: [number of queue instances, feature_dim]
        queue_labels: [number of queue instances]

        foreground: gt_label과 0.8이상 맞춰져야 contrastive loss
        background: gt_label과 0.4이하로 맞춰지면 backgroun로 
        '''

        if torch.any(queue_labels == -1):
            return 0
        
        gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        iou = torch.cat([p.iou for p in proposals], dim=0) if len(proposals) else torch.empty(0)

        # gt_bbox와 proposalbox가 0.7이상 겹칠때 contrastive learning에 사용.
        if branch == "contrastive_label":
            select = (iou > 0.7)
        elif branch == "contrastive_unlabel":
            select = (iou > 0.8)
        else:
            raise NotImplementedError
        num_proposals = gt_classes.shape[0]

        # remove low iou proposal
        gt_classes = gt_classes[select]
        iou = iou[select]
        projection_features = projection_features[select]

        if select.sum() == 0:
            return 0

        match_mask = torch.eq(gt_classes.view(-1,1), queue_labels.view(1,-1).cuda()) # [number of features, number of queue instances]
        num_matched = match_mask.sum(dim=1)

        # remove unmatched proposal
        keep = num_matched != 0
        match_mask = match_mask[keep]

        if keep.sum() == 0:
            return 0

        moco_logits = torch.mm(projection_features[keep], queue_features.cuda().T.clone().detach()) # [number of features, number of queue instances]
        moco_logits /= temperature

        logits_row_max, _ = torch.max(moco_logits, dim=1, keepdim=True)

        moco_logits = moco_logits - logits_row_max.detach()

        # 같은 class인 경우 0을 곱해줘서 분모에서 영향을 미치지 못하도록 함
        # exp전에 해야하나 exp 후에 해야하나
        # exp 후에 하는게 맞는
        log_prob = moco_logits - torch.log((torch.exp(moco_logits) * (1 - match_mask.float())).sum(dim=1, keepdim=True))

        # 같은 class끼리 positive pair
        loss = -(log_prob * match_mask).sum(dim=1) / match_mask.sum(dim=1) 

        # reweight with iou  
        if self.class_score_weight == "none":
            loss = loss.view(-1)
        elif self.class_score_weight == "proposal":
            raise NotImplementedError 
        elif self.class_score_weight == "queue":
            raise NotImplementedError 
        elif self.class_score_weight == "both":
            raise NotImplementedError 
        else:
            raise NotImplementedError 
        
        return loss.mean()

# Follow the Bounding Box Regression with Uncertainty for Accurate Object Detection
# https://github.com/yihui-he/KL-Loss/blob/master/detectron/modeling/fast_rcnn_heads.py
class FastRCNNUncertaintyOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNUncertaintyOutputLayers, self).__init__(cfg, input_shape)

        # Add uncertainty branch
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.bbox_uncertainty = nn.Linear(input_size, self.num_classes * 4)
        nn.init.normal_(self.bbox_uncertainty.weight, std=0.0001)
        nn.init.constant_(self.bbox_uncertainty.bias, 0)

        self.loss_weight = {
            "loss_cls": cfg.MODEL.ROI_HEADS.BBOX_CLS_LOSS_WEIGHT,
            "loss_box_reg": cfg.MODEL.ROI_HEADS.BBOX_REG_UNCERTAINTY_LOSS_WEIGHT
        }
        self.uncertainty_loss_regularization_weight = cfg.MODEL.ROI_HEADS.BBOX_REG_UNCERTAINTY_LOSS_REGULARIZATION_WEIGHT

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        proposal_deltas_uncertainty = self.bbox_uncertainty(x)
        return scores, proposal_deltas, proposal_deltas_uncertainty

    # NegativeLogLikelihoodLoss
    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, pred_deltas_variance, gt_classes, beta=1.0):
        """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
            fg_pred_deltas_variance = pred_deltas_variance[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]
            fg_pred_deltas_variance = pred_deltas_variance.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ] 

        loss_box_reg = self.UncertaintyLoss(
            anchors=[proposal_boxes[fg_inds]],
            box2box_transform=self.box2box_transform,
            pred_anchor_deltas=[fg_pred_deltas.unsqueeze(0)],
            pred_anchor_deltas_variance=[fg_pred_deltas_variance.unsqueeze(0)],
            gt_boxes=[gt_boxes[fg_inds]],
            fg_mask=...,
            beta=beta,
        )
        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
            
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def UncertaintyLoss(
        self,
        anchors: List[Union[Boxes, torch.Tensor]],
        box2box_transform: Box2BoxTransform,
        pred_anchor_deltas: List[torch.Tensor],
        pred_anchor_deltas_variance: List[torch.Tensor], # log(variance**2)
        gt_boxes: List[torch.Tensor],
        fg_mask:torch.Tensor,
        beta: float = 1.0,
    ):
        if isinstance(anchors[0], Boxes):
            anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
        else:
            anchors = torch.cat(anchors)

        gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
      
        loss_box_reg = smooth_l1_loss(
            torch.cat(pred_anchor_deltas, dim=1)[fg_mask],
            gt_anchor_deltas[fg_mask],
            beta=1.0,
            reduction="none",
        )
        pred_anchor_deltas_variance = torch.cat(pred_anchor_deltas_variance, dim=1)[fg_mask]
        first_term = loss_box_reg * torch.exp(-pred_anchor_deltas_variance)
        second_term =  pred_anchor_deltas_variance * self.uncertainty_loss_regularization_weight
        loss_box_reg = first_term + second_term

        return loss_box_reg.sum()

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.
            bbox_uncertainty: uncertainty from uncertainty branch(log(variacne^2))
        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas, proposal_deltas_uncertainty = predictions

        # parse classification outputs
        gt_classes = (
            torch.cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = torch.cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = torch.cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, proposal_deltas_uncertainty, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        scores, proposal_deltas, proposal_deltas_uncertainty = predictions

        boxes = self.predict_boxes((scores, proposal_deltas), proposals)
        scores = self.predict_probs((scores, proposal_deltas), proposals)
        uncertainties = self.predict_uncertainty(proposal_deltas_uncertainty, proposals)

        image_shapes = [x.image_size for x in proposals]
        return self.fast_rcnn_inference(
            boxes,
            scores,
            uncertainties,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_uncertainty(self, proposal_deltas_uncertainty, proposals):
        if not len(proposals):
            return []
        num_prop_per_image = [len(p) for p in proposals]
        proposal_deltas_uncertainty = torch.sqrt(torch.exp(proposal_deltas_uncertainty)) # log(var^2) -> var
        return proposal_deltas_uncertainty.split(num_prop_per_image)

    def fast_rcnn_inference(
        self,
        boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        uncertainties,
        image_shapes: List[Tuple[int, int]],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
    ):
    
        result_per_image = [
            self.fast_rcnn_inference_single_image(
                boxes_per_image, scores_per_image, uncertainties_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
            )
            for scores_per_image, boxes_per_image, uncertainties_per_image, image_shape in zip(scores, boxes, uncertainties, image_shapes)
        ]
        return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

    def fast_rcnn_inference_single_image(
        self,
        boxes,
        scores,
        uncertainties, 
        image_shape: Tuple[int, int],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
    ):
        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            uncertainties = uncertainties[valid_mask]

        scores = scores[:, :-1]
        num_bbox_reg_classes = boxes.shape[1] // 4
        # Convert to Boxes to use the `clip` function ...
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

        uncertainties = uncertainties.view(-1, num_bbox_reg_classes, 4)

        # 1. Filter results based on detection scores. It can make NMS more efficient
        #    by filtering out low-confidence detections.
        filter_mask = scores > score_thresh  # R x K
        # R' x 2. First column contains indices of the R predictions;
        # Second column contains indices of classes.
        filter_inds = filter_mask.nonzero()
        if num_bbox_reg_classes == 1:
            boxes = boxes[filter_inds[:, 0], 0]
            uncertainties = uncertainties[filter_inds[:, 0], 0]
        else:
            boxes = boxes[filter_mask]
            uncertainties = uncertainties[filter_mask]
        scores = scores[filter_mask]

        # 2. Apply NMS for each class independently.
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
        uncertainties = uncertainties[keep]

        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = filter_inds[:, 1]
        result.uncertainties = uncertainties
        return result, filter_inds[:, 0]

# focal loss
class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions

        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        losses = FastRCNNFocalLoss(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            num_classes=self.num_classes,
        ).losses()

        return losses


##########################
######### loss ###########
##########################
class FastRCNNFocalLoss(FastRCNNOutputs):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        num_classes=80,
    ):
        super(FastRCNNFocalLoss, self).__init__(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
            box_reg_loss_type,
        )
        self.num_classes = num_classes

    def losses(self):
        return {
            "loss_cls": self.comput_focal_loss(),
            "loss_box_reg": self.box_reg_loss(),
        }

    def comput_focal_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            FC_loss = FocalLoss(
                gamma=1.5,
                num_classes=self.num_classes,
            )
            total_loss = FC_loss(input=self.pred_class_logits, target=self.gt_classes)
            total_loss = total_loss / self.gt_classes.shape[0]

            return total_loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target):
        # focal loss
        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.sum()
