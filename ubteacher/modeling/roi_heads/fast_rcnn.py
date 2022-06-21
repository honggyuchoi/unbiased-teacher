# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Union
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.structures import Boxes, Instances, pairwise_iou 
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from fvcore.nn import smooth_l1_loss
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    FastRCNNOutputs
)

from detectron2.utils.events import EventStorage
from detectron2.utils.events import get_event_storage

from ubteacher.modeling.box_regression import Box2BoxTransform_XYXY


# Compute supervised contrastive loss
def supcon_loss(feature_q, label_q, score_q, feature_k, label_k, score_k, temperature):
    match_mask = torch.eq(label_q.view(-1,1), label_k.view(1,-1).cuda())
    num_matched = match_mask.sum(dim=1)

    keep = num_matched != 0
    match_mask = match_mask[keep]

    if keep.sum() == 0:
        return 0

    cos_sim = torch.mm(feature_q[keep], feature_k.cuda().T.detach())
    cos_sim /= temperature # 0.07, 0.1, 0.2

    logits_row_max, _ = torch.max(cos_sim, dim=1, keepdim=True)

    cos_sim = cos_sim - logits_row_max.detach()

    # 현재 positive pair만 분모에서 제거
    log_prob = cos_sim - torch.log(torch.exp(cos_sim).sum(dim=1, keepdim=True) - torch.exp(cos_sim))
    
    weight = torch.mm(score_q[keep].view(-1,1).detach(), score_k[keep].view(1,-1).cuda().detach())
    
    weighted_log_prob = log_prob * weight 
    # 같은 class끼리 positive pair
    loss = -(weighted_log_prob * match_mask).sum(dim=1) / match_mask.sum(dim=1) 

    return loss.mean()

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
        self.classwise_queue_warmup_iter = cfg.MOCO.CLASSWISE_QUEUE_WARMUP 

        self.cont_iou_threshold = cfg.MOCO.CONT_IOU_THRESHOLD
        self.cont_class_threshold = cfg.MOCO.CONT_CLASS_THRESHOLD

        self.classwise_queue = cfg.MOCO.CLASSWISE_QUEUE

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
            return losses

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

    def contrastive_loss(self, projection_features, projection_labels, queue_obj, temperature, branch, queue_scores=None):
        # if queue_features.dim() == 3:
        #     queue = queue_features.view(-1)
        #     queue_labels = torch.arange(queue_features.shape[0]).view(-1,1).expand(-1,queue_features.shape[1]).reshape(-1)

        if self.contrastive_loss_version == 'v1':
            return self.moco_contrastive_loss_v1(projection_features, projection_labels, queue_obj, temperature, branch)
        elif self.contrastive_loss_version == 'v2':
            return self.moco_contrastive_loss_v2(projection_features, projection_labels, queue_obj, temperature, branch)
        elif self.contrastive_loss_version == 'v2_2':
            return self.moco_contrastive_loss_v2_2(projection_features, projection_labels, queue_obj, temperature, branch)
        elif self. contrastive_loss_version == 'l2_loss':
            return self.feature_l2_loss(projection_features, projection_labels, queue_obj, branch)
        else:
            raise NotImplementedError    
    # 같은 class object끼리도 negative pair로 사용됨 
    def moco_contrastive_loss_v1(self, projection_features, proposals, queue_obj, temperature, branch):
        '''
        projection_features: from student model = [number of features, feature_dim]
        proposals: [number of features, 1]
        queue_features: [number of queue instances, feature_dim]
        queue_labels: [number of queue instances]
        '''
        # class agnoistic queue
        if self.classwise_queue is False:
            queue_features = queue_obj.queue.clone()
            queue_labels = queue_obj.queue_label.clone()
            queue_scores = queue_obj.queue_score.clone()
            if torch.any(queue_labels == -1):
                return 0

        # classwise queue
        else:
            storage = get_event_storage()
            queue_labels = queue_obj.get_queue_label().clone()
            if (torch.any(queue_labels == -1)) and (storage.iter < self.classwise_queue_warmup_iter):
                return 0
            queue_features = queue_obj.get_queue().clone()
            queue_scores = queue_obj.get_queue_score().clone()

            # remove -1 class 
            select = (queue_labels != -1)
            queue_features = queue_features[select]
            queue_labels = queue_labels[select]
            queue_scores = queue_scores[select]

        gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        iou = torch.cat([p.iou for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        gt_scores = torch.cat([p.scores for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        # gt_bbox와 proposalbox가 0.7이상 겹칠때 contrastive learning에 사용.
        select = (iou > self.cont_iou_threshold) & (gt_scores > self.cont_class_threshold)

        num_proposals = gt_classes.shape[0]

        selected_gt_classes = gt_classes[select]
        selected_gt_scores = gt_scores[select]
        selected_projection_features = projection_features[select]

        if select.sum() == 0:
            return 0

        match_mask = torch.eq(selected_gt_classes.view(-1,1), queue_labels.view(1,-1).cuda()) # [number of features, number of queue instances]
        num_matched = match_mask.sum(dim=1)

        keep = num_matched != 0
        if keep.sum() == 0:
            return 0

        match_mask = match_mask[keep]
        selected_projection_features = selected_projection_features[keep]
        selected_gt_scores = selected_gt_scores[keep]
        

        moco_logits = torch.mm(selected_projection_features, queue_features.cuda().T.detach()) # [number of features, number of queue instances]
        moco_logits /= temperature

        logits_row_max, _ = torch.max(moco_logits, dim=1, keepdim=True)

        moco_logits = moco_logits - logits_row_max.detach()

        # 현재 positive pair만 분모에서 제거(supcont 논문을 따름)
        log_prob = moco_logits - torch.log(torch.exp(moco_logits).sum(dim=1, keepdim=True) - torch.exp(moco_logits))
        
        weight = torch.mm(selected_gt_scores.view(-1,1).detach(), queue_scores.view(1,-1).cuda().detach())
        
        weighted_log_prob = log_prob * weight 
        # 같은 class끼리 positive pair
        loss = -(weighted_log_prob * match_mask).sum(dim=1) / match_mask.sum(dim=1) 

        return loss.mean()

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
    # normalization v2와 다르게 
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
        # what is temperature?? smaller temperature? larger temperature 
        # 0.07 0.1 (smaller temperature make more peaky)
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

    def feature_l2_loss(self, prediction_features, proposals, queue_features, queue_labels, branch):
        storage = get_event_storage()
        if (torch.any(queue_labels == -1)) and (storage.iter < self.classwise_queue_warmup_iter):
            return 0
        # remove -1 class 
        select = (queue_labels != -1)
        queue_features = queue_features[select]
        queue_labels = queue_labels[select]

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
        prediction_features = prediction_features[select]

        if select.sum() == 0:
            return 0

        match_mask = torch.eq(gt_classes.view(-1,1), queue_labels.view(1,-1).cuda()) # [number of features, number of queue instances]
        num_matched = match_mask.sum(dim=1)

        # remove unmatched proposal
        keep = num_matched != 0
        match_mask = match_mask[keep]

        if keep.sum() == 0:
            return 0

        cos_sim = torch.mm(prediction_features[keep], queue_features.cuda().T.clone().detach()) # [number of features, number of queue instances]
        
        loss = -2 * (cos_sim * match_mask).sum(dim=1) / (match_mask.sum(dim=1) + 1e-6)

        return loss.mean()

    
# Follow the Bounding Box Regression with Uncertainty for Accurate Object Detection
# https://github.com/yihui-he/KL-Loss/blob/master/detectron/modeling/fast_rcnn_heads.py
class FastRCNNUncertaintyOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape, cls_loss_type="FocalLoss", box_encode_type="xyxy"):
        super(FastRCNNUncertaintyOutputLayers, self).__init__(cfg, input_shape)
        # Add uncertainty branch
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # class
        self.bbox_uncertainty = nn.Linear(input_size, self.num_classes * 4)
        # Follow the bounding box uncertainty paper 
        nn.init.normal_(self.bbox_uncertainty.weight, std=0.0001)
        nn.init.constant_(self.bbox_uncertainty.bias, 0)
        self.loss_weight = {
            "loss_cls": 1.0,
            "loss_box_reg": 1.0,
            "loss_box_reg_first_term": 1.0,
            "loss_box_reg_second_term": 1.0
        }
        # loss_weight for uncertainty loss's second term
        self.uncertainty_loss_regularization_weight = cfg.MODEL.ROI_HEADS.BBOX_REG_UNCERTAINTY_LOSS_REGULARIZATION_WEIGHT
        self.cls_loss_type = cls_loss_type
        self.uncertainty_start_iter = cfg.MODEL.ROI_HEADS.UNCERTAINTY_START_ITER
        
        if box_encode_type == "xyxy":
            self.box2box_transform = Box2BoxTransform_XYXY(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    def forward(self, x):
        
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        proposal_deltas_uncertainty = self.bbox_uncertainty(x)
        return scores, proposal_deltas, proposal_deltas_uncertainty

    # NegativeLogLikelihoodLoss
    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, pred_deltas_variance, gt_classes, branch, beta=1.0):
        
        storage = get_event_storage()
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if fg_inds.numel() == 0:
            return 0, 0, 0
        
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
            fg_pred_deltas_variance = pred_deltas_variance[fg_inds]
        else:
            # select foreground sample and assigned classes
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]
            fg_pred_deltas_variance = pred_deltas_variance.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ] 

        normalizer = max(gt_classes.numel(), 1.0)
        
        # remove unreliable pseudo label and compute smoothl1 loss
        # how to weighting
        if (branch == "unsup_forward") \
            or (branch=="val_loss_without_uncertainty") \
                or (branch=="val_loss") \
                    or (branch == "supervised_smoothL1"):
            loss = self.SmoothL1Loss(
                anchors=[proposal_boxes[fg_inds]],
                box2box_transform=self.box2box_transform,
                pred_anchor_deltas=[fg_pred_deltas.unsqueeze(0)],
                gt_boxes=[gt_boxes[fg_inds]],
                fg_mask=...,
                beta=beta, 
            )
            return loss / normalizer, loss.detach() / normalizer, 0
       
        elif (branch == "supervised") or (branch=="val_loss_with_uncertainty"):
            first_term, second_term = self.UncertaintyLoss(
                anchors=[proposal_boxes[fg_inds]],
                box2box_transform=self.box2box_transform,
                pred_anchor_deltas=[fg_pred_deltas.unsqueeze(0)],
                pred_anchor_deltas_variance=[fg_pred_deltas_variance.unsqueeze(0)],
                gt_boxes=[gt_boxes[fg_inds]],
                fg_mask=...,
                beta=beta,
            )
            first_term = first_term / normalizer
            second_term = second_term / normalizer
            loss = first_term + second_term
            return loss, first_term.detach(), second_term.detach()  # return 0 if empty
        else:
            raise NotImplementedError

    def SmoothL1Loss(
        self,
        anchors: List[Union[Boxes, torch.Tensor]],
        box2box_transform: Box2BoxTransform,
        pred_anchor_deltas: List[torch.Tensor],
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
        return loss_box_reg.sum()

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
        
        return first_term.sum(), second_term.sum()


    def losses(self, predictions, proposals, branch=None, compute_val_loss=False):
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
        if compute_val_loss is False:
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
            # if branch == "unsup_forward":
            #     gt_uncertainties = torch.cat([p.gt_uncertainties for p in proposals], dim=0).detach()
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)
            # if branch == "unsup_forward":
            #     gt_uncertainties =torch.empty((0, 4), device=proposal_deltas.device)

        if self.cls_loss_type == "CrossEntropy":
            loss_cls = cross_entropy(scores, gt_classes, reduction="mean")

        elif self.cls_loss_type == "FocalLoss":
            loss_cls = FastRCNNFocalLoss(
                self.box2box_transform,
                scores,
                proposal_deltas,
                proposals,
                self.smooth_l1_beta,
                self.box_reg_loss_type,
                num_classes=self.num_classes,
            ).comput_focal_loss()
        else:
            raise NotImplementedError

        
        if (branch == "unsup_forward") or (branch == "supervised_smoothL1"):
            # smooth l1 loss
            loss_reg, first_term, second_term = self.box_reg_loss(
                        proposal_boxes, 
                        gt_boxes, 
                        proposal_deltas, 
                        proposal_deltas_uncertainty, 
                        gt_classes,
                        branch=branch,
                    )

        elif branch == 'supervised':     
            # uncertainty loss
                loss_reg, first_term, second_term = self.box_reg_loss(
                            proposal_boxes, 
                            gt_boxes, 
                            proposal_deltas, 
                            proposal_deltas_uncertainty, 
                            gt_classes, 
                            branch=branch, 
                        )
        elif (branch == 'val_loss_with_uncertainty') or (branch == 'val_loss'):
            with torch.no_grad():
                loss_reg, first_term, second_term = self.box_reg_loss(
                                proposal_boxes, 
                                gt_boxes, 
                                proposal_deltas, 
                                proposal_deltas_uncertainty, 
                                gt_classes, 
                                branch=branch, 
                            )
        elif branch == 'val_loss_without_uncertainty':
            with torch.no_grad():
                loss_reg, first_term, second_term = self.box_reg_loss(
                                proposal_boxes, 
                                gt_boxes, 
                                proposal_deltas, 
                                proposal_deltas_uncertainty, 
                                gt_classes, 
                                branch=branch, 
                            )

        else:
            raise NotImplementedError

        losses = {
                    'loss_cls': loss_cls,
                    "loss_box_reg": loss_reg,
                    "loss_box_reg_first_term": first_term,
                    "loss_box_reg_second_term": second_term, 
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
        uncertainties = self.predict_uncertainty(proposal_deltas_uncertainty, proposals) #value of sigma
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
    # proposal_deltas_uncertainty(model output)(log(sigma^2)) -> sigma 
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
        score_type='classification'
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
        if score_type == "classification":
            keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
        elif score_type == "uncertainty_based_score":
            mean_uncertainty = uncertainty.mean(dim=-1)
            nms_scores = scores / 10 * (mean_uncertainties.view(-1) + 1e-6)
            keep = batched_nms(boxes, nms_scores, filter_inds[:, 1], nms_thresh)
        else:
            raise NotImplementedError

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
    def __init__(self, cfg, input_shape, box_encode_type=None):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        if box_encode_type == "xyxy":
            self.box2box_transform = Box2BoxTransform_XYXY(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)


    def losses(self, predictions, proposals, compute_val_loss=False):
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
        if compute_val_loss is False:
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



# TODO: implement IOU prediction loss 
class FastRCNNIoUOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape, cls_loss_type="FocalLoss",  box_encode_type=None):
        super(FastRCNNIoUOutputLayers, self).__init__(cfg, input_shape)

        # Add uncertainty branch
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # class
        self.bbox_iou = nn.Sequential(
            nn.Linear(input_size, self.num_classes),
            nn.Sigmoid()
        )
        for layer in self.bbox_iou:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.0001)
                nn.init.constant_(layer.bias, 0)   

        self.cls_loss_type = cls_loss_type
        if box_encode_type == "xyxy":
            self.box2box_transform = Box2BoxTransform_XYXY(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)


    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        # iou prediction 
        proposal_ious = self.bbox_iou(x) # N, num_classes
        return scores, proposal_deltas, proposal_ious

    def losses(self, predictions, proposals, branch=None, compute_val_loss=False):
        scores, proposal_deltas, _ = predictions
        # parse classification outputs
        gt_classes = (
            torch.cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        if compute_val_loss is False:
            _log_classification_stats(scores, gt_classes)
        # parse box regression outputs

        if self.cls_loss_type == "CrossEntropy":
            raise NotImplementedError
            #loss_cls = cross_entropy(scores, gt_classes, reduction="mean")

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
        else:
            raise NotImplementedError

        return losses

    # roi - gt label's maximum iou
    # remove background samples
    # remove low iou samples
    def compute_iou_loss(self, pred_ious, gt_ious, gt_classes):
        assert pred_ious.shape[0] == gt_ious.shape[0]
        normalizer = max(gt_classes.numel(), 1.0)
        
        # remove low iou samples(lower than iou=0.5)
        fg_inds = (gt_ious <= 1.0) & (gt_ious>=0.0)
        if fg_inds.sum() == 0.0:
            return torch.tensor(0.0)
        
        fg_pred_ious = pred_ious[fg_inds]
        fg_gt_ious = gt_ious[fg_inds]
        fg_gt_classes = gt_classes[fg_inds]

        fg_gt_classes = fg_gt_classes.unsqueeze(1)
        fg_pred_ious = fg_pred_ious.gather(1, fg_gt_classes)
    
        loss_iou = smooth_l1_loss(
            fg_pred_ious.view(-1),
            fg_gt_ious.view(-1),
            beta=1.0,
            reduction='sum'
        )

        return loss_iou / normalizer 

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
        scores, proposal_deltas, proposal_ious = predictions

        boxes = self.predict_boxes((scores, proposal_deltas), proposals)
        scores = self.predict_probs((scores, proposal_deltas), proposals)
        predict_ious = self.predict_ious(proposal_ious, proposals) #value of sigma
        image_shapes = [x.image_size for x in proposals]
        return self.fast_rcnn_inference(
            boxes,
            scores,
            predict_ious,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )
        
    def predict_ious(self, proposal_ious, proposals):
        if not len(proposals):
            return []
        num_prop_per_image = [len(p) for p in proposals]
        return proposal_ious.split(num_prop_per_image) # B,N,num_classes

    def fast_rcnn_inference(
        self,
        boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        predict_ious,
        image_shapes: List[Tuple[int, int]],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
    ):
    
        result_per_image = [
            self.fast_rcnn_inference_single_image(
                boxes_per_image, scores_per_image, predict_ious_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
            )
            for scores_per_image, boxes_per_image, predict_ious_per_image, image_shape in zip(scores, boxes, predict_ious, image_shapes)
        ]
        return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

    def fast_rcnn_inference_single_image(
        self,
        boxes,
        scores,
        predict_ious, 
        image_shape: Tuple[int, int],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int
    ):
        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            predict_ious = predict_ious[valid_mask]

        scores = scores[:, :-1]
        num_bbox_reg_classes = boxes.shape[1] // 4
        # Convert to Boxes to use the `clip` function ...
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

        # 1. Filter results based on detection scores. It can make NMS more efficient
        #    by filtering out low-confidence detections.
        filter_mask = scores > score_thresh  # R x K
        # R' x 2. First column contains indices of the R predictions;
        # Second column contains indices of classes.
        filter_inds = filter_mask.nonzero()
        if num_bbox_reg_classes == 1:
            boxes = boxes[filter_inds[:, 0], 0]
        else:
            boxes = boxes[filter_mask]
        scores = scores[filter_mask]
        predict_ious = predict_ious[filter_mask]
        
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)

        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
        predict_ious = predict_ious[keep]

        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = filter_inds[:, 1]
        result.predict_ious = predict_ious
        return result, filter_inds[:, 0]