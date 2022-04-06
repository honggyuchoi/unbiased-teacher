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


class SSPL(nn.module):
    def __init__(self, dim_in=1024):
        super().__init__()
        self.contrastive_head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
        )
        self.proposal_predict_head = nn.Sequntial([
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, 4),
            nn.Sigmoid()
        ])

        for layer in self.contrastive_head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

        for layer in self.proposal_predict_head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)



    def forward(self, original_box_features, noise_box_feature_list, proposal_boxes, image_size):
        num_noise_box_features = len(noise_box_feature_list)
        noise_box_feature_list.append(original_box_features)
        box_features_cat = torch.cat(noise_box_features_list, dim=0)
        
        feat = self.contrastive_head(box_features_cat)
        projected_features = F.normalize(feat, dim=1)

        

        
        return losses

    def losses(proposals, ):

        return {

        }

    def compute_proposal_location_loss():

        return 

    def compute_contrastive_loss():


        