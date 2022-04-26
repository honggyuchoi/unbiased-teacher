# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch
import torch.nn as nn
from ubteacher.utils import FeatureQueue, ClasswiseFeatureQueue

class EnsembleTSModel(nn.Module):
    def __init__(self, modelTeacher, modelStudent):
        super(EnsembleTSModel, self).__init__()

        if isinstance(modelTeacher, (DistributedDataParallel, DataParallel)):
            modelTeacher = modelTeacher.module
        if isinstance(modelStudent, (DistributedDataParallel, DataParallel)):
            modelStudent = modelStudent.module

        self.modelTeacher = modelTeacher
        self.modelStudent = modelStudent

        

    def _init_queue(self, cfg, classwise_queue=False):
        if classwise_queue is False:
            self.feat_queue = FeatureQueue(cfg)
            # self.register_buffer('queue', torch.randn(queue_size, contrastive_feature_dim))
            # self.register_buffer('queue_label', torch.empty(queue_size).fill_(-1).long())
            # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            # self.register_buffer('cycles', torch.zeros(1))

        else: # classwise queue
            self.feat_queue = ClasswiseFeatureQueue(cfg)
            # self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

            # self.max_queue_size = queue_size // self.num_classes
            # self.register_buffer('queue', torch.randn(self.num_classes, max_queue_size, contrastive_feature_dim))
            # self.register_buffer("queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
            # self.register_buffer('cycles', torch.zeros(self.num_classes))
            # self.register_buffer('queue_score', torch.empty(queue_size, max_queue_size).fill_(-1).float())

class MoCov1TSModel(nn.Module):
    def __init__(self, ROIHeadTeacher, modelStudent):
        super(EnsembleTSModel, self).__init__()

        if isinstance(ROIHeadTeacher, (DistributedDataParallel, DataParallel)):
            modelTeacher = ROIHeadTeacher.module
        if isinstance(modelStudent, (DistributedDataParallel, DataParallel)):
            modelStudent = modelStudent.module

        self.ROIHeadTeacher = ROIHeadTeacher
        self.modelStudent = modelStudent


    def _init_queue(self, queue_size, contrastive_feature_dim, classwise_queue=False):
        if classwise_queue is False:
            self.register_buffer('queue', torch.randn(queue_size, contrastive_feature_dim))
            self.register_buffer('queue_label', torch.empty(queue_size).fill_(-1).long())
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer('cycles', torch.zeros(1))
        else:
            raise NotImplementedError
            # self.register_buffer('queue', torch.randn(queue_size, contrastive_feature_dim))
            # self.register_buffer('queue_label', torch.empty(queue_size).fill_(-1).long())
            # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            # self.register_buffer('cycles', torch.zeros(1))