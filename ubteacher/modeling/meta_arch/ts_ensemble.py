# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch
import torch.nn as nn


class EnsembleTSModel(nn.Module):
    def __init__(self, modelTeacher, modelStudent):
        super(EnsembleTSModel, self).__init__()

        if isinstance(modelTeacher, (DistributedDataParallel, DataParallel)):
            modelTeacher = modelTeacher.module
        if isinstance(modelStudent, (DistributedDataParallel, DataParallel)):
            modelStudent = modelStudent.module

        self.modelTeacher = modelTeacher
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