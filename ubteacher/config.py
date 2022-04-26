# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_ubteacher_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.TEST.VAL_LOSS = True

    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

    _C.SOLVER.IMG_PER_BATCH_LABEL = 1
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
    _C.SOLVER.FACTOR_LIST = (1,)
    _C.SOLVER.CHECKPOINT_PERIOD = 10000

    _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    _C.DATASETS.CROSS_DATASET = False
    _C.TEST.EVALUATOR = "COCOeval"

    _C.SEMISUPNET = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128

    # Semi-supervised training
    _C.SEMISUPNET.Trainer = "ubteacher"
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    _C.SEMISUPNET.BURN_UP_STEP = 12000
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"
    _C.SEMISUPNET.BURN_UP_WITH_CONTRASTIVE = False

    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True

    # contrastive learning parameters
    _C.MOCO = CN()
    _C.MOCO.QUEUE_SIZE = 50000
    _C.MOCO.CONTRASTIVE_FEATURE_DIM = 128
    _C.MOCO.TEMPERATURE = 0.1
    _C.MOCO.LABELED_CONTRASTIVE_IOU_THRES = 0.7
    _C.MOCO.PSEUDO_LABEL_JITTERING = True
    _C.MOCO.CONTRASTIVE_LOSS_VERSION = 'v2'
    _C.MOCO.CONTRASTIVE_LOSS_WEIGHT = 0.1
    _C.MOCO.CLASSWISE_QUEUE = False
    _C.MOCO.QUEUE_UPDATE_LABEL_WITH_BACKGROUND = True
    _C.MOCO.CLASS_SCORE_WEIGHT = "none"

    # RoI heads regression uncertainty
    _C.MODEL.ROI_HEADS.BBOX_CLS_LOSS_WEIGHT = 1.0
    _C.MODEL.ROI_HEADS.BBOX_REG_UNCERTAINTY_LOSS_WEIGHT = 0.1
    _C.MODEL.ROI_HEADS.BBOX_REG_UNCERTAINTY_LOSS_REGULARIZATION_WEIGHT = 0.5
    _C.MODEL.ROI_HEADS.UNCERTAINTY_START_ITER = 0

    _C.MODEL.ROI_BOX_HEAD.BOX_ENCODE_TYPE = "xywh"

    # For visualization 
    _C.VISUALIZATION = CN()
    _C.VISUALIZATION.TYPE = 'none' # 'none', 'rpn', 'roi', 'tsne'
    _C.VISUALIZATION.RPN_THRESHOLD = 0.9
    _C.VISUALIZATION.ROI_THRESHOLD = 0.9
    _C.VISUALIZATION.GT_BOXES = False

