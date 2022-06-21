#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import (
    UBTeacherTrainer, 
    BaselineTrainer
)
from ubteacher.engine.mocov2trainer import MoCov2Trainer
from ubteacher.engine.mocov1trainer import MoCov1Trainer
from ubteacher.engine.mocov3trainer import MoCov3Trainer
from ubteacher.engine.mocov3trainer_two_strong_aug import MoCov3Trainer_two_strong_aug
from ubteacher.engine.trainer_class_aware_cont import Trainer_class_aware_cont 
from ubteacher.engine.trainer_uncertainty import Trainer_uncertainty
from ubteacher.engine.trainer_cont_uncertainty import Trainer_cont_uncertainty
from ubteacher.engine.trainer_jittered_box_uncertainty import Trainer_jittered_box_uncertainty
from ubteacher.engine.trainer_predicted_iou import Trainer_predicted_IoU

# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import (
    TwoStagePseudoLabGeneralizedRCNN,
    TwoStagePseudoLabGeneralizedRCNN_MoCo,
    TwoStagePseudoLabGeneralizedRCNN_MoCov1,
    TwoStagePseudoLabGeneralizedRCNN_Uncertainty,
    TwoStagePseudoLabGeneralizedRCNN_cont,
)
from ubteacher.modeling.proposal_generator.rpn import (
    PseudoLabRPN,
    PseudoLabRPN_IoU
)
from ubteacher.modeling.roi_heads.roi_heads import (
    StandardROIHeadsPseudoLab,
    ContROIHeadsPseudoLab,
    StandardROIHeadsPseudoLabUncertainty,
    ContROIHeadsPseudoLabUncertainty,
    StandardROIHeadsPseudoLab_IoU
)

import ubteacher.data.datasets.builtin

from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    # teacher - student only roi_heads
    elif cfg.SEMISUPNET.Trainer == "mocov1":
        Trainer = MoCov1Trainer
    # teacher - student backbone to roi_heads 
    elif cfg.SEMISUPNET.Trainer == "mocov2":
        Trainer = MoCov2Trainer
    elif cfg.SEMISUPNET.Trainer == "mocov1":
        Trainer = MoCov1Trainer 
    elif cfg.SEMISUPNET.Trainer == "mocov3":
        Trainer = MoCov3Trainer
    elif cfg.SEMISUPNET.Trainer == "mocov3_two_strong_aug":
        Trainer = MoCov3Trainer_two_strong_aug
    elif cfg.SEMISUPNET.Trainer == "class_aware_cont":
        Trainer = Trainer_class_aware_cont
    elif cfg.SEMISUPNET.Trainer == "uncertainty":
        Trainer = Trainer_uncertainty
    elif cfg.SEMISUPNET.Trainer == "cont_uncertainty":
        Trainer = Trainer_cont_uncertainty
    elif cfg.SEMISUPNET.Trainer == "soft_teacher":
        Trainer = Trainer_jittered_box_uncertainty 
    elif cfg.SEMISUPNET.Trainer == "predicted_iou":
        Trainer = Trainer_predicted_IoU
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if (cfg.SEMISUPNET.Trainer == "ubteacher") or \
            (cfg.SEMISUPNET.Trainer == "mocov2") or \
            (cfg.SEMISUPNET.Trainer == "mocov1") or \
            (cfg.SEMISUPNET.Trainer == "mocov3") or \
            (cfg.SEMISUPNET.Trainer == "mocov3_two_strong_aug") or \
            (cfg.SEMISUPNET.Trainer == "class_aware_cont") or \
            (cfg.SEMISUPNET.Trainer == "uncertainty") or \
            (cfg.SEMISUPNET.Trainer == "cont_uncertainty") or \
            (cfg.SEMISUPNET.Trainer == "soft_teacher") or \
            (cfg.SEMISUPNET.Trainer == "predicted_iou"):

            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            if args.student == False:
                res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
            else:
                res = Trainer.test(cfg, ensem_ts_model.modelStudent)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--student', action='store_true')
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
