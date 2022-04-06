# CUDA_VISIBLE_DEVICES=3 python train_net.py \
#       --num-gpus 1 \
#       --config configs/mocov2/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
#        SOLVER.IMG_PER_BATCH_LABEL 6 SOLVER.IMG_PER_BATCH_UNLABEL 6 \
#        SEMISUPNET.BURN_UP_STEP 2000 \
#        SEMISUPNET.UNSUP_LOSS_WEIGHT 4.0 \
#        MOCO.CONTRASTIVE_LOSS_WEIGHT 0.5 \
#        OUTPUT_DIR './output/coco_1.0/mocov2_trial_3' \
#        MOCO.QUEUE_SIZE 50000;

# CUDA_VISIBLE_DEVICES=3 python train_net.py \
#       --num-gpus 1 \
#       --config configs/mocov2/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
#        SOLVER.IMG_PER_BATCH_LABEL 6 SOLVER.IMG_PER_BATCH_UNLABEL 6 \
#        SEMISUPNET.BURN_UP_STEP 2000 \
#        SEMISUPNET.UNSUP_LOSS_WEIGHT 2.0 \
#        MOCO.CONTRASTIVE_LOSS_WEIGHT 0.1 \
#        OUTPUT_DIR './output/coco_1.0/mocov2_trial_5' \
#        MOCO.QUEUE_SIZE 50000;

# CUDA_VISIBLE_DEVICES=1 python train_net.py \
#       --num-gpus 1 \
#       --config configs/mocov2/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
#       --resume \
#        SOLVER.IMG_PER_BATCH_LABEL 6 SOLVER.IMG_PER_BATCH_UNLABEL 6 \
#        SEMISUPNET.BURN_UP_STEP 2000 \
#        SEMISUPNET.UNSUP_LOSS_WEIGHT 2.0 \
#        MOCO.CONTRASTIVE_LOSS_WEIGHT 0.1 \
#        MOCO.QUEUE_UPDATE_LABEL_WITH_BACKGROUND False \
#        OUTPUT_DIR './output/coco_1.0/mocov2_trial_6_no_background' \
#        MOCO.QUEUE_SIZE 50000;

# ubteacher와 같은 setting, moco_loss 수정(num_proposal로 normalized), 
CUDA_VISIBLE_DEVICES=2 python train_net.py \
      --num-gpus 1 \
      --resume \
      --config configs/mocov2/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 6 SOLVER.IMG_PER_BATCH_UNLABEL 6 \
       SEMISUPNET.BURN_UP_STEP 3000 \
       SEMISUPNET.UNSUP_LOSS_WEIGHT 2.0 \
       MODEL.ROI_HEADS.LOSS "FocalLoss" \
       MOCO.CONTRASTIVE_LOSS_WEIGHT 0.1 \
       OUTPUT_DIR './output/coco_1.0/mocov2_trial_7_focal_loss' \
       MOCO.QUEUE_SIZE 50000;