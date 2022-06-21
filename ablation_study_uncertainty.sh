# Soft_teacher supervised 
# for threshold in 0.007 0.008 0.009 0.01
# do
#     CUDA_VISIBLE_DEVICES=0 python train_net.py \
#     --num-gpus 1 \
#     --config configs/soft_teacher/faster_rcnn_R_50_FPN_sup1_run1.yaml \
#     SEMISUPNET.Trainer "soft_teacher" \
#     SEMISUPNET.UNSUP_LOSS_WEIGHT 2.0 \
#     SOLVER.MAX_ITER 45001 \
#     SOLVER.STEPS "44990, 44995" \
#     SOLVER.IMG_PER_BATCH_LABEL 8 \
#     SOLVER.IMG_PER_BATCH_UNLABEL 8 \
#     SEMISUPNET.BURN_UP_STEP 5000 \
#     SOFT_TEACHER.UNCERTAINTY_THRESHOLD ${threshold} \
#     OUTPUT_DIR "./output/ablation_study/soft_teacher_${threshold}" \
#     SEED 1;
# done

# # Predicted_iou supervised
# CUDA_VISIBLE_DEVICES=1 python train_net.py \
# --num-gpus 1 \
# --config configs/predicted_iou/faster_rcnn_R_50_FPN_sup1_run1.yaml \
# SEMISUPNET.Trainer "predicted_iou" \
# SOLVER.MAX_ITER 45001 \
# SOLVER.STEPS "44990, 44995" \
# SOLVER.IMG_PER_BATCH_LABEL 6 \
# SOLVER.IMG_PER_BATCH_UNLABEL 6 \
# SOLVER.CHECKPOINT_PERIOD 1000 \
# SEMISUPNET.BURN_UP_STEP 45001 \
# IOUNET.IOU_THRESHOLD 0.7 \
# IOUNET.TRAINING_WITH_JITTERING True \
# IOUNET.JITTERING_TIMES 30 \
# IOUNET.JITTERING_FRAC 0.2 \
# IOUNET.IOULOSS_WEIGHT 1.0 \
# OUTPUT_DIR './output/ablation_study/predicted_iou_sup_1.0' \
# SEED 1;

# CUDA_VISIBLE_DEVICES=1 python train_net.py \
# --num-gpus 1 \
# --config configs/predicted_iou/faster_rcnn_R_50_FPN_sup1_run1.yaml \
# SEMISUPNET.Trainer "predicted_iou" \
# SOLVER.MAX_ITER 45001 \
# SOLVER.STEPS "44990, 44995" \
# SOLVER.IMG_PER_BATCH_LABEL 6 \
# SOLVER.IMG_PER_BATCH_UNLABEL 6 \
# SOLVER.CHECKPOINT_PERIOD 1000 \
# SEMISUPNET.BURN_UP_STEP 45001 \
# IOUNET.IOU_THRESHOLD 0.7 \
# IOUNET.TRAINING_WITH_JITTERING True \
# IOUNET.JITTERING_TIMES 30 \
# IOUNET.JITTERING_FRAC 0.2 \
# IOUNET.IOULOSS_WEIGHT 3.0 \
# OUTPUT_DIR './output/ablation_study/predicted_iou_sup_3.0' \
# SEED 1;

# CUDA_VISIBLE_DEVICES=1 python train_net.py \
# --num-gpus 1 \
# --config configs/predicted_iou/faster_rcnn_R_50_FPN_sup1_run1.yaml \
# SEMISUPNET.Trainer "predicted_iou" \
# SOLVER.MAX_ITER 45001 \
# SOLVER.STEPS "44990, 44995" \
# SOLVER.IMG_PER_BATCH_LABEL 6 \
# SOLVER.IMG_PER_BATCH_UNLABEL 6 \
# SOLVER.CHECKPOINT_PERIOD 1000 \
# SEMISUPNET.BURN_UP_STEP 45001 \
# IOUNET.IOU_THRESHOLD 0.7 \
# IOUNET.TRAINING_WITH_JITTERING True \
# IOUNET.JITTERING_TIMES 30 \
# IOUNET.JITTERING_FRAC 0.2 \
# IOUNET.IOULOSS_WEIGHT 5.0 \
# OUTPUT_DIR './output/ablation_study/predicted_iou_sup_5.0' \
# SEED 1;

# # Aleatoric_uncertainty
# CUDA_VISIBLE_DEVICES=1 python train_net.py \
# --num-gpus 1 \
# --config configs/uncertainty/voc/voc07_voc12.yaml \
# SEMISUPNET.Trainer "uncertainty" \
# SOLVER.MAX_ITER 45001 \
# SOLVER.STEPS "44990, 44995" \
# SOLVER.IMG_PER_BATCH_LABEL 6 \
# SOLVER.IMG_PER_BATCH_UNLABEL 6 \
# SEMISUPNET.BURN_UP_STEP 5000 \
# MODEL.ROI_HEADS.BBOX_REG_UNCERTAINTY_LOSS_WEIGHT 1.0 \
# MODEL.ROI_BOX_HEAD.BOX_ENCODE_TYPE "xyxy" \
# UNCERTAINTY.THRESHOLD 0.4 \
# UNCERTAINTY.UNLABEL_REG_LOSS_TYPE "uncertainty_threshold" \
# OUTPUT_DIR './output/ablation_study/aleatoric_uncertainty_0.4' \
# SEED 1;

# CUDA_VISIBLE_DEVICES=1 python train_net.py \
# --num-gpus 1 \
# --config configs/uncertainty/voc/voc07_voc12.yaml \
# SEMISUPNET.Trainer "uncertainty" \
# SOLVER.MAX_ITER 45001 \
# SOLVER.STEPS "44990, 44995" \
# SOLVER.IMG_PER_BATCH_LABEL 6 \
# SOLVER.IMG_PER_BATCH_UNLABEL 6 \
# SEMISUPNET.BURN_UP_STEP 5000 \
# MODEL.ROI_HEADS.BBOX_REG_UNCERTAINTY_LOSS_WEIGHT 1.0 \
# MODEL.ROI_BOX_HEAD.BOX_ENCODE_TYPE "xyxy" \
# UNCERTAINTY.THRESHOLD 0.3 \
# UNCERTAINTY.UNLABEL_REG_LOSS_TYPE "uncertainty_threshold" \
# OUTPUT_DIR './output/ablation_study/aleatoric_uncertainty_0.3' \
# SEED 1;

# CUDA_VISIBLE_DEVICES=1 python train_net.py \
# --num-gpus 1 \
# --config configs/uncertainty/voc/voc07_voc12.yaml \
# SEMISUPNET.Trainer "uncertainty" \
# SOLVER.MAX_ITER 45001 \
# SOLVER.STEPS "44990, 44995" \
# SOLVER.IMG_PER_BATCH_LABEL 6 \
# SOLVER.IMG_PER_BATCH_UNLABEL 6 \
# SEMISUPNET.BURN_UP_STEP 5000 \
# MODEL.ROI_HEADS.BBOX_REG_UNCERTAINTY_LOSS_WEIGHT 1.0 \
# MODEL.ROI_BOX_HEAD.BOX_ENCODE_TYPE "xyxy" \
# UNCERTAINTY.THRESHOLD 0.3 \
# UNCERTAINTY.UNLABEL_REG_LOSS_TYPE "uncertainty_threshold" \
# OUTPUT_DIR './output/ablation_study/aleatoric_uncertainty_0.5' \
# SEED 1;

# CUDA_VISIBLE_DEVICES=1 python train_net.py \
# --num-gpus 1 \
# --config configs/uncertainty/voc/voc07_voc12.yaml \
# SEMISUPNET.Trainer "uncertainty" \
# SOLVER.MAX_ITER 45001 \
# SOLVER.STEPS "44990, 44995" \
# SOLVER.IMG_PER_BATCH_LABEL 6 \
# SOLVER.IMG_PER_BATCH_UNLABEL 6 \
# SEMISUPNET.BURN_UP_STEP 5000 \
# MODEL.ROI_HEADS.BBOX_REG_UNCERTAINTY_LOSS_WEIGHT 1.0 \
# MODEL.ROI_BOX_HEAD.BOX_ENCODE_TYPE "xyxy" \
# UNCERTAINTY.THRESHOLD 0.3 \
# UNCERTAINTY.UNLABEL_REG_LOSS_TYPE "uncertainty_threshold" \
# OUTPUT_DIR './output/ablation_study/aleatoric_uncertainty_0.6' \
# SEED 1;
       
# No threshold
# remove contrastive learning
# No threshold
# CUDA_VISIBLE_DEVICES=2,3 python train_net.py \
# --num-gpus 2 \
# --config configs/class_aware_cont/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
# --dist-url 'tcp://0.0.0.0:12344' \
# SOLVER.IMG_PER_BATCH_LABEL 12 SOLVER.IMG_PER_BATCH_UNLABEL 12 \
# SOLVER.STEPS '44990, 44995' \
# SOLVER.MAX_ITER 45001 \
# SOLVER.CHECKPOINT_PERIOD 2000 \
# SEMISUPNET.BURN_UP_STEP 5000 \
# SEMISUPNET.UNSUP_LOSS_WEIGHT 4.0 \
# MOCO.WARMUP_EMA False \
# MOCO.CONTRASTIVE_LEARNING False \
# UNCERTAINTY.PSEUDO_LABEL_REG False \
# MODEL.ROI_HEADS.NAME 'ContROIHeadsPseudoLab' \
# MODEL.ROI_BOX_HEAD.BOX_ENCODE_TYPE "xyxy" \
# SEED 1 \
# OUTPUT_DIR './output/ablation_study/no_threshold';


CUDA_VISIBLE_DEVICES=2,3 python train_net.py \
--num-gpus 2 \
--config configs/predicted_iou/faster_rcnn_R_50_FPN_sup1_run1.yaml \
--dist-url 'tcp://0.0.0.0:12344' \
SEMISUPNET.Trainer "predicted_iou" \
SOLVER.MAX_ITER 45001 \
SOLVER.STEPS "44990, 44995" \
SOLVER.IMG_PER_BATCH_LABEL 12 \
SOLVER.IMG_PER_BATCH_UNLABEL 12 \
SOLVER.CHECKPOINT_PERIOD 1000 \
SEMISUPNET.BURN_UP_STEP 5000 \
SEMISUPNET.UNSUP_LOSS_WEIGHT 4.0 \
IOUNET.IOU_THRESHOLD 0.9 \
IOUNET.TRAINING_WITH_JITTERING True \
IOUNET.JITTERING_TIMES 30 \
IOUNET.JITTERING_FRAC 0.2 \
IOUNET.IOULOSS_WEIGHT 1.0 \
MOCO.CONTRASTIVE_LEARNING False \
UNCERTAINTY.PSEUDO_LABEL_REG True \
MODEL.ROI_BOX_HEAD.BOX_ENCODE_TYPE "xyxy" \
MODEL.WEIGHTS 'output/ablation_study/predicted_iou_0.9_batch_12/model_burnup.pth' \
MOCO.RESUME_AFTER_BURNUP True \
SEED 1 \
OUTPUT_DIR './output/ablation_study/predicted_iou_0.9_batch_12';