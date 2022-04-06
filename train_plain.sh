# Only labeled data
CUDA_VISIBLE_DEVICES=0 python train_net.py \
      --num-gpus 1 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup05_run1.yaml \
      MODEL.RPN.POSITIVE_FRACTION 0.5 \
      MODEL.ROI_HEADS.LOSS "CrossEntropy" \
      MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 256 \
      SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 \
      SOLVER.MAX_ITER 60000 SOLVER.STEPS '59990, 59995' \
      SEMISUPNET.BURN_UP_STEP 60000 \
      DATALOADER.RANDOM_DATA_SEED 1 \
      OUTPUT_DIR './output/coco_0.5/plain_iter60000_burnupstep' &


# Only labeled data
CUDA_VISIBLE_DEVICES=1 python train_net.py \
      --num-gpus 1 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
      MODEL.RPN.POSITIVE_FRACTION 0.5 \
      MODEL.ROI_HEADS.LOSS "CrossEntropy" \
      MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 256 \
      SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 \
      SOLVER.MAX_ITER 60000 SOLVER.STEPS '59990, 59995' \
      SEMISUPNET.BURN_UP_STEP 60000 \
      DATALOADER.RANDOM_DATA_SEED 1 \
      OUTPUT_DIR './output/coco_1.0/plain_iter60000_burnupstep' &


# # Only labeled data
# CUDA_VISIBLE_DEVICES=3 python train_net.py \
#       --num-gpus 1 \
#       --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup2_run1.yaml \
#       MODEL.RPN.POSITIVE_FRACTION 0.5 \
#       MODEL.ROI_HEADS.LOSS "CrossEntropy" \
#       MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 256 \
#       SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 \
#       SOLVER.MAX_ITER 60000 SOLVER.STEPS '59990, 59995' \
#       SEMISUPNET.BURN_UP_STEP 60000 \
#       DATALOADER.RANDOM_DATA_SEED 1 \
#       TEST.OUTPUT_DIR './output/coco_2.0/plain_iter60000' &