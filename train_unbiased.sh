# Only labeled data
CUDA_VISIBLE_DEVICES=2 python train_net.py \
      --num-gpus 1 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
      MODEL.RPN.POSITIVE_FRACTION 0.25 \
      MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 512 \
      SOLVER.IMG_PER_BATCH_LABEL 6 SOLVER.IMG_PER_BATCH_UNLABEL 6 \
      SOLVER.MAX_ITER 60000 SOLVER.STEPS '59990, 59995' \
      SEMISUPNET.BURN_UP_STEP 2000 \
      SEMISUPNET.UNSUP_LOSS_WEIGHT 2.5 \
      DATALOADER.RANDOM_DATA_SEED 1 \
      OUTPUT_DIR './output/coco_1.0/unbiased_teacher_unsuplossweight_2.5_positive_fraction_0.25_512';

CUDA_VISIBLE_DEVICES=3 python train_net.py \
      --num-gpus 1 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
      MODEL.RPN.POSITIVE_FRACTION 0.25 \
      MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 512 \
      SOLVER.IMG_PER_BATCH_LABEL 6 SOLVER.IMG_PER_BATCH_UNLABEL 6 \
      SOLVER.MAX_ITER 60000 SOLVER.STEPS '59990, 59995' \
      SEMISUPNET.BURN_UP_STEP 3000 \
      SEMISUPNET.UNSUP_LOSS_WEIGHT 2.0 \
      DATALOADER.RANDOM_DATA_SEED 1 \
      OUTPUT_DIR './output/coco_1.0/unbiased_teacher_unsuplossweight_2.0_positive_fraction_0.25_512_burnup_3000';

CUDA_VISIBLE_DEVICES=0 python train_net.py \
      --num-gpus 1 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
      MODEL.RPN.POSITIVE_FRACTION 0.25 \
      MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 512 \
      SOLVER.IMG_PER_BATCH_LABEL 6 SOLVER.IMG_PER_BATCH_UNLABEL 6 \
      SOLVER.MAX_ITER 60000 SOLVER.STEPS '59990, 59995' \
      SEMISUPNET.BURN_UP_STEP 3000 \
      SEMISUPNET.UNSUP_LOSS_WEIGHT 1.5 \
      DATALOADER.RANDOM_DATA_SEED 1 \
      OUTPUT_DIR './output/coco_1.0/unbiased_teacher_unsuplossweight_1.5_positive_fraction_0.25_512_burnup_3000';