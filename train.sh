CUDA_VISIBLE_DEVICES=0 python train_net.py \
      --num-gpus 1 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 \
       SEMISUPNET.UNSUP_LOSS_WEIGHT 3.0 &

CUDA_VISIBLE_DEVICES=1 python train_net.py \
      --num-gpus 1 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 \
       SEMISUPNET.UNSUP_LOSS_WEIGHT 2.0;
