# Only labeled data
CUDA_VISIBLE_DEVICES=0 python train_net.py \
      --num-gpus 1 \
      --config configs/plain_faster_rcnn/coco_supervision/faster_rcnn_R_50_FPN_sup05_run1.yaml \
      OUTPUT_DIR './output/coco_0.5/baseline_60000' &

# Only labeled data
CUDA_VISIBLE_DEVICES=1 python train_net.py \
      --num-gpus 1 \
      --config configs/plain_faster_rcnn/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
      OUTPUT_DIR './output/coco_1.0/baseline_60000' &