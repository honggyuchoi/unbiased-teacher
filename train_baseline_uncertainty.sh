# Only labeled data
CUDA_VISIBLE_DEVICES=1 python train_net.py \
      --num-gpus 1 \
      --config configs/plain_faster_rcnn/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
      MODEL.ROI_HEADS.NAME "StandardROIHeadsPseudoLabUncertainty" \
      MODEL.ROI_HEADS.BBOX_REG_UNCERTAINTY_LOSS_WEIGHT 0.1 \
      OUTPUT_DIR './output/coco_1.0/uncertainty_loss_weight_0.1' &&

# Only labeled data
CUDA_VISIBLE_DEVICES=1 python train_net.py \
      --num-gpus 1 \
      --config configs/plain_faster_rcnn/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
      MODEL.ROI_HEADS.NAME "StandardROIHeadsPseudoLabUncertainty" \
      MODEL.ROI_HEADS.BBOX_REG_UNCERTAINTY_LOSS_WEIGHT 0.5 \
      OUTPUT_DIR './output/coco_1.0/uncertainty_loss_weight_0.5' &&

# Only labeled data
CUDA_VISIBLE_DEVICES=1 python train_net.py \
      --num-gpus 1 \
      --config configs/plain_faster_rcnn/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
      MODEL.ROI_HEADS.NAME "StandardROIHeadsPseudoLabUncertainty" \
      MODEL.ROI_HEADS.BBOX_REG_UNCERTAINTY_LOSS_WEIGHT 1.0 \
      OUTPUT_DIR './output/coco_1.0/uncertainty_loss_weight_1.0';
