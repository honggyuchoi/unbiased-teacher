CUDA_VISIBLE_DEVICES=3 python train_net.py \
--config-file configs/plain_faster_rcnn/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
--eval-only MODEL.WEIGHTS weights/coco_1.0_x3.pkl \
OUTPUT_DIR './output/coco_100.0/ubteacher_weights_x3';