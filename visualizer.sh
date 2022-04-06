# 'none', 'rpn', 'roi', 'tsne', 'conf_mat'

# CUDA_VISIBLE_DEVICES=3 python draw_box.py \
# --config-file configs/plain_faster_rcnn/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
# MODEL.WEIGHTS output/coco_1.0/baseline_60000/model_best.pth \
# VISUALIZATION.TYPE "rpn";

# draw predictions
# CUDA_VISIBLE_DEVICES=3 python visualize_detection_results.py \
# --config-file configs/plain_faster_rcnn/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
# MODEL.WEIGHTS output/coco_1.0/baseline_60000/model_best.pth \
# VISUALIZATION.TYPE "roi" \
# VISUALIZATION.RPN_THRESHOLD 0.9 \
# VISUALIZATION.ROI_THRESHOLD 0.7;

# Generate confusion matrix
# CUDA_VISIBLE_DEVICES=3 python visualize_detection_results.py \
# --num-gpus 1 \
# --config-file configs/plain_faster_rcnn/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
# MODEL.WEIGHTS weights/coco_1.0_x3.pkl \
# VISUALIZATION.TYPE 'conf_mat';

CUDA_VISIBLE_DEVICES=3 python visualize_detection_results.py \
--num-gpus 1 \
--config-file configs/plain_faster_rcnn/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
MODEL.WEIGHTS weights/coco_1.0_x3.pkl \
VISUALIZATION.TYPE 'cosine_sim';

CUDA_VISIBLE_DEVICES=3 python visualize_detection_results.py \
--num-gpus 1 \
--config-file configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
MODEL.WEIGHTS weights/coco_0.1.pth \
VISUALIZATION.TYPE 'cosine_sim';


CUDA_VISIBLE_DEVICES=3 python visualize_detection_results.py \
--num-gpus 1 \
--config-file configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
MODEL.WEIGHTS weights/coco_0.01.pth \
VISUALIZATION.TYPE 'cosine_sim';
