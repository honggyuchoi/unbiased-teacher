#configs/plain_faster_rcnn/voc/voc07_voc12.yaml
#

CUDA_VISIBLE_DEVICES=2,3 python train_net.py \
--config-file configs/class_aware_cont/voc/voc07_voc12.yaml \
--eval-only \
MODEL.WEIGHTS 'output/ablation_study/voc_07_12coco20_ours/model_0073999.pth' \
TEST.EVALUATOR 'voc' \
MODEL.ROI_BOX_HEAD.BOX_ENCODE_TYPE "xyxy" \
OUTPUT_DIR 'eval/';

# output/ablation_study/voc_07_12coco20_ours/model_0065999.pth
# output/ablation_study/voc_07_12coco20_ours/model_0067999.pth
# output/ablation_study/voc_07_12coco20_ours/model_0069999.pth
# output/ablation_study/voc_07_12coco20_ours/model_0071999.pth
# output/ablation_study/voc_07_12coco20_ours/model_0073999.pth