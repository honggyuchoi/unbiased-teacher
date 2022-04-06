CUDA_VISIBLE_DEVICES=0 python train_net.py \
      --num-gpus 1 \
      --config configs/voc/voc07_voc12.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 6 SOLVER.IMG_PER_BATCH_UNLABEL 6 \
       SEMISUPNET.BURN_UP_STEP 180000 \
       TEST.EVAL_PERIOD 10000 \
       SOLVER.CHECKPOINT_PERIOD 10000 \
       OUTPUT_DIR './output/voc07_voc12/supervised' \
       SEED 1;