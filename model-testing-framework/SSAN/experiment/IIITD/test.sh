#!/bin/bash

cd ../../src

CUDA_LAUNCH_BLOCKING=1 python test.py --model_name 'SSAN' \
--test_dataset "CUHK-PEDES" \
--GPU_id 7 \
--part 6 \
--lr 0.001 \
--dataset 'IIITD' \
--dataroot '' \
--vocab_size 3374 \
--cross_dataset True \
--feature_length 1024 \
--class_num 15000 \
--mode 'test'