#!/bin/bash

cd ../../src

python test.py --model_name 'SSAN' \
--test_dataset 'ZURU' \
--GPU_id 4 \
--part 6 \
--lr 0.001 \
--dataset 'CUHK-PEDES' \
--dataroot '' \
--vocab_size 5000 \
--feature_length 1024 \
--class_num 11000 \
--cross_dataset True \
--mode 'test' 