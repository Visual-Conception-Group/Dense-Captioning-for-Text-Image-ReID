#!/bin/bash

cd ../../src

python test_lgur.py --model_name 'SSAN' \
--test_dataset 'ZURU' \
--GPU_id 2 \
--part 6 \
--lr 0.001 \
--dataset 'CUHK-PEDES' \
--dataroot '' \
--vocab_size 5000 \
--feature_length 1024 \
--class_num 11000 \
--mode 'test' 