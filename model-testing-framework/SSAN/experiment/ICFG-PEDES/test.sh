#!/bin/bash

cd ../../src

python test.py --model_name 'SSAN' \
--test_dataset 'IIITD' \
--GPU_id 4 \
--part 6 \
--lr 0.001 \
--dataset 'ICFG-PEDES' \
--dataroot '' \
--vocab_size 2500 \
--feature_length 1024 \
--class_num 11000 \
--cross_dataset True \
--mode 'test' 