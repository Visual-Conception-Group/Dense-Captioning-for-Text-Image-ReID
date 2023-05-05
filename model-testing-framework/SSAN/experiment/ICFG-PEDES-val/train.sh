#!/bin/bash

cd ../../src

python train.py --model_name 'SSAN' \
--GPU_id 5 \
--part 6 \
--lr 0.001 \
--dataset 'ICFG-PEDES-val' \
--epoch 60 \
--dataroot '' \
--class_num 2602 \
--vocab_size 2380 \
--feature_length 1024 \
--batch_size 64 \
--mode 'train' \
--cr_beta 0.1
