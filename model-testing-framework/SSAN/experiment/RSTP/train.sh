#!/bin/bash

cd ../../src

python train.py --model_name 'SSAN' \
--GPU_id 4 \
--part 6 \
--lr 0.001 \
--dataset 'RSTP' \
--epoch 60 \
--dataroot '' \
--class_num 3701 \
--vocab_size 3000 \
--feature_length 1024 \
--mode 'train' \
--batch_size 64 \
--cr_beta 0.1
