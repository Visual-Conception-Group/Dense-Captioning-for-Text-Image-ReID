#!/bin/bash

cd ../../src

python train.py --model_name 'SSAN' \
--GPU_id 2 \
--part 6 \
--lr 0.001 \
--dataset 'ZURU' \
--epoch 60 \
--dataroot '' \
--class_num 15000 \
--vocab_size 3374 \
--feature_length 1024 \
--mode 'train' \
--batch_size 64 \
--cr_beta 0.1

# --vocab_size 3374 \