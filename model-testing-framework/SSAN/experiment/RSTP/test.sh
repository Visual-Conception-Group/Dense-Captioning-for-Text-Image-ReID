#!/bin/bash

cd ../../src

python test.py --model_name 'SSAN' \
--GPU_id 4 \
--test_dataset 'IIITD' \
--part 6 \
--lr 0.001 \
--dataset 'RSTP' \
--dataroot '' \
--vocab_size 3000 \
--feature_length 1024 \
--cross_dataset True \
--mode 'test'