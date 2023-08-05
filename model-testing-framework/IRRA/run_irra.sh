#!/bin/bash
DATASET_NAME="IIITD"

python train.py \
--GPU_ID 0 \
--name iiitd \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60 \
--val_dataset 'val' \
# --factor 1
