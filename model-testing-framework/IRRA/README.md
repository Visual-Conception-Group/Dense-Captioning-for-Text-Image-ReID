# Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval

## Usage

## Training

```python
python train.py \
--name irra \
--img_aug \
--batch_size 64 \
--MLM \
--loss_names 'sdm+mlm+id' \
--dataset_name '<DATASET>' \
--root_dir '<DATASET_ROOT_DIR>' \
--num_epoch 60
```

## Testing

```python
python test.py --config_file '<path/to/model_dir/configs.yaml>'
```

Note: When testing for cross-dataset setup, change ```dataset_name``` to the dataset you want to test with, and change ```loss_names``` to ```sdm+mlm``` in the ```configs.yaml``` file