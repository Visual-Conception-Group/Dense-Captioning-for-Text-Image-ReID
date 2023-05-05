# Learning Granularity-Unified Representations for Text-to-Image Person Re-identification

### Download DeiT-small weights
```bash
wget https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth
```
### Process image and text datasets
```bash
python processed_data_singledata_<DATASET>.py
```

### Train
```bash
python train_mydecoder_pixelvit_txtimg_3_bert.py
```
