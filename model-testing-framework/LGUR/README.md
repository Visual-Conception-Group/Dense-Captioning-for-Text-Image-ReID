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

Note: In the data/dataset.py code, we use the notation IIITD_BLIP_3 to reference the BLIP captions with original images, IIITD_BLIP_LDM_4 to reference B1, and IIITD_BLIP_LDM_5 to reference B2. IIITD_BLIP_1 and IIITD_BLIP_2 can be ignored. 