# from utils.read_write_data import read_json, makedir, save_dict, write_txt
# import argparse
# from collections import namedtuple
# import os
# import nltk
# from nltk.tag import StanfordPOSTagger
# from random import shuffle
# import numpy as np
# import pickle
# import transformers as ppb
# import time
import json

fd = open("../../Datasets/IIITD-IIITD-OUTPUTDELIVERY/JsonOutput/Filtered.json")
data = json.load(fd)
fd.close()

count = 0
maxim = 0
max_id = ""
for i in data:
    if len(data[i]['Description 1'].split(" ")) > 60:
        maxim = max(maxim, len(data[i]['Description 1'].split(" ")))
        max_id = data[i]["Image ID"]
# print(count)
print(maxim, max_id)

# reid_raw_data = read_json('./nouns_10_choose.json')
# print(len(reid_raw_data.keys()))
# ctr = 0
# cv = 0
# cte = 0
# fd = open('combined_datasets.json')
# for data in json.load(fd):
#     if data['split'] == 'test':
#         cte += 1
#     if data['split'] == 'val':
#         cv += 1
#     if data['split'] == 'train':
#         ctr += 1
# fd.close()
# print(ctr, cv, cte)

# fd = open('combined_datasets.json')
# unique_ids = set()
# for data in json.load(fd):
#     unique_ids.add(data['id'])

# for i in range(max(unique_ids)+1):
#     if i not in unique_ids:
#         print(i)

# print(len(unique_ids))
# fd.close()