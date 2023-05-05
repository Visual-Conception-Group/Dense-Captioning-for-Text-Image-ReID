import os
import json
import random
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
dataset_address_prefix = "/raid/home/vibhu20150"

# dataset_address_prefix = "/Users/vibster/Developer/avs"
# dataset_address_prefix = "/home/Vibhu"


def createJson(data_aug, data_og, file_name, end_train, end_val):
    """
    Creates .json files for the pre-train, train, validation, and test datasets.
    """

    if(not os.path.exists(dataset_address_prefix + "/temp")):
        os.mkdir(dataset_address_prefix + "/temp")
    if(os.path.exists(dataset_address_prefix + "/temp/"+file_name)):
        os.remove(f"{dataset_address_prefix}/temp/"+file_name)

    l = []

    for i in range(len(data_og)):
        modified1 = dict()
        modified2 = dict()

        path = f"{dataset_address_prefix}/Datasets/ZURU_Combined/"+data_og[i]['image_path']

        if os.path.exists(path):
            modified1['image'] = path
            modified2['image'] = path
        else:
            print(i)
            print("ERROR: image file not found")
            return -1

        modified1['caption'] = data_og[i]['Description_1']
        modified2['caption'] = data_og[i]['Description_2']
        modified1['image_id'] = data_og[i]['image_id']
        modified2['image_id'] = data_og[i]['image_id']
        
        tokens_1 = []
        tokens_1.append(word_tokenize(modified1['caption']))
        tokens_2 = []
        tokens_2.append(word_tokenize(modified2['caption']))

        if (0 <= data_og[i]['image_index'] and data_og[i]['image_index'] < end_train) or (20_000 <= data_og[i]['image_index'] and data_og[i]['image_index'] < 22_500):
                split = 'train'

        elif (end_train <= data_og[i]['image_index'] and data_og[i]['image_index'] < end_val) or (22_500 <= data_og[i]['image_index'] and data_og[i]['image_index'] < 23_000):
                tokens_1 = [tokens_1,]
                tokens_2 = [tokens_2,]
                split = "val"
        
        elif (end_val <= data_og[i]['image_index'] and data_og[i]['image_index'] < 20_000) or (23_000 <= data_og[i]['image_index'] and data_og[i]['image_index'] < len(data_og)):
                tokens_1 = [tokens_1,]
                tokens_2 = [tokens_2,]
                split = "test"
        
        caption_1 = []
        caption_1.append(modified1['caption'])
        caption_2 = []
        caption_2.append(modified2['caption'])

        data_save_1 = {
            'file_path': modified1['image'],
            'id': data_og[i]['image_index'],
            'processed_tokens': tokens_1,
            'captions': caption_1,
            'split': split
        }

        data_save_2 = {
            'file_path': modified2['image'],
            'id': data_og[i]['image_index'],
            'processed_tokens': tokens_2,
            'captions': caption_2,
            'split': split
        }

        l.append(data_save_1)
        l.append(data_save_2)
    
    for i in range(len(data_aug)):
        modified1 = dict()

        path = f"{dataset_address_prefix}/instruct-pix2pix/"+data_aug[i]['image_path'][2:]

        if os.path.exists(path):
            modified1['image'] = path
        else:
            print(i)
            print("ERROR: image file not found")
            return -1

        modified1['caption'] = data_aug[i]['Description_1']
        modified1['image_id'] = data_aug[i]['image_id']
        
        tokens_1 = []
        tokens_1.append(word_tokenize(modified1['caption']))

        split = 'train'

        caption_1 = []
        caption_1.append(modified1['caption'])

        data_save_1 = {
            'file_path': modified1['image'],
            'id': i+len(data_og),
            'processed_tokens': tokens_1,
            'captions': caption_1,
            'split': split
        }

        l.append(data_save_1)
    
    random.Random(1).shuffle(l)    

    os.chdir(f"{dataset_address_prefix}/Person-Re-ID/LGUR")
    with open(file_name, "w") as outfile:
        json.dump(l, outfile)
    
    print("SUCCESS")
    return 0

def main():
    # open filter json file
    f = open(f"{dataset_address_prefix}/instruct-pix2pix/AUG-LIST_2.json")
    data_aug = json.load(f)
    f.close()

    f = open(f"{dataset_address_prefix}/Datasets/ZURU_Combined/Batch_Combined.json")
    data_og = json.load(f)
    f.close()

    print("data length: ", len(data_aug), len(data_og))
    createJson(data_aug, data_og, "ZURU-AUGMENTED-APPENDED-ICFG-FORMAT.json", end_train=15_000, end_val=17_500)

if __name__ == '__main__':
    main()