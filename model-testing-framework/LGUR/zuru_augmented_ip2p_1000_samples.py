import os
import json
import random
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
dataset_address_prefix = "/raid/home/vibhu20150"

# dataset_address_prefix = "/Users/vibster/Developer/avs"
# dataset_address_prefix = "/home/Vibhu"


def createJson(data, file_name, end_train, end_val):
    """
    Creates .json files for the pre-train, train, validation, and test datasets.
    """

    if(not os.path.exists(dataset_address_prefix + "/temp")):
        os.mkdir(dataset_address_prefix + "/temp")
    if(os.path.exists(dataset_address_prefix + "/temp/"+file_name)):
        os.remove(f"{dataset_address_prefix}/temp/"+file_name)

    l = []

    for i in range(len(data)):
        modified1 = dict()
        # modified2 = dict()

        path = f"{dataset_address_prefix}/instruct-pix2pix/"+data[i]['image_path'][2:]

        if os.path.exists(path):
            modified1['image'] = path
            # modified2['image'] = path
        else:
            print(i)
            print("ERROR: image file not found")
            return -1

        modified1['caption'] = data[i]['Description_1']
        # modified2['caption'] = data[i]['Description_2']
        modified1['image_id'] = data[i]['image_id']
        # modified2['image_id'] = data[i]['image_id']
        
        tokens_1 = []
        tokens_1.append(word_tokenize(modified1['caption']))
        # tokens_2 = []
        # tokens_2.append(word_tokenize(modified2['caption']))

        if (0 <= int(data[i]['input_image_id'][5:]) and int(data[i]['input_image_id'][5:]) < end_train):
                split = 'train'

        elif (end_train <= int(data[i]['input_image_id'][5:]) and int(data[i]['input_image_id'][5:]) < end_val):
                tokens_1 = [tokens_1,]
                # tokens_2 = [tokens_2,]
                split = "val"
        
        elif (end_val <= int(data[i]['input_image_id'][5:]) and int(data[i]['input_image_id'][5:]) < len(data)):
                tokens_1 = [tokens_1,]
                # tokens_2 = [tokens_2,]
                split = "test"
        
        caption_1 = []
        caption_1.append(modified1['caption'])
        # caption_2 = []
        # caption_2.append(modified2['caption'])

        data_save_1 = {
            'file_path': modified1['image'],
            'id': i,
            'processed_tokens': tokens_1,
            'captions': caption_1,
            'split': split
        }

        # data_save_2 = {
        #     'file_path': modified2['image'],
        #     'id': data[i]['image_index'],
        #     'processed_tokens': tokens_2,
        #     'captions': caption_2,
        #     'split': split
        # }

        l.append(data_save_1)
        # l.append(data_save_2)

    random.Random(1).shuffle(l)

    os.chdir(f"{dataset_address_prefix}/Person-Re-ID/LGUR")
    with open(file_name, "w") as outfile:
        json.dump(l, outfile)
    
    print("SUCCESS")
    return 0

def main():
    # open filter json file
    f = open(f"{dataset_address_prefix}/instruct-pix2pix/AUG-LIST_2.json")
    data = json.load(f)
    f.close()

    print("data length: ", len(data))
    createJson(data, "ZURU-AUGMENTED-ICFG-FORMAT.json", end_train=500, end_val=600)

    """
    train = id from 0 to 499
    val = id from 500 to 599
    test = remaining
    """

if __name__ == '__main__':
    main()