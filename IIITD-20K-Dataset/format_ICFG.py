import os
import json
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# dataset_address_prefix = "/raid/home/vibhu20150/Datasets/IIITD-20K/IIITD-20K-Dataset/"
dataset_address_prefix = ""


def createJson(data, fileName, endTrainInx, endValInx):
  """
  Creates .json files for the train, validation, and test datasets.
  """

  l = []
  person_id = 0

  for i in data.keys():


    image_path = ""
    
    jpeg_url = f"{dataset_address_prefix}IIITD-20K/"+data[i]['Image ID']+".jpeg"
    jpg_url = f"{dataset_address_prefix}IIITD-20K/"+data[i]['Image ID']+".jpg"

    if os.path.exists(jpeg_url):
      image_path = jpeg_url
    elif os.path.exists(jpg_url):
      image_path = jpg_url
    else:
      print("ERROR: image file not found")
      return -1
    
    tokens_1 = [word_tokenize(data[i]['Description 1']),]
    tokens_2 = [word_tokenize(data[i]['Description 2']),]

    split = "train"

    if endTrainInx <= person_id and person_id < endValInx:
      tokens_1 = [tokens_1,]
      tokens_2 = [tokens_2,]
      split = "val"
    elif endValInx <= person_id:
      tokens_1 = [tokens_1,]
      tokens_2 = [tokens_2,]
      split = "test"
    
    caption_1 = [data[i]['Description 1'], ]
    caption_2 = [data[i]['Description 2'], ]


    data_save_1 = {
      'file_path': image_path,
      'id': person_id,
      'processed_tokens': tokens_1,
      'captions': caption_1,
      'split': split
    }


    data_save_2 = {
      'file_path': image_path,
      'id': person_id,
      'processed_tokens': tokens_2,
      'captions': caption_2,
      'split': split
    }


    l.append(data_save_1)
    l.append(data_save_2)
    person_id += 1

  with open(fileName, "w") as outfile:
    json.dump(l, outfile)
  return 0


def main():
  # open filter json file
  f = open(f"{dataset_address_prefix}Filtered.json")
  data = json.load(f)
  f.close()
  print("Data length: ", len(data))
  createJson(data, "IIITD-20K-ICFG-FORMAT.json", 15000, 17500)
  return 

if __name__ == '__main__':
  main()
 