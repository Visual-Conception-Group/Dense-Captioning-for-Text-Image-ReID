import os
import json
import random
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

dataset_address_prefix = "/raid/home/vibhu20150"

def whichBatch(image_ID):
	"""
	Function to determine which batch the data belongs to in the dataset.
	"""
	idx = int(image_ID[5:])
	if(idx >= 0 and idx <= 1331):
		return "1"
	elif(idx >= 1332 and idx <= 6212):
		return "2"
	elif (idx >= 6213 and idx <= 15877):
		return "3"
	elif (idx >= 15878 and idx <= 22445):
		return "4"
	elif (idx >= 22446 and idx <= 22571):
		return "5"
	elif (idx >= 22572 and idx <= 22712):
		return "6"
	else:
		return "-1"



def createJson(data, fileName):
  """
  Creates .json files for the pre-train, train, validation, and test datasets.
  """
  # if(not os.path.exists(dataset_address_prefix + "/temp")):
  #   os.mkdir(dataset_address_prefix + "/temp")
  # if(os.path.exists(dataset_address_prefix + "/temp/"+fileName)):
  #   os.remove(f"{dataset_address_prefix}/temp/"+fileName)

  l = []
  person_id = 0

  for i in range(len(data)):
    modified1 = dict()
    # modified2 = dict()
    batch = whichBatch(data[i]['Image ID'])
    if(batch == "-1"):
      return -1
    
    jpeg_url = f"{dataset_address_prefix}/Datasets/ZURU-IIITD-OUTPUTDELIVERY/Batch "+batch+"/"+data[i]['Image ID']+".jpeg"
    jpg_url = f"{dataset_address_prefix}/Datasets/ZURU-IIITD-OUTPUTDELIVERY/Batch "+batch+"/"+data[i]['Image ID']+".jpg"


    if os.path.exists(jpeg_url):
      modified1['image'] = jpeg_url
      # modified2['image'] = jpeg_url
    elif os.path.exists(jpg_url):
      modified1['image'] = jpg_url
      # modified2['image'] = jpg_url
    else:
      print("ERROR: image file not found")
      return -1

    modified1['caption'] = data[i]['Description 1']
    # modified2['caption'] = data[i]['Description 2']
    modified1['image_id'] = data[i]['Image ID']
    # modified2['image_id'] = data[i]['Image ID']

    tokens_1 = []
    tokens_1.append(word_tokenize(modified1['caption']))
    # tokens_2 = []
    # tokens_2.append(word_tokenize(modified2['caption']))

    # tokens_2 = [tokens_2,]
    split = "train"
    
    caption_1 = []
    caption_1.append(modified1['caption'])
    # caption_2 = []
    # caption_2.append(modified2['caption'])


    data_save_1 = {
      'file_path': modified1['image'],
      'id': person_id,
      'processed_tokens': tokens_1,
      'captions': caption_1,
      'split': split
    }


    # data_save_2 = {
    #   'file_path': modified2['image'],
    #   'id': person_id,
    #   'processed_tokens': tokens_2,
    #   'captions': caption_2,
    #   'split': split
    # }

    l.append(data_save_1)
    # l.append(data_save_2)
    person_id += 1

  # random.Random(1).shuffle(l)

  # os.chdir(f"{dataset_address_prefix}/temp/")
  return l



def checkDataNotCorrupted(fileName):
  """
  Checks if the data has been loaded correctly. If it has not, create the file once again.
  If it is loaded correctly, it returns the size of the data list.
  """
  f = open (f"{dataset_address_prefix}/temp/"+fileName)
  data_list = json.load(f)
  f.close()
  return data_list

def addValAndTest(l, file):
  fd = open(file, 'r')
  data = json.load(fd)
  fd.close()

  for i in range(len(data)):
    if data[i]['split'] == 'val' or data[i]['split'] == 'test':
      l.append(data[i])
  
  return l
    


def main():
  # open filter json file
  f = open(f"{dataset_address_prefix}/Datasets/ZURU-IIITD-OUTPUTDELIVERY/JsonOutput/BLIP_gen_0_14999_4.json")
  data = json.load(f)
  f.close()
  print("data length: ", len(data))
  l = createJson(data, "ZURU-BLIP-3-ICFG-FORMAT.json")
  l = addValAndTest(l, 'ZURU-ICFG-FORMAT.json')
  print(len(l))
  with open("ZURU-BLIP-3-ICFG-FORMAT.json", "w") as outfile:
    json.dump(l, outfile)

if __name__ == '__main__':
  main()
 