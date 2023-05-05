import os
import pickle
# from nltk import word_tokenize
from copy import deepcopy


def read_dict(root):
    with open(root, 'rb') as f:
        data = pickle.load(f)

    return data


def save_dict(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)



class Word2Index(object):

    def __init__(self, vocab):
        self._vocab = {w: index + 1 for index, w in enumerate(vocab)}
        self.unk_id = len(vocab) + 1

    def __call__(self, word):
        if word not in self._vocab:
            return self.unk_id
        return self._vocab[word]

def main(og_dict, word2Ind_new: Word2Index, new_dataset):

    for i in range(len(og_dict["tokens"])):
        new_caption_ids = []
        for word in og_dict["tokens"][i]:
            # print(word)
            new_caption_ids.append(word2Ind_new(word))
        
        un_idx = word2Ind_new.unk_id
        if un_idx in new_caption_ids:
            new_caption_ids = list(filter(lambda x: x != un_idx, new_caption_ids))

        # new_caption_ids = list(filter(lambda x: x <= 2500 and x >= 0, new_caption_ids))

        og_dict["lstm_caption_id"][i] = new_caption_ids
    
    save_dict(og_dict, new_dataset + "_test_save")

if __name__ == '__main__':
    og_dict = read_dict("test_save.pkl")
    new_datasets = ["CUHK-PEDES", "ICFG-PEDES", "RSTP", "ZURU"]
    for new_dataset in new_datasets:
        word2id_file_path = f"../processed_data_{new_dataset}/word2Ind.pkl"

        word2Ind_new = read_dict(word2id_file_path)

        # print(type(word2Ind_new), word2Ind_new.unk_id)

        # print(og_dict.keys())
        # print()
        # for key, val in og_dict.items():
        #     print(key, len(val))
        # print()

        # print(og_dict["lstm_caption_id"][0])
        print(len(og_dict["lstm_caption_id"][0]))
        # print(len(word_tokenize(og_dict["captions"][0])))
        print(len(og_dict["captions"][0].split()))
        print(len(og_dict["tokens"]))
        print(len(og_dict["id"]))

        print("Starting", new_dataset)
        main(deepcopy(og_dict), word2Ind_new, new_dataset)
        print("DONE!")
