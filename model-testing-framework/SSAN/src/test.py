# -*- coding: utf-8 -*-
"""
@author: zifyloo
"""


from option.options import options
from data.dataloader import get_dataloader
import torch
from model.model import TextImgPersonReidNet
import os
from test_during_train import test


def save_checkpoint(state, opt):

    filename = os.path.join(opt.save_path, 'model/best.pth.tar')
    torch.save(state, filename)


def load_checkpoint(opt):
    filename = os.path.join(opt.save_path, 'model/best.pth.tar')
    state = torch.load(filename)
    print('Load the {} epoch parameter successfully'.format(state['epoch']))

    return state


def main(opt):
    opt.device = torch.device('cuda:{}'.format(opt.GPU_id))

    opt.save_path = './checkpoints/{}/'.format(opt.dataset) + opt.model_name

    print(opt.mode)

    if opt.test_dataset == "":
        test_img_dataloader, test_txt_dataloader = get_dataloader(opt)
    else:
        train_dataset = opt.dataset
        opt.dataset = opt.test_dataset
        opt.test_train_dataset = train_dataset
        print("TEST TRAIN DATASET", opt.test_train_dataset)
        test_img_dataloader, test_txt_dataloader = get_dataloader(opt)
        opt.dataset = train_dataset

    network = TextImgPersonReidNet(opt).to(opt.device)

    test_best = 0
    state = load_checkpoint(opt)
    missing_keys = network.load_state_dict(state['network'])
    print("Missing Keys: ", missing_keys)
    epoch = state['epoch']

    print(opt.model_name)
    network.eval()
    test(opt, epoch + 1, network, test_img_dataloader, test_txt_dataloader, test_best, False)
    network.train()


if __name__ == '__main__':
    opt = options().opt
    print(opt.cross_dataset)
    main(opt)





