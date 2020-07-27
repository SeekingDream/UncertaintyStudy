from typing import *
from BasicalClass import BasicModule
from BasicalClass import common_get_maxpos
import torch.nn as nn
import torch.optim as optim
import argparse
import torch


class PVScore:
    def __init__(self, instance: BasicModule, device):
        self.instance = instance
        self.device = device
        self.train_batch_size = self.instance.train_batch_size
        self.test_batch_size = self.instance.test_batch_size
        self.model = self.instance.model
        self.name = self.instance.name
        self.class_num = self.instance.class_num
        self.train_y, self.val_y, self.test_y = \
            self.instance.train_y, self.instance.val_y, self.instance.test_y

        self.train_pred_pos, self.train_pred_y =\
            self.instance.train_pred_pos, self.instance.train_pred_y
        self.val_pred_pos, self.val_pred_y = \
            self.instance.val_pred_pos, self.instance.val_pred_y
        self.test_pred_pos, self.test_pred_y = \
            self.instance.test_pred_pos, self.instance.test_pred_y

        self.softmax = nn.Softmax(dim=1)

        res = [
            common_get_maxpos(self.train_pred_pos),
            common_get_maxpos(self.val_pred_pos),
            common_get_maxpos(self.test_pred_pos),
        ]
        torch.save(res, './Result/' + self.name + '/viallina.res')
        print('get result for viallina')

    def run(self, feature_name):
        test_pred_pos, _ = torch.max(self.softmax(self.test_pred_pos), dim = 1)
        #common_get_auc(ground_truth.data.numpy(), test_pred_pos.data.numpy(), self.__class__.__name__)



