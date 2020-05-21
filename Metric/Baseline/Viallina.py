from typing import *
from utils import *
import torch.nn as nn
import torch.optim as optim
import argparse
import torch

class Viallina:
    def __init__(self, instance : CIFAR10_Module,
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.instance = instance
        self.device = device
        self.train_batch_size = self.instance.train_batch_size
        self.test_batch_size = self.instance.test_batch_size
        self.model = self.instance.model
        self.name = self.instance.name
        self.class_num = self.instance.class_num
        self.train_x, self.train_y = self.instance.train_x,  self.instance.train_y
        self.test_x, self.test_y = self.instance.test_x,  self.instance.test_y
        self.train_pred_pos, self.train_pred_y =\
            self.instance.train_pred_pos,self.instance.train_pred_y
        self.test_pred_pos, self.test_pred_y = \
            self.instance.test_pred_pos, self.instance.test_pred_y
        self.softmax = nn.Softmax(dim = 1)
        self.ground_truth = self.test_pred_y.eq(self.test_y)

    def run(self, feature_name):
        ground_truth = self.test_y.eq(self.test_pred_y).view([-1])
        test_pred_pos, _ = torch.max(self.softmax(self.test_pred_pos), dim = 1)
        common_get_auc(ground_truth.data.numpy(), test_pred_pos.data.numpy(), self.__class__.__name__)



