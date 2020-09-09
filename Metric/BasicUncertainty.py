from abc import ABCMeta, abstractmethod
import numpy as np
from utils import common_ten2numpy, common_predict
from utils import BasicModule
import torch.nn as nn
import torch
import os


class BasicUncertainty(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, instance: BasicModule, device):
        super(BasicUncertainty, self).__init__()
        self.instance = instance
        self.device = device
        self.train_batch_size = self.instance.train_batch_size
        self.test_batch_size = self.instance.test_batch_size
        self.model = self.instance.model.to(self.device)
        self.class_num = self.instance.class_num

        self.train_y, self.val_y, self.test_y = \
            self.instance.train_y, self.instance.val_y, self.instance.test_y

        self.train_pred_pos, self.train_pred_y =\
            self.instance.train_pred_pos, self.instance.train_pred_y
        self.val_pred_pos, self.val_pred_y = \
            self.instance.val_pred_pos, self.instance.val_pred_y
        self.test_pred_pos, self.test_pred_y = \
            self.instance.test_pred_pos, self.instance.test_pred_y

        self.train_loader = instance.train_loader
        self.val_loader = instance.val_loader
        self.test_loader = instance.test_loader

        self.train_num = len(self.train_y)
        self.val_num = len(self.val_y)
        self.test_num = len(self.test_y)

        self.train_oracle = np.int32(
            common_ten2numpy(self.train_pred_y).reshape([-1]) == common_ten2numpy(self.train_y).reshape([-1])
        )
        self.val_oracle = np.int32(
            common_ten2numpy(self.val_pred_y).reshape([-1]) == common_ten2numpy(self.val_y).reshape([-1])
        )
        self.test_oracle = np.int32(
            common_ten2numpy(self.test_pred_y).reshape([-1]) == common_ten2numpy(self.test_y).reshape([-1])
        )

        self.softmax = nn.Softmax(dim=1)

    @abstractmethod
    def _uncertainty_calculate(self, data_loader):
        return common_predict(data_loader, self.model, self.device)

    def run(self):
        score = self.get_uncertainty()
        self.save_uncertaity_file(score)
        print('finish score extract for class', self.__class__.__name__)
        return score

    def get_uncertainty(self):
        train_score = self._uncertainty_calculate(self.train_loader)
        val_score = self._uncertainty_calculate(self.val_loader)
        test_score = self._uncertainty_calculate(self.test_loader)
        result = {
            'train': train_score,
            'val': val_score,
            'test': test_score
        }
        return result

    def save_uncertaity_file(self, score_dict):
        data_name = self.instance.__class__.__name__
        uncertainty_type = self.__class__.__name__
        save_name = '../Result/' + data_name + '/' + uncertainty_type + '.res'
        if not os.path.isdir('../Result/' + data_name):
            os.mkdir('../Result/' + data_name)
        torch.save(score_dict, save_name)
        print('get result for dataset %s, uncertainty type is %s' % (data_name, uncertainty_type))
