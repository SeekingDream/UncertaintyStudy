import torch
from BasicalClass import BasicModule
from torch.nn import functional as F
from BasicalClass import common_predict, ten2numpy
import numpy as np
from utils import  IS_DEBUG, DEBUG_NUM

class Mahalanobis():
    def __init__(self, instance : BasicModule, device, iter_time=500):
        self.instance = instance
        self.model = instance.model
        self.device = device
        self.iter_time = iter_time
        self.val_y, self.test_y = instance.val_y, instance.test_y
        self.name = instance.name
        self.class_num = instance.class_num

    def get_penultimate(self, data_loader):
        res, y_list = [], []
        for i, (x, y) in enumerate(data_loader):
            x = x.to(self.device)
            self.model.to(self.device)
            fx = self.model.get_penultimate(x)
            res.append(fx)
            y_list.append(y)
            if IS_DEBUG and i >= DEBUG_NUM:
                break
        res = torch.cat(res, dim=0)
        y_list =  torch.cat(y_list, dim=0)
        return res, y_list



    def extract_metric(self, data_loader):
        fx, y = self.get_penultimate(data_loader)
        u_list, std_list = [],[]
        for target in range(self.class_num):
            fx_tar = fx[torch.where(y == target)]
            mean_val = torch.mean(fx_tar)
            std_val = (fx_tar - mean_val).transpose().mm((fx_tar - mean_val))
            u_list.append(mean_val)
            std_list.append(std_val)
        std_value = sum(std_list) / len(data_loader)
        return u_list, std_value



    def run_experiment(self, val_loader, test_loader):
        val_res = self.extract_metric(val_loader)
        test_res = self.extract_metric(test_loader)

        res = [
            val_res,
            test_res,
        ]
        torch.save(res, './Result/' + self.name + '/dropout.res')
        print('get result for MCDrop')
