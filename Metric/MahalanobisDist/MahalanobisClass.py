import torch
from BasicalClass import BasicModule
from torch.nn import functional as F
from BasicalClass import common_predict, ten2numpy
import numpy as np
from utils import  IS_DEBUG, DEBUG_NUM

class Mahalanobis():
    def __init__(self, instance : BasicModule, device):
        self.instance = instance
        self.model = instance.model
        self.device = device
        self.name = instance.name
        self.class_num = instance.class_num
        self.u_list, self.std_value = self.preprocess(instance.val_loader)

    def preprocess(self, data_loader):
        fx, y = self.get_penultimate(data_loader)
        u_list, std_list = [],[]
        for target in range(self.class_num):
            fx_tar = fx[torch.where(y == target)]
            mean_val = torch.mean(fx_tar)
            std_val = (fx_tar - mean_val).transpose(dim0=0, dim1= 1).mm((fx_tar - mean_val))
            u_list.append(mean_val)
            std_list.append(std_val)
        std_value = sum(std_list) / len(data_loader)
        std_value = torch.inverse(std_value)
        return u_list, std_value


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
        fx, _ = self.get_penultimate(data_loader)
        score = []
        for target in range(self.class_num):
            tmp = (fx - self.u_list[target]).mm(self.std_value)
            tmp = tmp.mm( (fx - self.u_list[target]).transpose(dim0=0, dim1=1) )
            tmp = tmp.diagonal().reshape([-1, 1])
            score.append(-tmp)
        score = torch.cat(score, dim = 1)
        return ten2numpy(torch.max(score, dim = 1)[0])

    def run_experiment(self, val_loader, test_loader):
        val_res = self.extract_metric(val_loader)
        test_res = self.extract_metric(test_loader)
        res = [
            val_res,
            test_res,
        ]
        torch.save(res, './Result/' + self.name + '/mahalanobis.res')
        print('get result for Mahalanobis')
