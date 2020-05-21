import torch
from BasicalClass import BasicModule
from torch.nn import functional as F
from BasicalClass import common_predict, ten2numpy
import numpy as np

class ModelActivateDropout():
    def __init__(self, instance : BasicModule, device, iter_time=500):
        self.model = instance.model
        self.device = device
        self.iter_time = iter_time
        self.name = instance.name

    def extract_metric(self, data_loader):
        res = 0
        self.model.train()
        for _ in range(self.iter_time):
            pos, _, _ = common_predict(data_loader, self.model, self.device)
            pos = F.softmax(pos, dim=1)
            res += pos
        res = ten2numpy(res / self.iter_time)
        res = -np.sum(res * np.log(res + 1e-18), axis=1)
        return res

    def run_experiment(self, val_loader, test_loader):
        val_res = self.extract_metric(val_loader)
        test_res = self.extract_metric(test_loader)

        res = [
            val_res,
            test_res,
        ]
        torch.save(res, './Result/' + self.name + '/dropout.res')
        print('get result for MCDrop')
