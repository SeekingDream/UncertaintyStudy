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
        self.val_pred = instance.val_pred_y
        self.test_pred = instance.test_pred_y


    def extract_metric(self, data_loader, orig_pred_y):
        res = 0
        self.model.train()
        for _ in range(self.iter_time):
            _, pred, _ = common_predict(data_loader, self.model, self.device)
            res= res + pred.eq(orig_pred_y)
        self.model.eval()
        res = ten2numpy(res.float() / self.iter_time)
        return res

    def run_experiment(self, val_loader, test_loader):
        val_res = self.extract_metric(val_loader, self.val_pred)
        test_res = self.extract_metric(test_loader, self.test_pred)

        res = [
            val_res,
            test_res,
        ]
        torch.save(res, './Result/' + self.name + '/dropout.res')
        print('get result for MCDrop')
