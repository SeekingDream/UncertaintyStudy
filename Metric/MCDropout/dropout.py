import torch
from BasicalClass import BasicModule
from torch.nn import functional as F
from BasicalClass import common_predict, common_ten2numpy
import numpy as np
from Metric import BasicUncertainty


class ModelActivateDropout(BasicUncertainty):
    def __init__(self, instance: BasicModule, device, iter_time):
        super(ModelActivateDropout, self).__init__(instance, device)
        self.iter_time = iter_time

    def extract_metric(self, data_loader, orig_pred_y):
        res = 0
        self.model.train()
        for _ in range(self.iter_time):
            _, pred, _ = common_predict(data_loader, self.model, self.device)
            res = res + pred.eq(orig_pred_y)
        self.model.eval()
        res = common_ten2numpy(res.float() / self.iter_time)
        return res

    def _predict_result(self, data_loader, model):
        pred_pos, pred_list, y_list = [], [], []
        model.to(self.device)
        for i, (x, y) in enumerate(data_loader):
            torch.cuda.empty_cache()
            x = x.to(self.device)
            output = model(x)
            pos, pred_y = torch.max(output, dim=1)
            pred_list.append(pred_y.detach())
            pred_pos.append(output.detach())
            y_list.append(y.detach())
        return torch.cat(pred_pos, dim=0).cpu(), torch.cat(pred_list, dim=0).cpu(), torch.cat(y_list, dim=0).cpu()

    @staticmethod
    def label_chgrate(orig_pred, prediction):
        _, repeat_num = np.shape(prediction)
        tmp = np.tile(orig_pred.reshape([-1, 1]), (1, repeat_num))
        return np.sum(tmp == prediction, axis=1, dtype=np.float) / repeat_num

    def _uncertainty_calculate(self, data_loader):
        self.model.eval()
        _, orig_pred, _ = self._predict_result(data_loader, self.model)
        mc_result = []
        self.model.train()
        for i in range(self.iter_time):
            _, res, _ = self._predict_result(data_loader, self.model)
            mc_result.append(common_ten2numpy(res).reshape([-1, 1]))
        mc_result = np.concatenate(mc_result, axis=1)
        score = self.label_chgrate(orig_pred, mc_result)
        return score
