from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader, Subset
from BasicalClass.common_function import common_predict, common_ten2numpy
import torch
import os


class BasicModule:
    __metaclass__ = ABCMeta

    def __init__(self, device, load_poor):
        self.device = device
        self.load_poor = load_poor
        self.train_batch_size = 64
        self.test_batch_size = 64
        self.model = self.get_model()
        self.class_num = 10
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        if not os.path.isdir('../Result/' + self.__class__.__name__):
            os.mkdir('../Result/' + self.__class__.__name__)

    def get_model(self):
        if not self.load_poor:
            model = self.load_model()
        else:
            model = self.load_poor_model()
        model.to(self.device)
        model.eval()
        print('model name is ', model.__class__.__name__)
        return model

    @abstractmethod
    def load_model(self):
        return None

    @abstractmethod
    def load_poor_model(self):
        return None

    def get_hiddenstate(self, dataloader, device):
        sub_num = self.model.sub_num
        hidden_res, label_res = [[] for _ in sub_num], []
        for x, y in dataloader:
            x = x.to(device)
            res = self.model.get_hidden(x)
            for i, r in enumerate(res):
                hidden_res[i].append(r)
            label_res.append(y)
        hidden_res = [torch.cat(tmp, dim=0) for tmp in hidden_res]
        return hidden_res, sub_num, torch.cat(label_res)

    def get_loader(self, train_db, val_db, test_db ):
        train_loader = DataLoader(
            train_db, batch_size=self.train_batch_size,
            shuffle=False, collate_fn=None)
        val_loader = DataLoader(
            val_db, batch_size=self.test_batch_size,
            shuffle=False, collate_fn=None)
        test_loader = DataLoader(
            test_db, batch_size=self.test_batch_size,
            shuffle=False, collate_fn=None)
        return train_loader, val_loader, test_loader

    def get_information(self):
        self.train_pred_pos, self.train_pred_y, self.train_y = \
            common_predict(self.train_loader, self.model, self.device)

        self.val_pred_pos, self.val_pred_y, self.val_y = \
            common_predict(self.val_loader, self.model, self.device)

        self.test_pred_pos, self.test_pred_y, self.test_y = \
            common_predict(self.test_loader, self.model, self.device)

    def save_truth(self):
        self.train_truth = self.train_pred_y.eq(self.train_y)
        self.val_truth = self.val_pred_y.eq(self.val_y)
        self.test_truth = self.test_pred_y.eq(self.test_y)
        truth = [
            common_ten2numpy(self.train_truth),
            common_ten2numpy(self.val_truth),
            common_ten2numpy(self.test_truth)
        ]
        torch.save(truth, '../Result/' + self.__class__.__name__ + '/truth.res')