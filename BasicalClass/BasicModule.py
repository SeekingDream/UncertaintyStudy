from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader, Subset
from BasicalClass.common_function import common_predict, common_ten2numpy
import torch
import os


class BasicModule:
    __metaclass__ = ABCMeta
    SAVE_DIR = 'Uncertainty_Results'
    DATA_DIR = 'java_dataset/data/code_summary-preprocess'
    RES_DIR = 'java_dataset/se_tasks/code_summary/result'

    def __init__(self, device, load_poor):
        self.tk_path = os.path.join(self.DATA_DIR, 'tk.pkl')
        self.train_path = os.path.join(self.DATA_DIR, 'train.pkl')
        self.shift1_path = os.path.join(self.DATA_DIR, 'shift1.pkl')
        self.shift2_path = os.path.join(self.DATA_DIR, 'shift2.pkl')
        self.val_path = os.path.join(self.DATA_DIR, 'val.pkl')
        self.vec_path = 'java_dataset/embedding_vec/100_2/Doc2VecEmbedding0.vec'
        self.embed_dim = 100
        self.out_dir = self.RES_DIR
        self.max_size = 50000 # use 50000 of the data
        self.embed_type = 1
        self.device = device
        self.load_poor = load_poor
        self.train_batch_size = 128
        self.test_batch_size = self.train_batch_size
        self.model = self.get_model()
        # self.class_num = 48557 # max target vocab size
        self.train_loader = None
        self.val_loader = None
        self.shift1_loader = None
        self.shift2_loader = None
        
        if not os.path.isdir(self.SAVE_DIR):
            os.mkdir(self.SAVE_DIR)
        if not os.path.isdir(
            os.path.join(self.SAVE_DIR, self.__class__.__name__)
        ):
            os.mkdir(os.path.join(self.SAVE_DIR, self.__class__.__name__))

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
    
        for (sts, paths, eds), y, length in dataloader:
            sts = sts.to(device)
            paths = paths.to(device)
            eds = eds.to(device)
            res = self.model.get_hidden(sts, paths, eds, length, device)
            y = torch.tensor(y, dtype=torch.long) # convert tuple to tensor
            # detach
            sts = sts.detach().cpu()
            paths = paths.detach().cpu()
            eds = eds.detach().cpu()
            res = [s.detach().cpu() for s in res]

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

        self.shift1_pred_pos, self.shift1_pred_y, self.shift1_y = \
            common_predict(self.shift1_loader, self.model, self.device)

        self.shift2_pred_pos, self.shift2_pred_y, self.shift2_y = \
            common_predict(self.shift2_loader, self.model, self.device)
        print(f'train class num: {self.train_pred_pos.size(1)}, val class num: {self.val_pred_pos.size(1)}, shift1 class num: {self.shift1_pred_pos.size(1)}, shift2 class num: {self.shift2_pred_pos.size(1)}')
        self.class_num = self.train_pred_pos.size(1) # setting the class_num

    def save_truth(self):
        self.train_truth = self.train_pred_y.eq(self.train_y)
        self.val_truth = self.val_pred_y.eq(self.val_y)
        self.shift1_truth = self.shift1_pred_y.eq(self.shift1_y)
        self.shift2_truth = self.shift2_pred_y.eq(self.shift2_y)
        truth = [
            common_ten2numpy(self.train_truth), # torch to numpy cpu
            common_ten2numpy(self.val_truth),
            common_ten2numpy(self.shift1_truth),
            common_ten2numpy(self.shift2_truth)
        ]
        torch.save(
            truth, 
            os.path.join(self.SAVE_DIR, self.__class__.__name__) + '/truth.res'
        )