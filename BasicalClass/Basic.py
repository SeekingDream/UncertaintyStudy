from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader,Subset
from BasicalClass.common_function import common_predict

class BasicModule:
    __metaclass__ = ABCMeta

    def __init__(self,device, load_poor):
        self.device = device
        self.load_poor = load_poor
        self.train_batch_size = 1000
        self.test_batch_size = 1000

    def get_model(self):
        if not self.load_poor:
            model = self.load_model().to(self.device)
        else:
            model = self.load_poor_model().to(self.device)
        model.eval()
        print('model name is ', model.__class__.__name__)
        return model

    @abstractmethod
    def load_model(self):
        return None

    @abstractmethod
    def load_poor_model(self):
        return None

    def get_loader(self, train_db, val_db, test_db):
        train_loader = DataLoader(
            train_db, batch_size=self.train_batch_size,
            shuffle=True, collate_fn=None)
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
            common_predict(self.validation_loader, self.model, self.device)

        self.test_pred_pos, self.test_pred_y, self.test_y = \
            common_predict(self.test_loader, self.model, self.device)
