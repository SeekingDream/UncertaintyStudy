from model.fashion import *
import torchvision
from torchvision import transforms
import torch.nn as nn
from BasicalClass.common_function import *
from BasicalClass.Basic import BasicModule
import torch.optim as optim


class Fashion_Module(BasicModule):
    def __init__(self, device, load_poor = False):
        super(Fashion_Module, self).__init__(device, load_poor)
        self.mean = (0,) #(0.1307,)
        self.std =  (1,) #(0.3081,)
        min_val = 0 #(0 - np.array(self.mean)) / np.array(self.std)
        max_val = 1 #(1 - np.array(self.mean)) / np.array(self.std)
        self.clip = (min_val, max_val)

        self.transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]
        )


        self.train_batch_size = 256
        self.test_batch_size = 256 if IS_DEBUG else 5000
        self.name = 'Fashion'
        self.train_loader, self.val_loader, self.test_loader = self.load_data()

        self.get_information()

        self.input_shape = (1, 28, 28)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.class_num = 10
        self.acc = common_cal_accuracy(self.test_pred_y, self.test_y)
        self.train_acc = common_cal_accuracy(self.train_pred_y, self.train_y)
        self.eps = 1.0
        if not os.path.isdir('./Result/' + self.name):
            os.mkdir('./Result/' + self.name)
        self.save_truth()
        print('construct the module', self.name, 'the accuracy is %0.3f, %0.3f' % (self.train_acc, self.acc))


    def load_model(self):
        model = Fashion_CNN()
        model.load_state_dict(
            torch.load('./model_weight/fashion/' + model.name + '.h5', map_location=self.device)
        )
        return model

    def load_poor_model(self):
        model = Fashion_MLP()
        model.load_state_dict(
            torch.load('./model_weight/fashion/' + model.name + '.h5', map_location=self.device)
        )
        return model

    def load_data(self):
        train_db = torch.load('./data/fashion/' + 'train.pt')
        val_db = torch.load('./data/fashion/' + 'val.pt')
        test_db = torch.load('./data/fashion/' + 'test.pt')
        return self.get_loader(train_db, val_db, test_db)

    def get_hiddenstate(self, dataset):
        data_loader = DataLoader(
            dataset, batch_size=self.train_batch_size,
            shuffle=False, collate_fn=None,
        )
        data_num = 0
        sub_num = self.model.sub_num
        sub_res_list = [[] for _ in sub_num]
        for i, x in enumerate(data_loader):
            data_num += len(x)
            x = x.to(self.device)
            self.model.to(self.device)
            sub_y = self.model.get_hidden(x)
            for j in range(len(sub_num)):
                sub_res_list[j].append(sub_y[j])
            if IS_DEBUG and i >= DEBUG_NUM:
                break
        sub_res_list = [torch.cat(i, dim=0) for i in sub_res_list]
        return sub_res_list, sub_num



if __name__ == '__main__':

    Fashion_Module(DEVICE)
