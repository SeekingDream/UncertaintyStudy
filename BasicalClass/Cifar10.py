from model.cifar_10 import *
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from BasicalClass.common_function import *
from BasicalClass.Basic import BasicModule


class CIFAR10_Module(BasicModule):
    def __init__(self, device, load_poor=False):
        super(CIFAR10_Module, self).__init__(device, load_poor)
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        min_val = (0 - np.array(self.mean)) / np.array(self.std)
        max_val = (1 - np.array(self.mean)) / np.array(self.std)
        self.clip = (min(min_val), max(max_val))

        self.model = self.get_model()

        self.transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]
        )
        self.train_batch_size = 32
        self.test_batch_size = 32 if IS_DEBUG else 2000
        self.name = 'CIFAR10'
        self.loss = nn.CrossEntropyLoss()
        train_dataset = self.load_data(True)
        self.train_x, self.train_y = common_get_xy(train_dataset, self.test_batch_size, self.device)

        self.train_pred_pos, self.train_pred_y = \
            common_predict(self.train_x, self.model, self.train_batch_size, self.device)

        test_dataset = self.load_data(False)
        self.test_x, self.test_y = common_get_xy(test_dataset, self.test_batch_size, self.device)
        self.test_pred_pos, self.test_pred_y = \
            common_predict(self.test_x, self.model, self.train_batch_size, self.device)

        self.ground_truth = (self.test_pred_y == self.test_y).reshape([-1]).int()

        self.train_acc = common_cal_accuracy(self.train_pred_y, self.train_y)
        self.acc = common_cal_accuracy(self.test_pred_y, self.test_y)
        self.input_shape = (3, 32, 32)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer =  optim.Adam(self.model.parameters(), lr=0.01)
        self.class_num = 10
        self.eps = 0.3

        if not os.path.isdir('./' + self.name):
            os.mkdir('./' + self.name)

        print('construct the module', self.name, 'the accuracy is %0.3f, %0.3f' % (self.train_acc, self.acc))
        print('training data number is', len(self.train_x), 'test data number is ', len(self.test_x))

    def load_model(self):
        model = resnet18()
        state_dict = torch.load('../model_weight/cifar_10/resnet18.pt', map_location=self.device)
        model.load_state_dict(state_dict)
        return model

    def load_poor_model(self):
        model = AlexNet()
        state_dict = torch.load('../model_weight/cifar_10/AlexNet.h5', map_location=self.device)
        model.load_state_dict(state_dict)
        return model

    def load_data(self, is_train=True):
        return torchvision.datasets.CIFAR10(
            'data/cifar_10', train=is_train, transform=self.transform_train, target_transform=None, download=True)

    def get_hiddenstate(self, dataset):
        data_loader = DataLoader(dataset, batch_size=self.train_batch_size,
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
        sub_res_list = [torch.cat(i, dim = 0) for i in sub_res_list]
        return sub_res_list, sub_num

