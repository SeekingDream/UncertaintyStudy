from model.android import *
import numpy as np
import torch
from utils import IS_DEBUG
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.optim as optim
import os

my_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def malware_loader(x, y):
    res = torch.cat((y.float(), x), dim = 1)
    data_loader = DataLoader(
        res, batch_size=2048,
        shuffle=True, collate_fn=my_collate_fn,
    )
    return data_loader

def my_collate_fn(batch):
    x = [data[1:] for data in batch]
    y = [data[:1].long() for data in batch]
    return torch.stack(x, dim=0), torch.stack(y, dim = 0)

def train_model():
    x = np.load('data/android_malware/x_train.npy')
    y = np.load('data/android_malware/y_train.npy')
    if IS_DEBUG:
        x = x[:1000, :500]
        y = y[:1000]
    data_num, feature_num = np.shape(x)
    print(data_num, feature_num)
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    train_loader = malware_loader(x, y)
    model = Android_Poor(feature_num).to(my_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(5):
        sum_loss = 0
        for i, (x, y) in enumerate(train_loader):
            print(i)
            optimizer.zero_grad()
            x = x.to(my_device)
            y = y.to(my_device).view([-1])
            pred_y = model(x)
            loss = criterion(pred_y, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss
        print('loss is ', sum_loss/data_num)

    model.eval()

    x = np.load('data/android_malware/x_test.npy')
    y = np.load('data/android_malware/y_test.npy')
    if IS_DEBUG:
        x = x[:1000, :500]
        y = y[:1000]
    data_num, feature_num = np.shape(x)
    print(data_num, feature_num)
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    train_loader = malware_loader(x, y)
    y_list, pred_list = [], []
    for (x, y) in train_loader:
        optimizer.zero_grad()
        x = x.to(my_device)
        y = y.to(my_device)
        pred_y = model(x)
        _, pred_y = torch.max(pred_y, dim=1)
        y_list.append(y)
        pred_list.append(pred_y)
    pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy().reshape([-1])
    y_list = torch.cat(y_list, dim=0).detach().cpu().numpy().reshape([-1])
    print(np.sum(pred_list == y_list)/data_num)
    path = './model_weight/android/'
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(model.state_dict(), path + model.__class__.__name__ + '.h5')
    print('save model !', path + model.__class__.__name__ + '.h5')


if __name__ == '__main__':
    train_model()