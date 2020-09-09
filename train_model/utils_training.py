import torch
import os
import torch.nn as nn
from utils import RAND_SEED
import argparse


def train_model(model, train_loader, optimizer, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)       # negative log likelihood loss(nll_loss), sum up batch cross entropy
        loss.backward()
        optimizer.step()                        # 根据parameter的梯度更新parameter的值
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return None


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():       #无需计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def save_model(model, save_dir):
    if not os.path.isdir('../model_weight/'):
        os.mkdir('../model_weight/')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_name = save_dir + model.__class__.__name__ + '.h5'
    torch.save(model.state_dict(), save_name)


def get_loader(data_dir):
    train_db = torch.load(data_dir + '_train.pt')
    val_db = torch.load(data_dir + '_val.pt')
    test_db = torch.load(data_dir + '_test.pt')
    return train_db, val_db, test_db


def get_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-device', type=int, default=2, help='the gpu id')
    parser.add_argument('-worker', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-epoch', type=int, default=100, help='epoch for training')
    parser.add_argument('-batch', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
    args = parser.parse_args()
    return args