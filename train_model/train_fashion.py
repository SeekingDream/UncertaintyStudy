import argparse                         # 加载处理命令行参数的库
import torch                            # 引入相关的包
import torch.optim as optim
from torchvision import datasets, transforms  # 加载pytorch官方提供的dataset
from utils import DEVICE, RAND_SEED
from torch.utils.data import DataLoader
from model.fashion import *
import torch.nn as nn
import os

def train_model( model, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)       # negative log likelihood loss(nll_loss), sum up batch cross entropy
        loss.backward()
        optimizer.step()                        # 根据parameter的梯度更新parameter的值
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():       #无需计算梯度
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    torch.manual_seed(RAND_SEED)

    data_dir = './data/fashion/'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)


    train_db = torch.load(data_dir + 'train.pt')
    val_db = torch.load(data_dir + 'val.pt')
    test_db = torch.load(data_dir + 'test.pt')

    train_loader = DataLoader(train_db,batch_size=1024, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_db,batch_size=2000, shuffle=True)

    model = Fashion_CNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)   #optimizer存储了所有parameters的引用，每个parameter都包含gradient

    for epoch in range(1, 20):
        train_model(model, train_loader, optimizer, epoch)
        test_model(model, test_loader)

    save_dir = './model_weight/fashion/'

    if not os.path.isdir('./model_weight/'):
        os.mkdir('./model_weight/')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_name = save_dir + model.name + '.h5'
    torch.save(model.state_dict(), save_name)


if __name__ == '__main__':
    main()