import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from utils import RAND_SEED
from torch.utils.data import DataLoader
from model.fashion import *
from train_model.utils_training import *
import os


def main():
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if device.type != 'cpu':
        torch.cuda.set_device(device=device)

    torch.manual_seed(RAND_SEED)

    data_dir = '../data/fashion'
    train_db, val_db, test_db = get_loader(data_dir)

    train_loader = DataLoader(train_db, batch_size=args.batch, shuffle=args.s, num_workers=args.worker)
    test_loader = DataLoader(test_db, batch_size=2000, shuffle=args.s, num_workers=args.worker)
    model = Fashion_CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, args.epoch):
        train_model(model, train_loader, optimizer, epoch, device)
        test_model(model, test_loader, device)

    save_dir = '../model_weight/fashion/'
    save_model(model, save_dir)


if __name__ == '__main__':
    args = get_input()
    main()
