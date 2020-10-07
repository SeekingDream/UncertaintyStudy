import torchvision
from torchvision import transforms
import torch
import torchtext
from torchtext import data
import numpy as np
import os

# image datasets
def preprocess_img(load_func, store_name):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    orig_db = load_func(
        './data/' + store_name, train=True, transform=transform, target_transform=None, download=True)
    test_db = load_func(
        './data/' + store_name, train=False, transform=transform, target_transform=None, download=True)
    train_size, val_size = int(len(orig_db) * 5 / 6), len(orig_db) - int(len(orig_db) * 5 / 6)
    train_db, val_db = torch.utils.data.random_split(orig_db, [train_size, val_size])

    torch.save(train_db, './data/' + store_name + '_train.pt')
    torch.save(val_db, './data/' + store_name + '_val.pt')
    torch.save(test_db, './data/' + store_name + '_test.pt')
    print('successful', store_name)


def process_image():
    func_list = [
        torchvision.datasets.FashionMNIST,
        torchvision.datasets.CIFAR10,
        torchvision.datasets.CIFAR100
    ]
    store_list = [
        'fashion', 'cifar10', 'cifar100'
    ]
    for i in range(3):
        preprocess_img(func_list[i], store_list[i])

# nlp datasets
def preprocess_text(dataset_name):
    dataset_path = './data/' + dataset_name
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)
    orig_db, test_db = \
        torchtext.datasets.text_classification.DATASETS[dataset_name](root=dataset_path)

    train_size, val_size = int(len(orig_db) * 4 / 5), len(orig_db) - int(len(orig_db) * 4 / 5)
    train_db, val_db = torch.utils.data.random_split(orig_db, [train_size, val_size])

    torch.save(train_db, './data/' + dataset_name + '_train.pt')
    torch.save(val_db, './data/' + dataset_name + '_val.pt')
    torch.save(test_db, './data/' + dataset_name + '_test.pt')
    print('successful', dataset_name)


def process_nlp():
    dataset_list = [
        'AG_NEWS', 'DBpedia', 'SogouNews'
    ]
    for data_name in dataset_list:
        preprocess_text(data_name)


if __name__ == '__main__':
    #process_image()
    process_nlp()