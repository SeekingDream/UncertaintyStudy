import torchvision
from torchvision import transforms
import torch


def preprocess(load_func, store_name):
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


def main():
    func_list = [
        torchvision.datasets.FashionMNIST,
        torchvision.datasets.CIFAR10,
        torchvision.datasets.CIFAR100
    ]
    store_list = [
        'fashion', 'cifar10', 'cifar100'
    ]
    for i in range(3):
        preprocess(func_list[i], store_list[i])


if __name__ == '__main__':
    main()