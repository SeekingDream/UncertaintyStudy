import torchvision
from torchvision import transforms
import torch


def preprocess():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ]
    )
    orig_db = torchvision.datasets.FashionMNIST(
        './data/fashion/', train=True, transform = transform, target_transform=None, download=True)
    test_db = torchvision.datasets.FashionMNIST(
        './data/fashion/', train=False, transform=transform, target_transform=None, download=True)
    train_db, val_db = torch.utils.data.random_split(orig_db, [50000, 10000])

    torch.save(train_db, './data/fashion/' + 'train.pt')
    torch.save(val_db, './data/fashion/' + 'val.pt')
    torch.save(test_db, './data/fashion/' + 'test.pt')
    print('successful')

preprocess()