from  BasicalClass import *


MODULE_LIST = [
    Fashion_Module,
    CIFAR10_Module,
    CIFAR100_Module,
    Android_Module,
]

if not os.path.isdir('./Result/'):
    os.mkdir('./Result/')









def main():
    m = CIFAR10_Module()
    print()


if __name__ == '__main__':
    main()
