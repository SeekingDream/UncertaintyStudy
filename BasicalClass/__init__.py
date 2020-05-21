from .Basic import BasicModule
from .AndroidMalware import Android_Module
from .Cifar10 import CIFAR10_Module
from .Cifar100 import CIFAR100_Module
from .Fashion import Fashion_Module
from .common_function import *

MODULE_LIST = [
    Fashion_Module,
    CIFAR10_Module,
    CIFAR100_Module,
    Android_Module,
]

if not os.path.isdir('./Result/'):
    os.mkdir('./Result/')