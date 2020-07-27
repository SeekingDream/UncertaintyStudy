#!/bin/bash


python ./train_fashion.py -device=1 -epoch=60 -lr=0.01
python ./train_cifar10.py -device=2 -epoch=100 -lr=0.01
python ./train_cifar100.py -device=6 -epoch=100 -lr=0.01