#!/bin/bash

DEVICE=1

python  ./Baseline/testScript.py --data_type=0 --device_id=$DEVICE
echo 'end the filter module 0'

python  ./Baseline/testScript.py --data_type=1 --device_id=$DEVICE
echo 'end the filter module 1'


python  ./Baseline/testScript.py --data_type=2 --device_id=$DEVICE
echo 'end the filter module 2'


python  ./Baseline/testScript.py --data_type=3 --device_id=$DEVICE
echo 'end the filter module 3'