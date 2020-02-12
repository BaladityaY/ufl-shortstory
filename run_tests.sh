#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

kval=50

python main_bala.py /raid/etegent/bala/data/areds/ --arch resnet18 --low-dim 32 -j 75 --K ${kval} -b 200 --lr 0.001 --evaluate --resume batch200_temp0.07_epoch63_checkpoint.pth.tar

python main_bala_coarse.py /raid/etegent/bala/data/areds/ --arch resnet18 --low-dim 32 -j 75 --K ${kval} -b 200 --lr 0.001 --evaluate --resume batch200_temp0.07_epoch63_checkpoint.pth.tar
