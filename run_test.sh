#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

python main_bala.py /raid/etegent/bala/data/soft_story/new_buildings/ --arch resnet50 --low-dim 32 -j 75 --K 50 -b 64 --lr 0.001 --evaluate --resume batch64_temp0.07_epoch0_checkpoint.pth.tar