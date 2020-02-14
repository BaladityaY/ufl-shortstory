#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
python main_bala.py /raid/etegent/bala/data/soft_story/new_buildings/ --arch resnet50 --low-dim 32 -j 1 --K 50 -b 64 --lr 0.001 --epochs 1 --fine_tune lemniscate32_resnet50.pth.tar