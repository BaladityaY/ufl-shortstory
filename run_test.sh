#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,3 

#counter=0
#until [ $counter -gt 100 ]
#do
python main_bala.py /raid/etegent/bala/data/areds/ --arch resnet18 --low-dim 32 -j 75 --K 50 -b 200 --lr 0.001 --evaluate --resume 'batch200_temp0.07_epoch63_checkpoint.pth.tar' #batch200_temp0.07_epoch${counter}_checkpoint.pth.tar
#((counter=counter+3))
#done