#!/bin/bash

DATASET=mvtec
CLASS_NAME=bottle

python3 eval_WinCLIP.py --dataset ${DATASET} --class-name ${CLASS_NAME} --use-adapter true --adapter-checkpoint ./result_adapter/mvtec-k-0/logger/bottle/adapter_ckpts/bottle_adapter_best.pt
