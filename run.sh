#!/bin/bash

DATASET=mvtec
CLASS_NAME=bottle

python3 eval_WinCLIP.py --dataset ${DATASET} --class-name ${CLASS_NAME}