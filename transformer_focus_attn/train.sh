#!/bin/bash
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python main.py -train -logfile train.log -exp_name train -pointer_gen True -batch_size 16 -load train_model/20w_1.616693model.pth
