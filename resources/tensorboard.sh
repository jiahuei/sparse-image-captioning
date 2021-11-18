#!/bin/bash

export CUDA_VISIBLE_DEVICES=""

LOG_DIR="/home/jiahuei/Documents/1_TF_files/relation_trans/mscoco_v1"

tensorboard --logdir=${LOG_DIR} --host 0.0.0.0








