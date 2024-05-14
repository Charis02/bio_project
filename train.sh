#!/bin/sh

source /etc/profile

module load anaconda/2023a-pytorch
module load cuda/11.8 
module load nccl/2.18.1-cuda11.8
export LD_LIBRARY_PATH=/usr/local/pkg/cuda/cuda-11.8/lib64

python train_nn.py