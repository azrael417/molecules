#!/bin/bash

# set device
export CUDA_VISIBLE_DEVICES=2,3

# connect to wandb
wandb login 6c8b9db0b520487f05d32ebc76fcea156bd85d58

# run stuff
python -m torch.distributed.launch --nproc_per_node=1 example_vae.py \
       -i /data/spike/rep1_closed.h5 \
       -s -a \
       -t resnet \
       -o /data/runs/ -m spike-cmaps-1 \
       --wandb_project_name covid_dl \
       -e 100 \
       -b 4 \
       -E 0 -D 1 \
       -h 3768 -w 3768 -d 471
