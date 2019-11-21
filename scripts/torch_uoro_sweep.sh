#!/usr/bin/env bash
lr_list=(0.001)
for lr in ${lr_list[@]}; do
    python3 main_torch.py --dataset='cycleworld' --data_size 50000 --num_units=4 --model_name=uoro --state_update True --lr=$lr --num_run=10 --use_gpu=False --verbose=True
done
