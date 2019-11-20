#!/usr/bin/env bash
lr_list=(0.01 0.003 0.001 0.0003 0.0001)
for lr in ${lr_list[@]}; do
    python3 main_uoro.py --total_length=5000 --dataset='cycleworld' --cycleworld_size=10 --n_input=2 --n_output=2 --num_units=4 --lr=$lr --runs=10 --verbose=False &
done
