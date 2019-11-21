#!/usr/bin/env bash
T_list=(8)
M_list=(1)
B_list=(4)
lr_list=(0.001)
for T in ${T_list[@]}; do
    for M in ${M_list[@]}; do
        for B in ${B_list[@]}; do
            for lr in ${lr_list[@]}; do
                python3 main_torch.py --dataset='cycleworld' --data_size 10000 --num_units=4 --model_name=fpp --state_update True --T $T --num_update $M --batch_size $B --lr=$lr --num_run=10 --use_gpu=False --verbose=True
            done
            wait
        done
    done
done