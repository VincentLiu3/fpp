#!/usr/bin/env bash
T_list=(1 2 4 8)
M_list=(1 2 4)
B_list=(1 2 4 8)
lr_list=(0.01 0.003 0.001 0.0003 0.0001)
for T in ${T_list[@]}; do
    for M in ${M_list[@]}; do
        for B in ${B_list[@]}; do
            for lr in ${lr_list[@]}; do
                python3 main_uoro.py --dataset='cycleworld' --data_size 10000 --env_size=10 --num_units=4 --state_update True --T $T --num_update $M --batch_size $B --lr=$lr --num_run=10 --verbose=False &
            done
            wait
        done
    done
done