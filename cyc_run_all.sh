#!/usr/bin/env bash
buffer_list=(1000)
T_list=(1 2 4 8 16)
M_list=(16 32)
B_list=(1)
lr_list=(0.0003 0.001 0.003 0.01 0.03 0.1)
for buffer_size in ${buffer_list[@]}; do
    for T in ${T_list[@]}; do
        for M in ${M_list[@]}; do
            for B in ${B_list[@]}; do
                for lr in ${lr_list[@]}; do
                    python3 main.py --total_length=10000 --use_bptt=False --use_buffer_bptt=False --use_hybrid=True --buffer_length=$buffer_size --updates_per_step=$T --num_update=$M --batch_size=$B --num_units=4 --dataset='cycleworld' --cycleworld_size=6 --learning_rate=$lr --output_learning_rate=$lr --state_learning_rate=$lr --runs=10 --verbose=False &
                done
                wait
            done
        done
	done
done