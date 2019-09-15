#
#!/usr/bin/env bash
lambda_series=(0 0.1 0.5 1 5 10 15)
lr_series=(0.001 0.003 0.01 0.03 0.1 0.3)
for lambda in ${lambda_series[@]}; do
	for lr in ${lr_series[@]}; do
  		python3 main.py --total_length=100000 --use_bptt=False --use_prioritized_exp_replay=False --use_hybrid=True --anneal_thresh_value=1.0 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=1 --state_updates_per_step=0  --batch_size=1 --num_units=4 --dataset='cycleworld' --cycleworld_size=6 --learning_rate=$lr --output_learning_rate=$lr --state_learning_rate=0.01 --lambda=$lambda --runs=30 &
  		python3 main.py --total_length=100000 --use_bptt=False --use_prioritized_exp_replay=False --use_hybrid=True --anneal_thresh_value=1.0 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=2 --state_updates_per_step=0  --batch_size=1 --num_units=4 --dataset='cycleworld' --cycleworld_size=6 --learning_rate=$lr --output_learning_rate=$lr --state_learning_rate=0.01 --lambda=$lambda --runs=30 &
  		python3 main.py --total_length=100000 --use_bptt=False --use_prioritized_exp_replay=False --use_hybrid=True --anneal_thresh_value=1.0 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=3 --state_updates_per_step=0  --batch_size=1 --num_units=4 --dataset='cycleworld' --cycleworld_size=6 --learning_rate=$lr --output_learning_rate=$lr --state_learning_rate=0.01 --lambda=$lambda --runs=30 &
		wait
	done
done