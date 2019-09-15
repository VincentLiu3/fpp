# python3 main.py --use_bptt=True --time_steps=1 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=True --time_steps=5 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=True --time_steps=10 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# # python3 main.py --use_bptt=True --time_steps=15 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# # python3 main.py --use_bptt=True --time_steps=20 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# # python3 main.py --use_bptt=True --time_steps=25 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# # python3 main.py --use_bptt=True --time_steps=30 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# # python3 main.py --use_bptt=True --time_steps=35 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# wait

# python3 main.py --use_bptt=False --use_prioritized_exp_replay=False --use_hybrid=False --buffer_length=1000 --updates_per_step=1 --state_updates_per_step=0  --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --use_prioritized_exp_replay=False --use_hybrid=False --buffer_length=1000 --updates_per_step=5 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --use_prioritized_exp_replay=False --use_hybrid=False --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &

# wait

python3 main.py --use_bptt=False --use_prioritized_exp_replay=True --alpha=0.5 --use_hybrid=False --buffer_length=1000 --updates_per_step=1 --state_updates_per_step=0  --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
python3 main.py --use_bptt=False --use_prioritized_exp_replay=True --alpha=0.5 --use_hybrid=False --buffer_length=1000 --updates_per_step=5 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
python3 main.py --use_bptt=False --use_prioritized_exp_replay=True --alpha=0.5 --use_hybrid=False --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &

wait

python3 main.py --use_bptt=False --use_prioritized_exp_replay=True --alpha=1.0 --use_hybrid=False --buffer_length=1000 --updates_per_step=1 --state_updates_per_step=0  --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
python3 main.py --use_bptt=False --use_prioritized_exp_replay=True --alpha=1.0 --use_hybrid=False --buffer_length=1000 --updates_per_step=5 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
python3 main.py --use_bptt=False --use_prioritized_exp_replay=True --alpha=1.0 --use_hybrid=False --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &

wait

python3 main.py --use_bptt=False --use_prioritized_exp_replay=True --alpha=0.0 --use_hybrid=False --buffer_length=1000 --updates_per_step=1 --state_updates_per_step=0  --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
python3 main.py --use_bptt=False --use_prioritized_exp_replay=True --alpha=0.0 --use_hybrid=False --buffer_length=1000 --updates_per_step=5 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
python3 main.py --use_bptt=False --use_prioritized_exp_replay=True --alpha=0.0 --use_hybrid=False --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &


# wait

# python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=2 --state_updates_per_step=0  --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=3 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=4 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=2 --state_updates_per_step=0  --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=3 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=4 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &

# wait

# python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=1 --state_updates_per_step=0  --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=5 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=15 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &

# wait

# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=1 --state_updates_per_step=0  --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=5 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=15 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &

# wait

# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=1 --state_updates_per_step=1  --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=5 --state_updates_per_step=5 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=10 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=15 --state_updates_per_step=15 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &

# wait

# python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=1 --state_updates_per_step=1  --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=5 --state_updates_per_step=5 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=10 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=15 --state_updates_per_step=15 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &

# wait

# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=20 --state_updates_per_step=0  --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=25 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=30 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=35 --state_updates_per_step=0 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
