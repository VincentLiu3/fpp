python3 main.py --verbose=True --use_hybrid=False --use_lstm=True --use_bptt=False --use_prioritized_exp_replay=False --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=0 --total_length=1000000 --batch_size=100 --num_units=10 --dataset='stochastic_dataset' --runs=1



# python3 main.py --use_bptt=True --time_steps=1 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=2 &
# python3 main.py --use_bptt=True --time_steps=5 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=2 &
# # python3 main.py --use_bptt=True --time_steps=10 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=2 &
# python3 main.py --use_bptt=True --time_steps=15 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=2 &
# python3 main.py --use_bptt=True --time_steps=20 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=2 &
# python3 main.py --use_bptt=True --time_steps=25 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=2 &
# python3 main.py --use_bptt=True --time_steps=30 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=2 &

# python3 main.py --use_bptt=False --anneal_thresh_value=1.5 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=1 --state_updates_per_step=0  --batch_size=100 --num_units=64 --dataset='stochastic_dataset' --runs=2 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1.5 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=0 --batch_size=100 --num_units=64 --dataset='stochastic_dataset' --runs=2 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1.5 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=20 --state_updates_per_step=0 --batch_size=100 --num_units=64 --dataset='stochastic_dataset' --runs=2 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1.5 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=30 --state_updates_per_step=0 --batch_size=100 --num_units=64 --dataset='stochastic_dataset' --runs=2 &

# python3 main.py --use_bptt=False --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=0  --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=10 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=20 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &

# python3 main.py --use_bptt=False --buffer_length=1000 --updates_per_step=20 --state_updates_per_step=0  --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --buffer_length=1000 --updates_per_step=20 --state_updates_per_step=10 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &

# python3 main.py --use_bptt=False --buffer_length=1000 --updates_per_step=15 --state_updates_per_step=0  --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --buffer_length=1000 --updates_per_step=20 --state_updates_per_step=0  --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --buffer_length=1000 --updates_per_step=15 --state_updates_per_step=50 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --buffer_length=1000 --updates_per_step=20 --state_updates_per_step=100 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &
# python3 main.py --use_bptt=False --buffer_length=1000 --updates_per_step=50 --state_updates_per_step=1000 --batch_size=100 --num_units=16 --dataset='stochastic_dataset' --runs=5 &


# python3 main.py --use_bptt=True --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 --time_steps=5 &
# python3 main.py --use_bptt=True --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 --time_steps=15 &
# python3 main.py --use_bptt=False --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 --buffer_length=1000 --updates_per_step=1 --state_updates_per_step=0 &
# python3 main.py --use_bptt=False --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=0 &
# python3 main.py --use_bptt=False --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=10 &
# python3 main.py --use_bptt=False --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 --buffer_length=1000 --updates_per_step=15 --state_updates_per_step=15 &
# python3 main.py --use_bptt=False --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 --buffer_length=1000 --updates_per_step=20 --state_updates_per_step=20 &

wait
echo "DONE"