python3 main.py --use_bptt=True --time_steps=1 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=True --time_steps=2 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=True --time_steps=3 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=True --time_steps=5 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=True --time_steps=6 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &

wait

python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=1 --state_updates_per_step=0  --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=3 --state_updates_per_step=0 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=5 --state_updates_per_step=0 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=0 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &

wait

python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=1 --state_updates_per_step=0  --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=3 --state_updates_per_step=0 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=5 --state_updates_per_step=0 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=0 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &

wait

python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=1 --state_updates_per_step=1  --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=3 --state_updates_per_step=3 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=5 --state_updates_per_step=5 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=10 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &

wait

python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=1 --state_updates_per_step=1  --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=3 --state_updates_per_step=3 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=5 --state_updates_per_step=5 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
python3 main.py --use_bptt=False --anneal_thresh_value=1.2 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=10 --state_updates_per_step=10 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &

# wait

# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=20 --state_updates_per_step=0  --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=25 --state_updates_per_step=0 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=30 --state_updates_per_step=0 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
# python3 main.py --use_bptt=False --anneal_thresh_value=1 --anneal_thresh_steps=499 --buffer_length=1000 --updates_per_step=35 --state_updates_per_step=0 --batch_size=1 --num_units=4 --dataset='cycleworld' --runs=5 &
