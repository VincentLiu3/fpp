# FPP
Pytorch Implementation of 
1. Training Recurrent Neural Networks Online by Learning Explicit State Variables
2. Unbiased Online Recurrent Optimization

## Running
```
python3 main_torch.py --dataset='cycleworld' --data_size 10000 --num_units=4 --model_name='fpp' --state_update True --T 10 --num_update 1 --batch_size 1 --lr=0.001 --num_run=10 --use_gpu=False --verbose=True
```
You can see the optional arguments by running:
```
python main_torch.py --help
```
