# FPP
Pytorch Implementation of two ICLR papers
1. [Training Recurrent Neural Networks Online by Learning Explicit State Variables](https://openreview.net/pdf?id=SJgmR0NKPr). Somjit Nath, Vincent Liu, Alan Chan, Xin Li, Adam White and Martha White.
2. [Unbiased Online Recurrent Optimization](https://arxiv.org/pdf/1702.05043.pdf). Corentin Tallec, Yann Ollivier.

## Running
```
python main_torch.py --dataset cw --data_size 10000 --model fpp --state_update True --overlap True --T 10 --num_update 1 --batch_size 1 --lr 0.001 --buffer_size 100 --num_run 10
```
You can see the optional arguments by running:
```
python main_torch.py --help
```
