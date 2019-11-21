import argparse
import logging


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def get_parser():
    """
    Instruction: USE these commands to get args
    parser = get_parser()
    args = parser.parse_args('')
    args = add_args(args)
    """
    parser = argparse.ArgumentParser(description='Pytorch Experiment')
    # dataset
    parser.add_argument('--dataset', type=str, default='cycleworld', help='Dataset: cycleworld, stochastic_dataset, lsd, ptb')
    parser.add_argument('--env_size', type=int, default=10, help='Dataset size (default: 10 for cycleworld)')
    parser.add_argument('--data_size', type=int, default=10000, help='Size of dataset')
    parser.add_argument('--num_trajectory', type=int, default=1, help='Number of trajectory (default: 1)')
    parser.add_argument('--num_run', type=int, default=1, help='Number of runs')
    parser.add_argument('--verbose', type=str2bool, default='True', help='Print or write to a file')
    # representation
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for RNN')
    parser.add_argument('--reg_lambda', type=float, default=1.0, help='Lambda for FPP')
    # RNN
    parser.add_argument('--n_input', type=int, default=2, help='Dimension of input')
    parser.add_argument('--n_output', type=int, default=2, help='Dimension of output')
    parser.add_argument('--num_units', type=int, default=4, help='Number of hidden units')
    parser.add_argument('--use_lstm', type=str2bool, default='False', help='Use LSTM')
    # model
    parser.add_argument('--model_name', type=str, default='fpp', help='Model: uoro or fpp')
    parser.add_argument('--state_update', type=str2bool, default='True', help='FPP: w/ or w/o state update')
    parser.add_argument('--T', type=int, default=10, help='Truncate parameter')
    parser.add_argument('--buffer_size', type=int, default=1000, help='Buffer size')
    parser.add_argument('--num_update', type=int, default=1, help='Number of updates per step M')
    parser.add_argument('--batch_size', type=int, default=1, help='Mini-batch size B')
    parser.add_argument('--use_gpu', type=str2bool, default='False', help='Use GPU')
    return parser


def add_args(args):
    assert args.num_trajectory == 1, 'set num_trajectory to 1.'
    return args


def print_msg(file_name, message, verbose):
    if verbose:
        logging.info(message)
    else:
        with open(file_name, 'a') as f:
            f.write('{}\n'.format(message))