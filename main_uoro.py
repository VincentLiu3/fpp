import tensorflow as tf
import torch
from env.data import *
from model.uoro import UORO_Model

import os
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# from tensorflow.python.util import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow loading GPU messages
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('verbose', True, 'Print Log')
flags.DEFINE_string('dataset', 'cycleworld', 'Name of dataset: cycleworld, stochastic_dataset, ptb')
flags.DEFINE_integer('cycleworld_size', 10, 'CycleWorld Size')
flags.DEFINE_integer('total_length', 10000, 'Length of entire dataset')
flags.DEFINE_integer('num_trajectory', 1, 'Number of trajectory')
flags.DEFINE_integer('runs', 1, 'Number of Runs')

flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_integer('n_input', 2, 'Dimension of Input')
flags.DEFINE_integer('n_output', 2, 'Dimension of Output')
flags.DEFINE_integer('num_units', 16, 'Hidden Units')
flags.DEFINE_bool('use_lstm', False, 'LSTM mode')


def print_msg(file_name, message, verbose):
    with open(file_name, 'a') as f:
        f.write('{}\n'.format(message))

    if verbose:
        logging.info(message)


assert FLAGS.num_trajectory == 1, 'set FLAGS.num_trajectory to 1.'

if FLAGS.dataset == 'cycleworld':
    data_name = '{}_cw'.format(FLAGS.cycleworld_size)
    dir_name = 'results/results-cw'
elif FLAGS.dataset in ['sd', 'stochastic_dataset']:
    data_name = 'sd'
    dir_name = 'results/results-sd'
elif FLAGS.dataset in ['lsd']:
    data_name = 'lsd'
    dir_name = 'results/results-lsd'
else:
    assert False, 'unknown dataset'

mini_dir_name = 'uoro'
filename = 'test'
# filename = '{},{},{},{},{},{},{},{},{},{}'.format(data_name, FLAGS.use_buffer_bptt, FLAGS.updates_per_step,
#                                                   FLAGS.num_update, FLAGS.batch_size, FLAGS.learning_rate,
#                                                   FLAGS.lambda_state, FLAGS.buffer_length, FLAGS.fix_buffer,
#                                                   FLAGS.runs)

os.makedirs('{}/{}/'.format(dir_name, mini_dir_name), exist_ok=True)
os.makedirs('logs/{}'.format(mini_dir_name), exist_ok=True)
log_file = 'logs/{}/{}.log'.format(mini_dir_name, filename)
data_file = '{}/{}/{}'.format(dir_name, mini_dir_name, filename)
result_file = '{}/{}_sweep.txt'.format(dir_name, mini_dir_name)

total_length = FLAGS.total_length
eval_size = 100
accuracy_series = np.zeros(shape=(FLAGS.runs, total_length//eval_size))
loss_series = np.zeros(shape=(FLAGS.runs, total_length//eval_size))

for run_no in range(FLAGS.runs):
    constant_seed = 0
    np.random.seed(constant_seed + run_no)

    msg = 'Run={}. cw_size={}.'.format(FLAGS.cycleworld_size, run_no)
    print_msg(log_file, msg, FLAGS.verbose)

    model = UORO_Model(FLAGS.n_input, FLAGS.num_units, FLAGS.n_output, FLAGS.lr)

    # output_op = model.output
    # train_op =  # use train for BRTT and train_seq for FPP
    # loss_op = model.loss
    # state_op = model.state

    # generate dataset
    if FLAGS.dataset == 'cycleworld':
        X, Y = generate_cw(FLAGS.cycleworld_size, FLAGS.num_trajectory, total_length)
        # X.shape = (1, 100000, 2)
        # Y.shape = (1, 100000, 2)
    elif FLAGS.dataset in ['sd', 'stochastic_dataset']:
        X, Y = generate_stochastic_data(FLAGS.num_trajectory, total_length)
    elif FLAGS.dataset in ['lsd']:
        X, Y = generate_stochastic_data(FLAGS.num_trajectory, total_length, is_short=False)

    X = torch.from_numpy(X).float()
    # Y = torch.from_numpy(Y).float()
    Y = torch.from_numpy(np.argmax(Y, axis=2)).long()

    # training
    iter_id = 0
    sum_acc = 0
    sum_loss = 0
    count = 0

    pred_series = []
    losses = []

    state = model.initialize_state()

    while iter_id < total_length:
        # get slice from time (iter-T to iter)
        x_t = X[:, iter_id:(iter_id+1)].reshape([1, 1, FLAGS.n_input])
        y_t = Y[:, iter_id:(iter_id+1)].reshape([1])

        loss, acc, state = model.train(x_t, state, y_t)

        sum_acc += acc
        sum_loss += loss

        if (iter_id+1) % 100 == 0:
            pred_series.append(sum_acc/count)
            losses.append(sum_loss/count)

            msg = 'Steps {:5d}. Accuracy {:.2f}. Loss {:4f}.'.format(iter_id+1, sum_acc/count, sum_loss/count)
            print_msg(log_file, msg, FLAGS.verbose)

            # corr = 0
            sum_acc = 0
            sum_loss = 0
            count = 0

        count += 1
        iter_id += 1

    accuracy_series[run_no] = pred_series
    loss_series[run_no] = losses

# save result
# with open(result_file, 'a') as f:
#     if FLAGS.total_length == 10000:
#         performance = '{:.4f},{:.4f}'.format(np.mean(accuracy_series), np.mean(loss_series))
#     else:
#         performance = '{:.4f},{:.4f}'.format(np.mean(accuracy_series), np.mean(accuracy_series[:, -10:]))
#     f.write('{},{}\n'.format(filename, performance))
#     if FLAGS.verbose:
#         logging.info('{},{}'.format(filename, performance))

# # np.save('results/baseline_{}-cw'.format(FLAGS.cycleworld_size),baseline)
# save_dict = {
#     'acc': accuracy_series,
#     'loss': loss_series
# }
# np.save(data_file, save_dict)
# logging.info('writing data to {}'.format(data_file))
