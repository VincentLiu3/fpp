import tensorflow as tf
from data import *
from model.replay_buffer import Replay_Buffer
from model.fpt import FPT_Model
# from model.bptt import BPTT_Model

import os
import logging

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

'''
Note on tf version:
1. pip install numpy==1.16.4
'''

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow loading GPU messages
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('verbose', False, 'Print Log')
flags.DEFINE_string('dataset', 'stochastic_dataset', 'Name of dataset: cycleworld, stochastic_dataset, ptb')
flags.DEFINE_integer('cycleworld_size', 10, 'CycleWorld Size')
flags.DEFINE_integer('total_length', 100000, 'Length of entire dataset')
flags.DEFINE_integer('num_trajectory', 1, 'Number of trajectory')
flags.DEFINE_integer('runs', 1, 'Number of Runs')

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_float('output_learning_rate', 0.001, 'Learning rate for Output Weights')
flags.DEFINE_float('state_learning_rate', 0.001, 'Learning rate for States')
flags.DEFINE_integer('n_input', 2, 'Dimension of Input')
flags.DEFINE_integer('n_classes', 2, 'Dimension of Output')
flags.DEFINE_integer('num_units', 16, 'Hidden Units')
flags.DEFINE_float('lambda_state', 1, 'Lambda')

flags.DEFINE_bool('use_hybrid', True, 'Hybrid mode')
flags.DEFINE_bool('use_buffer_bptt', False, 'Buffer BPTT')
flags.DEFINE_bool('fix_buffer', False, 'for fixed buffer experiments')

flags.DEFINE_integer('time_steps', 1, 'Truncate Parameter')
flags.DEFINE_integer('buffer_length', 1000, 'Buffer Length')
flags.DEFINE_integer('updates_per_step', 10, 'Number of Updates per Step T')
flags.DEFINE_integer('num_update', 1, 'Number of State Updates per Step M')
flags.DEFINE_integer('batch_size', 1, 'Mini-batch Size B')

flags.DEFINE_integer('offline_update', 10000, 'Number of offline update')

flags.DEFINE_bool('use_lstm', False, 'LSTM mode')
flags.DEFINE_bool('use_bptt', False, 'BPTT mode')
flags.DEFINE_bool('clip_gradients', False, 'Clip Gradients')
flags.DEFINE_bool('use_prioritized_exp_replay', False, 'Use Prioritized Experience Replay')
flags.DEFINE_float('alpha', 0.5, 'Alpha for PER')
flags.DEFINE_integer('anneal_thresh_steps', 499, 'Steps after which to anneal threshold')
flags.DEFINE_float('anneal_thresh_value', 1.0, 'Value by which threshold will be annealed')
flags.DEFINE_integer('state_updates_per_step', 0, 'Number of State Updates per Step')

assert FLAGS.num_trajectory == 1, 'set FLAGS.num_trajectory to 1.'
assert FLAGS.time_steps == 1, 'set FLAGS.time_steps to 1.'
assert FLAGS.use_hybrid is True, 'use hybrid.'
assert FLAGS.use_bptt is False, 'do not use bptt.'
# assert FLAGS.learning_rate == FLAGS.output_learning_rate and FLAGS.output_learning_rate == FLAGS.state_learning_rate, 'set lr.'
if FLAGS.fix_buffer:
    assert FLAGS.buffer_length == FLAGS.total_length, 'set total_length = buffer_length for fix buffer.'

if FLAGS.dataset == 'cycleworld':
    data_name = '{}_cw'.format(FLAGS.cycleworld_size)
    dir_name = 'results/results-cw/'
elif FLAGS.dataset == 'stochastic_dataset':
    data_name = 'sd'
    dir_name = 'results/results-sd/'
elif FLAGS.dataset == 'ptb':
    data_name = 'ptb'
    dir_name = 'results/results-ptb/'
else:
    assert False, 'unknown dataset'

if FLAGS.use_buffer_bptt:
    pathname = dir_name + 'buffer_bptt/'

else:
    pathname = dir_name + 'fpp/'

filename = '{},{},{},{},{},{},{},{},{},{}'.format(data_name, FLAGS.use_buffer_bptt, FLAGS.updates_per_step,
                                                  FLAGS.num_update, FLAGS.batch_size, FLAGS.learning_rate,
                                                  FLAGS.lambda_state, FLAGS.buffer_length, FLAGS.fix_buffer,
                                                  FLAGS.runs)

os.makedirs(pathname, exist_ok=True)
os.makedirs('logs', exist_ok=True)
if FLAGS.fix_buffer:
    log_file = "logs/fix-{}.log".format(filename)
else:
    log_file = "logs/{}.log".format(filename)

num_batches = FLAGS.total_length
accuracy_series = np.zeros(shape=(FLAGS.runs, num_batches//100))
loss_series = np.zeros(shape=(FLAGS.runs, num_batches//100))
ave_val_acc = np.zeros(shape=(FLAGS.runs, FLAGS.offline_update//500))
ave_val_loss = np.zeros(shape=(FLAGS.runs, FLAGS.offline_update//500))

for run_no in range(FLAGS.runs):
    constant_seed = 0
    tf.reset_default_graph()
    tf.set_random_seed(constant_seed + run_no)
    np.random.seed(constant_seed + run_no)

    if FLAGS.use_bptt:
        with open(log_file, 'a') as f:
            f.write('BPTT size: {} time_steps: {} run_no: {}\n'.format(FLAGS.cycleworld_size, FLAGS.time_steps, run_no))
    else:
        msg = 'cw_size={}. buffer_size={}. T={}. M={}. Run={}.'.format(FLAGS.cycleworld_size, FLAGS.buffer_length,
                                                                       FLAGS.num_update, FLAGS.updates_per_step, run_no)
        with open(log_file, 'a') as f:
            f.write('{}\n'.format(msg))

        if FLAGS.verbose:
            logging.info(msg)

    model = FPT_Model(FLAGS.dataset, FLAGS.use_buffer_bptt, FLAGS.fix_buffer,
                      FLAGS.n_input, FLAGS.n_classes, FLAGS.num_units, FLAGS.batch_size, FLAGS.updates_per_step,
                      FLAGS.learning_rate, FLAGS.clip_gradients, FLAGS.lambda_state, run_no)

    # if FLAGS.use_prioritized_exp_replay:
    #     buffer = Prioritized_Replay_Buffer(FLAGS.buffer_length, FLAGS.alpha)
    # else:
    buffer = Replay_Buffer(FLAGS.buffer_length)

    output_op = model.output
    # train_op =  # use train for BRTT and train_seq for FPP
    loss_op = model.loss
    state_op = model.state

    with tf.Session() as sess:  
        init = tf.global_variables_initializer()
        sess.run(init)
        # X,Y -> [num_trajectory, num_batches, n_input]
        if FLAGS.dataset == 'cycleworld':
            X, Y = generate_cw(FLAGS.cycleworld_size, FLAGS.num_trajectory, num_batches)
            # X.shape = (1, 100000, 2)
            # Y.shape = (1, 100000, 2)
        elif FLAGS.dataset == 'stochastic_dataset':
            X, Y = generate_stochastic_data(FLAGS.num_trajectory, num_batches)
        elif FLAGS.dataset == 'ptb':
            pass

        iter_id = 0  # FLAGS.time_steps
        # corr = 0
        # pred_zero = 0

        # total loss/accuracy over 100 iterations
        sum_acc = 0
        sum_loss = 0
        count = 0

        pred_series = []
        losses = []

        if FLAGS.use_lstm:
            state = np.zeros(shape=[2, FLAGS.num_trajectory, FLAGS.num_units])
        else:
            # initialize s_0 as zero vector
            state = np.zeros(shape=[FLAGS.num_trajectory, FLAGS.num_units])

        while iter_id < num_batches:
            # get slice from time (iter-T to iter)
            x_t = X[:, iter_id:(iter_id+1)].reshape([1, FLAGS.num_trajectory, FLAGS.n_input])
            y_t = Y[:, iter_id:(iter_id+1)].reshape([FLAGS.num_trajectory, FLAGS.n_classes])
            # x_t = batch_x[:, 0, :].reshape([FLAGS.time_steps, FLAGS.num_trajectory, FLAGS.n_input])
            # y_t = batch_y[:, 0, :]

            state_tm1 = state  # save previous state vector
            output, state, acc, loss = sess.run([model.output, model.state, model.accuracy, model.loss],
                                                feed_dict={
                                                model.x: x_t,               # shape = (None, 1, n_input)
                                                model.y: y_t,               # shape = (None, n_class)
                                                model.init_state: state     # shape = (1, n_unit)
                                                })

            data = np.squeeze(x_t, axis=0), state_tm1, state, y_t  # (o_t, s_t-1, s_t, y_t) in the paper
            buffer.add(data)
            # print('add', x_t.shape)
            # print('add', state_tm1.shape)
            # print('add', state.shape)
            # print('add', y_t.shape)

            # x_t.shape = [1, 1, n_input]
            # state.shape = [1, n_units]
            # y_t.shape = [1, n_class[]

            sum_acc += acc
            sum_loss += loss

            if FLAGS.fix_buffer is False:
            # if True:
                # TRAINING
                if iter_id >= FLAGS.updates_per_step:  # updates_per_step = T
                    for _ in range(FLAGS.num_update):
                        x_t_series, s_tm1_series, s_t_series, y_t_series, idx_series = buffer.sample_batch(FLAGS.batch_size, FLAGS.updates_per_step)
                        # print('sample', x_t_series.shape)
                        # print('sample', s_tm1_series.shape)
                        # print('sample', s_t_series.shape)
                        # print('sample', y_t_series.shape)
                        # print('sample', idx_series)

                        _, new_s_tm1, new_s_t = sess.run([model.train_seq, model.state_tm1_c_seq, model.state_t_c_seq],
                                                         feed_dict={
                                                            model.x_t: x_t_series,  # [T, B, n_input]
                                                            model.y_t: y_t_series,  # [T, B, n_class]
                                                            model.s_tm1: s_tm1_series,  # [T, B, n_unit]
                                                            model.s_t: s_t_series  # [T, B, n_unit]
                                                         })

                        for b in range(idx_series.shape[0]):
                            # print(b)
                            new_s_tm1_series = np.expand_dims(s_tm1_series[:, b, :], axis=1)
                            new_s_t_series = np.expand_dims(s_t_series[:, b, :], axis=1)
                            new_s_tm1_series[0] = np.expand_dims(new_s_tm1[b], axis=0)
                            new_s_t_series[-1] = np.expand_dims(new_s_t[b], axis=0)

                            new_x_t_series = np.expand_dims(x_t_series[:, b, :], axis=1)
                            new_y_t_series = np.expand_dims(y_t_series[:, b, :], axis=1)

                            # print('replace', new_x_t_series.shape)
                            # print('replace', new_s_tm1_series.shape)
                            # print('replace', new_s_t_series.shape)
                            # print('replace', new_y_t_series.shape)
                            # update S_{t-T} and S_t in the buffer
                            buffer.replace(idx_series[b], new_x_t_series, new_s_tm1_series, new_s_t_series,
                                           new_y_t_series, FLAGS.updates_per_step)

            if (iter_id+1) % 100 == 0:
                # steps.append(iter)-
                pred_series.append(sum_acc/count)
                losses.append(sum_loss/count)

                with open(log_file, 'a') as f:
                    f.write('Steps {:5d}. Accuracy {:.2f}. Loss {:4f}.\n'.format(iter_id+1, sum_acc/count, sum_loss/count))

                if FLAGS.verbose:
                    logging.info('Steps {:5d}. Accuracy {:.2f}. Loss {:4f}.'.format(iter_id+1, sum_acc/count, sum_loss/count))

                # corr = 0
                sum_acc = 0
                sum_loss = 0
                count = 0

            count += 1
            iter_id += 1

        if FLAGS.fix_buffer:
            T = FLAGS.updates_per_step
            use_state = True

            online_loss = 0
            val_acc_list = []
            val_loss_list = []

            for iter_id in range(FLAGS.offline_update):
                # Training
                x_t_series, s_tm1_series, s_t_series, y_t_series, idx_series = buffer.sample_batch(FLAGS.batch_size,
                                                                                                   FLAGS.updates_per_step)

                _, new_s_tm1, new_s_t = sess.run([model.train_seq, model.state_tm1_c_seq, model.state_t_c_seq],
                                                               feed_dict={
                                                                     model.x_t: x_t_series,  # [T, B, n_input]
                                                                     model.y_t: y_t_series,  # [T, B, n_class]
                                                                     model.s_tm1: s_tm1_series,  # [T, B, n_unit]
                                                                     model.s_t: s_t_series  # [T, B, n_unit]
                                                                })
                # model.pred_acc,
                # online_loss += online_loss_

                for b in range(idx_series.shape[0]):
                    new_s_tm1_series = np.expand_dims(s_tm1_series[:, b, :], axis=1)
                    new_s_t_series = np.expand_dims(s_t_series[:, b, :], axis=1)
                    new_s_tm1_series[0] = np.expand_dims(new_s_tm1[b], axis=0)
                    new_s_t_series[-1] = np.expand_dims(new_s_t[b], axis=0)

                    new_x_t_series = np.expand_dims(x_t_series[:, b, :], axis=1)
                    new_y_t_series = np.expand_dims(y_t_series[:, b, :], axis=1)

                    # update S_{t-T} and S_t in the buffer
                    buffer.replace(idx_series[b], new_x_t_series, new_s_tm1_series, new_s_t_series,
                                   new_y_t_series, FLAGS.updates_per_step)

                # Testing: what do we want to report the accuracy?
                if (iter_id+1) % 500 == 0:
                    # Testing
                    test_id = 0
                    val_loss = 0
                    val_acc = 0
                    count = 0

                    if use_state:
                        while test_id + T <= num_batches:
                            ob_batch, state_tm1_batch, _, y_batch, _ = buffer._encode_sample(np.arange(test_id, test_id+T))

                            val_loss_, val_acc_ = sess.run([model.loss_target_seq, model.pred_acc], feed_dict={
                                                            model.x_t: ob_batch,          # (None, 1, n_input)
                                                            model.y_t: y_batch,  # (None, n_class)
                                                            model.s_tm1: state_tm1_batch
                                                            # model.init_state: state_tm1_batch[0, :, :]  # (None, n_unit)
                                                          })

                            val_loss += val_loss_
                            val_acc += val_acc_
                            test_id += 1
                            count += 1

                    else:
                        training_state = np.zeros((1, FLAGS.num_units))

                        while test_id <= num_batches - T:
                            batch_x = X[:, test_id:(test_id+T)].reshape([T, 1, FLAGS.n_input])
                            batch_y = Y[:, test_id:(test_id+T)].reshape([T, 1, FLAGS.n_classes])

                            val_loss_, val_acc_, training_state = sess.run([model.val_loss, model.val_acc, model.val_state],
                                                                           feed_dict={
                                                                           model.val_input: batch_x,  # [T, 1, n_input]
                                                                           model.val_label: batch_y,  # [T, 1, n_class]
                                                                           model.init_state: training_state  # [1, n_unit]
                                                                           })
                            # print(val_acc_)
                            val_loss += val_loss_
                            val_acc += val_acc_ * T  # number of correct prediction
                            test_id += T
                            count += T

                    val_acc_list.append(val_acc/count)
                    val_loss_list.append(val_loss/count)

                    msg = 'Batch training: Steps {}. Batch Acc {:.2f}. Loss {:4f}.'.format(iter_id+1,
                                                                                           val_acc/count,
                                                                                           val_loss/count)
                    with open(log_file, 'a') as f:
                        f.write(msg)

                    if FLAGS.verbose:
                        logging.info(msg)

                    online_loss = 0

            ave_val_acc[run_no] = val_acc_list
            ave_val_loss[run_no] = val_loss_list

    accuracy_series[run_no] = pred_series
    loss_series[run_no] = losses

if FLAGS.fix_buffer:
    ave_acc = np.mean(ave_val_acc, axis=1)
    ave_loss = np.mean(ave_val_loss, axis=1)

    with open('{}/fix-sweep.txt'.format(dir_name), 'a') as f:
        performance = '{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(ave_acc), np.std(ave_acc), np.mean(ave_loss), np.std(ave_loss))
        f.write('{},{}\n'.format(filename, performance))
        if FLAGS.verbose:
            logging.info('{},{}'.format(filename, performance))

else:
    with open('{}/sweep.txt'.format(dir_name), 'a') as f:
        performance = '{:.4f},{:.4f}'.format(np.mean(accuracy_series), np.mean(loss_series))
        f.write('{},{}\n'.format(filename, performance))
        if FLAGS.verbose:
            logging.info('{},{}'.format(filename, performance))

    # np.save('results/baseline_{}-cw'.format(FLAGS.cycleworld_size),baseline)
    save_dict = {
        'acc': accuracy_series,
        'loss': loss_series
    }
    np.save('{}{}'.format(pathname, filename), save_dict)
    logging.info('writing data to {}{}'.format(pathname, filename))
