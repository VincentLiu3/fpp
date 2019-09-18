import tensorflow as tf
# import matplotlib.pyplot as plt

from data import *
from model.replay_buffer import Replay_Buffer
from model.bptt import BPTT_Model
from model.fpt import FPT_Model

import os
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

'''
Note on tf version:
1. pip install numpy==1.16.4
'''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow loading GPU messages
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('verbose', False, 'Print Log')
flags.DEFINE_string('dataset', 'stochastic_dataset', 'Name of dataset: cycleworld, stochastic_dataset, ptb')
flags.DEFINE_integer('cycleworld_size', 10, 'CycleWorld Size')
flags.DEFINE_integer('total_length', 100000, 'Length of entire dataset')
flags.DEFINE_integer('num_trajectory', 1, 'number of trajectory')
flags.DEFINE_integer('runs', 1, 'Number of Runs')

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_float('output_learning_rate', 0.001, 'Learning rate for Output Weights')
flags.DEFINE_float('state_learning_rate', 0.001, 'Learning rate for States')
flags.DEFINE_integer('n_input', 2, 'Dimension of Input')
flags.DEFINE_integer('n_classes', 2, 'Dimension of Output')
flags.DEFINE_integer('num_units', 16, 'Hidden Units')
flags.DEFINE_float('lambda_state', 1, 'Lambda')

flags.DEFINE_bool('use_hybrid', False, 'Hybrid mode')
flags.DEFINE_bool('use_buffer_bptt', False, 'Buffer BPTT')

flags.DEFINE_integer('time_steps', 1, 'Truncate Parameter')
flags.DEFINE_integer('buffer_length', 1000, 'Buffer Length')
flags.DEFINE_integer('updates_per_step', 10, 'Number of Updates per Step T')
flags.DEFINE_integer('num_update', 1, 'Number of State Updates per Step M')
flags.DEFINE_integer('batch_size', 1, 'Mini-batch Size B')

flags.DEFINE_bool('use_lstm', False, 'LSTM mode')
flags.DEFINE_bool('use_bptt', False, 'BPTT mode')
flags.DEFINE_bool('clip_gradients', False, 'Clip Gradients')
flags.DEFINE_bool('use_prioritized_exp_replay', False, 'Use Prioritized Experience Replay')
flags.DEFINE_float('alpha', 0.5, 'Alpha for PER')
flags.DEFINE_integer('anneal_thresh_steps', 499, 'Steps after which to anneal threshold')
flags.DEFINE_float('anneal_thresh_value', 1.0, 'Value by which threshold will be annealed')
flags.DEFINE_integer('state_updates_per_step', 0, 'Number of State Updates per Step')

assert FLAGS.time_steps == 1, 'set FLAGS.time_steps to 1.'
assert FLAGS.use_hybrid is True, 'use hybrid.'
assert FLAGS.learning_rate == FLAGS.output_learning_rate and FLAGS.output_learning_rate == FLAGS.state_learning_rate, 'lr'

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

if FLAGS.use_bptt is False:
    if FLAGS.use_hybrid is False:
        pathname = dir_name+'normal/'
        filename = data_name+'lr_{}{}_lamb_{}_fpt_n_{}_N_{}_s_{}'.format(FLAGS.learning_rate,FLAGS.state_learning_rate,FLAGS.lambda_state,FLAGS.buffer_length,FLAGS.updates_per_step,FLAGS.state_updates_per_step)
    else:
        if FLAGS.use_buffer_bptt:
            pathname = dir_name+'buffer_bptt/'
            # filename = data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}_a{}{}'.format(
            #     FLAGS.learning_rate, FLAGS.state_learning_rate, FLAGS.buffer_length, FLAGS.updates_per_step,
            #     FLAGS.state_updates_per_step, FLAGS.anneal_thresh_steps, FLAGS.anneal_thresh_value)
        else:
            pathname = dir_name+'fpp/'
            # filename = data_name+'lr_{}{}_lamb_{}_fpt_n_{}_N_{}_s_{}_a{}{}'.format(
            #     FLAGS.learning_rate,FLAGS.state_learning_rate,FLAGS.lambda_state,FLAGS.buffer_length,
            #     FLAGS.updates_per_step,FLAGS.state_updates_per_step,FLAGS.anneal_thresh_steps,FLAGS.anneal_thresh_value)

        filename = '{},{},{},{},{},{},{},{}'.format(data_name, FLAGS.use_buffer_bptt, FLAGS.updates_per_step,
                                                FLAGS.num_update, FLAGS.batch_size,
                                                FLAGS.learning_rate, FLAGS.lambda_state, FLAGS.buffer_length)

else:
    pathname = dir_name + 'bptt/'
    filename = data_name + 'lr_{}_bptt_T_{}'.format(FLAGS.learning_rate, FLAGS.time_steps)

log_file = "logs/{}_log.txt".format(filename)
os.makedirs(pathname, exist_ok=True)

if FLAGS.dataset == 'ptb':
    X, Y = data_ptb(batch_size=FLAGS.num_trajectory)
    num_batches = np.shape(X)[0]
    accuracy_series = np.zeros(shape=(FLAGS.runs,num_batches//100))
    loss_series = np.zeros(shape=(FLAGS.runs,num_batches//100))
    # num_batches = 1000
else:
    num_batches = FLAGS.total_length // FLAGS.num_trajectory
    accuracy_series = np.zeros(shape=(FLAGS.runs, num_batches//100-1))
    loss_series = np.zeros(shape=(FLAGS.runs, num_batches//100-1))

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

    if FLAGS.use_bptt:
        model = BPTT_Model(
                           FLAGS.dataset,
                           FLAGS.use_lstm,
                           FLAGS.n_input,
                           FLAGS.n_classes,
                           FLAGS.num_units,
                           FLAGS.num_trajectory,
                           FLAGS.time_steps,
                           FLAGS.learning_rate,
                           FLAGS.clip_gradients,
                           run_no)
    else:
        model = FPT_Model(FLAGS.dataset, FLAGS.use_lstm, FLAGS.n_input, FLAGS.n_classes, FLAGS.num_units,
                          FLAGS.num_trajectory, FLAGS.batch_size, FLAGS.updates_per_step, FLAGS.state_updates_per_step,
                          FLAGS.learning_rate, FLAGS.output_learning_rate, FLAGS.state_learning_rate,
                          FLAGS.clip_gradients, FLAGS.use_buffer_bptt, FLAGS.lambda_state, run_no)

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
            ##### Check both fpt and bptt for vocab size
            # X, Y = data_ptb(vocab_size = 10000, batch_size=FLAGS.batch_size,num_steps=1)
            # num_batches = 10

        iter = FLAGS.time_steps
        # corr = 0
        # pred_zero = 0

        # total loss/accuracy over 100 iterations
        sum_acc = 0
        sum_loss = 0
        loss_all = 0

        count = 0
        # use_hybrid is True for normal FPP
        # if FLAGS.use_hybrid is False:
        #     random_no = 1.0
        #     threshold = 0.0
        # else:
        # threshold = 1.0

        pred_series = []
        losses = []
        # baseline = []
        # steps = []

        if FLAGS.use_lstm:
            state = np.zeros(shape=[2, FLAGS.num_trajectory, FLAGS.num_units])
        else:
            # initialize s_0 as zero vector
            state = np.zeros(shape=[FLAGS.num_trajectory, FLAGS.num_units])

        while iter < num_batches:
            if FLAGS.dataset == 'ptb':
                batch_x = X[iter-FLAGS.time_steps:iter].T
                batch_y = Y[iter-FLAGS.time_steps:iter].T
                
                if FLAGS.use_bptt is False:
                    x_t = batch_x[0].reshape([FLAGS.num_trajectory,1])
                    y_t = batch_y[0].reshape([FLAGS.num_trajectory,1])

                batch_x = np.squeeze(batch_x,axis=0)
                batch_y = np.squeeze(batch_y,axis=0)

            else:
                # get slice from time (iter-T to iter)
                batch_x = X[:, iter-FLAGS.time_steps:iter].reshape([FLAGS.num_trajectory, FLAGS.time_steps, FLAGS.n_input])
                batch_y = Y[:, iter-FLAGS.time_steps:iter].reshape([FLAGS.num_trajectory, FLAGS.time_steps, FLAGS.n_classes])
                x_t = batch_x[:, 0, :].reshape([FLAGS.num_trajectory, 1, FLAGS.n_input])
                y_t = batch_y[:, 0, :]
                
            if FLAGS.use_bptt:
                output, loss, state, _, acc = sess.run([output_op, loss_op, model.state_last, model.train, model.accuracy],
                                                       feed_dict={
                                                        model.x: batch_x,
                                                        model.y: batch_y,
                                                        model.state_placeholder: state
                                                       })
                # print(np.shape(batch_x))
                # print(batch_x[0])
                # input()
                # print(state.shape)
                sum_acc += acc
                sum_loss += loss
                loss_all += loss
            else:
                state_tm1 = state  # save previous state vector
                output, state, acc, loss = sess.run([model.output, model.state, model.accuracy, model.loss],
                                                    feed_dict={
                                                    model.x: x_t,  # shape = (None, 1, n_input)
                                                    model.y: y_t,  # shape = (None, n_class)
                                                    model.state_placeholder: state
                                                    })

                data = x_t, state_tm1, state, y_t  # (o_t, s_t-1, s_t, y_t) in the paper
                buffer.add(data)
                # if FLAGS.use_prioritized_exp_replay is False:
                #     buffer.add(data)
                # else:
                #     buffer.add(data, loss)

                sum_acc += acc
                sum_loss += loss/FLAGS.time_steps
                loss_all += loss/FLAGS.time_steps

                # TRAINING
                if iter > FLAGS.updates_per_step:  # updates_per_step = T
                    if FLAGS.use_hybrid:
                        # random_no = abs(np.random.random())
                        # if iter % FLAGS.anneal_thresh_steps == 0:
                        #     # if FLAGS.updates_per_step != 1:
                        #     #     FLAGS.updates_per_step -= 1
                        #     #     model.change_sample_updates(FLAGS.updates_per_step)
                        #     threshold /= FLAGS.anneal_thresh_value
                        # if random_no < threshold:
                        # this will be true always for normal FPP
                        for _ in range(FLAGS.num_update):
                            # x_t_series, s_tm1_series, s_t_series, y_t_series, idx_series = buffer.sample_successive(FLAGS.updates_per_step)
                            x_t_series, s_tm1_series, s_t_series, y_t_series, idx_series = buffer.sample_batch(FLAGS.batch_size, FLAGS.updates_per_step)

                            _, new_s_tm1, new_s_t = sess.run([model.train_seq, model.state_tm1_c_seq, model.state_t_c_seq],
                                                                 feed_dict={
                                                                    model.x_t: x_t_series,  # [T, B, 1, n_input]
                                                                    model.y_t: y_t_series,  # [T, B, n_class]
                                                                    model.s_tm1: s_tm1_series,  # [T, B, n_unit]
                                                                    model.s_t: s_t_series  # [T, B, n_unit]
                                                                 })

                            for b in range(idx_series.shape[0]):
                                new_s_tm1_series = np.expand_dims(s_tm1_series[:, b, :], axis=1)
                                new_s_t_series = np.expand_dims(s_t_series[:, b, :], axis=1)
                                new_s_tm1_series[0] = np.expand_dims(new_s_tm1[b], axis=0)
                                new_s_t_series[-1] = np.expand_dims(new_s_t[b], axis=0)

                                new_x_t_series = np.expand_dims(x_t_series[:, b, :, :], axis=1)
                                new_y_t_series = np.expand_dims(y_t_series[:, b, :], axis=1)
                                # update S_{t-T} and S_t in the buffer
                                buffer.replace(idx_series[b], new_x_t_series, new_s_tm1_series, new_s_t_series, new_y_t_series, FLAGS.updates_per_step)

                    else:
                        assert False, 'wrong implemntation'
                        for _ in range(FLAGS.updates_per_step):
                            x_t_series, s_tm1_series,s_t_series, y_t_series, idx_series = buffer.sample(FLAGS.updates_per_step)

                            _,s_tm1_c_series,s_t_c_series = sess.run([model.train, model.state_tm1_c, model.state_t_c],
                                                        feed_dict={
                                                        model.x_t: x_t_series,
                                                        model.y_t:y_t_series ,
                                                        model.s_tm1:s_tm1_series,
                                                        model.s_t:s_t_series,
                                                        })

                        if FLAGS.use_lstm:
                            s_tm1_c_series = np.reshape(s_tm1_c_series, [FLAGS.updates_per_step,2,FLAGS.num_trajectory, FLAGS.num_units])
                            s_t_c_series = np.reshape(s_t_c_series, [FLAGS.updates_per_step,2,FLAGS.num_trajectory, FLAGS.num_units])

                        else:
                            s_tm1_c_series = np.reshape(s_tm1_c_series, [FLAGS.updates_per_step,FLAGS.batch_size,FLAGS.num_units])
                            s_t_c_series = np.reshape(s_t_c_series, [FLAGS.updates_per_step,FLAGS.batch_size,FLAGS.num_units])

                        buffer.replace(idx_series, x_t_series, s_tm1_c_series, s_t_c_series, y_t_series, FLAGS.updates_per_step)
                    
            # if iter % 1000 == 0:
            #     if FLAGS.use_bptt is False:
            #         # decrease the learfning rate. Why do we do this?
            #         FLAGS.state_learning_rate /= 3
            #         model.update_state_lr(FLAGS.state_learning_rate)

            if iter % 100 == 0:
                # steps.append(iter)
                if FLAGS.use_bptt:
                    pred_series.append(sum_acc/100)
                    losses.append(sum_loss/100)
                    with open(log_file, 'a') as f:
                        if FLAGS.dataset == 'ptb':
                            print("Predictions at step",iter,"Correct: {}, Loss: {}, Perp: {}".format(sum_acc,sum_loss/count,np.exp(loss_all/iter)),file = f)
                        else:
                            print("Predictions at step",iter,"Correct: {}, Loss: {}".format(sum_acc/100,sum_loss/100), file=f)
                    
                    if FLAGS.verbose:
                        # print to std_out
                        if FLAGS.dataset == 'ptb':
                            print('Steps:',iter,'Accuracy:',sum_acc/(100),'Loss:{}, Perp: {}'.format(sum_loss/(100),np.exp(loss_all/iter)))
                        else:
                            print('Steps:',iter,'Accuracy:',sum_acc/(100),'Loss:',sum_loss/(100))
                    # print(grad)
                else:
                    pred_series.append(sum_acc)
                    losses.append(sum_loss/count)

                    with open(log_file, 'a') as f:
                        # if FLAGS.dataset == 'ptb':
                        #     print("Predictions at step",iter,"Correct: {}, Loss: {}, Perp: {}".format(sum_acc,sum_loss/count,np.exp(loss_all/iter)),file = f)
                        # else:
                        f.write('Steps {}. Accuracy {:.2f}. Loss {:4f}.\n'.format(iter, sum_acc/100, sum_loss/count))

                    # print([(v.name,sess.run(v)) for v in tf.trainable_variables() if v.name.startswith('rnn')])
                    # # print([(v.name,sess.run(v)) for v in tf.trainable_variables() if v.name.startswith('fully')])
                    # print(target)
                    # print(predicted.T)
                    if FLAGS.verbose:
                        # if FLAGS.dataset == 'ptb':
                        #     print('Steps:',iter,'Accuracy:',sum_acc/100,'Loss:{}, Perp: {}'.format(sum_loss/(count),np.exp(loss_all/iter)))
                        # else:
                        logging.info('Steps {}. Accuracy {:.2f}. Loss {:4f}.'.format(iter, sum_acc/100, sum_loss/count))
                    # print(grad)

                # corr = 0
                sum_acc = 0
                sum_loss = 0
                count = 0

            count += 1
            iter += 1

    accuracy_series[run_no] = pred_series
    loss_series[run_no] = losses

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
