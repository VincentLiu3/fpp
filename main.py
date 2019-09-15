import tensorflow as tf 
import numpy as np 
# import matplotlib.pyplot as plt

from data import *
from replay_buffer import Replay_Buffer,Prioritized_Replay_Buffer
from bptt import BPTT_Model
from fpt import FPT_Model


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow loading GPU messages
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= ""

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('verbose',False,'Print Log')
flags.DEFINE_string('dataset','stochastic_dataset','Name of dataset: cycleworld, stochastic_dataset,ptb')
flags.DEFINE_integer('total_length', 1000000, 'Length of entire dataset')
flags.DEFINE_integer('batch_size', 100, 'Batch Size')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_float('output_learning_rate', 0.01, 'Learning rate for Output Weights')
flags.DEFINE_float('state_learning_rate', 0.01, 'Learning rate for States')
flags.DEFINE_integer('n_input', 2, 'Dimension of Input')
flags.DEFINE_integer('n_classes', 2, 'Dimension of Output')
flags.DEFINE_integer('num_units', 16, 'Hidden Units')
flags.DEFINE_float('lambda_state',1, 'Lambda')
flags.DEFINE_bool('use_lstm',False,'LSTM mode')
flags.DEFINE_bool('use_bptt',False,'BPTT mode')
flags.DEFINE_bool('clip_gradients',False,'Clip Gradients')
flags.DEFINE_bool('use_prioritized_exp_replay',False,'Use Prioritized Experience Replay')
flags.DEFINE_float('alpha', 0.5,'Alpha for PER')
flags.DEFINE_integer('anneal_thresh_steps',499,'Steps after which to anneal threshold')
flags.DEFINE_float('anneal_thresh_value',1.0,'Value by which threshold will be annealed')
flags.DEFINE_bool('use_hybrid',False,'Hybrid mode')
flags.DEFINE_bool('use_buffer_bptt',False,'Buffer BPTT')

flags.DEFINE_integer('cycleworld_size', 10, 'CycleWorld Size')
flags.DEFINE_integer('runs',1,'Number of Runs')

flags.DEFINE_integer('time_steps', 10, 'Truncate Parameter')

flags.DEFINE_integer('buffer_length',1000, 'Buffer Length')
flags.DEFINE_integer('updates_per_step', 10, 'Number of Updates per Step')
flags.DEFINE_integer('state_updates_per_step', 0, 'Number of State Updates per Step')

if FLAGS.dataset == 'cycleworld':
    data_name = '{}_cw_'.format(FLAGS.cycleworld_size)
    pathname = 'results-iclr-slr/results-cw/'
elif FLAGS.dataset == 'stochastic_dataset':
    data_name = 'sd_'
    pathname = 'results-iclr-slr/results-sd/'
elif FLAGS.dataset == 'ptb':
    data_name = 'ptb_'
    pathname = 'results-iclr/results-ptb/'


if FLAGS.use_bptt == False:
    # if FLAGS.dataset != 'ptb':
    FLAGS.time_steps = 1    
    if FLAGS.use_hybrid == False:
        pathname = pathname+'normal/'
        filename = data_name+'lr_{}{}_lamb_{}_fpt_n_{}_N_{}_s_{}'.format(FLAGS.learning_rate,FLAGS.state_learning_rate,FLAGS.lambda_state,FLAGS.buffer_length,FLAGS.updates_per_step,FLAGS.state_updates_per_step)
    else:
        if FLAGS.use_buffer_bptt == True:
            pathname = pathname+'buffer_bptt/'
            filename = data_name+'lr_{}{}_fpt_n_{}_N_{}_s_{}_a{}{}'.format(
                FLAGS.learning_rate,FLAGS.state_learning_rate,FLAGS.buffer_length,FLAGS.updates_per_step,
                FLAGS.state_updates_per_step,FLAGS.anneal_thresh_steps,FLAGS.anneal_thresh_value)
        else:
            pathname = pathname+'hybrid/'
            filename = data_name+'lr_{}{}_lamb_{}_fpt_n_{}_N_{}_s_{}_a{}{}'.format(
                FLAGS.learning_rate,FLAGS.state_learning_rate,FLAGS.lambda_state,FLAGS.buffer_length,FLAGS.updates_per_step,
                FLAGS.state_updates_per_step,FLAGS.anneal_thresh_steps,FLAGS.anneal_thresh_value)
    if FLAGS.use_prioritized_exp_replay == True:
        pathname = pathname+'per/'
        filename = filename+'exp_'+str(FLAGS.alpha)

else:
    pathname = pathname+'bptt/'
    filename = data_name+'lr_{}_bptt_T_{}'.format(FLAGS.learning_rate, FLAGS.time_steps)



log_file = "logs/"+filename+"log.txt"
os.makedirs(pathname,exist_ok=True)
result_file = pathname+filename
if FLAGS.dataset == 'ptb':
    X, Y = data_ptb(batch_size=FLAGS.batch_size)
    num_batches = np.shape(X)[0]
    accuracy_series = np.zeros(shape=(FLAGS.runs,num_batches//100))
    loss_series = np.zeros(shape=(FLAGS.runs,num_batches//100))
    # num_batches = 1000
else:
    num_batches = FLAGS.total_length // FLAGS.batch_size
    accuracy_series = np.zeros(shape=(FLAGS.runs,num_batches//100-1))
    loss_series = np.zeros(shape=(FLAGS.runs,num_batches//100-1))

for run_no in range(FLAGS.runs):
    tf.reset_default_graph()
    tf.set_random_seed(run_no)
    np.random.seed(run_no)

    if FLAGS.use_bptt:
        with open(log_file, 'a') as f:
            print('BPTT', 
                "size:", FLAGS.cycleworld_size,
                "time_steps:",FLAGS.time_steps,
                "run_no:",run_no,
                file=f)
    else:
        with open(log_file, 'a') as f:
            print('Fixed Point', 
                "size:", FLAGS.cycleworld_size,
                "buffer:",FLAGS.buffer_length,
                "updates:",FLAGS.updates_per_step,
                "run_no:",run_no,
                file=f)

    if FLAGS.use_bptt:
        model = BPTT_Model(
                           FLAGS.dataset,
                           FLAGS.use_lstm,
                           FLAGS.n_input,
                           FLAGS.n_classes,
                           FLAGS.num_units,
                           FLAGS.batch_size,
                           FLAGS.time_steps,
                           FLAGS.learning_rate,
                           FLAGS.clip_gradients,
                           run_no)
    else:
        model = FPT_Model(
                           FLAGS.dataset,
                           FLAGS.use_lstm,
                           FLAGS.n_input,
                           FLAGS.n_classes,
                           FLAGS.num_units,
                           FLAGS.batch_size,
                           FLAGS.updates_per_step,
                           FLAGS.state_updates_per_step,
                           FLAGS.learning_rate,
                           FLAGS.output_learning_rate,
                           FLAGS.state_learning_rate,
                           FLAGS.clip_gradients,
                           FLAGS.use_buffer_bptt,
                           FLAGS.lambda_state,
                           run_no)

    if FLAGS.use_prioritized_exp_replay == False:
        buffer = Replay_Buffer(FLAGS.buffer_length)
    else:
        buffer = Prioritized_Replay_Buffer(FLAGS.buffer_length, FLAGS.alpha)

    output_op = model.output
    train_op = model.train
    loss_op = model.loss
    state_op = model.state

    with tf.Session() as sess:  
        init = tf.global_variables_initializer()
        sess.run(init)
        ## X,Y -> [batch_size, num_batches]
        if FLAGS.dataset == 'cycleworld':
            X, Y = generate_cw(FLAGS.cycleworld_size,FLAGS.batch_size,num_batches)
        elif FLAGS.dataset == 'stochastic_dataset':
            X, Y = generate_stochastic_data(FLAGS.batch_size,num_batches)
        elif FLAGS.dataset == 'ptb':
            pass
            ##### Check both fpt and bptt for vocab size
            # X, Y = data_ptb(vocab_size = 10000,batch_size=FLAGS.batch_size,num_steps=1)
            # num_batches = 10

        
        iter = FLAGS.time_steps
        corr = 0
        pred_zero = 0
        sum_acc = 0
        sum_loss = 0
        loss_all = 0
        count = 0
        if FLAGS.use_hybrid == False:
            random_no = 1.0
            threshold = 0.0
        else:
            threshold = 1.0
        pred_series = []
        losses = []
        baseline = []
        steps = []

        if FLAGS.use_lstm:
            state = np.zeros(shape=[2,FLAGS.batch_size,FLAGS.num_units])
        else:
            state = np.zeros(shape=[FLAGS.batch_size,FLAGS.num_units])

        while iter<num_batches:
            if FLAGS.dataset == 'ptb':
                batch_x = X[iter-FLAGS.time_steps:iter].T
                batch_y = Y[iter-FLAGS.time_steps:iter].T
                
                if FLAGS.use_bptt == False:
                    x_t = batch_x[0].reshape([FLAGS.batch_size,1])
                    y_t = batch_y[0].reshape([FLAGS.batch_size,1])
                    

                batch_x = np.squeeze(batch_x,axis=0)
                batch_y = np.squeeze(batch_y,axis=0)
                

            else:
                batch_x = X[:,iter-FLAGS.time_steps:iter].reshape([FLAGS.batch_size,FLAGS.time_steps,FLAGS.n_input])
                batch_y = Y[:,iter-FLAGS.time_steps:iter].reshape([FLAGS.batch_size,FLAGS.time_steps,FLAGS.n_classes])
                x_t = batch_x[:,0,:].reshape([FLAGS.batch_size,1,FLAGS.n_input])
                y_t = batch_y[:,0,:]
                
            if FLAGS.use_bptt:
                output,loss,state,_,acc = sess.run([output_op,loss_op,model.state_last,train_op,model.accuracy],
                feed_dict={
                model.x: batch_x,
                model.y: batch_y,
                model.state_placeholder:state
                })
                # print(np.shape(batch_x))
                # print(batch_x[0])
                # input()
                # print(state.shape)
                sum_acc += acc
                sum_loss += loss
                loss_all += loss
            else:
                state_tm1 = state
                output,state,acc,loss = sess.run([output_op,state_op,model.accuracy,model.loss],
                feed_dict={
                model.x: x_t,
                model.y: y_t,
                model.state_placeholder:state
                })

                data = x_t, state_tm1, state, y_t
                if FLAGS.use_prioritized_exp_replay == False:
                    buffer.add(data)
                else:
                    buffer.add(data,loss)

                # x_t = batch_x[:,index].reshape([FLAGS.batch_size,1])
                # y_t = batch_y[:,index].reshape([FLAGS.batch_size,1])
                # print(x_t)
                # input()

                sum_acc += acc
                sum_loss += loss/FLAGS.time_steps
                loss_all += loss/FLAGS.time_steps

                ##TRAINING
                if iter > FLAGS.updates_per_step:
                    if FLAGS.use_hybrid:
                        random_no = abs(np.random.random())
                        if iter % FLAGS.anneal_thresh_steps == 0:
                            # if FLAGS.updates_per_step != 1:
                            #     FLAGS.updates_per_step -= 1
                            #     model.change_sample_updates(FLAGS.updates_per_step)
                            threshold /= FLAGS.anneal_thresh_value
                    if random_no < threshold:
                        x_t_series, s_tm1_series,s_t_series, y_t_series, idx_series = buffer.sample_successive(FLAGS.updates_per_step)
                        # print(np.shape(x_t_series))
                        # print(np.squeeze(x_t_series))
                        # input()
                        _,s_tm1_c_new,s_t_c_new = sess.run([model.train_seq,model.state_tm1_c_seq,model.state_t_c_seq],
                                                    feed_dict={
                                                    model.x_t: x_t_series,
                                                    model.y_t:y_t_series ,
                                                    model.s_tm1:s_tm1_series,
                                                    model.s_t:s_t_series
                                                    })
                        s_tm1_c_series = s_tm1_series
                        s_t_c_series = s_t_series
                        s_tm1_c_series[0] = s_tm1_c_new
                        s_t_c_series[-1] = s_t_c_new

                        buffer.replace(idx_series, x_t_series, s_tm1_c_series,s_t_c_series, y_t_series, FLAGS.updates_per_step)

                    else:
                        for _ in range(FLAGS.updates_per_step):
                            x_t_series, s_tm1_series,s_t_series, y_t_series, idx_series = buffer.sample(FLAGS.updates_per_step)

                            _,s_tm1_c_series,s_t_c_series = sess.run([train_op,model.state_tm1_c,model.state_t_c],
                                                        feed_dict={
                                                        model.x_t: x_t_series,
                                                        model.y_t:y_t_series ,
                                                        model.s_tm1:s_tm1_series,
                                                        model.s_t:s_t_series,
                                                        })

                        if FLAGS.use_lstm:
                            s_tm1_c_series = np.reshape(s_tm1_c_series, [FLAGS.updates_per_step,2,FLAGS.batch_size,FLAGS.num_units])
                            s_t_c_series = np.reshape(s_t_c_series, [FLAGS.updates_per_step,2,FLAGS.batch_size,FLAGS.num_units])

                        else:
                            s_tm1_c_series = np.reshape(s_tm1_c_series, [FLAGS.updates_per_step,FLAGS.batch_size,FLAGS.num_units])
                            s_t_c_series = np.reshape(s_t_c_series, [FLAGS.updates_per_step,FLAGS.batch_size,FLAGS.num_units])

                        buffer.replace(idx_series, x_t_series, s_tm1_c_series,s_t_c_series, y_t_series, FLAGS.updates_per_step)
                    
            if iter%1000 == 0:
                if FLAGS.use_bptt == False:
                    FLAGS.state_learning_rate /= 3
                    model.update_state_lr(FLAGS.state_learning_rate)

            if iter % 100 == 0:
                steps.append(iter)
                if FLAGS.use_bptt == True:
                    pred_series.append(sum_acc/(100))
                    losses.append(sum_loss/(100))
                    with open(log_file, 'a') as f:
                        if FLAGS.dataset == 'ptb':
                            print("Predictions at step",iter,"Correct: {}, Loss: {}, Perp: {}".format(sum_acc,sum_loss/count,np.exp(loss_all/iter)),file = f)
                        else:
                            print("Predictions at step",iter,"Correct: {}, Loss: {}".format(sum_acc/(100),sum_loss/(100)),file = f)
                    
                    if FLAGS.verbose == True:
                        if FLAGS.dataset == 'ptb':
                            print('Steps:',iter,'Accuracy:',sum_acc/(100),'Loss:{}, Perp: {}'.format(sum_loss/(100),np.exp(loss_all/iter)))
                        else:
                            print('Steps:',iter,'Accuracy:',sum_acc/(100),'Loss:',sum_loss/(100))
                    # print(grad)
                else:
                    pred_series.append(sum_acc)
                    losses.append(sum_loss/(count))
                # baseline.append(pred_zero/FLAGS.batch_size)
                    with open(log_file, 'a') as f:
                        if FLAGS.dataset == 'ptb':
                            print("Predictions at step",iter,"Correct: {}, Loss: {}, Perp: {}".format(sum_acc,sum_loss/count,np.exp(loss_all/iter)),file = f)
                        else:
                            print("Predictions at step",iter,"Correct: {}, Loss: {}".format(sum_acc,sum_loss/count),file = f)
                    # print("Predictions at step",iter,"Correct: {}".format(corr/FLAGS.batch_size))
                    # print([(v.name,sess.run(v)) for v in tf.trainable_variables() if v.name.startswith('rnn')])
                    # # print([(v.name,sess.run(v)) for v in tf.trainable_variables() if v.name.startswith('fully')])
                    # print(target)
                    # print(predicted.T)
                    if FLAGS.verbose == True:
                        if FLAGS.dataset == 'ptb':
                            print('Steps:',iter,'Accuracy:',sum_acc/(100),'Loss:{}, Perp: {}'.format(sum_loss/(count),np.exp(loss_all/iter)))
                        else:
                            print('Steps:',iter,'Accuracy:',sum_acc/(100),'Loss:',sum_loss/(count))
                    # print(grad)
                corr = 0
                sum_acc = 0
                sum_loss = 0
                count = 0

            count += 1
            iter += 1
    accuracy_series[run_no] = pred_series
    loss_series[run_no] = losses
# np.save('results/baseline_{}-cw'.format(FLAGS.cycleworld_size),baseline)
np.save(result_file,accuracy_series)
np.save(result_file+'_loss',loss_series)