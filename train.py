import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow loading GPU messages
# available_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
from util import ReplayBuffer
from fixedpoint import FixedPoint
from bptt import BPTT
from fixedpoint_prioritized import FixedPointPrioritized
# from proportional import Experience

def save_run_results(filename, run_results):
    """ Saves hyperparameter sweeps results """
    #np.save(filename, [x._asdict() for x in run_results])
    np.save(filename, run_results)

def load_run_results(filename):
    results = np.load(filename)
    return results

def print_variables(sess, variable_list):
    variable_names = [v.name for v in variable_list]
    values = sess.run(variable_names)
    for k, v in zip(variable_names, values):
        print("Variable:", k)
        print("Shape: ", v.shape)
        print(np.sum(v > 0))

def fixed_pt_train(rnn_params,fpt_params, filename,X,Y):
    start_time = time.time()
    #Load params
    input_size = rnn_params['input_size']
    state_size = rnn_params['state_size']
    num_classes = rnn_params['num_classes']
    total_series_length = rnn_params['total_series_length']
    batch_size = rnn_params['batch_size']
    cycleworld_size = rnn_params['cycleworld_size']

    runs = rnn_params['runs']

    learning_rate = fpt_params['learning_rate']
    state_update = fpt_params['state_update']
    n = fpt_params['buffer_length']
    N = fpt_params['updates_per_step']

    avg_pred = np.zeros(shape=(runs,total_series_length//batch_size-1))
    avg_losses = np.zeros(shape=(runs,total_series_length//batch_size-1))

    #Logfile
    log_file = "logs/"+filename+"log.txt"

    for run_no in range(runs):
        tf.reset_default_graph()
        tf.set_random_seed(run_no)
        np.random.seed(run_no)

        #Start Printing to the logfile
        with open(log_file, 'a') as f:
            print('Fixed Point', str(datetime.now()), 
                "size: ", cycleworld_size,
                "run: ", run_no , 
                "params:",fpt_params,
                file=f)


        #Experience Replay
        replay_buffer = ReplayBuffer(n)

        #Open Session
        sess = tf.Session()
        
        #Graph Building
        fpt = FixedPoint(input_size, state_size, num_classes)
        prop = fpt.forward_propagation(input_size, state_size, num_classes)
        train = fpt.train(1, state_size, num_classes, learning_rate)
        buffer_update = fpt.buffer_update(1, state_size, num_classes)
        state_update = fpt.state_update(1, state_size, num_classes, learning_rate)
        #Initialize Variables
        sess.run(tf.global_variables_initializer())

        training_losses = []
        out_series = []
        correct_series = []
        incorrect_series = []
        baseline = []
        tp_series = []
        correct = 0
        incorrect = 0
        training_loss = 0
        zeros = 0
        #Init Training State
        training_state = np.zeros((input_size, state_size))
        time_estimated = np.inf

        #Start Training
        for i in range(0,total_series_length-1):
            #Input Data
            x = np.atleast_2d(X[i])
            y = np.atleast_2d(Y[i])

            sm1 = training_state

            #forward propagation
            training_state, output_t = \
                            sess.run([fpt.final_state,
                                      fpt.predictions],
                                          feed_dict={
                                          fpt.x:x, 
                                          fpt.y:y, 
                                          fpt.init_state:training_state})

            out = np.argmax(output_t,axis=2)
            for j in range(len(y)):
                if y[0][j] == 0:
                    zeros += 1
                if y[0][j] == out[j]:
                    correct += 1
                else:
                    incorrect += 1

            #get next input
            xp1 = np.atleast_2d(X[i+1])
            yp1 = np.atleast_2d(Y[i+1])

            #get next state
            sp1, output_tp1 = \
                            sess.run([fpt.final_state,
                                      fpt.predictions],
                                          feed_dict={fpt.x:xp1, fpt.y:yp1, fpt.init_state:training_state})

            #add to replay buffer
            replay_buffer.add(x, sm1, sp1, xp1, y)

            # #Sample and Train from Buffer
            # if i>N:
            #     x_s, sm1_s, sp1_s, xp1_s, y_s, idx_s = replay_buffer.sample(N)
            #     x_train = np.reshape([x_s,xp1_s],[N,2])
            #     sm1_s_r = np.reshape(sm1_s,[N,state_size])
            #     sp1_s_r = np.reshape(sp1_s,[N,state_size])
            #     y_s_r = np.reshape(y_s,[N,1])
            #     tr_losses, training_loss_, state, sp1_c, _ , _, output,l_1,l_2 = \
            #                 sess.run([fpt.losses,
            #                           fpt.total_loss,
            #                           fpt.final_state_train,
            #                           fpt.final_state_correct,
            #                           fpt.train_step,
            #                           fpt.train_step_1,
            #                           fpt.predictions_train,
            #                           fpt.losses_s,
            #                           fpt.losses_v],
            #                               feed_dict={fpt.x_train:x_train, 
            #                                          fpt.y_t_s:y_s_r, 
            #                                          fpt.sm1_s:sm1_s_r,
            #                                          fpt.sp1_s:sp1_s_r})

            #     sp1_c = np.reshape(sp1_c, [N,1,state_size])
            #     replay_buffer.replace(idx_s, x_s, sm1_s, sp1_c, xp1_s, y_s,N)

            ##TRAIN   
            for j in range(min(i,N)):
                x_s, sm1_s, sp1_s, xp1_s, y_s, idx_s = replay_buffer.one_sample()
                x_train = np.reshape([x_s,xp1_s],[1,2])
                sm1_s_r = np.reshape(sm1_s,[1,state_size])
                sp1_s_r = np.reshape(sp1_s,[1,state_size])
                y_s_r = np.reshape(y_s,[1,1])
                if i % 100 == 0:
                    tr_losses, training_loss_, state, sp1_c, _ , _, output,l_1,l_2 = \
                                sess.run([fpt.losses,
                                          fpt.total_loss,
                                          fpt.final_state_train,
                                          fpt.final_state_correct,
                                          fpt.train_step,
                                          fpt.train_step_1,
                                          fpt.predictions_train,
                                          fpt.losses_s,
                                          fpt.losses_v],
                                              feed_dict={fpt.x_train:x_train, 
                                                         fpt.y_t_s:y_s_r, 
                                                         fpt.sm1_s:sm1_s_r,
                                                         fpt.sp1_s:sp1_s_r})
                else:
                    tr_losses, training_loss_, state, sp1_c, _ , output,l_1,l_2 = \
                                sess.run([fpt.losses,
                                          fpt.total_loss,
                                          fpt.final_state_train,
                                          fpt.final_state_correct,
                                          fpt.train_step,
                                          fpt.predictions_train,
                                          fpt.losses_s,
                                          fpt.losses_v],
                                              feed_dict={fpt.x_train:x_train, 
                                                         fpt.y_t_s:y_s_r, 
                                                         fpt.sm1_s:sm1_s_r,
                                                         fpt.sp1_s:sp1_s_r})

                if state_update == True:
                    sp1_c = sess.run(fpt.new_state,
                                        feed_dict = {fpt.y_state:y_s_r,
                                                     fpt.state:sp1_c})

                sp1_c = np.reshape(sp1_c, [1,1,state_size])


                training_loss += training_loss_
                #Replace buffer with new data
                replay_buffer.replace_one(idx_s, x_s, sm1_s, sp1_c, xp1_s, y_s)

            #Buffer Update
            for j in range(min(i,N)):
                x_s, sm1_s, sp1_s, xp1_s, y_s, idx_s = replay_buffer.one_sample()
                x_train = np.reshape([x_s,xp1_s],[1,2])
                sm1_s_r = np.reshape(sm1_s,[1,state_size])
                sp1_s_r = np.reshape(sp1_s,[1,state_size])
                y_s_r = np.reshape(y_s,[1,1])
                sp1_c = \
                            sess.run(fpt.final_state_train_update,
                                          feed_dict={fpt.x_train_update:x_train, 
                                                     fpt.y_t_s_update:y_s_r, 
                                                     fpt.sm1_s_update:sm1_s_r,
                                                     fpt.sp1_s_update:sp1_s_r})
                            
                sp1_c = np.reshape(sp1_c, [1,1,state_size])


                training_loss += training_loss_
                #Replace buffer with new data
                replay_buffer.replace_one(idx_s, x_s, sm1_s, sp1_c, xp1_s, y_s)

            if i % batch_size == 0 and i > 0:
                if i%1000 == 0:
                    time_elapsed = (time.time() - start_time)
                    speed = (i+1)/(time_elapsed)
                    time_estimated = ((total_series_length-i)/speed)/60
                    with open(log_file, 'a') as f:
                        print("Predictions at step", i,"Correct: {}, Incorrect: {}, Time est: {} min".format(correct,incorrect,time_estimated),file = f)
                    print("Predictions at step", i,"Correct: {}, Incorrect: {}, Time est: {} min".format(correct,incorrect,time_estimated))
                training_losses.append(training_loss/100)
                training_loss = 0
                correct_series.append(correct)
                incorrect_series.append(incorrect)
                baseline.append(zeros)
                correct = 0
                incorrect = 0
                zeros = 0

        avg_losses[run_no] = training_losses
        avg_pred[run_no] = correct_series

        sess.close()

        np.save('results/'+filename+'_loss',avg_losses)
        np.save('results/'+filename+'_pred',avg_pred)
    np.save('baseline',baseline)
    
    print("Time: %.8s sec" % (time.time() - start_time))


def fixed_pt_train_prioritized(rnn_params,fpt_params, filename,X,Y):
    start_time = time.time()

    input_size = rnn_params['input_size']
    state_size = rnn_params['state_size']
    num_classes = rnn_params['num_classes']
    total_series_length = rnn_params['total_series_length']
    batch_size = rnn_params['batch_size']
    cycleworld_size = rnn_params['cycleworld_size']

    runs = rnn_params['runs']

    learning_rate = fpt_params['learning_rate']
    n = fpt_params['buffer_length']
    N = fpt_params['updates_per_step']
    alpha = fpt_params['alpha']

    avg_pred = np.zeros(shape=(runs,total_series_length//batch_size-1))
    avg_losses = np.zeros(shape=(runs,total_series_length//batch_size-1))

    # f = open("fixedpointprioritized.txt", "a")

    for run_no in range(runs):
        tf.reset_default_graph()
        tf.set_random_seed(run_no+1)
        np.random.seed(run_no+1)

        #Experience Replay
        replay_buffer = Experience(n,N,alpha)

        sess = tf.Session()
        start_time = time.time()
        
        fpt = FixedPointPrioritized(input_size, state_size, num_classes)
        prop = fpt.forward_propagation(input_size, state_size, num_classes)
        priority = fpt.priority(1, state_size, num_classes)
        train = fpt.train(1, state_size, num_classes, learning_rate)
        sess.run(tf.global_variables_initializer())
        training_losses = []
        out_series = []
        correct_series = []
        incorrect_series = []
        baseline = []
        tp_series = []
        correct = 0
        incorrect = 0
        tp = 0
        training_loss = 0
        zeros = 0
        training_state = np.zeros((input_size, state_size))
        time_estimated = np.inf

        for i in range(0,total_series_length-1):
            #Input Data
            x = np.atleast_2d(X[i])
            y = np.atleast_2d(Y[i])

            sm1 = training_state

            training_state, output_t = \
                            sess.run([fpt.final_state,
                                      fpt.predictions],
                                          feed_dict={fpt.x:x, fpt.y:y, fpt.init_state:training_state})

            xp1 = np.atleast_2d(X[i+1])
            yp1 = np.atleast_2d(Y[i+1])

            out = np.argmax(output_t,axis=2)
            for j in range(len(y)):
                if y[0][j] == 0:
                    zeros += 1
                if y[0][j] == out[j]:
                    correct += 1
                else:
                    incorrect += 1

            sp1, output_tp1 = \
                            sess.run([fpt.final_state,
                                      fpt.predictions],
                                          feed_dict={fpt.x:xp1, fpt.y:yp1, fpt.init_state:training_state})
            
            sm1 = np.reshape(sm1,[1,state_size])
            sp1 = np.reshape(sp1,[1,state_size])
            
            priority = \
                            sess.run([fpt.total_loss_p
                                      ],
                                          feed_dict={fpt.x_p:x, 
                                                     fpt.y_p:y, 
                                                     fpt.sm1_p:sm1,
                                                     fpt.sp1_p:sp1})

            add_data = (x, sm1, sp1, xp1, y)

            priority = np.sum(priority)

            replay_buffer.add(add_data,priority)


            if i >= N:
                out, weights, indices = replay_buffer.select(0)

                for j in range(N):
                    x_s, sm1_s, sp1_s, xp1_s, y_s = out[j]
                    idx = indices[j]

                    x_train = np.reshape([x_s,xp1_s],[1,2])
                    sm1_s_r = np.reshape(sm1_s,[1,state_size])
                    sp1_s_r = np.reshape(sp1_s,[1,state_size])
                    y_s_r = np.reshape(y_s,[1,1])
                    tr_losses, training_loss_, state, sp1_c, _ , output,l_1,l_2 = \
                                sess.run([fpt.losses,
                                          fpt.total_loss,
                                          fpt.final_state_train,
                                          fpt.final_state_correct,
                                          fpt.train_step,
                                          fpt.predictions_train,
                                          fpt.losses_s,
                                          fpt.losses_v],
                                              feed_dict={fpt.x_train:x_train, 
                                                         fpt.y_t_s:y_s_r, 
                                                         fpt.sm1_s:sm1_s_r,
                                                         fpt.sp1_s:sp1_s_r})

                    training_loss += training_loss_
                    data_corrected = (x_s, sm1_s, sp1_c, xp1_s, y_s)
                    replay_buffer.replace(idx, data_corrected)

            if i % batch_size == 0 and i > 0:
                time_elapsed = (time.time() - start_time)
                speed = (i+1)/(time_elapsed)
                time_estimated = ((total_series_length-i)/speed)/60
                if verbose:   
                    print("Predictions at step", i,"Correct: {}, Incorrect: {}".format(correct,incorrect))
                # f.write("Predictions at step:{}, Correct: {}, Incorrect: {}, Time: {}".format(i,correct,incorrect,time_elapsed))
                training_losses.append(training_loss/100)
                training_loss = 0
                # print('X:',X)
                # print('Y:',Y)
                # print('O:',out.T .astype(float))
                correct_series.append(correct)
                incorrect_series.append(incorrect)
                baseline.append(zeros)
                # tp_series.append(tp)
                correct = 0
                incorrect = 0
                zeros = 0
                # tp = 0

            
            print("Run: {}/{}, Time Estimated: {} min \t".format(i+1,total_series_length,round(time_estimated,2)), end="\r")
            
        
        sess.close()
        print()
        avg_losses[run_no] = training_losses
        avg_pred[run_no] = correct_series     
      
    np.save('results/'+filename+'losses_fpt_p',np.mean(avg_losses,axis=0))
    np.save('results/'+filename+'correct_fpt_p',np.mean(avg_pred,axis=0))
    np.save('baseline',baseline)
    
    print("Time: %.8s sec" % (time.time() - start_time))

    # plt.title('{}-CycleWorld (FPT)'.format(cycleworld_size))
    # plt.subplot(211)
    # plt.ylim(0,110)
    # plt.ylabel('Good predictions in last {} steps'.format(batch_size))
    # plt.plot(correct_series)
    # plt.plot(baseline)
    # plt.subplot(212)
    # plt.ylabel('Loss in last {} steps'.format(batch_size))
    # plt.plot(training_losses)
    # plt.savefig('plot_fpt.png')
    # plt.show()




def bptt_train(rnn_params, bptt_params, filename, X, Y):
    start_time = time.time()
    #Params
    input_size = rnn_params['input_size']
    state_size = rnn_params['state_size']
    num_classes = rnn_params['num_classes']
    total_series_length = rnn_params['total_series_length']
    batch_size = rnn_params['batch_size']
    cycleworld_size = rnn_params['cycleworld_size']

    runs = rnn_params['runs']

    learning_rate = bptt_params['learning_rate']
    truncate_bptt = bptt_params['truncate_bptt']

    avg_pred = np.zeros(shape=(runs,total_series_length//batch_size-1))
    avg_losses = np.zeros(shape=(runs,total_series_length//batch_size-1))

    log_file = "logs/"+filename+"log.txt"
    
    for run_no in range(runs):
        tf.reset_default_graph()
        tf.set_random_seed(run_no)
        np.random.seed(run_no)

        #Start Printing to the logfile
        with open(log_file, 'a') as f:
            print('Fixed Point', str(datetime.now()), 
                "size: ", cycleworld_size,
                "run: ", run_no , 
                "params:",bptt_params,
                file=f)    


        sess = tf.Session()
        
        bptt = BPTT(input_size, state_size, truncate_bptt, num_classes)
        train = bptt.train(learning_rate, input_size, state_size, truncate_bptt, num_classes)
        sess.run(tf.global_variables_initializer())
        training_losses = []
        out_series = []
        correct_series = []
        incorrect_series = []
        baseline = []
        tp_series = []
        correct = 0
        incorrect = 0
        tp = 0
        training_loss = 0
        zeros = 0
        training_state = np.zeros((input_size, state_size))
        time_estimated = np.inf

        for i in range(total_series_length-truncate_bptt):
            #Input Data
            x = X[i:truncate_bptt+i].T
            y = Y[i:truncate_bptt+i].T

            tr_losses, training_loss_, training_state, _ , output = \
                            sess.run([bptt.losses,
                                      bptt.total_loss,
                                      bptt.final_state,
                                      bptt.train_step,
                                      bptt.predictions],
                                          feed_dict={bptt.x:x, bptt.y:y, bptt.init_state:training_state})
            out = np.argmax(output,axis=2)
            for j in range(len(y)):
                if y[0][j] == 0:
                    zeros += 1
                if y[0][j] == out[j]:
                    correct += 1
                else:
                    incorrect += 1

            training_loss += training_loss_

            if i % batch_size == 0 and i > 0:
                if i%1000 == 0:
                    time_elapsed = (time.time() - start_time)
                    speed = (i+1)/(time_elapsed)
                    time_estimated = ((total_series_length-i)/speed)/60
                    with open(log_file, 'a') as f: 
                        print("Predictions at step", i,"Correct: {}, Incorrect: {}, Time est: {} min".format(correct,incorrect,time_estimated),file = f)
                training_losses.append(training_loss/100)
                training_loss = 0
                correct_series.append(correct)
                incorrect_series.append(incorrect)
                baseline.append(zeros)
                correct = 0
                incorrect = 0
                zeros = 0

        avg_losses[run_no] = training_losses
        avg_pred[run_no] = correct_series
        sess.close()

    np.save('results/'+filename+'_loss',avg_losses)
    np.save('results/'+filename+'_pred',avg_pred)
    np.save('baseline',baseline)
    
    print("Time: %.8s sec" % (time.time() - start_time))