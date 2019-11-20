import torch
from env.data import *
from model.uoro import UORO_Model
from model.torch_fpp import FPP_Model
from model.torch_buffer import Replay_Buffer
from model.utils import get_parser, add_args, print_msg
import os
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def get_file_name(FLAGS):
    if FLAGS.dataset == 'cycleworld':
        data_name = '{}_cw'.format(FLAGS.env_size)
        dir_name = 'results/results-cw'
    elif FLAGS.dataset in ['stochastic_dataset']:
        data_name = 'sd'
        dir_name = 'results/results-sd'
    elif FLAGS.dataset in ['sd', 'lsd', 'anbn']:
        data_name = FLAGS.dataset
        dir_name = 'results/results-{}'.format(data_name)
    else:
        assert False, 'unknown dataset'

    if FLAGS.model_name == 'uoro':
        mini_dir_name = 'uoro'
        filename = '{},{},{},{}'.format(data_name, FLAGS.data_size, FLAGS.num_run, FLAGS.lr)
    elif FLAGS.model_name == 'fpp':
        if FLAGS.state_update:
            mini_dir_name = 'fpp'
        else:
            mini_dir_name = 'buffer_bptt'
        filename = '{},{},{},{},{},{},{},{},{}'.format(data_name, FLAGS.data_size, FLAGS.num_run, FLAGS.reg_lambda,
                                                       FLAGS.buffer_size, FLAGS.T, FLAGS.num_update, FLAGS.batch_size,
                                                       FLAGS.lr)
    else:
        assert False, 'unknown model name'

    os.makedirs('{}/{}/'.format(dir_name, mini_dir_name), exist_ok=True)
    os.makedirs('logs/{}'.format(mini_dir_name), exist_ok=True)
    log_file = 'logs/{}/{}.log'.format(mini_dir_name, filename)
    data_file = '{}/{}/{}'.format(dir_name, mini_dir_name, filename)
    result_file = '{}/{}_sweep.txt'.format(dir_name, mini_dir_name)
    return data_name, filename, log_file, data_file, result_file


if __name__ == '__main__':
    parser = get_parser()
    FLAGS = parser.parse_args()
    FLAGS = add_args(FLAGS)

    data_name, filename, log_file, data_file, result_file = get_file_name(FLAGS)
    data_size = FLAGS.data_size
    eval_size = 100
    accuracy_series = np.zeros(shape=(FLAGS.num_run, data_size//eval_size))
    loss_series = np.zeros(shape=(FLAGS.num_run, data_size//eval_size))

    for run_no in range(FLAGS.num_run):
        constant_seed = 0
        np.random.seed(constant_seed + run_no)
        torch.manual_seed(constant_seed + run_no)

        msg = '{}: Run={}. lr={}'.format(data_name, run_no, FLAGS.lr)
        print_msg(log_file, msg, FLAGS.verbose)

        # generate dataset
        # X.shape = (1, 100000, 2)
        # Y.shape = (1, 100000, 2)
        if FLAGS.dataset == 'cycleworld':
            X, Y = generate_cw(FLAGS.env_size, FLAGS.num_trajectory, data_size)
            FLAGS.n_input = FLAGS.n_output = 2
        elif FLAGS.dataset in ['sd', 'stochastic_dataset']:
            X, Y = generate_stochastic_data(FLAGS.num_trajectory, data_size)
            FLAGS.n_input = FLAGS.n_output = 2
        elif FLAGS.dataset in ['lsd']:
            X, Y = generate_stochastic_data(FLAGS.num_trajectory, data_size, is_short=False)
            FLAGS.n_input = FLAGS.n_output = 2
        elif FLAGS.dataset in ['anbn']:
            from env.anbn import generate_anbn_data
            X, Y = generate_anbn_data(data_size, k=1, l=4, num_class=3)
            FLAGS.n_input = FLAGS.n_output = 3
        else:
            assert False, 'unknown dataset'

        X = torch.from_numpy(X).float()
        # Y = torch.from_numpy(Y).float()
        Y = torch.from_numpy(np.argmax(Y, axis=2)).long()

        # define model
        if FLAGS.model_name == 'fpp':
            model = FPP_Model(FLAGS.n_input, FLAGS.num_units, FLAGS.n_output, FLAGS.lr, FLAGS.state_update,
                              FLAGS.batch_size, FLAGS.T, FLAGS.reg_lambda)
            buffer = Replay_Buffer(FLAGS.buffer_size)
        else:
            model = UORO_Model(FLAGS.n_input, FLAGS.num_units, FLAGS.n_output, FLAGS.lr)

        # training
        iter_id = 0
        sum_acc = 0
        sum_loss = 0
        sum_trn_loss = 0
        count = 0
        update_count = 0

        pred_series = []
        losses = []

        model.initialize_state()
        while iter_id < data_size:
            # print(iter_id)
            # get slice from time (iter-T to iter)
            x_t = X[:, iter_id:(iter_id+1)].reshape([1, 1, FLAGS.n_input])
            y_t = Y[:, iter_id:(iter_id+1)].reshape([1])

            loss, acc, state_old, state_new = model.forward(x_t, y_t)

            sum_acc += acc
            sum_loss += loss
            count += 1

            # for FPP
            if FLAGS.model_name == 'fpp':
                # add to buffer
                data = x_t, state_old, state_new, y_t
                buffer.add(data)

                if iter_id >= FLAGS.T:
                    # sample from buffer
                    for _ in range(FLAGS.num_update):
                        x_batch, state_old_batch, state_new_batch, y_batch, idx_series = buffer.sample_batch(FLAGS.batch_size,
                                                                                                             FLAGS.T)
                        # update FPP
                        trn_loss, state_old_updated, state_new_updated = model.train(x_batch, state_old_batch, state_new_batch, y_batch)
                        sum_trn_loss += trn_loss
                        update_count += 1

                    if FLAGS.state_update:
                        # update buffer
                        for b in range(idx_series.shape[0]):
                            buffer.replace_old(idx_series[b][0], state_old_updated[:, b:(b+1), :])
                            buffer.replace_new(idx_series[b][-1], state_new_updated[:, b:(b + 1), :])

            if (iter_id+1) % 100 == 0:
                pred_series.append(sum_acc/count)
                losses.append(sum_loss/count)

                if FLAGS.model_name == 'fpp':
                    msg = 'Steps {:5d}. Training loss {:.4f}. Accuracy {:.2f}. Loss {:4f}.'.format(iter_id + 1,
                                                                                                   sum_trn_loss / update_count,
                                                                                                   sum_acc / count,
                                                                                                   sum_loss / count)
                else:
                    msg = 'Steps {:5d}. Accuracy {:.2f}. Loss {:4f}.'.format(iter_id+1, sum_acc/count, sum_loss/count)
                print_msg(log_file, msg, FLAGS.verbose)

                # corr = 0
                sum_acc = 0
                sum_loss = 0
                sum_trn_loss = 0
                count = 0
                update_count = 0

            iter_id += 1

        accuracy_series[run_no] = pred_series
        loss_series[run_no] = losses

    # save result
    with open(result_file, 'a') as f:
        performance = '{:.4f},{:.4f}'.format(np.mean(accuracy_series), np.mean(loss_series))
        f.write('{},{}\n'.format(filename, performance))
        if FLAGS.verbose:
            logging.info('{},{}'.format(filename, performance))

    # np.save('results/baseline_{}-cw'.format(FLAGS.env_size),baseline)
    save_dict = {
        'acc': accuracy_series,
        'loss': loss_series
    }
    np.save(data_file, save_dict)
    msg = 'writing data to {}'.format(data_file)
    print_msg(log_file, msg, FLAGS.verbose)
