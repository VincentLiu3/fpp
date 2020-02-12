import torch
import numpy as np
from model.torch_uoro import UOROModel
from model.torch_fpp import FPPModel
from model.torch_tbptt import TBPTTModel
from model.torch_buffer import ReplayBuffer
from model.utils import get_parser, check_args, print_msg
import os
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def get_file_name(FLAGS):
    if FLAGS.dataset in ['cycleworld', 'cw']:
        data_name = '{}_cw'.format(FLAGS.env_size)
        dir_name = 'results/results-cw'
    elif FLAGS.dataset in ['stochastic_dataset']:
        data_name = 'sd'
        dir_name = 'results/results-{}'.format(data_name)
    elif FLAGS.dataset in ['sd', 'lsd', 'anbn']:
        data_name = FLAGS.dataset
        dir_name = 'results/results-{}'.format(data_name)
    elif FLAGS.dataset in ['sequential_mnist', 'mnist']:
        data_name = 'mnist'
        dir_name = 'results/results-{}'.format(data_name)
    else:
        assert False, 'unknown dataset'

    if FLAGS.model_name == 'uoro':
        mini_dir_name = 'uoro'
        filename = '{},{},{},{}'.format(data_name, FLAGS.data_size, FLAGS.num_run, FLAGS.lr)
    else:
        mini_dir_name = FLAGS.model_name
        filename = '{},{},{},{},{},{},{},{},{},{},{}'.format(data_name, FLAGS.data_size, FLAGS.num_run,
                                                             FLAGS.state_update, FLAGS.overlap,
                                                             FLAGS.reg_lambda, FLAGS.buffer_size, FLAGS.T,
                                                             FLAGS.num_update, FLAGS.batch_size, FLAGS.lr)

    os.makedirs('{}/{}/'.format(dir_name, mini_dir_name), exist_ok=True)
    os.makedirs('logs/{}'.format(mini_dir_name), exist_ok=True)
    log_file = 'logs/{}/{}.log'.format(mini_dir_name, filename)
    data_file = '{}/{}/{}'.format(dir_name, mini_dir_name, filename)
    result_file = '{}/{}_sweep.txt'.format(dir_name, mini_dir_name)
    return data_name, filename, log_file, data_file, result_file


if __name__ == '__main__':
    parser = get_parser()
    FLAGS = parser.parse_args()
    FLAGS = check_args(FLAGS)

    if FLAGS.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data_name, filename, log_file, data_file, result_file = get_file_name(FLAGS)
    data_size = FLAGS.data_size
    constant_seed = 0
    eval_size = 100
    accuracy_series = np.zeros(shape=(FLAGS.num_run, data_size//eval_size))
    loss_series = np.zeros(shape=(FLAGS.num_run, data_size//eval_size))

    for run_no in range(FLAGS.num_run):
        np.random.seed(constant_seed + run_no)
        torch.manual_seed(constant_seed + run_no)

        msg = '[Run {}] {} on {}: Run={}. lr={}'.format(run_no, FLAGS.model_name, data_name, run_no, FLAGS.lr)
        print_msg(log_file, msg, FLAGS.verbose)

        # generate dataset
        if FLAGS.dataset in ['cycleworld', 'cw']:
            from env.data import generate_cw
            X, Y = generate_cw(FLAGS.env_size, FLAGS.num_trajectory, data_size)
            # X.shape = (1, data_size, 2)
            # Y.shape = (1, data_size, 2)
            FLAGS.n_input = FLAGS.n_output = 2
        elif FLAGS.dataset in ['sd', 'stochastic_dataset']:
            from env.data import generate_stochastic_data
            X, Y = generate_stochastic_data(FLAGS.num_trajectory, data_size)
            FLAGS.n_input = FLAGS.n_output = 2
        elif FLAGS.dataset in ['lsd']:
            from env.data import generate_stochastic_data
            X, Y = generate_stochastic_data(FLAGS.num_trajectory, data_size, is_short=False)
            FLAGS.n_input = FLAGS.n_output = 2
        elif FLAGS.dataset in ['anbn']:
            from env.anbn import generate_anbn_data
            X, Y = generate_anbn_data(data_size, k=1, l=4, num_class=3)
            FLAGS.n_input = FLAGS.n_output = 3
        elif FLAGS.dataset in ['sequential_mnist', 'mnist']:
            from env.sequential_mnist import MnistDataset
            mnist_dataset = MnistDataset(dataset_folder='./data')
            FLAGS.n_input = 28
            FLAGS.n_output = 10
        else:
            assert False, 'unknown dataset'

        if FLAGS.dataset not in ['sequential_mnist', 'mnist']:
            X = torch.from_numpy(X).float()
            # Y = torch.from_numpy(Y).float()
            Y = torch.from_numpy(np.argmax(Y, axis=2)).long()

        # define model
        if FLAGS.model_name == 'fpp':
            model = FPPModel(FLAGS.n_input, FLAGS.num_units, FLAGS.n_output, FLAGS.lr, FLAGS.state_update,
                             FLAGS.batch_size, FLAGS.T, FLAGS.reg_lambda, device)
            buffer = ReplayBuffer(FLAGS.buffer_size)

        elif FLAGS.model_name == 'uoro':
            model = UOROModel(FLAGS.n_input, FLAGS.num_units, FLAGS.n_output, FLAGS.lr, device)

        elif FLAGS.model_name == 't-bptt':
            model = TBPTTModel(FLAGS.n_input, FLAGS.num_units, FLAGS.n_output, FLAGS.lr, FLAGS.state_update,
                               FLAGS.batch_size, FLAGS.T, FLAGS.overlap, device)
            buffer = ReplayBuffer(FLAGS.T)

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
            if FLAGS.dataset in ['sequential_mnist', 'mnist']:
                x_t, y_t = mnist_dataset[iter_id]
            else:
                # get slice from time (iter-T to iter)
                x_t = X[:, iter_id:(iter_id+1)].reshape([1, 1, FLAGS.n_input])
                y_t = Y[:, iter_id:(iter_id+1)].reshape([1])

            loss, acc, state_old, state_new = model.forward(x_t, y_t)
            # print(y_t)

            if FLAGS.dataset in ['sequential_mnist', 'mnist']:
                # only compute loss for the last 14 time steps
                if iter_id % 28 >= 15:
                    sum_acc += acc
                    sum_loss += loss
                    count += 1
            else:
                sum_acc += acc
                sum_loss += loss
                count += 1

            if FLAGS.model_name == 'fpp':
                # add data to buffer
                data = x_t, state_old, state_new, y_t
                buffer.add(data)

                if iter_id >= FLAGS.T:
                    for _ in range(FLAGS.num_update):
                        # sample from buffer
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
                            buffer.replace_new(idx_series[b][-1], state_new_updated[:, b:(b+1), :])

            elif FLAGS.model_name == 't-bptt':
                data = x_t, state_old, state_new, y_t
                buffer.add(data)

                if (FLAGS.overlap and iter_id >= FLAGS.T) or (FLAGS.overlap is False and (iter_id+1) % FLAGS.T == 0):
                    x_batch, state_old_batch, state_new_batch, y_batch, idx_series = buffer.sample_all()
                    # what should be the initial state for T-BPTT? zero vector?
                    trn_loss, state_old_updated, state_new_updated = model.train(x_batch, state_old_batch, state_new_batch, y_batch)
                    sum_trn_loss += trn_loss
                    update_count += 1

            # evaluation
            if (iter_id+1) % 100 == 0:
                pred_series.append(sum_acc/count)
                losses.append(sum_loss/count)

                if FLAGS.model_name in ['fpp', 't-bptt']:
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

    # save result and data
    with open(result_file, 'a') as f:
        performance = '{:.4f},{:.4f}'.format(np.mean(accuracy_series), np.mean(loss_series))
        f.write('{},{}\n'.format(filename, performance))
        if FLAGS.verbose:
            logging.info('{},{}'.format(filename, performance))

    save_dict = {
        'acc': accuracy_series,
        'loss': loss_series
    }
    np.save(data_file, save_dict)
    msg = 'writing data to {}'.format(data_file)
    print_msg(log_file, msg, FLAGS.verbose)
