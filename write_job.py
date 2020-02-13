from itertools import product
import numpy as np
import argparse

save_in_folder = "scripts/"
NUM_STEP = 10000


def write_run_all_jobs(count, init_count=0):
    """
    generate a .sh file to run all experiments
    """
    with open('{}run_all.sh'.format(save_in_folder), 'w') as f:
        f.write('#!/usr/bin/env bash\nchmod +x ./{}tasks_*.sh\n'.format(save_in_folder))
        for i in range(count-init_count):
            f.write('./{}tasks_{}.sh &> {}log{}.txt &\n'.format(save_in_folder, init_count+i, save_in_folder, i))
            if (i+1) % 10 == 0:
                f.write('wait\n')


def write_jobs(all_comb, count, verbose):
    cmd = "python main_torch.py --dataset {} --data_size {} --model {} --state_update {} --overlap {} " \
          "--T {} --num_update {} --batch_size {} --lr {} --buffer_size 100 --num_run 10\n"
    # init_count = count
    for domain, num_step, model, state_update, overlap, T, M, B, lr in all_comb:
        new_cmd = cmd.format(domain, num_step, model, state_update, overlap, T, M, B, lr)
        with open("{}tasks_{}.sh".format(save_in_folder, count), 'w') as f:
            f.write(new_cmd)
        if verbose == "True":
            print(count, new_cmd)
        count += 1
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default="True", type=str)
    parser.add_argument('--start', default=0, type=int)
    args = parser.parse_args()

    count = args.start

    # cw
    # domain_lst = ['cw']
    # # model_lst = ['fpp']
    # # state_update_lst = ['True', 'False']
    # # overlap_lst = ['True']
    # # T_lst = [1, 2, 4]
    # # M_lst = [1]
    # # B_lst = [1, 2, 4, 6, 16]
    # lr_lst = [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
    # num_step_lst = [5000]

    # all_comb = list(product(domain_lst, num_step_lst, model_lst, state_update_lst, overlap_lst, T_lst, M_lst, B_lst, lr_lst))
    # count = write_jobs(all_comb, count, args.verbose)
    #
    # model_lst = ['t-bptt']
    # state_update_lst = ['True']
    # overlap_lst = ['True', 'False']
    # M_lst = [1]
    # B_lst = [1]
    # all_comb = list(product(domain_lst, num_step_lst, model_lst, state_update_lst, overlap_lst, T_lst, M_lst, B_lst, lr_lst))
    # count = write_jobs(all_comb, count, args.verbose)

    # model_lst = ['uoro']
    # state_update_lst = ['True']
    # overlap_lst = ['True']
    # T_lst = [1]
    # M_lst = [1]
    # B_lst = [1]
    # all_comb = list(
    #     product(domain_lst, num_step_lst, model_lst, state_update_lst, overlap_lst, T_lst, M_lst, B_lst, lr_lst))
    # count = write_jobs(all_comb, count, args.verbose)

    # lsd
    # domain_lst = ['lsd']
    # model_lst = ['fpp']
    # state_update_lst = ['True', 'False']
    # overlap_lst = ['True']
    # T_lst = [8, 16, 32]
    # M_lst = [1]
    # B_lst = [1, 2, 4, 8, 16]
    # num_step_lst = [10000]

    # all_comb = list(product(domain_lst, num_step_lst, model_lst, state_update_lst, overlap_lst, T_lst, M_lst, B_lst, lr_lst))
    # count = write_jobs(all_comb, count, args.verbose)

    # model_lst = ['t-bptt']
    # state_update_lst = ['True']
    # overlap_lst = ['True', 'False']
    # M_lst = [1]
    # B_lst = [1]
    # all_comb = list(product(domain_lst, num_step_lst, model_lst, state_update_lst, overlap_lst, T_lst, M_lst, B_lst, lr_lst))
    # count = write_jobs(all_comb, count, args.verbose)

    # model_lst = ['uoro']
    # state_update_lst = ['True']
    # overlap_lst = ['True']
    # T_lst = [1]
    # M_lst = [1]
    # B_lst = [1]
    # all_comb = list(
    #     product(domain_lst, num_step_lst, model_lst, state_update_lst, overlap_lst, T_lst, M_lst, B_lst, lr_lst))
    # count = write_jobs(all_comb, count, args.verbose)

    # mnist
    domain_lst = ['mnist']
    model_lst = ['fpp']
    state_update_lst = ['True']
    overlap_lst = ['True']
    T_lst = [7, 14, 21, 28]
    M_lst = [1]
    B_lst = [1]
    lr_lst = [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
    num_step_lst = [28000]

    all_comb = list(
        product(domain_lst, num_step_lst, model_lst, state_update_lst, overlap_lst, T_lst, M_lst, B_lst, lr_lst))
    count = write_jobs(all_comb, count, args.verbose)

    model_lst = ['t-bptt']
    state_update_lst = ['True']
    overlap_lst = ['True', 'False']
    M_lst = [1]
    B_lst = [1]
    all_comb = list(
        product(domain_lst, num_step_lst, model_lst, state_update_lst, overlap_lst, T_lst, M_lst, B_lst, lr_lst))
    count = write_jobs(all_comb, count, args.verbose)

    model_lst = ['uoro']
    state_update_lst = ['True']
    overlap_lst = ['True']
    T_lst = [1]
    M_lst = [1]
    B_lst = [1]
    all_comb = list(
        product(domain_lst, num_step_lst, model_lst, state_update_lst, overlap_lst, T_lst, M_lst, B_lst, lr_lst))
    count = write_jobs(all_comb, count, args.verbose)

    domain_lst = ['mnist']
    model_lst = ['fpp']
    state_update_lst = ['True']
    overlap_lst = ['True']
    T_lst = [28]
    M_lst = [1]
    B_lst = [8, 16]
    lr_lst = [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
    num_step_lst = [28000]

    all_comb = list(
        product(domain_lst, num_step_lst, model_lst, state_update_lst, overlap_lst, T_lst, M_lst, B_lst, lr_lst))
    count = write_jobs(all_comb, count, args.verbose)

    print('sbatch --array={}-{} ./run.sh'.format(0, count-1))
    write_run_all_jobs(count)