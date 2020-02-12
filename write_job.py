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
          "--T {} --batch_size {} --num_update {} --lr {} --num_units {} --num_run 10\n"
    init_count = count
    for domain, model, state_update, overlap, T, M, B, lr, num_unit in all_comb:
        new_cmd = cmd.format(domain, NUM_STEP, model, state_update, overlap, T, M, B, lr, num_unit)
        with open("{}tasks_{}.sh".format(save_in_folder, count), 'w') as f:
            f.write(new_cmd)
        if verbose == "True":
            print(count, new_cmd)
        count += 1
    print('sbatch --array={}-{} ./run.sh'.format(init_count, count-1))
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default="True", type=str)
    parser.add_argument('--start', default=0, type=int)
    args = parser.parse_args()

    count = args.start

    # TPA
    domain_lst = ['cw']
    model_lst = ['fpp']
    state_update_lst = ['True', 'False']
    overlap_lst = ['True']
    T_lst = [10]
    M_lst = [1]
    B_lst = [1]
    lr_lst = [0.001, 0.0003]
    num_unit_lst = [4]
    run_id_lst = np.arange(10)

    all_comb = list(product(domain_lst, model_lst, state_update_lst, overlap_lst, T_lst, M_lst, B_lst, lr_lst,
                            num_unit_lst))
    count = write_jobs(all_comb, count, args.verbose)

    model_lst = ['t-bptt']
    state_update_lst = ['True']
    overlap_lst = ['True', 'False']
    all_comb = list(product(domain_lst, model_lst, state_update_lst, overlap_lst, T_lst, M_lst, B_lst, lr_lst,
                            num_unit_lst))
    count = write_jobs(all_comb, count, args.verbose)

    model_lst = ['uoro']
    state_update_lst = ['True']
    overlap_lst = ['True']
    all_comb = list(product(domain_lst, model_lst, state_update_lst, overlap_lst, T_lst, M_lst, B_lst, lr_lst,
                            num_unit_lst))
    count = write_jobs(all_comb, count, args.verbose)

    write_run_all_jobs(count)