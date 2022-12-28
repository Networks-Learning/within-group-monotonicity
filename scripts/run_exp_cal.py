"""
Run the experiments where we vary the amount of calibration data.
"""
import os
from exp_utils import generate_commands, submit_commands
from params_exp_cal import *
if __name__ == "__main__":
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    train_cal_raw_path = "./data/data_normal_train_cal_raw_q_ratio_{}_test_ratio_{}.pkl".format(q_ratio, test_ratio)
    test_raw_path = "./data/data_normal_test_raw_q_ratio_{}_test_ratio_{}.pkl".format(q_ratio, test_ratio)
    if prepare_data:
        prepare_data_command = "python ./scripts/prepare_data.py --train_cal_raw_path {} --test_raw_path {} " \
                               "--q_ratio {} --test_ratio {}".format(train_cal_raw_path, test_raw_path,
                                                                     q_ratio, test_ratio)
        os.system(prepare_data_command)
    commands = generate_commands(exp_dir, Z, n_trains, n_cals, n_test, lbds, runs, n_runs_test, k, alpha, classifier_type,
                                 umb_num_bins, train_cal_raw_path, test_raw_path, noise_ratios,generate_data,train_LR, train_umb)
    # print(len(commands))
    if submit:
        submit_commands(exp_token, exp_dir, split_size, commands, submit)