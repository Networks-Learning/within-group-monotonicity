"""
Utils for running the experiments
"""
import os
import random
import numpy as np
from sklearn.utils import shuffle


def generate_commands(exp_dir, Z, n_trains, n_cals, n_test, lbds, runs, n_runs_test, k, classifier_type,
                      umb_num_bins, train_cal_raw_path, test_raw_path):
    """
    generate a list of commands from the experiment setup
    """
    commands = []
    for Z_indices in Z:
        for n_train in n_trains:
            for n_cal in n_cals:
                for lbd in lbds:
                    for run in runs:
                        exp_identity_string = "_".join(["_".join([str(index) for index in Z_indices]), str(n_train), str(n_cal), lbd, str(run)])
                        print("adding Experiment: " + exp_identity_string)
                        train_data_path = os.path.join(exp_dir, exp_identity_string + "_train_data.pkl")
                        cal_data_path = os.path.join(exp_dir, exp_identity_string + "_cal_data.pkl")
                        scaler_path = os.path.join(exp_dir, exp_identity_string + "_scaler.pkl")
                        data_generation_command = "python ./scripts/generate_data.py --n_train {} --n_cal {} " \
                                                  "--train_cal_raw_path {} --train_data_path {} --cal_data_path {} " \
                                                  "--scaler_path {}".format(n_train, n_cal, train_cal_raw_path,
                                                                         train_data_path, cal_data_path, scaler_path)


                        classifier_path = os.path.join(exp_dir, exp_identity_string + "_classifier.pkl")
                        if classifier_type == "LR":
                            train_classifier_command = "python ./src/train_LR.py --Z_indices {} --train_data_path {} --cal_data_path {} --lbd {} " \
                                                       "--classifier_path {}".format("_".join([str(index) for index in Z_indices]), train_data_path, cal_data_path, lbd,
                                                                                     classifier_path)
                        else:
                            raise ValueError("Classifier {} not supported".format(classifier_type))



                        exp_commands = [data_generation_command, train_classifier_command]
                        for umb_num_bin in umb_num_bins:
                            umb_result_path = os.path.join(exp_dir, exp_identity_string +
                                                           "_umb_{}_result.pkl".format(umb_num_bin))
                            umb_path = os.path.join(exp_dir, exp_identity_string + "_umb.pkl")
                            umb_prediction_command = "python ./src/umb.py --Z_indices {} --cal_data_path {} " \
                                                     "--test_raw_path {} --classifier_path {} --umb_path {} --result_path {} --k {}" \
                                                     " --m {} --B {} " \
                                                     "--scaler_path {} --n_runs_test {}".format("_".join([str(index) for index in Z_indices]), cal_data_path, test_raw_path,
                                                                               classifier_path, umb_path, umb_result_path, k, n_test,
                                                                                umb_num_bin, scaler_path,n_runs_test)
                            exp_commands.append(umb_prediction_command)



                            wgm_path = os.path.join(exp_dir, exp_identity_string + "_wgm.pkl")
                            wgm_result_path = os.path.join(exp_dir, exp_identity_string + "_wgm_{}_result.pkl".format(umb_num_bin))
                            wgm_command = "python ./src/wg_monotone.py --Z_indices {} --cal_data_path {} --test_raw_path {} --classifier_path {}" \
                                          " --wgm_path {} --result_path {} --k {} --m {}  --B {} " \
                                          "--scaler_path {} --n_runs_test {}".format("_".join([str(index) for index in Z_indices]),cal_data_path, test_raw_path, classifier_path, wgm_path,
                                                                    wgm_result_path, k, n_test,umb_num_bin, scaler_path,n_runs_test)
                            exp_commands.append(wgm_command)


                            wgc_path = os.path.join(exp_dir, exp_identity_string + "_wgc.pkl")
                            wgc_result_path = os.path.join(exp_dir,
                                                           exp_identity_string + "_wgc_{}_result.pkl".format(
                                                               umb_num_bin))
                            wgc_command = "python ./src/wg_calibrated.py --Z_indices {} --cal_data_path {} --test_raw_path {} --classifier_path {}" \
                                          " --wgc_path {} --result_path {} --k {} --m {}  --B {} " \
                                          "--scaler_path {} --n_runs_test {}".format(
                                "_".join([str(index) for index in Z_indices]), cal_data_path, test_raw_path,
                                classifier_path, wgc_path,
                                wgc_result_path, k, n_test, umb_num_bin, scaler_path, n_runs_test)
                            exp_commands.append(wgc_command)

                            pav_path = os.path.join(exp_dir, exp_identity_string + "_pav.pkl")
                            pav_result_path = os.path.join(exp_dir,
                                                           exp_identity_string + "_pav_{}_result.pkl".format(
                                                               umb_num_bin))
                            pav_command = "python ./src/pav.py --Z_indices {} --cal_data_path {} --test_raw_path {} --classifier_path {}" \
                                          " --pav_path {} --result_path {} --k {} --m {}  --B {} " \
                                          "--scaler_path {} --n_runs_test {}".format(
                                "_".join([str(index) for index in Z_indices]), cal_data_path, test_raw_path,
                                classifier_path, pav_path,
                                pav_result_path, k, n_test, umb_num_bin, scaler_path,
                                n_runs_test)
                            exp_commands.append(pav_command)

                        commands.append(exp_commands)
    return commands



def generate_commands_discrimination(exp_dir, Z, n_trains, n_cals, n_test, lbds, runs, n_runs_test, k, classifier_type,
                      umb_num_bins, train_cal_raw_path, test_raw_path):
    """
    generate a list of commands from the experiment setup
    """
    commands = []
    for Z_indices in Z:
        for n_train in n_trains:
            for n_cal in n_cals:
                for lbd in lbds:
                    for run in runs:
                        exp_identity_string = "_".join(["_".join([str(index) for index in Z_indices]), str(n_train), str(n_cal), lbd, str(run)])
                        print("adding Experiment: " + exp_identity_string)
                        train_data_path = os.path.join(exp_dir, exp_identity_string + "_train_data.pkl")
                        cal_data_path = os.path.join(exp_dir, exp_identity_string + "_cal_data.pkl")
                        scaler_path = os.path.join(exp_dir, exp_identity_string + "_scaler.pkl")
                        data_generation_command = "python ./scripts/generate_data.py --n_train {} --n_cal {} " \
                                                  "--train_cal_raw_path {} --train_data_path {} --cal_data_path {} " \
                                                  "--scaler_path {}".format(n_train, n_cal, train_cal_raw_path,
                                                                         train_data_path, cal_data_path, scaler_path)

                        classifier_path = os.path.join(exp_dir, exp_identity_string + "_classifier.pkl")
                        if classifier_type == "LR":
                            train_classifier_command = "python ./src/train_LR.py --Z_indices {} --train_data_path {} --cal_data_path {} --lbd {} " \
                                                       "--classifier_path {}".format("_".join([str(index) for index in Z_indices]), train_data_path, cal_data_path, lbd,
                                                                                     classifier_path)
                        else:
                            raise ValueError("Classifier {} not supported".format(classifier_type))


                        exp_commands = [data_generation_command, train_classifier_command]
                        for umb_num_bin in umb_num_bins:
                            umb_result_path = os.path.join(exp_dir, exp_identity_string +
                                                           "_umb_{}_result.pkl".format(umb_num_bin))
                            umb_path = os.path.join(exp_dir, exp_identity_string + "_umb.pkl")
                            umb_prediction_command = "python ./src/umb.py --Z_indices {} --cal_data_path {} " \
                                                     "--test_raw_path {} --classifier_path {} --umb_path {} --result_path {} --k {}" \
                                                     " --m {} --B {} " \
                                                     "--scaler_path {} --n_runs_test {}".format("_".join([str(index) for index in Z_indices]), cal_data_path, test_raw_path,
                                                                               classifier_path, umb_path, umb_result_path, k, n_test,
                                                                                umb_num_bin, scaler_path,n_runs_test)
                            exp_commands.append(umb_prediction_command)

                        commands.append(exp_commands)
    return commands


def submit_commands(exp_token, exp_dir, split_size, commands, submit):
    """
    submit commands to server
    """
    perm = list(range(len(commands)))
    random.shuffle(perm)
    commands = [commands[idx] for idx in perm]
    split_len = int((len(commands) - 1) / split_size) + 1
    current_idx = 0
    while True:
        stop = 0
        start = current_idx * split_len
        end = (current_idx + 1) * split_len
        if end >= len(commands):
            stop = 1
            end = len(commands)
        with open(os.path.join(exp_dir, "scripts{}.sh".format(current_idx)), "w") as f:
            for exp_commands in commands[start:end]:
                for exp_command in exp_commands:
                    f.write(exp_command + "\n")
        current_idx += 1
        if stop:
            break

    scripts = [os.path.join(exp_dir, "scripts{}.sh".format(idx)) for idx in range(current_idx)]
    cnt = 0
    for script in scripts:
        submission_command = "sbatch -N1 -n1 -c1 --mem=12G -t 72:00:00 " \
                             "-J %s -o %s.o -e %s.e --wrap=\"sh %s\"" % (exp_token + str(cnt), script, script, script)
        cnt += 1
        if submit:
            os.system(submission_command)
    return


def transform_except_last_dim(data, scaler):
    return np.concatenate((scaler.transform(data[:, :-1]), data[:, -1:]), axis=1)
