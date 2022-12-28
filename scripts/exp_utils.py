"""
Utils for running the experiments
"""
import os
import random
import numpy as np
from sklearn.utils import shuffle


def generate_commands(exp_dir, Z, n_trains, n_cals, n_test, lbds, runs, n_runs_test, k, alpha, classifier_type,
                      umb_num_bins, train_cal_raw_path, test_raw_path, noise_ratios=[-1],generate_data = False, train_LR = False, train_umb = False):
    """
    generate a list of commands from the experiment setup
    """
    commands = []
    print(Z)
    for Z_indices in Z:
        for n_train in n_trains:
            for noise_ratio in noise_ratios:
                for n_cal in n_cals:
                    for lbd in lbds:
                        for run in runs:
                            exp_identity_string = "_".join(["_".join([str(index) for index in Z_indices]), str(n_train), str(noise_ratio), str(n_cal), lbd, str(run)])
                            print("adding Experiment: " + exp_identity_string)
                            train_data_path = os.path.join(exp_dir, exp_identity_string + "_train_data.pkl")
                            cal_data_path = os.path.join(exp_dir, exp_identity_string + "_cal_data.pkl")
                            scaler_path = os.path.join(exp_dir, exp_identity_string + "_scaler.pkl")
                            data_generation_command = "python ./scripts/generate_data.py --n_train {} --n_cal {} " \
                                                      "--train_cal_raw_path {} --train_data_path {} --cal_data_path {} " \
                                                      "--scaler_path {}".format(n_train, n_cal, train_cal_raw_path,
                                                                             train_data_path, cal_data_path, scaler_path)

                            if generate_data:
                                # print("generating data...")
                                # if os.system(data_generation_command)==256:
                                #     return
                                commands.append(data_generation_command)
                            classifier_path = os.path.join(exp_dir, exp_identity_string + "_classifier.pkl")
                            if classifier_type == "LR":
                                train_classifier_command = "python ./src/train_LR.py --Z_indices {} --train_data_path {} --cal_data_path {} --lbd {} " \
                                                           "--noise_ratio_maj {} --noise_ratio_min {} " \
                                                           "--classifier_path {}".format("_".join([str(index) for index in Z_indices]), train_data_path, cal_data_path, lbd, noise_ratio,
                                                                                         noise_ratio, classifier_path)
                            else:
                                raise ValueError("Classifier {} not supported".format(classifier_type))



                            if train_LR:
                                commands.append(train_classifier_command)
                                # print("training LR...")
                                # if os.system(train_classifier_command)==256:
                                #     return

                            # css_result_path = os.path.join(exp_dir, exp_identity_string + "_css_result.pkl")
                            # css_command = "python ./src/css.py --cal_data_path {} --test_raw_path {}" \
                            #               " --classifier_path {} --result_path {} --k {} --m {} --alpha {} " \
                            #               "--scaler_path {}".format(cal_data_path, test_raw_path, classifier_path,
                            #                                         css_result_path, k, n_test, alpha, scaler_path)
                            # ucss_result_path = os.path.join(exp_dir, exp_identity_string + "_ucss_result.pkl")
                            # ucss_command = "python ./src/ucss.py --test_raw_path {} " \
                            #                "--classifier_path {} --result_path {} --k {} --m {} " \
                            #                "--n_runs_test {} --scaler_path {}".format(test_raw_path, classifier_path,
                            #                                                           ucss_result_path, k, n_test, n_runs_test,
                            #                                                           scaler_path)
                            # iso_reg_ss_result_path = os.path.join(exp_dir, exp_identity_string + "_iso_reg_ss_result.pkl")
                            # iso_reg_ss_command = "python ./src/iso_reg_ss.py --cal_data_path {} " \
                            #                      "--test_raw_path {} --classifier_path {} --result_path {} --k {} --m {} " \
                            #                      "--scaler_path {}".format(cal_data_path, test_raw_path, classifier_path,
                            #                                                iso_reg_ss_result_path, k, n_test, scaler_path)
                            #
                            # platt_scal_ss_result_path = os.path.join(exp_dir, exp_identity_string +
                            #                                             "_platt_scal_ss_result.pkl")
                            # platt_scal_ss_command = "python ./src/platt_scal_ss.py --cal_data_path {} " \
                            #                        "--test_raw_path {} --classifier_path {} --result_path {} --k {} --m {} " \
                            #                        "--n_runs_test {} --scaler_path {}".format(cal_data_path, test_raw_path,
                            #                                                               classifier_path,
                            #                                                               platt_scal_ss_result_path, k, n_test,
                            #                                                               n_runs_test, scaler_path)
                            #
                            # exp_commands = [data_generation_command, train_classifier_command, css_command, ucss_command,
                            #                 iso_reg_ss_command, platt_scal_ss_command]
                            for umb_num_bin in umb_num_bins:
                                umb_result_path = os.path.join(exp_dir, exp_identity_string +
                                                               "_umb_{}_result.pkl".format(umb_num_bin))
                                umb_path = os.path.join(exp_dir, exp_identity_string + "_umb.pkl")
                                umb_prediction_command = "python ./src/umb_ss.py --Z_indices {} --cal_data_path {} " \
                                                         "--test_raw_path {} --classifier_path {} --umb_path {} --result_path {} --k {}" \
                                                         " --m {} --alpha {} --B {} " \
                                                         "--scaler_path {}".format("_".join([str(index) for index in Z_indices]), cal_data_path, test_raw_path,
                                                                                   classifier_path, umb_path, umb_result_path, k, n_test,
                                                                                   alpha, umb_num_bin, scaler_path)

                                if train_umb:
                                    commands.append(umb_prediction_command)
                                    # print("training umb with {} bins".format(umb_num_bin))
                                    # if os.system(umb_prediction_command)==256:
                                    #     return

                                wgm_path = os.path.join(exp_dir, exp_identity_string + "_wgm.pkl")
                                wgm_result_path = os.path.join(exp_dir, exp_identity_string + "_wgm_{}_result.pkl".format(umb_num_bin))
                                wgm_command = "python ./src/wg_monotone.py --Z_indices {} --cal_data_path {} --test_raw_path {} --classifier_path {}" \
                                              " --wgm_path {} --result_path {} --k {} --m {} --alpha {} --B {} " \
                                              "--scaler_path {}".format("_".join([str(index) for index in Z_indices]),cal_data_path, test_raw_path, classifier_path, wgm_path,
                                                                        wgm_result_path, k, n_test, alpha,umb_num_bin, scaler_path)
                                commands.append(wgm_command)
                                # print("training wgm starting from umb with {} bins".format(umb_num_bin))
                                # if os.system(wgm_command)==256:
                                #     return

                            #     exp_commands.append(umb_prediction_command)
                            # commands.append(exp_commands)
    return commands


def generate_commands_diversity(exp_dir, n_train, n_trains_min, n_cal_maj, n_cals_min, n_test, n_test_maj,
                                n_test_min, lbds, runs, n_runs_test, k_maj, k_min, alpha, classifier_type,
                                umb_num_bins, train_cal_maj_raw_path, train_cal_min_raw_path, test_raw_path,
                                noise_ratio_maj=0, noise_ratios_min=[-1]):
    """
    generate a list of commands from the diversity experiment setup
    """
    commands = []
    for n_train_min in n_trains_min:
        n_train_maj = n_train - n_train_min
        for noise_ratio_min in noise_ratios_min:
            for n_cal_min in n_cals_min:
                for lbd in lbds:
                    for run in runs:
                        exp_identity_string = "_".join([str(n_train_min), str(noise_ratio_min), str(n_cal_min), lbd, str(run)])
                        train_data_path = os.path.join(exp_dir, exp_identity_string + "_train_data.pkl")
                        cal_data_maj_path = os.path.join(exp_dir, exp_identity_string + "_cal_data_maj.pkl")
                        cal_data_min_path = os.path.join(exp_dir, exp_identity_string + "_cal_data_min.pkl")
                        scaler_path = os.path.join(exp_dir, exp_identity_string + "_scaler.pkl")
                        data_generation_command = "python ./scripts/generate_data_diversity.py --n_train_maj {} " \
                                                  "--n_train_min {} --n_cal_maj {} --n_cal_min {} " \
                                                  "--train_cal_maj_raw_path {} --train_cal_min_raw_path {} " \
                                                  "--train_data_path {} --cal_data_maj_path {} --cal_data_min_path {} " \
                                                  "--scaler_path {}".format(n_train_maj, n_train_min, n_cal_maj, n_cal_min,
                                                                            train_cal_maj_raw_path, train_cal_min_raw_path,
                                                                            train_data_path, cal_data_maj_path,
                                                                            cal_data_min_path, scaler_path)
                        classifier_path = os.path.join(exp_dir, exp_identity_string + "_classifier.pkl")
                        if classifier_type == "LR":
                            train_classifier_command = "python ./src/train_LR.py --train_data_path {} --lbd {} " \
                                                       "--noise_ratio_maj {} --noise_ratio_min {} " \
                                                       "--classifier_path {}".format(train_data_path, lbd,
                                                                                     noise_ratio_maj, noise_ratio_min,
                                                                                     classifier_path)
                        elif classifier_type == "MLP":
                            train_classifier_command = "python ./src/train_MLP.py --train_data_path {} --lbd {} " \
                                                       "--classifier_path {}".format(train_data_path, lbd, classifier_path)
                        elif classifier_type == "NB":
                            train_classifier_command = "python ./src/train_NB.py --train_data_path {} " \
                                                       "--classifier_path {}".format(train_data_path, classifier_path)
                        else:
                            raise ValueError("Classifier {} not supported".format(classifier_type))
                        css_result_path = os.path.join(exp_dir, exp_identity_string + "_css_result.pkl")
                        css_command = "python ./src/css_diversity.py --cal_data_maj_path {} --cal_data_min_path {} " \
                                      "--test_raw_path {} --classifier_path {} --result_path {} --k_maj {} --k_min {} " \
                                      "--m_maj {} --m_min {} --alpha {} --scaler_path " \
                                      "{}".format(cal_data_maj_path, cal_data_min_path, test_raw_path, classifier_path,
                                                  css_result_path, k_maj, k_min, n_test_maj, n_test_min, alpha, scaler_path)
                        css_naive_result_path = os.path.join(exp_dir, exp_identity_string + "_css_naive_result.pkl")
                        css_naive_command = "python ./src/css_diversity_naive.py --cal_data_maj_path {} " \
                                            "--cal_data_min_path {} --test_raw_path {} --classifier_path {} " \
                                            "--result_path {} --k_maj {} --k_min {} --m_maj {} --m_min {} --alpha {} " \
                                            "--scaler_path {}".format(cal_data_maj_path, cal_data_min_path, test_raw_path,
                                                                      classifier_path, css_naive_result_path, k_maj, k_min,
                                                                      n_test_maj, n_test_min, alpha, scaler_path)
                        ucss_result_path = os.path.join(exp_dir, exp_identity_string + "_ucss_result.pkl")
                        ucss_command = "python ./src/ucss_diversity.py --test_raw_path {} " \
                                       "--classifier_path {} --result_path {} --k_maj {} --k_min {} --m {} " \
                                       "--n_runs_test {} --scaler_path {}".format(test_raw_path, classifier_path,
                                                                                  ucss_result_path, k_maj, k_min, n_test,
                                                                                  n_runs_test, scaler_path)
                        iso_reg_ss_result_path = os.path.join(exp_dir, exp_identity_string + "_iso_reg_ss_result.pkl")
                        iso_reg_ss_command = "python ./src/iso_reg_ss_diversity.py --cal_data_maj_path {} " \
                                             "--cal_data_min_path {} --test_raw_path {} --classifier_path {} " \
                                             "--result_path {} --k_maj {} --k_min {} --m_maj {} --m_min {} " \
                                             "--scaler_path {}".format(cal_data_maj_path, cal_data_min_path, test_raw_path,
                                                                       classifier_path, iso_reg_ss_result_path,  k_maj,
                                                                       k_min, n_test_maj, n_test_min, scaler_path)
                        platt_scal_ss_result_path = os.path.join(exp_dir, exp_identity_string + "_platt_scal_ss_result.pkl")
                        platt_scal_ss_command = "python ./src/platt_scal_ss_diversity.py --cal_data_maj_path {} " \
                                               "--cal_data_min_path {} --test_raw_path {} --classifier_path {} " \
                                               "--result_path {} --k_maj {} --k_min {} --m {} --n_runs_test {} " \
                                               "--scaler_path {}".format(cal_data_maj_path, cal_data_min_path,
                                                                         test_raw_path, classifier_path,
                                                                         platt_scal_ss_result_path, k_maj, k_min, n_test,
                                                                         n_runs_test, scaler_path)
                        exp_commands = [data_generation_command, train_classifier_command, css_command, css_naive_command,
                                        ucss_command, iso_reg_ss_command, platt_scal_ss_command]
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


def satisfy_ratio(X, y, q_ratio):
    index_q = []
    index_uq = []
    for (i, label) in enumerate(y):
        if label:
            index_q.append(i)
        else:
            index_uq.append(i)
    X_q = X[index_q, :]
    y_q = y[index_q]
    X_uq = X[index_uq, :]
    y_uq = y[index_uq]
    n_q = y_q.size
    n_uq = y_uq.size
    if (n_q * 1.) / (n_q + n_uq) > q_ratio:
        n_q = int(n_uq * q_ratio / (1 - q_ratio))
        X_q, y_q = shuffle(X_q, y_q)
        X_q, y_q = X_q[:n_q], y_q[:n_q]
    else:
        n_uq = int(n_q * (1 - q_ratio) / q_ratio)
        X_uq, y_uq = shuffle(X_uq, y_uq)
        X_uq, y_uq = X_uq[:n_uq], y_uq[:n_uq]
    X, y = np.concatenate((X_q, X_uq), axis=0), np.concatenate((y_q, y_uq))
    X, y = shuffle(X, y)
    return X, y


def satisfy_rho(X, y):
    index_q = []
    index_uq = []
    for (i, label) in enumerate(y):
        if label:
            index_q.append(i)
        else:
            index_uq.append(i)
    X_q = X[index_q, :]
    y_q = y[index_q]
    X_uq = X[index_uq, :]
    y_uq = y[index_uq]
    n_q = y_q.size
    n_uq = y_uq.size
    if (n_q * 1.) / (n_q + n_uq) > q_ratio:
        n_q = int(n_uq * q_ratio / (1 - q_ratio))
        X_q, y_q = shuffle(X_q, y_q)
        X_q, y_q = X_q[:n_q], y_q[:n_q]
    else:
        n_uq = int(n_q * (1 - q_ratio) / q_ratio)
        X_uq, y_uq = shuffle(X_uq, y_uq)
        X_uq, y_uq = X_uq[:n_uq], y_uq[:n_uq]
    X, y = np.concatenate((X_q, X_uq), axis=0), np.concatenate((y_q, y_uq))
    X, y = shuffle(X, y)
    return X, y


def transform_except_last_dim(data, scaler):
    return np.concatenate((scaler.transform(data[:, :-1]), data[:, -1:]), axis=1)
