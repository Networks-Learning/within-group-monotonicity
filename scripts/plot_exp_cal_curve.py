import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn import preprocessing as p
import numpy as np
import os
from plot_constants import *
plt.rcParams.update(params)
plt.rc('font', family='serif')

if __name__ == "__main__":
    from params_exp_bins import *
    from matplotlib.ticker import StrMethodFormatter

    if not os.path.exists('./plots'):
        os.mkdir('./plots')

    for umb_num_bin in umb_num_bins:
        algorithm_labels["umb_" + str(umb_num_bin)] = "UMB"
        algorithm_labels["wgm_" + str(umb_num_bin)] = "WGM"
        algorithm_labels["wgc_" + str(umb_num_bin)] = "WGC"
        algorithm_labels["pav_" + str(umb_num_bin)] = "PAV"
        algorithm_colors["umb_" + str(umb_num_bin)] = "tab:green"
        algorithm_colors["wgm_" + str(umb_num_bin)] = "tab:blue"
        algorithm_colors["wgc_" + str(umb_num_bin)] = "tab:red"
        algorithm_colors["pav_" + str(umb_num_bin)] = "tab:orange"
        algorithm_markers["umb_" + str(umb_num_bin)] = 8
        algorithm_markers["wgm_" + str(umb_num_bin)] = 10
        algorithm_markers["wgc_" + str(umb_num_bin)] = 9
        algorithm_markers["pav_" + str(umb_num_bin)] = 11

    the_n_cal = n_cals[0]  # for one calibration set



    for k_idx,k in enumerate(ks):
        for z, Z_indices in enumerate(Z):
            fig, axs = plt.subplots(1, 2)
            fig.set_size_inches(fig_width, fig_height)
            Z_str = "_".join([str(index) for index in Z_indices])  # for one set of groups
            num_groups = Z_labels[Z_indices[0]]["num_groups"]
            handles = []
            algorithms = []
            results = {}

            algorithms.append("wgm")
            algorithms.append("umb")
            algorithms.append("pav")
            algorithms.append("wgc")

            metrics = ["f1_score","accuracy"]

            the_n_cal = n_cals[0]

            for umb_num_bin in umb_num_bins:
                results[umb_num_bin] = {}
                for algorithm in algorithms:
                    results[umb_num_bin][algorithm] = {}
                    for metric in metrics:
                        results[umb_num_bin][algorithm][metric] = {}
                        results[umb_num_bin][algorithm][metric]["values"] = []

            for umb_num_bin in umb_num_bins:
                for run in runs:
                    for algorithm in algorithms:
                        exp_identity_string = "_".join(
                            [Z_str, str(n_train), str(the_n_cal), lbd, str(run)])
                        result_path = os.path.join(exp_dir,
                                                   exp_identity_string + "_{}_{}_result.pkl".format(algorithm,
                                                                                                    umb_num_bin))
                        # collect_results_quantitative_exp(result_path, umb_num_bin, algorithm, results, metrics)
                        with open(result_path, 'rb') as f:
                            result = pickle.load(f)
                            # print(algorithm,result.keys())
                        for metric in metrics:
                            results[umb_num_bin][algorithm][metric]["values"].append(result[metric][k_idx])

            for umb_num_bin in umb_num_bins:
                for algorithm in algorithms:
                    for metric in metrics:
                        assert len(results[umb_num_bin][algorithm][metric]["values"]) == n_runs
                        results[umb_num_bin][algorithm][metric]["mean"] = np.mean(
                            results[umb_num_bin][algorithm][metric]["values"],axis=0)
                        results[umb_num_bin][algorithm][metric]["std"] = np.std(
                            results[umb_num_bin][algorithm][metric]["values"],axis=0,ddof=1)


            for idx,metric in enumerate(metrics):
                handles = []
                for algorithm in algorithms:
                    if metric=="alpha" and algorithm!="wgc":
                        continue
                    # print(algorithm,results[algorithm]["prob_true"]["values"])
                    mean_pred = np.array([results[umb_num_bin][algorithm][metric]["mean"] for umb_num_bin
                                               in umb_num_bins])
                    denom = np.sqrt(n_runs)
                    # if metric!="alpha":
                    #     denom = np.sqrt(n_runs)
                    std_pred = np.array([results[umb_num_bin][algorithm][metric]["std"] /denom for umb_num_bin
                                              in umb_num_bins])

                    line = axs[idx].plot(umb_num_bins,
                                mean_pred, linewidth=line_width,
                                       label=algorithm_labels["{}_{}".format(algorithm, str(umb_num_bins[0]))],
                                       color=algorithm_colors["{}_{}".format(algorithm, str(umb_num_bins[0]))],
                                       marker=algorithm_markers["{}_{}".format(algorithm, str(umb_num_bins[
                                                                                                  0]))])
                    handles.append(line[0])

                    axs[idx].fill_between(umb_num_bins, mean_pred-std_pred,
                                                  mean_pred+std_pred, alpha=transparency,
                                                  color=algorithm_colors[
                                                      "{}_{}".format(algorithm, str(umb_num_bins[0]))])
                    axs[idx].set_xticks(umb_num_bins)
                    axs[idx].set_ylabel(metric_labels[metric])
                    axs[idx].set_xlabel(xlabels["n_bins"])

            plt.tight_layout(rect=[0, 0, 1, 1])
            fig.savefig("./plots/exp_cal_curve_{}_k_{}.pdf".format(Z_indices[0],str(k)), format="pdf")