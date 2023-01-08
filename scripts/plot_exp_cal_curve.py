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


    fig, axs = plt.subplots(1, len(Z))
    fig.set_size_inches(fig_width, fig_height)

    Z = [[6], [15]]
    for z, Z_indices in enumerate(Z):
        Z_str = "_".join([str(index) for index in Z_indices])  # for one set of groups
        num_groups = Z_labels[Z_indices[0]]["num_groups"]
        handles = []
        algorithms = []
        results = {}

        algorithms.append("wgc")
        algorithms.append("pav")
        algorithms.append("umb")
        algorithms.append("wgm")

        metrics = ["f1_score"]

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
                        [Z_str, str(n_train), str(noise_ratio), str(the_n_cal), lbd, str(run)])
                    result_path = os.path.join(exp_dir,
                                               exp_identity_string + "_{}_{}_result.pkl".format(algorithm,
                                                                                                umb_num_bin))
                    collect_results_quantitative_exp(result_path, umb_num_bin, algorithm, results, metrics)
                    # with open(result_path, 'rb') as f:
                    #     result = pickle.load(f)
                    # results[umb_num_bin][algorithm]["ECE"]["values"].append(result["ECE"])

        for umb_num_bin in umb_num_bins:
            for algorithm in algorithms:
                for metric in metrics:
                    assert len(results[umb_num_bin][algorithm][metric]["values"]) == n_runs
                    results[umb_num_bin][algorithm][metric]["mean"] = np.mean(
                        results[umb_num_bin][algorithm][metric]["values"],axis=0)
                    results[umb_num_bin][algorithm][metric]["std"] = np.std(
                        results[umb_num_bin][algorithm][metric]["values"],axis=0,ddof=1)


        for algorithm in algorithms:
            for metric in metrics:
                # print(algorithm,results[algorithm]["prob_true"]["values"])
                mean_pred = np.array([results[umb_num_bin][algorithm][metric]["mean"] for umb_num_bin
                                           in umb_num_bins])
                std_pred = np.array([results[umb_num_bin][algorithm][metric]["std"] /np.sqrt(n_runs) for umb_num_bin
                                          in umb_num_bins])

                line = axs[z].plot(umb_num_bins,
                            mean_pred, linewidth=line_width,
                                   label=algorithm_labels["{}_{}".format(algorithm, str(umb_num_bins[0]))],
                                   color=algorithm_colors["{}_{}".format(algorithm, str(umb_num_bins[0]))],
                                   marker=algorithm_markers["{}_{}".format(algorithm, str(umb_num_bins[
                                                                                              0]))])
                handles.append(line[0])

                axs[z].fill_between(umb_num_bins, mean_pred-std_pred,
                                              mean_pred+std_pred, alpha=transparency,
                                              color=algorithm_colors[
                                                  "{}_{}".format(algorithm, str(umb_num_bins[0]))])

                # axs[z].errorbar(umb_num_bins, mean_pred,
                #                     std_pred,capthick=capthick,
                #                 label=algorithm_labels["{}_{}".format(algorithm, str(umb_num_bins[0]))],
                #                 color=algorithm_colors["{}_{}".format(algorithm, str(umb_num_bins[0]))],
                #                 marker=algorithm_markers["{}_{}".format(algorithm, str(umb_num_bins[
                #                                                                            0]))])

        axs[z].set_xlabel(xlabels["n_bins"])
        # axs[z].set_xticks([round(float(label), 2) for label in results["umb"]["prob_true"]["mean"]])
        # axs[z].set_xticklabels([str(round(float(label), 2)) for label in results["umb"]["prob_true"]["mean"]])

        fig.legend(handles=handles,loc='upper center', bbox_to_anchor=(0.52, 1.03), ncol=4)
    plt.figtext(x=0.21, y=0.82, s=Z_labels[Z[0][0]]["feature"], fontsize=font_size)
    plt.figtext(x=0.73, y=0.82, s=Z_labels[Z[1][0]]["feature"], fontsize=font_size)
    axs[0].set_ylabel(metric_labels[metrics[0]])

    plt.tight_layout(rect=[0, 0, 1, 0.82])
    fig.savefig("./plots/exp_cal_curve.pdf", format="pdf")

