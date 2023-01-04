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

    algorithm_labels = {}
    algorithm_colors = {}
    algorithm_markers = {}

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

    fig, axs = plt.subplots(2, len(Z))
    fig.set_size_inches(fig_width, fig_height*2)
    for z,Z_indices in enumerate(Z):
        Z_str = "_".join([str(index) for index in Z_indices])  # for one set of groups
        # plotting num bins of wgm vs umb number of bins for different umb bin numbers
        algorithms = []
        results = {}

        algorithms.append("wgm")
        algorithms.append("umb")
        algorithms.append("pav")
        algorithms.append("wgc")


        metrics = ["n_bins", "num_selected"]  #"alpha","accuracy"

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

        for umb_num_bin in umb_num_bins:
            for algorithm in algorithms:
                for metric in metrics:
                    assert len(results[umb_num_bin][algorithm][metric]["values"])==n_runs
                    results[umb_num_bin][algorithm][metric]["mean"] = np.mean(
                        results[umb_num_bin][algorithm][metric]["values"])
                    results[umb_num_bin][algorithm][metric]["std"] = np.std(
                        results[umb_num_bin][algorithm][metric]["values"],
                        ddof=1)
                # assert (np.array(results[umb_num_bins][algorithm][metric]["values"]) >= 0).all()

        for row,metric in enumerate(metrics):
            handles = []
            for algorithm in algorithms:
                if metric=="n_bins" and algorithm=="umb":
                    continue

                if metric=="num_selected" and algorithm=="wgc":
                    print("here")
                    continue

                mean_algorithm = np.array([results[umb_num_bin][algorithm][metric]["mean"] for umb_num_bin
                                           in umb_num_bins])
                std_algorithm = np.array([results[umb_num_bin][algorithm][metric]["std"] for umb_num_bin
                                          in umb_num_bins])

                line = axs[row][z].plot(umb_num_bins, mean_algorithm,
                                        linewidth=line_width,
                                        label=algorithm_labels["{}_{}".format(algorithm, str(umb_num_bins[0]))],
                                        color=algorithm_colors["{}_{}".format(algorithm, str(umb_num_bins[0]))],
                                        marker=algorithm_markers["{}_{}".format(algorithm, str(umb_num_bins[
                                                                                                   0]))])  # , color=group_colors[i], marker=group_markers[i])
                handles.append(line[0])

                axs[row][z].fill_between(umb_num_bins, mean_algorithm - std_algorithm,
                                         mean_algorithm + std_algorithm, alpha=transparency,
                                         color=algorithm_colors[
                                             "{}_{}".format(algorithm, str(umb_num_bins[0]))])

                axs[row][z].set_xticks(umb_num_bins)
                axs[row][z].set_xlabel(xlabels["n_bins"])
                if row==0:
                    title = axs[row][z].set_title(Z_labels[Z_indices[0]]["feature"],y=1)
                # title.set_position([0.5,0.8])
                # axs[row][z].set_yticks([])
                axs[row][0].set_ylabel(metric_labels[metric])

                # axs[2][z].set_ylim((5, 15))
                # axs[3][z].set_ylim((6, 7))
    fig.legend(handles=handles,loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4)

    # axs[0].legend( loc='center right', bbox_to_anchor=(-0.12, 0.5), ncol=1)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    fig.savefig("./plots/exp_bins.pdf", format="pdf")