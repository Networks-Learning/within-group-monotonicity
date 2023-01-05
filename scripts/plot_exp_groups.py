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

    fig, axs = plt.subplots(1, len(Z))
    fig.set_size_inches(fig_width, fig_height*1)
    for z, Z_indices in enumerate(Z):
        Z_str = "_".join([str(index) for index in Z_indices])  # for one set of groups
        num_groups = Z_labels[Z_indices[0]]["num_groups"]
        algorithms = []
        results = {}
        handles = []
        the_n_cal = n_cals[0]  # for one calibration set
        the_umb_num_bin = 15
        algorithms.append("umb_" + str(the_umb_num_bin))
        algorithms.append("wgm_" + str(the_umb_num_bin))
        algorithms.append("pav_" + str(the_umb_num_bin))
        algorithms.append("wgc_" + str(the_umb_num_bin))

        metrics = ["group_accuracy"]

        for grp in range(num_groups):
            results[grp] = {}
            for algorithm in algorithms:
                results[grp][algorithm] = {}
                for metric in metrics:
                    results[grp][algorithm][metric] = {}
                    results[grp][algorithm][metric]["values"] = []

        for grp in range(num_groups):
            for run in runs:
                exp_identity_string = "_".join(
                    [Z_str, str(n_train), str(noise_ratio), str(the_n_cal), lbd, str(run)])
                for algorithm in algorithms:
                    result_path = os.path.join(exp_dir,
                                               exp_identity_string + "_{}_result.pkl".format(algorithm))
                    collect_results_normal_exp(result_path, grp, algorithm, results, metrics)

        for grp in range(num_groups):
            for algorithm in algorithms:
                for metric in metrics:
                    assert len(results[grp][algorithm][metric]["values"]) == n_runs
                    results[grp][algorithm][metric]["mean"] = np.mean(results[grp][algorithm][metric]["values"],
                                                                      )
                    results[grp][algorithm][metric]["std"] = np.std(results[grp][algorithm][metric]["values"],
                                                                    ddof=1)
                    assert (np.array(results[grp][algorithm][metric]["mean"]) >= 0).all()

        for metric in metrics:
            for alg_idx, algorithm in enumerate(algorithms):
                mean_algorithm = np.array([results[grp][algorithm][metric]["mean"] for grp in range(num_groups)])
                std_algorithm = np.array([results[grp][algorithm][metric]["std"] for grp in range(num_groups)])

                bars = axs[z].bar(np.arange(num_groups) - ((alg_idx - 1) * 0.2), mean_algorithm, align='edge',
                                       width=0.1, label=algorithm_labels[algorithm],
                                       color=algorithm_colors[algorithm])

                if z == 0:
                    axs[z].legend(loc='center right', bbox_to_anchor=(-0.12, 0.5), ncol=1)

                # axs[row][z].errorbar(np.arange(num_groups) - ((alg_idx - 1) * 0.2),mean_algorithm,std_algorithm,color=algorithm_colors[algorithm]\
                #                       ,linewidth=line_width,capthick=capthick)
                axs[0].set_ylabel(metric_labels[metric])

                axs[z].set_xticks(range(num_groups))
                axs[z].set_xticklabels([Z_labels[Z_indices[0]][i] for i in range(num_groups)])
                # axs[alg][z].set_yticks([])
                # axs[alg][z].set_ylim((0,1))

            # axs[alg][0].yaxis.set_major_locator(ticker.MultipleLocator(0.25))

    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4)

    # axs[0].legend( loc='center right', bbox_to_anchor=(-0.12, 0.5), ncol=1)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    fig.savefig("./plots/exp_groups.pdf", format="pdf")