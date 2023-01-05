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


    the_umb_num_bin = 30

    algorithm_labels["umb_" + str(the_umb_num_bin)] = "UMB"
    algorithm_labels["wgm_" + str(the_umb_num_bin)] = "WGM"
    algorithm_labels["wgc_" + str(the_umb_num_bin)] = "WGC"
    algorithm_labels["pav_" + str(the_umb_num_bin)] = "PAV"
    algorithm_colors["umb_" + str(the_umb_num_bin)] = "tab:green"
    algorithm_colors["wgm_" + str(the_umb_num_bin)] = "tab:blue"
    algorithm_colors["wgc_" + str(the_umb_num_bin)] = "tab:red"
    algorithm_colors["pav_" + str(the_umb_num_bin)] = "tab:orange"
    algorithm_markers["umb_" + str(the_umb_num_bin)] = 8
    algorithm_markers["wgm_" + str(the_umb_num_bin)] = 10
    algorithm_markers["wgc_" + str(the_umb_num_bin)] = 9
    algorithm_markers["pav_" + str(the_umb_num_bin)] = 11


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

        metrics = ["prob_true", "prob_pred"]

        the_n_cal = n_cals[0]
        the_run = 0

        for algorithm in algorithms:
            results[algorithm] = {}
            for metric in metrics:
                results[algorithm][metric] = {}
                results[algorithm][metric]["values"] = []

        for algorithm in algorithms:
            exp_identity_string = "_".join(
                [Z_str, str(n_train), str(noise_ratio), str(the_n_cal), lbd, str(the_run)])
            result_path = os.path.join(exp_dir,
                                       exp_identity_string + "_{}_{}_result.pkl".format(algorithm,
                                                                                        the_umb_num_bin))
            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            for metric in metrics:

                results[algorithm][metric]["values"].append(result[metric])

        for algorithm in algorithms:
            for metric in metrics:
                assert len(results[algorithm][metric]["values"]) == n_runs
                results[algorithm][metric]["mean"] = np.mean(
                    results[algorithm][metric]["values"],axis=0)
                results[algorithm][metric]["std"] = np.std(
                    results[algorithm][metric]["values"],axis=0)


        for algorithm in algorithms:
            # print(algorithm,results[algorithm]["prob_true"]["values"])
            assert(results[algorithm]["prob_true"]["mean"].shape[0]==the_umb_num_bin)
            line = axs[z].plot(np.array(results[algorithm]["prob_true"]["mean"]),
                        np.array(results[algorithm]["prob_pred"]["mean"]), linewidth=line_width,
                        label=algorithm_labels["{}_{}".format(algorithm, str(the_umb_num_bin))],
                        color=algorithm_colors["{}_{}".format(algorithm, str(the_umb_num_bin))],
                        marker=algorithm_markers["{}_{}".format(algorithm, str(the_umb_num_bin))]
                        )
            handles.append(line[0])

            axs[z].fill_between(results[algorithm]["prob_true"]["mean"], results[algorithm]["prob_pred"]["mean"] - results[algorithm]["prob_pred"]["std"],
                                          results[algorithm]["prob_pred"]["mean"] + results[algorithm]["prob_pred"]["std"], alpha=transparency,
                                          color=algorithm_colors[
                                              "{}_{}".format(algorithm, str(the_umb_num_bin))])

        axs[z].set_xlabel(xlabels["prob_true"])
        # axs[z].set_xticks([round(float(label), 2) for label in results["umb"]["prob_true"]["mean"]])
        # axs[z].set_xticklabels([str(round(float(label), 2)) for label in results["umb"]["prob_true"]["mean"]])

        fig.legend(handles=handles,loc='upper center', bbox_to_anchor=(0.52, 1.03), ncol=4)
    plt.figtext(x=0.21, y=0.82, s=Z_labels[Z[0][0]]["feature"], fontsize=font_size)
    plt.figtext(x=0.73, y=0.82, s=Z_labels[Z[1][0]]["feature"], fontsize=font_size)
    axs[0].set_ylabel(metric_labels["prob_pred"])

    plt.tight_layout(rect=[0, 0, 1, 0.82])
    fig.savefig("./plots/exp_cal_curve_{}.pdf".format(the_umb_num_bin), format="pdf")

