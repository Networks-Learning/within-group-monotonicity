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
        algorithm_labels["umb_" + str(umb_num_bin)] = r"$f$"
        algorithm_labels["wgm_" + str(umb_num_bin)] = r"$f_{\mathcal{B}^*}$"
        algorithm_labels["wgc_" + str(umb_num_bin)] = r"$f_{\mathcal{B}^{*}_{\epsilon-cal}}$"
        algorithm_labels["pav_" + str(umb_num_bin)] = r"$f_{\mathcal{B}_{pav}}$"
        algorithm_colors["umb_" + str(umb_num_bin)] = "tab:green"
        algorithm_colors["wgm_" + str(umb_num_bin)] = "tab:blue"
        algorithm_colors["wgc_" + str(umb_num_bin)] = "tab:red"
        algorithm_colors["pav_" + str(umb_num_bin)] = "tab:orange"
        algorithm_markers["umb_" + str(umb_num_bin)] = 8
        algorithm_markers["wgm_" + str(umb_num_bin)] = 10
        algorithm_markers["wgc_" + str(umb_num_bin)] = 9
        algorithm_markers["pav_" + str(umb_num_bin)] = 11

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(fig_width,fig_height+1)


    # plotting num bins of wgm vs umb number of bins for different umb bin numbers
    algorithm = "umb"
    results = {}
    # algorithms.append("wgm")
    # algorithms.append("pav")
    # algorithms.append("wgc")


    metrics = ["discriminated_against","group_num_in_bin","pool_discriminated"]#,"log_loss","accuracy"]  #"alpha","accuracy"

    the_n_cal = n_cals[0]

    for umb_num_bin in umb_num_bins:
        results[umb_num_bin] = {}
        for z,Z_indices in enumerate(Z):
            results[umb_num_bin][z] = {}
            for metric in metrics:
                results[umb_num_bin][z][metric] = {}
                results[umb_num_bin][z][metric]["values"] = []

    for umb_num_bin in umb_num_bins:
        for run in runs:
            for z,Z_indices in enumerate(Z):
                Z_str = "_".join([str(index) for index in Z_indices])  # for one set of groups
                exp_identity_string = "_".join(
                    [Z_str, str(n_train), str(noise_ratio), str(the_n_cal), lbd, str(run)])
                result_path = os.path.join(exp_dir,
                                           exp_identity_string + "_{}_{}_result.pkl".format(algorithm,
                                                                                            umb_num_bin))
                collect_results_quantitative_exp(result_path, umb_num_bin, z, results, metrics)



    for umb_num_bin in umb_num_bins:
        for run in runs:
            for z,Z_indices in enumerate(Z):
                results[umb_num_bin][z]["group_num_in_bin"]["values"][run] = np.sum(np.where(results[umb_num_bin][z]["discriminated_against"]["values"][run],\
                                                                                               results[umb_num_bin][z]["group_num_in_bin"]["values"][run],\
                                                                                                np.zeros(results[umb_num_bin][z]["group_num_in_bin"]["values"][run].shape)))/the_n_cal


    for umb_num_bin in umb_num_bins:
        for z,Z_indices in enumerate(Z):
            for metric in ["group_num_in_bin","pool_discriminated"]:
                assert len(results[umb_num_bin][z][metric]["values"])==n_runs
                results[umb_num_bin][z][metric]["mean"] = np.mean(
                    results[umb_num_bin][z][metric]["values"])
                results[umb_num_bin][z][metric]["std"] = np.std(
                    results[umb_num_bin][z][metric]["values"],ddof=1)
            # assert (np.array(results[umb_num_bins][algorithm][metric]["values"]) >= 0).all()
    # fig_legend = plt.figure(figsize=(fig_width,0.8))
    for idx,metric in enumerate(["group_num_in_bin","pool_discriminated"]):
        handles = []
        for z,Z_indices in enumerate(Z):
            mean_algorithm = np.array([results[umb_num_bin][z][metric]["mean"] for umb_num_bin
                                       in umb_num_bins])
            std_algorithm = np.array([results[umb_num_bin][z][metric]["std"]/np.sqrt(n_runs) for umb_num_bin
                                      in umb_num_bins])

            line = axs[idx].plot(umb_num_bins, mean_algorithm,
                                    linewidth=line_width,
                                    label=Z_labels[Z_indices[0]]["feature"],
                                    color=Z_labels[Z_indices[0]]["color"],
                                    marker=Z_labels[Z_indices[0]]["marker"])
            handles.append(line[0])
            # if metric=="n_bins":
            axs[idx].fill_between(umb_num_bins, mean_algorithm - std_algorithm,
                                     mean_algorithm + std_algorithm, alpha=transparency,
                                     color=Z_labels[Z_indices[0]]["color"])

            # axs[z * 2 + idx].errorbar(umb_num_bins, mean_algorithm,
            #                        std_algorithm,
            #                        color=algorithm_colors[
            #                            "{}_{}".format(algorithm, str(umb_num_bins[0]))])

            axs[idx].set_xticks(umb_num_bins)

            # title = axs[0][z*2].set_title(Z_labels[Z_indices[0]]["feature"],y=1,x=1)
            # title.set_position([0.5,0.8])
            # axs[row][z].set_yticks([])
            axs[idx].set_ylabel(metric_labels[metric])
            axs[idx].set_xlabel(xlabels["n_bins"])

    # fig_legend.legend(handles=handles,loc='center', ncol=4)
    # fig_legend.savefig('./plots/legend.pdf')
    fig.legend(handles=handles,loc='lower center', bbox_to_anchor=(0.5, 0.85), ncol=2)

    # plt.figtext(x=0.21, y=0.82, s=Z_labels[Z[0][0]]["feature"], fontsize=font_size)
    # plt.figtext(x=0.73, y=0.82, s=Z_labels[Z[1][0]]["feature"], fontsize=font_size)

    # axs[0].legend( loc='center right', bbox_to_anchor=(-0.12, 0.5), ncol=1)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig("./plots/exp_discrimination.pdf", format="pdf")