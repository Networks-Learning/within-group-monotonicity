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
    from params_exp_discrimination import *
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




    # plotting num bins of wgm vs umb number of bins for different umb bin numbers
    algorithm = "umb"
    results = {}


    metrics = ["discriminated_against","group_num_in_bin","bin_values","group_num_positives_in_bin"]#,"log_loss","accuracy"]  #"alpha","accuracy"

    the_n_cal = n_cals[0]
    umb_num_bin = umb_num_bins[0]

    for z, Z_indices in enumerate(Z):
        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(fig_width / 2, fig_height + 0.5)

        num_groups = Z_labels[Z_indices[0]]["num_groups"]
        for group in range(num_groups):
            results[group] = {}
            for metric in metrics:
                results[group][metric] = {}
                results[group][metric]["values"] = []

        for run in runs:
            for group in range(num_groups):
                Z_str = "_".join([str(index) for index in Z_indices])  # for one set of groups
                exp_identity_string = "_".join(
                    [Z_str, str(n_train), str(the_n_cal), lbd, str(run)])
                result_path = os.path.join(exp_dir,
                                           exp_identity_string + "_{}_{}_result.pkl".format(algorithm,
                                                                                            umb_num_bin))
                collect_results_group_exp(result_path, group, results, metrics)



        for run in runs:
            for group in range(num_groups):
                results[group]["group_num_positives_in_bin"]["values"][run] = np.sum(results[group]["group_num_in_bin"]["values"][run])/the_n_cal
                results[group]["bin_values"]["values"][run] = np.sum(results[group]["bin_values"]["values"][run]*results[group]["group_num_in_bin"]["values"][run])/np.sum(results[group]["group_num_in_bin"]["values"][run])
                results[group]["group_num_in_bin"]["values"][run] = np.sum(np.where(results[group]["discriminated_against"]["values"][run],\
                                                                                               results[group]["group_num_in_bin"]["values"][run],\
                                                                                                np.zeros(results[group]["group_num_in_bin"]["values"][run].shape)))/np.sum(results[group]["group_num_in_bin"]["values"][run])



        for group in range(num_groups):
            for metric in ["group_num_in_bin","bin_values","group_num_positives_in_bin"]:
                assert len(results[group][metric]["values"])==n_runs
                results[group][metric]["mean"] = np.mean(
                    results[group][metric]["values"])
                results[group][metric]["std"] = np.std(
                    results[group][metric]["values"],ddof=1)
            # assert (np.array(results[umb_num_bins][algorithm][metric]["values"]) >= 0).all()
        # fig_legend = plt.figure(figsize=(fig_width,0.8))
        handles = []
        mean_group_bin_value = np.array([results[group]["group_num_positives_in_bin"]["mean"] for group in range(num_groups)])
        args_sored = np.argsort(mean_group_bin_value)

        mean_algorithm = np.array([results[group]["group_num_in_bin"]["mean"] for group in range(num_groups)])[args_sored]
        std_algorithm = np.array([results[group]["group_num_in_bin"]["std"]/np.sqrt(n_runs) for group in range(num_groups)])[args_sored]
        std_group_bin_value = np.array([results[group]["group_num_positives_in_bin"]["std"]/np.sqrt(n_runs) for group in range(num_groups)])[args_sored]
        mean_group_bin_value = mean_group_bin_value[args_sored]

        for group in range(num_groups):
            line = axs.bar(group, mean_algorithm[group],width=0.2,
                                    linewidth=line_width,
                                    label=Z_labels[Z_indices[0]][args_sored[group]],
                                    color=group_colors[args_sored[group]],
                                    )#marker=Z_labels[Z_indices[0]]["marker"]
            handles.append(line)
            # if metric=="n_bins":
            axs.errorbar(group, mean_algorithm[group], yerr=std_algorithm[group],
                    label=Z_labels[Z_indices[0]][args_sored[group]],
                    color='lightslategrey',
                    )  # marker=Z_labels[Z_indices[0]]["marker"]

        axs.set_xticks(range(num_groups))
        axs.set_xticklabels([str(round(float(label), 2)) for label in mean_group_bin_value])

        # title = axs[0][z*2].set_title(Z_labels[Z_indices[0]]["feature"],y=1,x=1)
        # title.set_position([0.5,0.8])
        # axs[row][z].set_yticks([])
        axs.set_ylabel(r"$p_{d|z}$",fontsize=34)
        axs.set_xlabel(xlabels["group_rho"])
        legend = axs.legend(handles=handles,fontsize=17,title = Z_labels[Z_indices[0]]["feature"])
        plt.setp(legend.get_title(), fontsize=18)
        # axs.set_ylim(0,0.35)
        axs.yaxis.set_major_locator(ticker.MultipleLocator(0.1))


        plt.tight_layout(rect=[0, 0, 1, 1])
        fig.savefig("./plots/exp_group_discrimination_{}.pdf".format(Z[z][0]), format="pdf")