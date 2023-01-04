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
    from params_exp_violations import *
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


    the_run = 0
    the_n_cal = n_cals[0]  # for one calibration set
    the_umb_num_bin = 15

    algorithms = []
    algorithms.append("umb_" + str(the_umb_num_bin))
    algorithms.append("wgm_" + str(the_umb_num_bin))
    # algorithms.append("pav_" + str(the_umb_num_bin))
    # algorithms.append("wgc_" + str(the_umb_num_bin))

    fig, axs = plt.subplots(len(algorithms), len(Z))
    fig.set_size_inches(fig_width, fig_height * len(algorithms))

    Z = [[6], [15]]
    for z,Z_indices in enumerate(Z):
        Z_str = "_".join([str(index) for index in Z_indices])  # for one set of groups
        num_groups = Z_labels[Z_indices[0]]["num_groups"]

        # plot group bin values for one run, one calibration set, for each algorithms across bin values

        results = {}
        num_bins = {}

        metrics = ["group_bin_values", "bin_values", "bin_rho","discriminated_against"]

        handles = []
        for row,algorithm in enumerate(algorithms):
            exp_identity_string = "_".join([Z_str, str(n_train), str(noise_ratio), str(the_n_cal), lbd, str(the_run)])
            result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))

            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            num_bins = len(result["bin_values"])


            for bin in range(num_bins):
                results[bin] = {}
                # results[bin][algorithm] = {}
                for metric in metrics:
                    results[bin][metric] = {}
                    results[bin][metric]["values"] = []

            for bin in range(num_bins):
                for metric in metrics:
                    results[bin][metric]["values"] = result[metric][bin]
                # collect_results_normal_exp(result_path, bin, algorithm, results, metrics)

            # for bin in range(num_bins[algorithm]):
            #     for metric in metrics:
            #         results[bin][algorithm][metric]["mean"] = np.mean(results[bin][algorithm][metric]["values"],axis=0)
            #         results[bin][algorithm][metric]["std"] = np.std(results[bin][algorithm][metric]["values"],
            #                                                           ddof=1,axis=0)
            #         assert (np.array(results[bin][algorithm][metric]["values"]) >= 0).all()

            mean_algorithm = np.array([results[bin]["group_bin_values"]["values"] for bin
                                                    in range(num_bins)])

            bin_value_algorithm = np.array([results[bin]["bin_values"]["values"] for bin
                                        in range(num_bins)])

            disc_algorithm = np.array([results[bin]["discriminated_against"]["values"] for bin
                                            in range(num_bins)])
            rho_algorithm = np.array([results[bin]["bin_rho"]["values"] for bin
                                       in range(num_bins)])

            import matplotlib.colors as mcolors

            for i in range(num_groups):
                mean = mean_algorithm[:,i]
                # std = std_algorithm[:,i]
                # alpha = alpha_algorithm[:,i]
                disc = disc_algorithm[:,i]
                # rgba_colors = np.zeros(shape=(alpha.shape[0],4))
                # rgba_colors[:,:3] = mcolors.to_rgb(group_colors[i])
                # rgba_colors[:,3] = [1 if dis else 0.7 for dis in disc]

                legend_bars = axs[row][z].bar(np.arange(num_bins) - ((i - 1) * 0.2), mean, align='edge',
                                linewidth=disc, width=0.1, color=group_colors[i],
                                label=Z_labels[Z_indices[0]][i])
                handles.append(legend_bars)

                bars = axs[row][z].bar(np.arange(num_bins)-((i-1)*0.2), mean,align='edge',
                            linewidth=disc,width=0.1,edgecolor='black',color=group_colors[i])

                if row==0:
                    legend = axs[row][z].legend(handles = handles, loc='upper center', bbox_to_anchor=(0.5, 1.2),ncol=2, title = Z_labels[Z_indices[0]]["feature"])
                    plt.setp(legend.get_title(), fontsize=params['legend.fontsize'])

                hatch = ['//' if dis else '' for dis in disc]
                for bar, h in zip(bars, hatch):
                    bar.set_hatch(h)

                axs[row][z].set_xticks(range(num_bins))
                axs[row][z].set_xticklabels([str(round(float(label), 2)) for label in bin_value_algorithm])

                axs[row][z].set_yticks([])
                # axs[alg][z].set_ylim((0,1))

            axs[row][0].yaxis.set_major_locator(ticker.MultipleLocator(0.25))
            if algorithm.startswith("umb"):
                axs[row][0].set_ylabel(r'$\Pr(Y=1|f(X),Z)$')
                axs[row][z].set_xlabel(r'$\Pr(Y=1|f(X))$')
            if algorithm.startswith("wgm"):
                axs[row][0].set_ylabel(r'$\Pr(Y=1|f_{\mathcal{B}^*}(X),Z)$')
                axs[row][z].set_xlabel(r'$\Pr(Y=1|f_{\mathcal{B}^*}(X))$')
            if algorithm.startswith("wgc"):
                axs[row][0].set_ylabel(r'$\Pr(Y=1|f_{\mathcal{B}_{cal}}(X),Z)$')
                axs[row][z].set_xlabel(r'$\Pr(Y=1|f_{\mathcal{B}_{cal}}(X))$')

            if algorithm.startswith("pav"):
                axs[row][0].set_ylabel(r'$\Pr(Y=1|f_{\mathcal{B}_{pav}}(X),Z)$')
                axs[row][z].set_xlabel(r'$\Pr(Y=1|f_{\mathcal{B}_{pav}}(X))$')

    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig("./plots/exp_violations_{}.pdf".format('_'.join(algorithms)), format="pdf")


    # plotting ROC
    Z = [[6], [15]]
    umb_num_bins = [15]
    fig, axs = plt.subplots(1, len(Z)*len(umb_num_bins))
    fig.set_size_inches(fig_width, fig_height)

    for z, Z_indices in enumerate(Z):
        Z_str = "_".join([str(index) for index in Z_indices])  # for one set of groups
        num_groups = Z_labels[Z_indices[0]]["num_groups"]

        algorithms = []
        results = {}

        algorithms.append("wgc")
        algorithms.append("pav")
        algorithms.append("umb")
        algorithms.append("wgm")

        metrics = ["fpr","tpr"]

        the_n_cal = n_cals[0]

        for umb_num_bin in umb_num_bins:
            results[umb_num_bin] = {}
            for algorithm in algorithms:
                results[umb_num_bin][algorithm] = {}
                for metric in metrics:
                    results[umb_num_bin][algorithm][metric] = {}
                    results[umb_num_bin][algorithm][metric]["values"] = []

        for umb_num_bin in umb_num_bins:
            for algorithm in algorithms:
                exp_identity_string = "_".join(
                    [Z_str, str(n_train), str(noise_ratio), str(the_n_cal), lbd, str(the_run)])
                result_path = os.path.join(exp_dir,
                                           exp_identity_string + "_{}_{}_result.pkl".format(algorithm,
                                                                                        umb_num_bin))
                with open(result_path, 'rb') as f:
                    result = pickle.load(f)
                for metric in metrics:
                    results[umb_num_bin][algorithm][metric]["values"] = result[metric]

        for idx,umb_num_bin in enumerate(umb_num_bins):
            for algorithm in algorithms:
                axs[z].plot(results[umb_num_bin][algorithm]["fpr"]["values"],
                                 results[umb_num_bin][algorithm]["tpr"]["values"], linewidth=line_width,
                                 label=algorithm_labels["{}_{}".format(algorithm, str(umb_num_bins[0]))],
                                 color=algorithm_colors["{}_{}".format(algorithm, str(umb_num_bins[0]))],
                                 marker=algorithm_markers["{}_{}".format(algorithm, str(umb_num_bins[
                                                                                            0]))]
                            )
                axs[z].set_xlabel(xlabels["fpr"])


    axs[0].set_ylabel(metric_labels["tpr"])
    axs[0].legend(loc='center right', bbox_to_anchor=(-0.12, 0.5), ncol=1)
    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig("./plots/exp_violations_ROC.pdf", format="pdf")


    # plot ROC per group
    # num_groups = 4
    plot_group = False
    if plot_group:
        fig, axs = plt.subplots(4, len(Z))
        fig.set_size_inches(fig_width, fig_height*4)

        for z, Z_indices in enumerate(Z):
            Z_str = "_".join([str(index) for index in Z_indices])  # for one set of groups
            num_groups = Z_labels[Z_indices[0]]["num_groups"]

            # plotting ROC
            row = 0
            algorithms = []
            results = {}

            algorithms.append("umb")
            algorithms.append("wgm")
            algorithms.append("pav")
            algorithms.append("wgc")

            metrics = ["fpr","tpr","group_fpr", "group_tpr"]

            the_n_cal = n_cals[0]

            # for umb_num_bin in umb_num_bins:
            #     results[umb_num_bin] = {}
            for algorithm in algorithms:
                results[algorithm] = {}
                for metric in metrics:
                    results[algorithm][metric] = {}
                    results[algorithm][metric]["values"] = []

            # for umb_num_bin in umb_num_bins:
            for algorithm in algorithms:
                exp_identity_string = "_".join(
                    [Z_str, str(n_train), str(noise_ratio), str(the_n_cal), lbd, str(the_run)])
                result_path = os.path.join(exp_dir,
                                           exp_identity_string + "_{}_{}_result.pkl".format(algorithm,
                                                                                            the_umb_num_bin))
                with open(result_path, 'rb') as f:
                    result = pickle.load(f)
                for metric in metrics:
                    results[algorithm][metric]["values"] = result[metric]

            for row,algorithm in enumerate(algorithms):
                axs[row][z].plot(results[algorithm]["fpr"]["values"],
                            results[algorithm]["tpr"]["values"], linewidth=line_width,
                            label=algorithm_labels["{}_{}".format(algorithm, str(umb_num_bins[0]))],
                            color=algorithm_colors["{}_{}".format(algorithm, str(umb_num_bins[0]))],
                            marker=algorithm_markers["{}_{}".format(algorithm, str(umb_num_bins[
                                                                                       0]))])
                for grp in range(num_groups):
                    axs[row][z].plot(results[algorithm]["group_fpr"]["values"][grp,:],
                                     results[algorithm]["group_tpr"]["values"][grp,:],
                                     linewidth=line_width,
                                     color=group_colors[grp],
                                     label=Z_labels[Z_indices[0]][grp],
                                     marker=algorithm_markers["{}_{}".format(algorithm, str(umb_num_bins[
                                                                                                0]))])
                    axs[row][z].set_xlabel(xlabels["fpr"])
                axs[row][0].set_ylabel(metric_labels["tpr"])
                axs[row][0].legend(loc='center right', bbox_to_anchor=(-0.12, 0.5), ncol=1)

        plt.tight_layout(rect=[0, 0, 1, 1])
        fig.savefig("./plots/exp_violations_group_ROC.pdf", format="pdf")


        # for algorithm in algorithms:
        #     axs[row][z].plot(results[algorithm]["fpr"]["values"],
        #                      results[algorithm]["tpr"]["values"], linewidth=line_width,
        #                      label=algorithm_labels["{}_{}".format(algorithm, str(umb_num_bins[0]))],
        #                      color=algorithm_colors["{}_{}".format(algorithm, str(umb_num_bins[0]))],
        #                      marker=algorithm_markers["{}_{}".format(algorithm, str(umb_num_bins[
        #                                                                                 0]))])
        #     for grp in range(num_groups):
        #         axs[row][z].plot(results[algorithm]["group_fpr"]["values"][grp,:],
        #                          results[algorithm]["group_tpr"]["values"][grp,:],
        #                          linewidth=line_width,
        #                          label=algorithm_labels["{}_{}".format(algorithm, str(umb_num_bins[0]))],
        #                          color=algorithm_colors["{}_{}".format(algorithm, str(umb_num_bins[0]))],
        #                          marker=algorithm_markers["{}_{}".format(algorithm, str(umb_num_bins[
        #                                                                                     0]))])
        #         axs[row][z].set_xlabel(xlabels["fpr"])
        #     axs[row][0].set_ylabel(metric_labels["tpr"])
        #     axs[row][0].legend(loc='center right', bbox_to_anchor=(-0.12, 0.5), ncol=1)
        #     row += 1

        # plot group accuracy, one calibration set, multiple runs, across algorithms

        # plt.tight_layout(rect=[0, 0, 1, 1])
        # if not os.path.exists('./plots'):
        #     os.mkdir('./plots')
        # fig.savefig("./plots/exp_violations.pdf", format="pdf")
