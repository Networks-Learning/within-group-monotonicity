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
    from params_exp_cal import *
    from matplotlib.ticker import StrMethodFormatter
    fig, axs = plt.subplots(9, len(Z))

    fig.set_size_inches(len(Z)*15, 30)

    Z_labels = {
        # 2: {0:"Married",1:"Widowed",2:"Divorced",3:"Separated",4:"Never married"},
        2: {0:"Married or Separated", 1: "Never married", "feature":"Marital status", "num_groups":2},
        4: {0: "With a disability", 1: "Without a disability", "feature": "Disability record","num_groups":2},
        6: {0:"Born in the US", 1:"Born in Unincorporated US", 2:"Born abroad", 3:"Not a US citizen", "feature":"Citizenship status", "num_groups":4},
        10: {0:"Native", 1:"Foreign born", "feature":"Nativity","num_groups":2},
        14: {0: "Male", 1:"Female", "feature":"Gender","num_groups":2},
        15: {0: "White", 1:"Black or African American", 2:"American Indian or Alaska", 3:"Asian, Native Hawaiian or other", "feature":"Race code","num_groups":4},
        1: {0:"No diploma", 1:"diploma", 2:"Associate or Bachelor degree", 3: "Masters or Doctorate degree", "feature":"Educational attainment","num_groups":4},
        0: {0:"0-25", 1:"26-50", 2:"51-75", 3:"76-99", "feature":"Age","num_groups":4}
    }
    algorithm_labels = {}
    algorithm_colors = {}
    algorithm_markers = {}
    metric_labels = {"group_accuracy": r'$\Pr(Y=S|Z)$', "n_bins":r'$|\mathcal{B}|$',"accuracy":r'$\Pr(Y=S)$', "num_selected": r'Shortlist Size',\
                     "alpha":r'$\alpha$'}
    for umb_num_bin in umb_num_bins:
        algorithm_labels["umb_" + str(umb_num_bin)] = "UMB {} Bins".format(umb_num_bin)
        algorithm_labels["wgm_" + str(umb_num_bin)] = "WGM"
        algorithm_labels["wgc_" + str(umb_num_bin)] = "WGC"
        algorithm_labels["pav_" + str(umb_num_bin)] = "PAV"
        algorithm_colors["umb_" + str(umb_num_bin)] = "tab:red"
        algorithm_colors["wgm_" + str(umb_num_bin)] = "tab:green"
        algorithm_colors["wgc_" + str(umb_num_bin)] = "tab:purple"
        algorithm_colors["pav_" + str(umb_num_bin)] = "tab:blue"
        algorithm_markers["umb_" + str(umb_num_bin)] = 10
        algorithm_markers["wgm_" + str(umb_num_bin)] = 11
        algorithm_markers["wgc_" + str(umb_num_bin)] = 9
        algorithm_markers["pav_" + str(umb_num_bin)] = 8


    for z,Z_indices in enumerate(Z):
        Z_str = "_".join([str(index) for index in Z_indices])  # for one set of groups
        num_groups = Z_labels[Z_indices[0]]["num_groups"]


        # plot group bin values for one run, one calibration set, for each algorithms across bin values
        algorithms = []
        results = {}
        num_bins = {}
        the_n_cal = n_cals[0]  # for one calibration set
        the_run = 0
        the_umb_num_bin = umb_num_bins[1]
        algorithms.append("umb_" + str(the_umb_num_bin))
        algorithms.append("wgm_" + str(the_umb_num_bin))
        algorithms.append("wgc_" + str(the_umb_num_bin))
        algorithms.append("pav_" + str(the_umb_num_bin))

        row = 0
        handles = []
        for algorithm in algorithms:
            exp_identity_string = "_".join([Z_str, str(n_train), str(noise_ratio), str(the_n_cal), lbd, str(the_run)])
            result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))

            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            num_bins[algorithm] = len(result["bin_values"])

            metrics = ["group_bin_values", "group_rho", "bin_values", "discriminated_against"]

            for bin in range(num_bins[algorithm]):
                results[bin] = {}
                results[bin][algorithm] = {}
                for metric in metrics:
                    results[bin][algorithm][metric] = {}
                    results[bin][algorithm][metric]["values"] = []

            for bin in range(num_bins[algorithm]):
                collect_results_normal_exp(result_path, bin, algorithm, results, metrics)

            for bin in range(num_bins[algorithm]):
                for metric in metrics:
                    results[bin][algorithm][metric]["mean"] = np.mean(results[bin][algorithm][metric]["values"],axis=0)
                    results[bin][algorithm][metric]["std"] = np.std(results[bin][algorithm][metric]["values"],
                                                                      ddof=1,axis=0)
                    assert (np.array(results[bin][algorithm][metric]["values"]) >= 0).all()

            mean_algorithm = np.array([results[bin][algorithm]["group_bin_values"]["mean"] for bin
                                                    in range(num_bins[algorithm])])
            std_algorithm = np.array([results[bin][algorithm]["group_bin_values"]["std"] for bin
                                                   in range(num_bins[algorithm])])
            alpha_algorithm = np.array([results[bin][algorithm]["group_rho"]["mean"] for bin
                                        in range(num_bins[algorithm])])
            bin_value_algorithm = np.array([results[bin][algorithm]["bin_values"]["mean"] for bin
                                        in range(num_bins[algorithm])])
            disc_algorithm = np.array([results[bin][algorithm]["discriminated_against"]["mean"] for bin
                                            in range(num_bins[algorithm])])

            import matplotlib.colors as mcolors

            for i in range(num_groups):
                mean = mean_algorithm[:,i]
                std = std_algorithm[:,i]
                alpha = alpha_algorithm[:,i]
                disc = disc_algorithm[:,i]
                rgba_colors = np.zeros(shape=(alpha.shape[0],4))
                rgba_colors[:,:3] = mcolors.to_rgb(group_colors[i])
                rgba_colors[:,3] = [1 if dis else 0.7 for dis in disc]

                legend_bars = axs[row][z].bar(np.arange(num_bins[algorithm]) - ((i - 1) * 0.2), mean, align='edge',
                                linewidth=disc, width=0.1, color=group_colors[i],
                                label=Z_labels[Z_indices[0]][i])
                handles.append(legend_bars)

                bars = axs[row][z].bar(np.arange(num_bins[algorithm])-((i-1)*0.2), mean,align='edge',
                            linewidth=disc,width=0.1,edgecolor='black',color=group_colors[i])

                if row==0:
                    legend = axs[row][z].legend(handles = handles, loc='upper center', bbox_to_anchor=(0.5, 1.5),ncol=2, title = Z_labels[Z_indices[0]]["feature"])
                    plt.setp(legend.get_title(), fontsize=params['legend.fontsize'])

                hatch = ['//' if dis else '' for dis in disc]
                for bar, h in zip(bars, hatch):
                    bar.set_hatch(h)

                axs[row][z].set_xticks(range(num_bins[algorithm]))
                axs[row][z].set_xticklabels([str(round(float(label), 2)) for label in bin_value_algorithm])

                # axs[alg][z].set_yticks([])
                # axs[alg][z].set_ylim((0,1))

            # axs[alg][0].yaxis.set_major_locator(ticker.MultipleLocator(0.25))
            if algorithm.startswith("umb"):
                axs[row][0].set_ylabel(r'$\Pr(Y=1|f_(X),Z)$')
                axs[row][z].set_xlabel(r'$\Pr(Y=1|f_(X))$')
            if algorithm.startswith("wgm"):
                axs[row][0].set_ylabel(r'$\Pr(Y=1|f_{{\mathcal{B}}^*}(X),Z)$')
                axs[row][z].set_xlabel(r'$\Pr(Y=1|f_{{\mathcal{B}}^*}(X))$')
            if algorithm.startswith("wgc"):
                axs[row][0].set_ylabel(r'$\Pr(Y=1|f_{{\mathcal{B}}_{cal}}(X),Z)$')
                axs[row][z].set_xlabel(r'$\Pr(Y=1|f_{{\mathcal{B}}_{cal}}(X))$')

            if algorithm.startswith("pav"):
                axs[row][0].set_ylabel(r'$\Pr(Y=1|f_{{\mathcal{B}}_{pav}}(X),Z)$')
                axs[row][z].set_xlabel(r'$\Pr(Y=1|f_{{\mathcal{B}}_{pav}}(X))$')


            row += 1



        # plotting num bins of wgm vs umb number of bins for different umb bin numbers
        algorithms = []
        results = {}

        algorithms.append("wgm")
        algorithms.append("umb")
        algorithms.append("wgc")
        algorithms.append("pav")

        metrics = ["n_bins", "accuracy", "num_selected", "alpha"]

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
                    results[umb_num_bin][algorithm][metric]["mean"] = np.mean(
                        results[umb_num_bin][algorithm][metric]["values"])
                    results[umb_num_bin][algorithm][metric]["std"] = np.std(
                        results[umb_num_bin][algorithm][metric]["values"],
                        ddof=1)
                # assert (np.array(results[umb_num_bins][algorithm][metric]["values"]) >= 0).all()

        for metric in metrics:
            handles = []
            for algorithm in algorithms:
                if metric=="n_bins" and algorithm=="umb":
                    continue
                if metric=="alpha" and algorithm!="wgc":
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
                # axs[2 + i][z].set_yticks([])

                # axs[2][z].set_ylim((5, 15))
                # axs[3][z].set_ylim((6, 7))
            axs[row][0].legend( loc='center right', bbox_to_anchor=(-0.12, 0.5), ncol=1)
            axs[row][0].set_ylabel(metric_labels[metric])
            axs[row][z].set_xlabel(r'n')

            # axs[3][0].set_ylabel(r'Shortlist Size')
            # axs[2][0].yaxis.set_major_locator(ticker.MultipleLocator(2))
            # axs[3][0].set_ylim(top=7)
            # axs[3][0].yaxis.set_major_locator(ticker.MultipleLocator(1))
            row += 1




        # plot group accuracy, one calibration set, multiple runs, across algorithms
        algorithms = []
        results = {}
        handles = []
        the_n_cal = n_cals[0]  # for one calibration set

        the_umb_num_bin = umb_num_bins[0]
        algorithms.append("umb_" + str(the_umb_num_bin))
        algorithms.append("wgm_" + str(the_umb_num_bin))
        algorithms.append("wgc_" + str(the_umb_num_bin))
        algorithms.append("pav_" + str(the_umb_num_bin))

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
                    result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))
                    collect_results_normal_exp(result_path, grp, algorithm, results, metrics)

        for grp in range(num_groups):
            for algorithm in algorithms:
                for metric in metrics:
                    results[grp][algorithm][metric]["mean"] = np.mean(results[grp][algorithm][metric]["values"],
                                                                      )
                    results[grp][algorithm][metric]["std"] = np.std(results[grp][algorithm][metric]["values"],
                                                                    ddof=1)
                    assert (np.array(results[grp][algorithm][metric]["mean"]) >= 0).all()

        for metric in metrics:
            for alg_idx,algorithm in enumerate(algorithms):
                    mean_algorithm = np.array([results[grp][algorithm][metric]["mean"] for grp in range(num_groups)])
                    std_algorithm = np.array([results[grp][algorithm][metric]["std"] for grp in range(num_groups)])

                    bars = axs[row][z].bar(np.arange(num_groups) - ((alg_idx - 1) * 0.2), mean_algorithm,align='edge',
                                            width=0.1,label=algorithm_labels[algorithm], color=algorithm_colors[algorithm])

                    if z==0:
                        axs[row][z].legend(loc='center right', bbox_to_anchor=(-0.12, 0.5), ncol=1)

                    # axs[row][z].errorbar(np.arange(num_groups) - ((alg_idx - 1) * 0.2),mean_algorithm,std_algorithm,color=algorithm_colors[algorithm]\
                    #                       ,linewidth=line_width,capthick=capthick)
                    axs[row][0].set_ylabel(metric_labels[metric])

                    axs[row][z].set_xticks(range(num_groups))
                    axs[row][z].set_xticklabels([Z_labels[Z_indices[0]][i] for i in range(num_groups)])
                    # axs[alg][z].set_yticks([])
                    # axs[alg][z].set_ylim((0,1))

                # axs[alg][0].yaxis.set_major_locator(ticker.MultipleLocator(0.25))
            row += 1



        # plotting num bins of wgm vs umb number of bins for different umb bin numbers
        if len(n_cals)>1:
            algorithms = []
            bin_count = {}
            results = {}
            for umb_num_bin in umb_num_bins:
                algorithms.append("wgm_" + str(umb_num_bin))
                bin_count["wgm_" + str(umb_num_bin)] = umb_num_bin
                algorithms.append("wgc_" + str(umb_num_bin))
                bin_count["wgc_" + str(umb_num_bin)] = umb_num_bin

            metrics = ["n_bins", "accuracy", "num_selected"]

            for n_cal in n_cals:
                results[n_cal] = {}
                for algorithm in algorithms:
                    results[n_cal][algorithm] = {}
                    for metric in metrics:
                        results[n_cal][algorithm][metric] = {}
                        results[n_cal][algorithm][metric]["values"] = []

            for n_cal in n_cals:
                for run in runs:
                    for algorithm in algorithms:
                        exp_identity_string = "_".join(
                            [Z_str, str(n_train), str(noise_ratio), str(n_cal), lbd, str(run)])
                        result_path = os.path.join(exp_dir,
                                                   exp_identity_string + "_{}_result.pkl".format(algorithm))
                        collect_results_quantitative_exp(result_path, n_cal, algorithm, results, metrics)

            for n_cal in n_cals:
                for algorithm in algorithms:
                    for metric in metrics:
                        results[n_cal][algorithm][metric]["mean"] = np.mean(
                            results[n_cal][algorithm][metric]["values"])
                        results[n_cal][algorithm][metric]["std"] = np.std(
                            results[n_cal][algorithm][metric]["values"],
                            ddof=1)
                    # assert (np.array(results[umb_num_bins][algorithm][metric]["values"]) >= 0).all()

            for i, metric in enumerate(metrics):
                handles = []
                for algorithm in algorithms:

                    mean_algorithm = np.array([results[n_cal][algorithm][metric]["mean"] for n_cal
                                               in n_cals])
                    std_algorithm = np.array([results[n_cal][algorithm][metric]["std"] for n_cal
                                              in n_cals])

                    line = axs[row][z].plot(n_cals_label, mean_algorithm,
                                            linewidth=line_width,
                                            label="n = {}".format(bin_count[algorithm]),
                                            color=umb_colors[bin_count[algorithm]],
                                            marker=umb_markers[bin_count[algorithm]])  # , color=group_colors[i], marker=group_markers[i])
                    handles.append(line[0])

                    axs[row][z].fill_between(n_cals_label, mean_algorithm - std_algorithm,
                                             mean_algorithm + std_algorithm, alpha=transparency,
                                             color=umb_colors[bin_count[algorithm]])
                    axs[row][z].set_xticks(n_cals_label)
                    # axs[2 + i][z].set_yticks([])

                    # axs[2][z].set_ylim((5, 15))
                    # axs[3][z].set_ylim((6, 7))

                axs[row][z].set_xlabel(r'Calibration Set Size')
                axs[row][0].legend( loc='center right', bbox_to_anchor=(-0.12, 0.5), ncol=1)
                axs[row][0].set_ylabel(metric_labels[metric])
                # axs[3][0].set_ylabel(r'Shortlist Size')
                # axs[2][0].yaxis.set_major_locator(ticker.MultipleLocator(2))
                # axs[3][0].set_ylim(top=7)
                # axs[3][0].yaxis.set_major_locator(ticker.MultipleLocator(1))
                row += 1

        # # plotting num bins of wgm vs umb number of bins for different umb bin numbers
        # break
        # row += 1
        # algorithms = []
        # for umb_num_bin in umb_num_bins:
        #     algorithms.append("wgm")
        #     algorithms.append("umb")
        #
        # metrics = ["accuracy"]  # num_selected
        # results = {}
        #
        # for umb_num_bin in umb_num_bins:
        #     results[umb_num_bin] = {}
        #     for algorithm in algorithms:
        #         results[umb_num_bin][algorithm] = {}
        #         for metric in metrics:
        #             results[umb_num_bin][algorithm][metric] = {}
        #             results[umb_num_bin][algorithm][metric]["values"] = []
        #
        # for umb_num_bin in umb_num_bins:
        #     for run in runs:
        #         for algorithm in algorithms:
        #             exp_identity_string = "_".join(
        #                 [Z_str, str(n_train), str(noise_ratio), str(the_n_cal), lbd, str(run)])
        #             result_path = os.path.join(exp_dir, exp_identity_string + "_{}_{}_result.pkl".format(algorithm,
        #                                                                                                  umb_num_bin))
        #             collect_results_quantitative_exp(result_path, umb_num_bin, algorithm, results, metrics)
        #
        # for umb_num_bin in umb_num_bins:
        #     for algorithm in algorithms:
        #         for metric in metrics:
        #             results[umb_num_bin][algorithm][metric]["mean"] = np.mean(
        #                 results[umb_num_bin][algorithm][metric]["values"])
        #             results[umb_num_bin][algorithm][metric]["std"] = np.std(
        #                 results[umb_num_bin][algorithm][metric]["values"],
        #                 ddof=1)
        #         # assert (np.array(results[umb_num_bins][algorithm][metric]["values"]) >= 0).all()
        # handles = []
        # for algorithm in algorithms:
        #     for i, metric in enumerate(metrics):
        #         mean_algorithm = np.array([results[umb_num_bin][algorithm][metric]["mean"] for umb_num_bin
        #                                    in umb_num_bins])
        #         std_algorithm = np.array([results[umb_num_bin][algorithm][metric]["std"] for umb_num_bin
        #                                   in umb_num_bins])
        #
        #         line = axs[row][z].plot(umb_num_bins, mean_algorithm,
        #                                 linewidth=line_width,
        #                                 label=algorithm_labels["{}_{}".format(algorithm, str(umb_num_bins[0]))],
        #                                 color=algorithm_colors["{}_{}".format(algorithm, str(umb_num_bins[0]))],
        #                                 marker=algorithm_markers["{}_{}".format(algorithm, str(
        #                                     umb_num_bins[0]))])  # , color=group_colors[i], marker=group_markers[i])
        #         handles.append(line[0])
        #
        #         axs[row][z].fill_between(umb_num_bins, mean_algorithm - std_algorithm,
        #                                  mean_algorithm + std_algorithm, alpha=transparency,
        #                                  color=algorithm_colors["{}_{}".format(algorithm, str(umb_num_bins[0]))])
        #
        #         axs[row][z].set_xticks(umb_num_bins)
        #         # axs[2 + i][z].set_yticks([])
        #
        #         # axs[2][z].set_ylim((5, 15))
        #         # axs[3][z].set_ylim((6, 7))
        #
        #         if i == 0:
        #             axs[row][0].legend(handles=handles, loc='center right', bbox_to_anchor=(-0.12, 0.5), ncol=1)
        #             handles = []
        #
        #         axs[row][z].set_xlabel(r'n')
        #     axs[row][0].set_ylabel(r'$|\mathcal{B}|$')
            # axs[3][0].set_ylabel(r'Shortlist Size')
            # axs[2][0].yaxis.set_major_locator(ticker.MultipleLocator(2))
            # axs[3][0].set_ylim(top=7)
            # axs[3][0].yaxis.set_major_locator(ticker.MultipleLocator(1))


        # #plotting accuracy for different algorithms across number of bins
        # algorithms = []
        # for umb_num_bin in umb_num_bins:
        #     algorithms.append("umb_" + str(umb_num_bin))
        #     # algorithms.append("wgm_" + str(umb_num_bin))
        #     algorithm_labels["umb_" + str(umb_num_bin)] = "n = {}".format(umb_num_bin)
        #     # algorithm_labels["wgm_" + str(umb_num_bin)] = "WGM {} Bins".format(umb_num_bin)
        #     algorithm_colors["umb_" + str(umb_num_bin)] = umb_colors[umb_num_bin]
        #     # algorithm_colors["wgm_" + str(umb_num_bin)] = umb_colors[umb_num_bin]
        # metrics = ["accuracy"]#["num_selected"]
        # results = {}
        #
        # for umb_num_bin, algorithm in zip(umb_num_bins, algorithms):
        #     results[umb_num_bin] = {}
        #     results[umb_num_bin][algorithm] = {}
        #     for metric in metrics:
        #         results[umb_num_bin][algorithm][metric] = {}
        #         results[umb_num_bin][algorithm][metric]["values"] = []
        #
        # for umb_num_bin, algorithm in zip(umb_num_bins, algorithms):
        #     for run in runs:
        #         exp_identity_string = "_".join(
        #             [Z_str, str(n_train), str(noise_ratio), str(the_n_cal), lbd, str(run)])
        #         result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))
        #         collect_results_quantitative_exp(result_path, umb_num_bin, algorithm, results, metrics)
        #
        # for umb_num_bin, algorithm in zip(umb_num_bins, algorithms):
        #     for metric in metrics:
        #         results[umb_num_bin][algorithm][metric]["mean"] = np.mean(
        #             results[umb_num_bin][algorithm][metric]["values"])
        #         results[umb_num_bin][algorithm][metric]["std"] = np.std(
        #             results[umb_num_bin][algorithm][metric]["values"],
        #             ddof=1)
        #         # assert (np.array(results[umb_num_bins][algorithm][metric]["values"]) >= 0).all()
        # # handles = []
        # # for algorithm in algorithms:
        # #     # plotting number of bins of wgm against umb
        # mean_algorithm = np.array([results[umb_num_bin][algorithm][metrics[0]]["mean"] for umb_num_bin, algorithm
        #                            in zip(umb_num_bins, algorithms)])
        # std_algorithm = np.array([results[umb_num_bin][algorithm][metrics[0]]["std"] for umb_num_bin, algorithm
        #                           in zip(umb_num_bins, algorithms)])
        #
        # # num_groups = mean_algorithm.shape[1]
        # import matplotlib.colors as mcolors
        #
        # line = axs[3][z].plot(umb_num_bins, mean_algorithm,
        #                       linewidth=line_width,label="UMB", color=algorithm_colors["UMB"],marker=algorithm_markers["UMB"])  # , color=group_colors[i], marker=group_markers[i])
        # handles.append(line[0])
        #
        # # axs[3][z].errorbar(umb_num_bins, mean_algorithm, std_algorithm, linewidth=line_width, capthick=capthick)
        # axs[3][z].fill_between(umb_num_bins, mean_algorithm-std_algorithm, mean_algorithm+std_algorithm, color=algorithm_colors["UMB"],alpha=transparency)
        #
        # # axs[3][z].set_xticks(umb_num_bins)
        # axs[3][0].legend(handles=handles, loc='center right', bbox_to_anchor=(-0.1, 0.5), ncol=1)




        # plotting num bins of wgm vs umb number of bins for different umb bin numbes
        # algorithms = []
        # for umb_num_bin in umb_num_bins:
        #     algorithms.append("wgm_" + str(umb_num_bin))
        #     algorithm_labels["wgm_" + str(umb_num_bin)] = "n = {}".format(umb_num_bin)
        #     algorithm_colors["wgm_" + str(umb_num_bin)] = umb_colors[umb_num_bin]
        #     algorithm_markers["wgm_" + str(umb_num_bin)] = umb_markers[umb_num_bin]
        # metrics = ["n_bins","accuracy"] #num_selected
        # results = {}
        #
        # for ncal in n_cals:
        #     results[ncal] = {}
        #     for algorithm in algorithms:
        #         results[ncal][algorithm] = {}
        #         for metric in metrics:
        #             results[ncal][algorithm][metric] = {}
        #             results[ncal][algorithm][metric]["values"] = []
        #
        # for ncal in n_cals:
        #     for run in runs:
        #         exp_identity_string = "_".join([Z_str,str(n_train), str(noise_ratio), str(ncal), lbd, str(run)])
        #         for algorithm in algorithms:
        #             result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))
        #             collect_results_quantitative_exp(result_path, ncal, algorithm, results, metrics)
        #
        # for ncal in n_cals:
        #     for algorithm in algorithms:
        #         for metric in metrics:
        #             results[ncal][algorithm][metric]["mean"] = np.mean(results[ncal][algorithm][metric]["values"])
        #             results[ncal][algorithm][metric]["std"] = np.std(results[ncal][algorithm][metric]["values"],
        #                                                         ddof=1)
        #             # assert (np.array(results[umb_num_bins][algorithm][metric]["values"]) >= 0).all()
        # handles = []
        # for algorithm in algorithms:
        #     for i,metric in enumerate(metrics):
        #         mean_algorithm = np.array([results[ncal][algorithm][metric]["mean"] for ncal
        #                                    in n_cals])
        #         std_algorithm = np.array([results[ncal][algorithm][metric]["std"] for ncal
        #                                   in n_cals])
        #
        #         # num_groups = mean_algorithm.shape[1]
        #         import matplotlib.colors as mcolors
        #
        #         line = axs[4+i][z].plot(n_cals_label, mean_algorithm,color=algorithm_colors[algorithm],label=algorithm_labels[algorithm],
        #                           linewidth=line_width,marker=algorithm_markers[algorithm])  # , color=group_colors[i], marker=group_markers[i])
        #         if i==0:
        #             handles.append(line[0])
        #
        #         axs[4+i][z].fill_between(n_cals_label, mean_algorithm-std_algorithm, mean_algorithm+std_algorithm, linewidth=line_width,\
        #                                color=algorithm_colors[algorithm],label=algorithm_labels[algorithm],alpha=transparency)
        #
        #         # axs[3][z].set_xticks(np.arange(len(n_cals)),n_cals_label)
        #         # axs[4+i][z].set_yticks([])
        #         axs[4+i][z].set_xlabel(r"$|\mathcal{X}|$")
        #
        #         # axs[4][z].set_ylim(4,20)
        #         # axs[5][z].set_ylim(6,7)
        #
        # axs[4][0].legend(handles=handles, loc='center right', bbox_to_anchor=(-0.1, 0.5), ncol=1)
        # axs[4][0].set_ylabel(r'$|\mathcal{B}|$')
        # axs[5][0].set_ylabel(r'Shortlist Size')
        # axs[4][0].yaxis.set_major_locator(ticker.MultipleLocator(2))
        # axs[5][0].set_ylim(top=7)
        # axs[5][0].yaxis.set_major_locator(ticker.MultipleLocator(1))


    # axs[2][0].legend(handles = handles, loc='center right', bbox_to_anchor=(-0.08, 0.5), ncol=1)
    # axs[2][0].yaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.tight_layout(rect=[0, 0, 1, 1])
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    fig.savefig("./plots/exp_wgm.pdf", format="pdf")
