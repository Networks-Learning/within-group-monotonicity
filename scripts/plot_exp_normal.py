import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from plot_constants import *
plt.rcParams.update(params)
plt.rc('font', family='serif')


if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(28, 6)
    from params_exp_noise import *
    algorithms = []
    algorithm_labels = {}
    algorithm_colors = {}
    algorithm_markers = {}
    # algorithm_labels = {}
    # algorithm_colors = {}
    # algorithm_markers = {}
    # algorithms = ["ucss", "iso_reg_ss", "platt_scal_ss"]
    # algorithm_df_guarantee = {
    #     "css": True,
    #     "ucss": False,
    #     "iso_reg_ss": False,
    #     "platt_scal_ss": False
    # }
    # algorithm_labels = {
    #     "css": "CSS",
    #     "ucss": "Uncalibrated",
    #     "iso_reg_ss": "Isotonic",
    #     "platt_scal_ss": "Platt"
    # }
    # algorithm_colors = {
    #     "css": "tab:blue",
    #     "ucss": "tab:red",
    #     "iso_reg_ss": "tab:purple",
    #     "platt_scal_ss": "tab:cyan"
    # }
    # algorithm_markers = {
    #     "css": "s",
    #     "ucss": 9,
    #     "iso_reg_ss": 10,
    #     "platt_scal_ss": 11
    # }
    Z_labels = {
        2: {0:"Married",1:"Widowed",2:"Divorced",3:"Separated",4:"Never married"},
        4: {0: "With a disability", 1: "Without a disability"},
        6: {0:"Born in US", 1:"Born in Puerto Rico", 2:"Born abroad", 3:"US citizen", 4:"Not a US citizen"},
        10: {0:"Native", 1:"Foreign born"},
        14: {0: "Male", 1:"Female"}
    }


    for umb_num_bin in umb_num_bins:
        algorithms.append("umb_" + str(umb_num_bin))
        algorithm_labels["umb_" + str(umb_num_bin)] = "UMB {} Bins".format(umb_num_bin)
        algorithm_colors["umb_" + str(umb_num_bin)] = umb_colors[umb_num_bin]
        # algorithm_df_guarantee["umb_" + str(umb_num_bin)] = True
        algorithm_markers["umb_" + str(umb_num_bin)] = umb_markers[umb_num_bin]
    # # algorithms.append("css")
    metrics = ["group_bin_value","group_rho"] #["num_selected", "num_qualified", "num_unqualified", "constraint_satisfied","num_positives_in_bin","num_in_bin","bin_value"]
    # results = {}
    # for Z_indices in Z:
    #     Z_str = "_".join([str(index) for index in Z_indices])
    #     results[Z_str] = {}
    #     for noise_ratio in noise_ratios:
    #         results[Z_str][noise_ratio] = {}
    #         for algorithm in algorithms:
    #             results[Z_str][noise_ratio][algorithm] = {}
    #             for metric in metrics:
    #                 results[Z_str][noise_ratio][algorithm][metric] = {}
    #                 results[Z_str][noise_ratio][algorithm][metric]["values"] = []
    #
    # for Z_indices in Z:
    #     Z_str = "_".join([str(index) for index in Z_indices])
    #     for noise_ratio in noise_ratios:
    #         for run in runs:
    #             exp_identity_string = "_".join([Z_str, str(n_train), str(noise_ratio), str(n_cal), lbd, str(run)])
    #             for algorithm in algorithms:
    #                 result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))
    #                 collect_results_normal_exp(result_path, noise_ratio,Z_str, algorithm, results)
    #
    # for Z_indices in Z:
    #     Z_str = "_".join([str(index) for index in Z_indices])
    #     for noise_ratio in noise_ratios:
    #         for algorithm in algorithms:
    #             for metric in metrics:
    #                 results[Z_str][noise_ratio][algorithm][metric]["mean"] = np.mean(results[Z_str][noise_ratio][algorithm][metric]["values"],axis=0)
    #                 results[Z_str][noise_ratio][algorithm][metric]["std"] = np.std(results[Z_str][noise_ratio][algorithm][metric]["values"],
    #                                                         ddof=1,axis=0)

    # plotting violations
    handles = []
    # for i,Z_indices in enumerate(Z):
    #     Z_str = "_".join([str(index) for index in Z_indices])
    #     for algorithm in algorithms:
    #         for noise_ratio in enumerate(noise_ratios):
    #             mean_algorithm = np.array([results[Z_str][noise_ratio][algorithm]["bin_value"]["mean"]]).squeeze()
    #             std_err_algorithm = np.array(
    #                 [results[Z_str][noise_ratio][algorithm]["bin_value"]["std"] / np.sqrt(n_runs)]).squeeze()
    #             line = axs[i].plot(range(umb_num_bin), mean_algorithm, color=algorithm_colors[algorithm],
    #                                marker=algorithm_markers[algorithm], linewidth=line_width,
    #                                label=algorithm_labels[algorithm])
    #             if algorithm == "css":
    #                 handles = [line[0]] + handles
    #             else:
    #                 handles.append(line[0])
    #             axs[i].errorbar(range(umb_num_bin), mean_algorithm, std_err_algorithm, color=algorithm_colors[algorithm],
    #                             marker=algorithm_markers[algorithm], linewidth=line_width,
    #                             label=algorithm_labels[algorithm], capthick=capthick)
    #             # axs[i].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    #             axs[i].xaxis.set_major_locator(ticker.MultipleLocator(1))
    #             axs[i].set_xlabel("$f(X)$", fontsize=font_size)
    #             axs[i].set_ylabel("$P(Y|f(X))$", fontsize=font_size)
    # fig.savefig("./plots/exp_normal.pdf", format="pdf")


    # plotting whether constraint is satisfied
    # handles = []
    # for algorithm in algorithms:
    #     mean_algorithm = np.array([results[noise_ratio][algorithm]["constraint_satisfied"]["mean"]
    #                                                     for noise_ratio in noise_ratios])
    #     std_err_algorithm = np.array(
    #         [results[noise_ratio][algorithm]["constraint_satisfied"]["std"] / np.sqrt(n_runs) for noise_ratio in noise_ratios])
    #     line = axs[0].plot(noise_ratios_label, mean_algorithm, color=algorithm_colors[algorithm],
    #                        marker=algorithm_markers[algorithm], linewidth=line_width,
    #                        label=algorithm_labels[algorithm])
    #     if algorithm == "css":
    #         handles = [line[0]] + handles
    #     else:
    #         handles.append(line[0])
    #     axs[0].errorbar(noise_ratios_label, mean_algorithm, std_err_algorithm, color=algorithm_colors[algorithm],
    #                        marker=algorithm_markers[algorithm], linewidth=line_width,
    #                        label=algorithm_labels[algorithm], capthick=capthick)
    # axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    # axs[0].set_xlabel("$r_{\mathrm{noise}}$", fontsize=font_size)
    # axs[0].set_ylabel("EQ", fontsize=font_size)
    #
    # # plotting the number of selected applicants
    # for algorithm in algorithms:
    #     if not algorithm_df_guarantee[algorithm]:
    #         continue
    #     mean_algorithm = np.array([results[noise_ratio][algorithm]["num_selected"]["mean"] for noise_ratio
    #                                             in noise_ratios])
    #     std_algorithm = np.array([results[noise_ratio][algorithm]["num_selected"]["std"] for noise_ratio
    #                                            in noise_ratios])
    #
    #     axs[1].plot(noise_ratios_label, mean_algorithm, linewidth=line_width, color=algorithm_colors[algorithm], marker=algorithm_markers[algorithm]
    #              , label=algorithm_labels[algorithm])
    #     axs[1].fill_between(noise_ratios_label, mean_algorithm - std_algorithm,
    #                      mean_algorithm + std_algorithm, alpha=transparency,
    #                      color=algorithm_colors[algorithm])
    # axs[1].set_xlabel("$r_{\mathrm{noise}}$", fontsize=font_size)
    # axs[1].set_ylabel("SS", fontsize=font_size)
    # axs[1].set_ylim(top=35)
    # axs[1].set_ylim(bottom=5)
    #

    #for the same Z and calibration set size multiple runs
    from params_exp_cal import *
    for j,Z_indices in enumerate(Z):
        Z_str = "_".join([str(index) for index in Z_indices])  #for one set of groups
        n_cal = n_cals[0]  # for one calibration set
        results = {}
        # results[n_cal] = {}

        for bin in range(umb_num_bins[0]):
            results[bin] = {}
            for algorithm in algorithms:
                results[bin][algorithm] = {}
                for metric in metrics:
                    results[bin][algorithm][metric] = {}
                    results[bin][algorithm][metric]["values"] = []

        for bin in range(umb_num_bins[0]):
            for run in runs:
                exp_identity_string = "_".join([Z_str,str(n_train), str(noise_ratio), str(n_cal), lbd, str(run)])
                for algorithm in algorithms:
                    result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))
                    collect_results_normal_exp(result_path, bin, algorithm, results)

        for bin in range(umb_num_bins[0]):
            for algorithm in algorithms:
                for metric in metrics:
                    print(np.array(results[bin][algorithm][metric]["values"]))
                    assert (np.array(results[bin][algorithm][metric]["values"])[:,:lim_num_groups]>=0).all()
                    results[bin][algorithm][metric]["mean"] = np.mean(results[bin][algorithm][metric]["values"],axis=0)
                    results[bin][algorithm][metric]["std"] = np.std(results[bin][algorithm][metric]["values"],
                                                                      ddof=1,axis=0)
                    print(results[bin][algorithm][metric]["std"].shape)
        # plotting whether constraint is satisfied
        # for algorithm in algorithms:
        #     # if algorithm_df_guarantee[algorithm] and algorithm != "css":
        #     #     continue
        #     mean_algorithm = np.array([results[n_cal][algorithm]["constraint_satisfied"]["mean"]
        #                                                     for n_cal in n_cals])
        #     std_err_algorithm = np.array(
        #         [results[n_cal][algorithm]["constraint_satisfied"]["std"] / np.sqrt(n_runs) for n_cal in n_cals])
        #     axs[2].errorbar(n_cals_label, mean_algorithm, std_err_algorithm, color=algorithm_colors[algorithm],
        #                     linewidth=line_width, label=algorithm_labels[algorithm], marker=algorithm_markers[algorithm],
        #                     capthick=capthick)
        # axs[2].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        # axs[2].set_xlabel("$n$", fontsize=font_size)
        # axs[2].set_ylabel("EQ", fontsize=font_size)
        #
        # # plotting the number of selected applicants
        for algorithm in algorithms:
            mean_algorithm = np.array([results[bin][algorithm]["group_bin_value"]["mean"] for bin
                                                    in range(umb_num_bins[0])])
            std_algorithm = np.array([results[bin][algorithm]["group_bin_value"]["std"] for bin
                                                   in range(umb_num_bins[0])])
            alpha_algorithm = np.array([results[bin][algorithm]["group_rho"]["mean"] for bin
                                                   in range(umb_num_bins[0])])


            num_groups = mean_algorithm.shape[1]
            import matplotlib.colors as mcolors

            for i in range(num_groups):
                mean = mean_algorithm[:,i]
                std = std_algorithm[:,i]
                alpha = alpha_algorithm[:,i]
                rgba_colors = np.zeros(shape=(alpha.shape[0],4))
                rgba_colors[:,:3] = mcolors.to_rgb(group_colors[i])
                rgba_colors[:,3] = alpha

                print(alpha)
                # np.where(mean >= 0, mean, np.empty(shape=mean.shape))
                # axs[j].plot(range(umb_num_bins[0]),mean , linewidth=line_width)#, color=group_colors[i], marker=group_markers[i])
                #          # , label=Z_labels[Z_indices[0]][i])
                axs[j].bar(np.arange(umb_num_bins[0])-((i-1)*0.2), mean,align='edge',
                            linewidth=line_width,width=0.1,color=group_colors[i],label=Z_labels[Z_indices[0]][i])  # , color=group_colors[i], marker=group_markers[i])
                # , label=Z_labels[Z_indices[0]][i])
                axs[j].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),ncol=3)
                axs[j].set_yticks([])
                axs[j].set_xticks(range(0,umb_num_bins[0],1),range(1,umb_num_bins[0]+1,1))
                # axs[j].fill_between(range(umb_num_bins[0]), mean - std,
                #                  mean + std, alpha=transparency,
                #                  color=group_colors[i])
        # axs[3].set_xlabel("$n$", fontsize=font_size)
        # axs[3].set_ylabel("SS", fontsize=font_size)
        # axs[3].set_ylim(top=35)
    # axs[3].set_ylim(bottom=5)
    #
    # fig.legend(handles=handles, bbox_to_anchor=(0.5, 1.02), loc="upper center", ncol=5)
    axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig("./plots/exp_normal.pdf", format="pdf")
