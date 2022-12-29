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
    fig, axs = plt.subplots(6, len(Z))
    fig.set_size_inches(len(Z)*10, 25)

    Z_labels = {
        # 2: {0:"Married",1:"Widowed",2:"Divorced",3:"Separated",4:"Never married"},
        2: {0:"Married or Separated", 1: "Never married", "feature":"Marital status"},
        4: {0: "With a disability", 1: "Without a disability", "feature": "Disability record"},
        6: {0:"Born in the US", 1:"Born in Unincorporated US", 2:"Born abroad", 3:"Not a US citizen", "feature":"Citizenship status"},
        10: {0:"Native", 1:"Foreign born", "feature":"Nativity"},
        14: {0: "Male", 1:"Female", "feature":"Gender"},
        15: {0: "White", 1:"Black or African American", 2:"American Indian or Alaska", 3:"Asian, Native Hawaiian or other", "feature":"Race code"},
        1: {0:"No diploma", 1:"diploma", 2:"Associate or Bachelor degree", 3: "Masters or Doctorate degree", "feature":"Educational attainment"},
        0: {0:"0-25", 1:"26-50", 2:"51-75", 3:"76-99", "feature":"Age"}
    }

    #plot bin values for one run, across algorithms and Z_indices

    for z,Z_indices in enumerate(Z):
        algorithms = []
        algorithm_labels = {}
        algorithm_colors = {"WGM":"tab:green", "UMB":"tab:red"}
        algorithm_markers = {"WGM":10, "UMB":11}
        num_bins = {}
        handles = []
        Z_str = "_".join([str(index) for index in Z_indices])  #for one set of groups
        the_n_cal = n_cals[0]  # for one calibration set
        the_run = 0
        results = {}
        # results[n_cal] = {}

        # for umb_num_bin in umb_num_bins:
        #     algorithms.append("umb_" + str(umb_num_bin))
        #     algorithms.append("wgm_" + str(umb_num_bin))
        #     algorithm_labels["umb_" + str(umb_num_bin)] = "UMB {} Bins".format(umb_num_bin)
        #     algorithm_labels["wgm_" + str(umb_num_bin)] = "WGM {} Bins".format(umb_num_bin)
        #     algorithm_colors["umb_" + str(umb_num_bin)] = umb_colors[umb_num_bin]
        #     algorithm_colors["wgm_" + str(umb_num_bin)] = umb_colors[umb_num_bin]
        the_umb_num_bin = umb_num_bins[1]
        algorithms.append("umb_" + str(the_umb_num_bin))
        algorithms.append("wgm_" + str(the_umb_num_bin))
        algorithm_labels["umb_" + str(the_umb_num_bin)] = "n = {}".format(the_umb_num_bin)
        algorithm_labels["wgm_" + str(the_umb_num_bin)] = "n = {}".format(the_umb_num_bin)
        algorithm_colors["umb_" + str(the_umb_num_bin)] = umb_colors[the_umb_num_bin]
        algorithm_colors["wgm_" + str(the_umb_num_bin)] = umb_colors[the_umb_num_bin]

        for alg,algorithm in enumerate(algorithms):
            exp_identity_string = "_".join([Z_str, str(n_train), str(noise_ratio), str(the_n_cal), lbd, str(the_run)])
            result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))

            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            num_bins[algorithm] = len(result["bin_values"])

            metrics = ["group_bin_values","group_rho"]

            for bin in range(num_bins[algorithm]):
                results[bin] = {}
                results[bin][algorithm] = {}
                for metric in metrics:
                    results[bin][algorithm][metric] = {}
                    results[bin][algorithm][metric]["values"] = []

            for bin in range(num_bins[algorithm]):
                for metric in metrics:
                    collect_results_normal_exp(result_path, bin, algorithm, results, metric)

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

            num_groups = mean_algorithm.shape[1]
            import matplotlib.colors as mcolors

            # axs[2][j].imshow(alpha_algorithm.swapaxes(1,0))
            # axs[2][j].set_yticks(np.arange(alpha_algorithm.shape[1]),labels=Z_labels[Z_indices[0]].values(),fontsize=12)
            # axs[2][j].set_xticks(range(0, umb_num_bins[0], 1), range(1, umb_num_bins[0] + 1, 1))
            for i in range(num_groups):
                mean = mean_algorithm[:,i]
                std = std_algorithm[:,i]
                alpha = alpha_algorithm[:,i]
                rgba_colors = np.zeros(shape=(alpha.shape[0],4))
                rgba_colors[:,:3] = mcolors.to_rgb(group_colors[i])
                rgba_colors[:,3] = alpha

                axs[alg][z].bar(np.arange(num_bins[algorithm])-((i-1)*0.2), mean,align='edge',
                            linewidth=line_width,width=0.1,color=group_colors[i],label=Z_labels[Z_indices[0]][i])  # , color=group_colors[i], marker=group_markers[i])

                if alg==0:
                    legend = axs[alg][z].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),ncol=2, title = Z_labels[Z_indices[0]]["feature"])
                    plt.setp(legend.get_title(), fontsize=params['legend.fontsize'])

                axs[alg][z].set_xticks(range(0, num_bins[algorithm], 1), range(1, num_bins[algorithm] + 1, 1))
                axs[alg][z].set_yticks([])


            axs[alg][0].yaxis.set_major_locator(ticker.MultipleLocator(0.25))
            if algorithm.startswith("umb"):
                axs[alg][0].set_ylabel(r'$\Pr(Y=1|a,z)$')
                axs[alg][z].set_xlabel(r'$a,z$')
            if algorithm.startswith("wgm"):
                axs[alg][0].set_ylabel(r'$\Pr(Y=1|a_\mathcal{A},z)$')
                axs[alg][z].set_xlabel(r'$a_\mathcal{A},z$')


        # plotting num bins of wgm vs umb number of bins for different umb bin numbes
        algorithms = []
        for umb_num_bin in umb_num_bins:
            # algorithms.append("umb_" + str(umb_num_bin))
            algorithms.append("wgm_" + str(umb_num_bin))
            # algorithm_labels["umb_" + str(umb_num_bin)] = "UMB {} Bins".format(umb_num_bin)
            algorithm_labels["wgm_" + str(umb_num_bin)] = "n = {}".format(umb_num_bin)
            # algorithm_colors["umb_" + str(umb_num_bin)] = umb_colors[umb_num_bin]
            algorithm_colors["wgm_" + str(umb_num_bin)] = umb_colors[umb_num_bin]
        metrics = ["n_bins","num_selected"]
        results = {}

        for umb_num_bin, algorithm in zip(umb_num_bins, algorithms):
            results[umb_num_bin] = {}
            results[umb_num_bin][algorithm] = {}
            for metric in metrics:
                results[umb_num_bin][algorithm][metric] = {}
                results[umb_num_bin][algorithm][metric]["values"] = []

        for umb_num_bin, algorithm in zip(umb_num_bins, algorithms):
            for run in runs:
                exp_identity_string = "_".join(
                    [Z_str, str(n_train), str(noise_ratio), str(the_n_cal), lbd, str(run)])
                result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))
                for metric in metrics:
                    collect_results_quantitative_exp(result_path, umb_num_bin, algorithm, results, metric)

        for umb_num_bin, algorithm in zip(umb_num_bins, algorithms):
            for metric in metrics:
                results[umb_num_bin][algorithm][metric]["mean"] = np.mean(
                    results[umb_num_bin][algorithm][metric]["values"])
                results[umb_num_bin][algorithm][metric]["std"] = np.std(
                    results[umb_num_bin][algorithm][metric]["values"],
                    ddof=1)
                # assert (np.array(results[umb_num_bins][algorithm][metric]["values"]) >= 0).all()
        handles = []
        # for algorithm in algorithms:
        #     # plotting number of bins of wgm against umb
        for i,metric in enumerate(metrics):
            mean_algorithm = np.array([results[umb_num_bin][algorithm][metric]["mean"] for umb_num_bin, algorithm
                                       in zip(umb_num_bins, algorithms)])
            std_algorithm = np.array([results[umb_num_bin][algorithm][metric]["std"] for umb_num_bin, algorithm
                                      in zip(umb_num_bins, algorithms)])

            # num_groups = mean_algorithm.shape[1]
            import matplotlib.colors as mcolors

            line = axs[2+i][z].plot(umb_num_bins, mean_algorithm,
                                  linewidth=line_width, label = "WGM",color = algorithm_colors["WGM"],marker = algorithm_markers["WGM"])  # , color=group_colors[i], marker=group_markers[i])
            handles.append(line[0])

            # axs[2+i][z].errorbar(umb_num_bins, mean_algorithm, std_algorithm, linewidth=line_width,color = algorithm_colors["WGM"],capthick=capthick)
            axs[2 + i][z].fill_between(umb_num_bins, mean_algorithm-std_algorithm, mean_algorithm+std_algorithm,alpha = transparency,
                                   color=algorithm_colors["WGM"])

            axs[2+i][z].set_xticks(umb_num_bins)
            axs[2 + i][z].set_yticks([])

            if i==0:
                axs[2][0].legend(handles=handles, loc='center right', bbox_to_anchor=(-0.12, 0.5), ncol=1)
                handles = []


            axs[2+i][z].set_xlabel(r'n')
        axs[2][0].set_ylabel(r'$|\mathcal{B}|$')
        axs[3][0].set_ylabel(r'Shortlist Size')
        axs[2][0].yaxis.set_major_locator(ticker.MultipleLocator(2))
        axs[3][0].yaxis.set_major_locator(ticker.MultipleLocator(1))

        algorithms = []
        for umb_num_bin in umb_num_bins:
            algorithms.append("umb_" + str(umb_num_bin))
            # algorithms.append("wgm_" + str(umb_num_bin))
            algorithm_labels["umb_" + str(umb_num_bin)] = "n = {}".format(umb_num_bin)
            # algorithm_labels["wgm_" + str(umb_num_bin)] = "WGM {} Bins".format(umb_num_bin)
            algorithm_colors["umb_" + str(umb_num_bin)] = umb_colors[umb_num_bin]
            # algorithm_colors["wgm_" + str(umb_num_bin)] = umb_colors[umb_num_bin]
        metrics = ["num_selected"]
        results = {}

        for umb_num_bin, algorithm in zip(umb_num_bins, algorithms):
            results[umb_num_bin] = {}
            results[umb_num_bin][algorithm] = {}
            for metric in metrics:
                results[umb_num_bin][algorithm][metric] = {}
                results[umb_num_bin][algorithm][metric]["values"] = []

        for umb_num_bin, algorithm in zip(umb_num_bins, algorithms):
            for run in runs:
                exp_identity_string = "_".join(
                    [Z_str, str(n_train), str(noise_ratio), str(the_n_cal), lbd, str(run)])
                result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))
                for metric in metrics:
                    collect_results_quantitative_exp(result_path, umb_num_bin, algorithm, results, metric)

        for umb_num_bin, algorithm in zip(umb_num_bins, algorithms):
            for metric in metrics:
                results[umb_num_bin][algorithm][metric]["mean"] = np.mean(
                    results[umb_num_bin][algorithm][metric]["values"])
                results[umb_num_bin][algorithm][metric]["std"] = np.std(
                    results[umb_num_bin][algorithm][metric]["values"],
                    ddof=1)
                # assert (np.array(results[umb_num_bins][algorithm][metric]["values"]) >= 0).all()
        # handles = []
        # for algorithm in algorithms:
        #     # plotting number of bins of wgm against umb
        mean_algorithm = np.array([results[umb_num_bin][algorithm]["num_selected"]["mean"] for umb_num_bin, algorithm
                                   in zip(umb_num_bins, algorithms)])
        std_algorithm = np.array([results[umb_num_bin][algorithm]["num_selected"]["std"] for umb_num_bin, algorithm
                                  in zip(umb_num_bins, algorithms)])

        # num_groups = mean_algorithm.shape[1]
        import matplotlib.colors as mcolors

        line = axs[3][z].plot(umb_num_bins, mean_algorithm,
                              linewidth=line_width,label="UMB", color=algorithm_colors["UMB"],marker=algorithm_markers["UMB"])  # , color=group_colors[i], marker=group_markers[i])
        handles.append(line[0])

        # axs[3][z].errorbar(umb_num_bins, mean_algorithm, std_algorithm, linewidth=line_width, capthick=capthick)
        axs[3][z].fill_between(umb_num_bins, mean_algorithm-std_algorithm, mean_algorithm+std_algorithm, color=algorithm_colors["UMB"],alpha=transparency)

        axs[3][z].set_xticks(umb_num_bins)
        axs[3][0].legend(handles=handles, loc='center right', bbox_to_anchor=(-0.1, 0.5), ncol=1)




        # plotting num bins of wgm vs umb number of bins for different umb bin numbes
        algorithms = []
        for umb_num_bin in umb_num_bins:
            algorithms.append("wgm_" + str(umb_num_bin))
            algorithm_labels["wgm_" + str(umb_num_bin)] = "n = {}".format(umb_num_bin)
            algorithm_colors["wgm_" + str(umb_num_bin)] = umb_colors[umb_num_bin]
            algorithm_markers["wgm_" + str(umb_num_bin)] = umb_markers[umb_num_bin]
        metrics = ["n_bins","num_selected"]
        results = {}

        for ncal in n_cals:
            results[ncal] = {}
            for algorithm in algorithms:
                results[ncal][algorithm] = {}
                for metric in metrics:
                    results[ncal][algorithm][metric] = {}
                    results[ncal][algorithm][metric]["values"] = []

        for ncal in n_cals:
            for run in runs:
                exp_identity_string = "_".join([Z_str,str(n_train), str(noise_ratio), str(ncal), lbd, str(run)])
                for algorithm in algorithms:
                    result_path = os.path.join(exp_dir, exp_identity_string + "_{}_result.pkl".format(algorithm))
                    for metric in metrics:
                        collect_results_quantitative_exp(result_path, ncal, algorithm, results, metric)

        for ncal in n_cals:
            for algorithm in algorithms:
                for metric in metrics:
                    results[ncal][algorithm][metric]["mean"] = np.mean(results[ncal][algorithm][metric]["values"])
                    results[ncal][algorithm][metric]["std"] = np.std(results[ncal][algorithm][metric]["values"],
                                                                ddof=1)
                    # assert (np.array(results[umb_num_bins][algorithm][metric]["values"]) >= 0).all()
        handles = []
        for algorithm in algorithms:
            for i,metric in enumerate(metrics):
                mean_algorithm = np.array([results[ncal][algorithm][metric]["mean"] for ncal
                                           in n_cals])
                std_algorithm = np.array([results[ncal][algorithm][metric]["std"] for ncal
                                          in n_cals])

                # num_groups = mean_algorithm.shape[1]
                import matplotlib.colors as mcolors

                line = axs[4+i][z].plot(n_cals_label, mean_algorithm,color=algorithm_colors[algorithm],label=algorithm_labels[algorithm],
                                  linewidth=line_width,marker=algorithm_markers[algorithm])  # , color=group_colors[i], marker=group_markers[i])
                if i==0:
                    handles.append(line[0])

                axs[4+i][z].fill_between(n_cals_label, mean_algorithm-std_algorithm, mean_algorithm+std_algorithm, linewidth=line_width,\
                                       color=algorithm_colors[algorithm],label=algorithm_labels[algorithm],alpha=transparency)

                # axs[3][z].set_xticks(np.arange(len(n_cals)),n_cals_label)
                axs[4+i][z].set_yticks([])
                axs[4+i][z].set_xlabel(r"$|\mathcal{X}|$")
        axs[4][0].legend(handles=handles, loc='center right', bbox_to_anchor=(-0.1, 0.5), ncol=1)
        axs[4][0].set_ylabel(r'$|\mathcal{B}|$')
        axs[5][0].set_ylabel(r'Shortlist Size')
        axs[4][0].yaxis.set_major_locator(ticker.MultipleLocator(2))
        axs[5][0].yaxis.set_major_locator(ticker.MultipleLocator(1))


    # axs[2][0].legend(handles = handles, loc='center right', bbox_to_anchor=(-0.08, 0.5), ncol=1)
    # axs[2][0].yaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.tight_layout(rect=[0, 0, 1, 1])
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    fig.savefig("./plots/exp_wgm.pdf", format="pdf")
