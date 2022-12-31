import pickle
params = {'legend.fontsize': 16,#28,
          'xtick.labelsize': 24,
          'ytick.labelsize': 28,
          'lines.markersize': 15,
          'errorbar.capsize': 8.0,
            'axes.labelsize' : 24,
            'text.usetex'  : True,
            'font.family': 'serif',
          }


line_width = 3.0
transparency = 0.1
font_size = 24
capthick = 3.0
dpi = 100


def collect_results_normal_exp(result_path, exp_parameter, algorithm, results, metrics):
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
    # print(result)
    # print(result["bin_value"].shape)
    # print(result["group_bin_value"][exp_parameter].shape)
    # results[exp_parameter][algorithm]["num_selected"]["values"].append(result["num_selected"])
    # results[exp_parameter][algorithm]["num_qualified"]["values"].append(result["num_qualified"])
    # results[exp_parameter][algorithm]["num_unqualified"]["values"].append(result["num_selected"] -
    #                                                                       result["num_qualified"])
    # results[exp_parameter][algorithm]["constraint_satisfied"]["values"].append(result["constraint_satisfied"])
    #
    # results[exp_parameter][algorithm]["num_positives_in_bin"]["values"].append(result["num_positives_in_bin"])
    # results[exp_parameter][algorithm]["num_in_bin"]["values"].append(result["num_in_bin"])
    # results[exp_parameter][algorithm]["bin_value"]["values"].append(result["bin_value"][exp_parameter])
    # results[exp_parameter][algorithm]["group_bin_values"]["values"].append(result["group_bin_values"][exp_parameter])
    # results[exp_parameter][algorithm]["group_rho"]["values"].append(result["group_rho"][exp_parameter])
    for metric in metrics:
        results[exp_parameter][algorithm][metric]["values"].append(result[metric][exp_parameter])





def collect_results_quantitative_exp(result_path, exp_parameter, algorithm, results, metrics):
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
    for metric in metrics:
        results[exp_parameter][algorithm][metric]["values"].append(result[metric])



def collect_results_diversity_exp(result_path, exp_parameter, algorithm, results):
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
    results[exp_parameter][algorithm]["num_selected_maj"]["values"].append(result["num_selected_maj"])
    results[exp_parameter][algorithm]["num_qualified_maj"]["values"].append(result["num_qualified_maj"])
    results[exp_parameter][algorithm]["num_unqualified_maj"]["values"].append(result["num_selected_maj"] -
                                                                          result["num_qualified_maj"])
    results[exp_parameter][algorithm]["constraint_satisfied_maj"]["values"].append(result["constraint_satisfied_maj"])
    results[exp_parameter][algorithm]["num_selected_min"]["values"].append(result["num_selected_min"])
    results[exp_parameter][algorithm]["num_qualified_min"]["values"].append(result["num_qualified_min"])
    results[exp_parameter][algorithm]["num_unqualified_min"]["values"].append(result["num_selected_min"] -
                                                                              result["num_qualified_min"])
    results[exp_parameter][algorithm]["constraint_satisfied_min"]["values"].append(result["constraint_satisfied_min"])
