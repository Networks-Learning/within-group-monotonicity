import pickle
params = {'legend.fontsize': 24,#24,
          'xtick.labelsize': 24,
          'ytick.labelsize': 24,
          'lines.markersize': 15,
          'errorbar.capsize': 8.0,
            'axes.labelsize' : 24,
            'text.usetex'  : True,
            'text.latex.preamble': r'\usepackage{amsmath}',
            'font.family': 'serif',
          'axes.titlesize':24,

          }
# 'figure.autolayout': True

line_width = 3.0
transparency = 0.1
font_size = 28
capthick = 3.0
dpi = 100
fig_width = 14
fig_height = 5

Z_labels = {
        # 2: {0:"Married",1:"Widowed",2:"Divorced",3:"Separated",4:"Never married"},
        2: {0:"Married or Separated", 1: "Never married", "feature":"Marital status (Z)", "num_groups":2},
        4: {0: "With a disability", 1: "Without a disability", "feature": "Disability record (Z)","num_groups":2,"color":"orchid", "marker":"v"},
        6: {0:"Born in the US", 1:"Born in Unincorporated US", 2:"Born abroad", 3:"Not a US citizen", "feature":"Citizenship status (Z)", "num_groups":4, "color":"deepskyblue","marker":"D"},
        10: {0:"Native", 1:"Foreign born", "feature":"Nativity (Z)","num_groups":2},
        14: {0: "Male", 1:"Female", "feature":"Gender (Z)","num_groups":2,"color":"royalblue", "marker":"s"},
        15: {0: "White", 1:"Black or African American", 2:"American Indian or Alaska", 3:"Asian, Native Hawaiian or other", "feature":"Race code (Z)","num_groups":4, "color":"lightcoral","marker":"h"},
        1: {0:"No diploma", 1:"diploma", 2:"Associate or Bachelor degree", 3: "Masters or Doctorate degree", "feature":"Educational attainment (Z)","num_groups":4},
        0: {0:"0-25", 1:"26-50", 2:"51-75", 3:"76-99", "feature":"Age (Z)","num_groups":4, "color":"seagreen","marker":"^"}
    }
algorithm_labels = {}
algorithm_colors = {}
algorithm_markers = {}
metric_labels = {"group_accuracy": r'$\Pr(Y=\hat{Y}|Z)$', "n_bins":r'$|\mathcal{B}|$',"accuracy":r'$\Pr(\hat{Y} = Y)$', "num_selected": r'Shortlist Size',\
                 "alpha":r'$\alpha_{WGC}$', "tpr":"True Positive Rate", "group_tpr":"True Positive Rate", "log_loss":"Cross Entropy Loss"\
                 ,"prob_pred":r"$\Pr(\hat{Y}=1|X)$","ECE":"ECE", "sharpness":"Sharpness Score", "f1_score":"F1 Score","group_num_in_bin":"Discrimination prob","pool_discriminated":"Pool Discrimination prob"}
xlabels = {"n_bins":r'$|\text{Range($f$)}|$', "fpr":"False Positive Rate", "group_fpr":"False Positive Rate",\
           "prob_true":r"$\Pr(Y=1|X)$", }

group_colors = {0: "tab:purple", 1: "tab:pink", 2: "tab:cyan", 3: "tab:olive" }
umb_colors = {5: "tab:orange", 10: "tab:brown", 15: "tab:pink", 20: "tab:gray", 25: "tab:olive", 30:"tab:blue"}
umb_markers = {5: 4, 10: 5, 15: 6, 20: 7, 25: 8, 30:9}
group_markers = {0:4, 1: 5, 2: 6, 3: 7, 4: 8}

def collect_results_normal_exp(result_path, exp_parameter, algorithm, results, metrics):
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
    for metric in metrics:
        results[exp_parameter][algorithm][metric]["values"].append(result[metric][exp_parameter])


def collect_results_quantitative_exp(result_path, exp_parameter, algorithm, results, metrics,k_idx=None):
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
        # print(algorithm,result.keys())
    for metric in metrics:
        if metric=="num_selected":
            results[exp_parameter][algorithm][metric]["values"].append(result[metric][k_idx])
        else:
            results[exp_parameter][algorithm][metric]["values"].append(result[metric])
