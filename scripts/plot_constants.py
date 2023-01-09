import pickle
params = {'legend.fontsize': 26,#28,
          'xtick.labelsize': 26,
          'ytick.labelsize': 28,
          'lines.markersize': 15,
          'errorbar.capsize': 8.0,
            'axes.labelsize' : 28,
            'text.usetex'  : True,
            'text.latex.preamble': r'\usepackage{amsmath}',
            'font.family': 'serif',
          'axes.titlesize':26,
          }


line_width = 3.0
transparency = 0.1
font_size = 28
capthick = 3.0
dpi = 100
fig_width = 28
fig_height = 6

Z_labels = {
        # 2: {0:"Married",1:"Widowed",2:"Divorced",3:"Separated",4:"Never married"},
        2: {0:"Married or Separated", 1: "Never married", "feature":"Marital status (Z)", "num_groups":2},
        4: {0: "With a disability", 1: "Without a disability", "feature": "Disability record (Z)","num_groups":2},
        6: {0:"Born in the US", 1:"Born in Unincorporated US", 2:"Born abroad", 3:"Not a US citizen", "feature":"Citizenship status (Z)", "num_groups":4},
        10: {0:"Native", 1:"Foreign born", "feature":"Nativity (Z)","num_groups":2},
        14: {0: "Male", 1:"Female", "feature":"Gender (Z)","num_groups":2},
        15: {0: "White", 1:"Black or African American", 2:"American Indian or Alaska", 3:"Asian, Native Hawaiian or other", "feature":"Race code (Z)","num_groups":4},
        1: {0:"No diploma", 1:"diploma", 2:"Associate or Bachelor degree", 3: "Masters or Doctorate degree", "feature":"Educational attainment (Z)","num_groups":4},
        0: {0:"0-25", 1:"26-50", 2:"51-75", 3:"76-99", "feature":"Age (Z)","num_groups":4}
    }
algorithm_labels = {}
algorithm_colors = {}
algorithm_markers = {}
metric_labels = {"group_accuracy": r'$\Pr(Y=\hat{Y}|Z)$', "n_bins":r'$|\mathcal{B}|$',"accuracy":r'$\Pr(\hat{Y} = Y)$', "num_selected": r'Shortlist Size',\
                 "alpha":r'$\alpha_{WGC}$', "tpr":"True Positive Rate", "group_tpr":"True Positive Rate", "log_loss":"Cross Entropy Loss"\
                 ,"prob_pred":r"$\Pr(\hat{Y}=1|X)$","ECE":"ECE", "sharpness":"Sharpness Score", "f1_score":"F1 Score"}
xlabels = {"n_bins":r'$|\text{Range($f$)}|$', "fpr":"False Positive Rate", "group_fpr":"False Positive Rate",\
           "prob_true":r"$\Pr(Y=1|X)$", }



def collect_results_normal_exp(result_path, exp_parameter, algorithm, results, metrics):
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
    for metric in metrics:
        results[exp_parameter][algorithm][metric]["values"].append(result[metric][exp_parameter])


def collect_results_quantitative_exp(result_path, exp_parameter, algorithm, results, metrics):
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
        # print(algorithm,result.keys())
    for metric in metrics:
        results[exp_parameter][algorithm][metric]["values"].append(result[metric])
