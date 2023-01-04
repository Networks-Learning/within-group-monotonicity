exp_token = "cz"
exp_dir = "./exp_violations"
q_ratio = "0.2"
test_ratio = "0.5"
prepare_data = False
submit = False
split_size = 1000
n_test = 100
k = 10
# Z = [[2],[4],[10],[6],[14]]   #valid ones 2,4,6,10,14
Z = [[6],[15]]
# Z = [[14],[15]]
# Z = [[2],[6],[10],[14]]   #valid ones
n_runs = 1
n_runs_test = 1
n_train = 100000
n_trains = [100000]
noise_ratio = -1
noise_ratios = [noise_ratio]
n_cals = [50000]
n_cals_label = ["5e4"]
runs = list(range(n_runs))
classifier_type = "LR"
lbd = "1e-6"
lbds = ["1e-6"]
umb_num_bins = [5,10,15,20,25,30]#, 2, 3, 4, 5]
umb_colors = {5: "tab:orange", 10: "tab:brown", 15: "tab:pink", 20: "tab:gray", 25: "tab:olive", 30:"tab:blue"}
group_markers = {0:4, 1: 5, 2: 6, 3: 7, 4: 8}
group_colors = {0: "tab:purple", 1: "tab:pink", 2: "tab:cyan", 3: "tab:olive" }
lim_num_groups = 20
umb_markers = {5: 4, 10: 5, 15: 6, 20: 7, 25: 8, 30:9}



#0: age
#1: school
#2: marital status
#4: disability
#6: citizenship
#14: sex
#15: race