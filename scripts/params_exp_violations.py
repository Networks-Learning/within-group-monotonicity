exp_token = "cz"
exp_dir = "./exp_violations"
q_ratio = "0.2"
test_ratio = "0.5"
prepare_data = False
submit = False
split_size = 1000
n_test = 100
k = 5
# Z = [[2],[4],[10],[6],[14]]   #valid ones 2,4,6,10,14
Z = [[6],[15],[14],[4]]#,[15],[0],[14]]
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
umb_num_bins =[15]# [5,10,15,20,25,30]#, 2, 3, 4, 5]

ks = [5,10,15]

#0: age
#1: school
#2: marital status
#4: disability
#6: citizenship
#14: sex
#15: race