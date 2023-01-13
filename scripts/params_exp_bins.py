exp_token = "cz"
exp_dir = "./exp_cal_curve"
q_ratio = "0.2"
test_ratio = "0.5"
prepare_data = False
submit = True
split_size = 1000
n_test = 100
k = 5
# Z = [[2],[4],[10],[6],[14]]   #valid ones 2,4,6,10,14
Z = [[6],[15],[14],[0],[4]]#,[15]]#,[0],[14]]#,[1],[2],[4],[6],[14],[15]]
# Z = [[14],[15]]
# Z = [[2],[6],[10],[14]]   #valid ones
n_runs = 100
n_runs_test = 100#000
n_train = 100000
n_trains = [100000]
noise_ratio = -1
noise_ratios = [noise_ratio]
n_cals = [50000]#[10000, 50000, 100000, 500000,1000000]#, 2000, 5000, 10000, 20000, 50000, 100000]
n_cals_label = ["5e4"]#, "5e4", "1e5", "5e5", "1e6"]
runs = list(range(n_runs))
classifier_type = "LR"
lbd = "1e-6"
lbds = ["1e-6"]
umb_num_bins = [5,10,15,20,25,30,35,40]

ks = [5,10,15]



#0: age
#1: school
#2: marital status
#4: disability
#6: citizenship
#14: sex
#15: race