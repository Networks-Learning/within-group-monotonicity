exp_token = "cz"
exp_dir = "./exp_bins"
test_ratio = "0.5"
# s = False
submit = True
split_size = 1000
n_test = 100
k = 5
Z = [[6],[15],[14],[4]]
n_runs = 100 # we used 100 in our paper and ran on a machine with 48 CPUs
n_runs_test = 100
n_train = 100000
n_trains = [100000]
n_cals = [50000]
n_cals_label = ["5e4"]
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