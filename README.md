# On the Within-Group Discrimination of Screening Classifiers

### Create Environment
On a machine with [conda](https://docs.conda.io/en/latest/) installed, run
```angular2html
conda env create -f environment.yml
source activate wgm
```
The parameters required for every set of experiments are in params_*.py files. 
Make sure you set prepare_data = True only the first time you run the experiments to download and prepare the data.
Set it to False if you have the data already downloaded. 

If you are using a machine with [Slurm](https://slurm.schedmd.com/documentation.html) workload manager set submit = True in params_*.py. 
You can then increase the n_runs parameter which specifies the number of runs.
Set submit = False if you run the experiments on your local machine and make sure the number of runs is small.

### Run Experiments

```angular2html
python ./scripts/run_exp_violations.py
python ./scripts/run_exp_discrimination.py
python ./scripts/run_exp_bins.py
```

### Plot Figures

#### To generate Figures 1, 5, 9
Run
```angular2html
python ./scripts/plot_exp_violations.py
```

#### To generate Figures 2, 7
Run
```angular2html
python ./scripts/plot_discrimination.py
python ./scripts/plot_exp_group_discrimination.py
```

#### To generate Figures 3, 4, 8
Run
```angular2html
python ./scripts/plot_exp_bins.py
```

#### To generate Figures 6
Run
```angular2html
python ./scripts/plot_wgc_eps.py
```
