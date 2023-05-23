# On the Within-Group Fairness of Screening Classifiers
This is a repository containing code for the paper:

> N. Okati, S. Tsirtsis, and M. Gomez Rodriguez. **_On the Within-Group Discrimination of Screening Classifiers._**

The paper is available [here](https://arxiv.org/abs/2302.00025).
### Pre-requisites
To install all the requirements, on a machine with [conda](https://docs.conda.io/en/latest/) installed, run
```angular2html
conda env create -f environment.yml
source activate wgm
```

### Preparation
The parameters required for every set of experiments are in ./scripts/params_*.py files. 
The current settings are those used in our experiments.
The first time that you run the experiments the data will be downloaded and saved in the data folder. 

If you are using a machine with [Slurm](https://slurm.schedmd.com/documentation.html) workload manager set submit = True in ./scripts/params_*.py. 
You can then increase the n_runs parameter which specifies the number of runs. We used 100 in our experiments.
Set submit = False if you run the experiments on your local machine and make sure the number of runs is small.

### Execution

```angular2html
python ./scripts/run_exp_bins.py
```

### Generate Figures

#### To generate Figures 1, 5, 9
```angular2html
python ./scripts/plot_exp_violations.py
```

#### To generate Figures 2, 7
```angular2html
python ./scripts/plot_discrimination.py
python ./scripts/plot_exp_group_discrimination.py
```

#### To generate Figures 3, 4, 8
```angular2html
python ./scripts/plot_exp_bins.py
```

#### To generate Figures 6
```angular2html
python ./scripts/plot_wgc_eps.py
```

## Citation
If you use parts of the code in this repository for your own research, please consider citing:

```
@article{okati2023on,
        title={On the Within-Group Discrimination of Screening Classifiers},
        author={Okati, Nastaran and Tsirtsis, Stratis and Gomez-Rodriguez, Manuel},
        journal={arXiv preprint arXiv:2302.00025},
        year={2023}
}
```
