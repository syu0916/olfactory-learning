# olfactory-learning
This repo contains all of the work that has been done thus far in building multi layer perceptrons of different configurations for the task of predicting cell experience (naive, quinine sucrose) given scRNA seq data from mouse piriform cortex neurons. The repo has four subdirectories, all of which have essentially the same structure and files:
* mlp_base
* mlp_multi-input
* **mlp_multi-label**
* mlp_timepoints

of all of the subdirectories, `mlp_multi-label` is the most developed and has the most recent and up-to-date updates with regards to shap experiments and scripts. Within the `mlp_multi-label` directory, there are a couple of directories and files:
* `csvs`: a directory that contains csvs of all of the top genes and results from other scripts
* `ct-models` (not in the github repo, but is referenced and will need to be made using `mkdir`): a directory that contains the `.h5` files for trained models
* `notebooks`: contains exploratory jupyter notebooks used for analysis of models and shap values. There are a couple of different notebooks that are regrettably a little messy:
    * `exploration.ipynb`: primarily used to explore the dataset and visualize venn diagrams about what the split between conditions/timepoints looked like
    * `interpretation.ipynb`: the biggest and most comprehensive notebook. primarily used to make visualizations related to shap values and confusion matrices for the model. 
    * `Variance.ipynb`: deprecated.
* `pickles` (not in the github repo, but is referenced and will need to be made using `mkdir`): where all of the pickles, used in saving train/test split data, shap values, etc. is stored. 
* python files: all of the python files used for model building, training, etc. have been documented with block comments at the top of the file in the **`mlp_multi-label`** directory. reference these comments to understand what the purpose of the file is. 
* `*.sh` files: most of them are used to interface with the gpu partition. `gpu_batch.sh` was what I was using to train my models, `interpretation_batch.sh` was what I was using to run shap analysis, but in reality both scripts do pretty much the same thing.

## trianing the model:
1. run `sbatch gpu_batch.sh` in terminal
2. see the slurm output: there will probably be a lot of errors regarding to unknown file paths
3. resolve file path issues and make dependent directories if needed (ie `pickles`, `ct-models`, etc.)

## doing analysis: 
1. run the shap script by running `sbatch interpretation_batch.sh`
2. reference `notebooks/interpretation.ipynb` to utilize existing shap analysis code.

If after examining this repo, things don't make sense, please let me know :D

