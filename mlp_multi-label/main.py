
import shutil
from preprocess import get_data
import os
# from models import GenomeMLP, test, train_best_model 
from model_tune import GenomeMLP

import tensorflow as tf
import hyperparameters as hp
import pandas as pd
# from tensorboard import data
import numpy as np
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
import pickle
import warnings
import keras_tuner as kt

"""
    main file used to run and train the file.
"""

def main_train():
    # Define a unique tuner directory
    # ["30", "90"] 
    # for time in [ "30", "90"]:
    #     for ct in ["IN_CGE_NPY", "IN_CGE_VIP", "IN_MGE-Pvalb",  "IN_MGE-SST", "Pyrlayer2a/b", "Pyrlayer3", "SL1",  "SL2",  "Vglut2"]:
    for time in ["30", "90"]:
        train_data, test_data = get_data(time)

        tuner_dir = f"tuner_logs/{time}min/"
        if os.path.exists(tuner_dir):
            shutil.rmtree(tuner_dir)  # Removes the entire tuner directory and its contents
            print(f"Deleted previous tuner directory: {tuner_dir}")
        tuner = kt.RandomSearch(
            GenomeMLP(),
            objective="val_loss",
            max_trials=10,  # Ensure each search does 10 trials
            executions_per_trial=1,
            directory=tuner_dir,  # Use a unique directory per run
            project_name="model_tuning"
        )
        
        tuner.search(train_data, validation_data=test_data)
        
        # Retrieve the best model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)

        # Train the best model on the full dataset
        best_model.fit(train_data, validation_data=test_data, epochs=best_hps.get("epochs"))
        best_model.save(f"ct-models/{time}min-model.h5")

        del best_model
        del tuner



main_train()
# test_from_pkl()