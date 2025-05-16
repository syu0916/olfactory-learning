
import shutil
from preprocess import get_data
import os
# from models import GenomeMLP, test, train_best_model 
from model_tune import GenomeMLP

import tensorflow as tf
import hyperparameters as hp
# from tensorboard import data
import numpy as np
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
import pickle
import warnings
import keras_tuner as kt

# Suppress specific warning
warnings.filterwarnings("ignore", message="build_fn will be renamed to model in a future release")



# def main_train():
    
#     train_data, test_data = get_data()
#     hp = kt.HyperParameters()
#     hypermodel = GenomeMLP()
#     model = hypermodel.build(hp)
#     hypermodel.fit(hp, model, test_data, train_data)

# def main_train():
#     train_data, test_data = get_data()
#     tuner = kt.RandomSearch(
#         GenomeMLP(),
#         objective="val_accuracy",  # Ensure tuning is based on validation accuracy
#         max_trials=50,  # Set how many trials to run
#         executions_per_trial=1,  # Number of times each model is trained per trial
#         directory="kt_tuning",
#         project_name="genome_mlp"
#     )

#     tuner.search(train_data, validation_data=test_data)
    
#     # Retrieve the best model
#     best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
#     best_model = tuner.hypermodel.build(best_hps)

#     # Train the best model on the full dataset
#     best_model.fit(train_data, validation_data=test_data, epochs=best_hps.get("epochs"))

#     return best_model

def main_train():
    # Define a unique tuner directory
    # ["30", "90"] 
    for time in [ "30", "90"]:
        for ct in ["IN_CGE_NPY", "IN_CGE_VIP", "IN_MGE-Pvalb",  "IN_MGE-SST", "Pyrlayer2a/b", "Pyrlayer3", "SL1",  "SL2",  "Vglut2"]:
    # for time in [ "90"]:
    #     for ct in [ "Pyrlayer2a/b", "Pyrlayer3", "SL1",  "SL2",  "Vglut2"]:
            train_data, test_data = get_data(time, ct)

            tuner_dir = f"tuner_logs/{time}min-{ct}/"
            if os.path.exists(tuner_dir):
                shutil.rmtree(tuner_dir)  # Removes the entire tuner directory and its contents
                print(f"Deleted previous tuner directory: {tuner_dir}")
            tuner = kt.RandomSearch(
                GenomeMLP(),
                objective="val_accuracy",
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
            best_model.save(f"ct-models/{time}min-{ct}-model.h5")

            del best_model
            del tuner


def test_from_pkl():
    model = GenomeMLP()

    with open('pickles/test_data.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('pickles/test_labels.pkl', 'rb') as f:
        obs_test = pickle.load(f)

    X_test = X_test.X.toarray()

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test.astype(np.float32), obs_test.astype(np.int64)))
    test_dataset = test_dataset.batch(hp.batch_size)

    # train_data, validate_data, test_data = get_data()
    print("model data split properly")

    # Build the model
    model(tf.keras.Input(shape=( 14050,)))
    model.load_weights('weights/best_weights.h5')
    model.summary()

    model.compile(
        optimizer = model.optimizer, 
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False), 
        metrics = [["sparse_categorical_accuracy"]]
    )
    print("model compiled properly, now testing.")

    # train(model, train_data, validate_data)

    # test(model, test_dataset)

main_train()
# test_from_pkl()