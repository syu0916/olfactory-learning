
import hyperparameters as hp
import numpy as np
import tensorflow as tf
from keras.models import load_model
import hyperparameters as hp
import numpy as np
import shap
import pickle

np.random.seed(hp.random_seed)

def generate_values(time, ct):

    # loading in the model
    model = load_model(f"ct-models/{time}min-{ct}-model.h5")
    print("UNLOAIDNG PICKLES")
    with open(f'pickles/{time}min-data-{ct}/train_data.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open(f'pickles/{time}min-data-{ct}/train_labels.pkl', 'rb') as f:
        obs_train = pickle.load(f)

    # reading in the data
    with open(f'pickles/{time}min-data-{ct}/test_data.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open(f'pickles/{time}min-data-{ct}/test_labels.pkl', 'rb') as f:
        obs_test = pickle.load(f)

    model.evaluate(X_test.X, obs_test)
    tf.config.run_functions_eagerly(True)

    X_train_subset = X_train.X
    if len(X_train) > 1000:
        # sample data
        subset_indices = np.random.choice(len(X_train), size=1000, replace=False)
        X_train_subset = X_train[subset_indices, :]
        print(X_train_subset.shape)
        X_train_subset = np.array(X_train_subset.X.toarray())

    # instantiating shap explainer
    dummy_input = tf.zeros(shape=(1, hp.num_features))
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    explainer = shap.DeepExplainer(model, X_train_subset)

    tf.config.run_functions_eagerly(False)  

    if len(X_test) > 300:
        subset_indices = np.random.choice(len(X_test), size=300, replace=False)
        X_test = X_test[subset_indices].copy()


    # task 1: generate shap values for the entire test set
    print("shap created: getting shap values")
    shap_vals = explainer.shap_values(X_test.X)
    print("shap values successfully got")
    with open(f'pickles/{time}min-data-{ct}/aggregate_shap.pkl', 'wb') as f:
        pickle.dump(shap_vals, f)
    with open(f'pickles/{time}min-data-{ct}/aggregate_shap_samples.pkl', 'wb') as f:
        pickle.dump(X_test, f)



mods = ["IN_CGE_NPY", "IN_CGE_VIP", "IN_MGE-Pvalb",  "IN_MGE-SST", "Pyrlayer2a/b", "Pyrlayer3", "SL1",  "SL2",  "Vglut2"]

for i in ["30", "90"]:
    for j in mods:
        generate_values(i, j)
# interpret()