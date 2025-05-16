
import h5py
from sklearn.model_selection import train_test_split
import hyperparameters as hp
import numpy as np
from models import GenomeMLP, test
import shap.maskers
import tensorflow as tf
from keras.models import load_model
import hyperparameters as hp
import numpy as np
import shap
import pickle

def interpret():
    with open('explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    with open('validation.pkl', 'rb') as f:
        val_data = pickle.load(f)
    shap_vals = explainer.shap_values(val_data)

    with open('shap_vals.pkl', 'wb') as f:
        pickle.dump(shap_vals, f)

# def read_in_model(which_model, layers, dropout):
#     model = GenomeMLP(layers, layers, layers, layers, dropout, dropout)


#     model.model(tf.keras.Input(shape=(hp.num_features, )))
#     model.model(tf.zeros((1, hp.num_features)))

#     model.model.load_weights(f'{which_model}-model/checkpoint.weights.h5')
#     model.summary()

#     model.build(input_shape=(None, hp.num_features))
#     model.compile(
#         optimizer = "adam", 
#         loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False), 
#         metrics = [["sparse_categorical_accuracy"]]
#     )

#     return model 

def read_in_model(time, ct):
    model = load_model(f"ct-models/{time}min-{ct}-model.h5")
    model.evaluate(data[time + ct][2].X, data[time + ct][3])

    return model


def main(which_model, layers, dropout):

    # loading in the model
    model = read_in_model(which_model=which_model, layers = layers, dropout = dropout)
    print("UNLOAIDNG PICKLES")
    with open(f'pickles/{which_model}-data/train_data.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open(f'pickles/{which_model}-data/train_labels.pkl', 'rb') as f:
        obs_train = pickle.load(f)

    # reading in the data
    with open(f'pickles/{which_model}-data/test_data.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open(f'pickles/{which_model}-data/test_labels.pkl', 'rb') as f:
        obs_test = pickle.load(f)

    tf.config.run_functions_eagerly(True)

    # sample data
    subset_indices = np.random.choice(len(X_train), size=1000, replace=False)
    X_train_subset = X_train[subset_indices, :]
    print(X_train_subset.shape)
    X_train_subset = np.array(X_train_subset.X.toarray())

    # instantiating shap explainer
    dummy_input = tf.zeros(shape=(1, hp.num_features))
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    explainer = shap.DeepExplainer(model.model, X_train_subset)

    tf.config.run_functions_eagerly(False)  


    # for each shap generation, try to get around 500 shap values.
    print(X_test)
    # the entire test set is about 19000
    _, shap_subset, _, _ = train_test_split(
        X_test.X.toarray(), 
        obs_test, 
        test_size = 0.1,
        random_state = hp.random_seed,
        stratify=obs_test
    )
    # shap_subset = shap_subset.raw.X.toarray()
    # shap_subset = np.expand_dims(shap_subset.raw.X.toarray(), axis = 2)

    # getting shap vals
    # task 1: generate shap values for the entire test set
    print("shap created: getting shap values")
    shap_vals = explainer.shap_values(shap_subset)
    print("shap values successfully got")
    with open(f'pickles/{which_model}-data/aggregate_shap.pkl', 'wb') as f:
        pickle.dump(shap_vals, f)
    with open(f'pickles/{which_model}-data/aggregate_shap_samples.pkl', 'wb') as f:
        pickle.dump(shap_subset, f)

    # prepping intersection samples

    # condition = {
    #     0:[], # early
    #     # 3:[], # early
    #     # 5:[], # early
    #     # 7:[], # early
    #     1:[], # late
    #     # 4:[], # late
    #     # 6:[], # late
    #     # 8:[] #late
    # } # early conditions

    # condition_intersection = [0, 1, 2, 3, 4, 5] # specifically in hibitory cells

    # preds = model.predict(X_test.raw.X.toarray())
    # preds = np.argmax(tf.nn.softmax(preds, axis = 1), axis = 1)
    # for cluster in condition_intersection:
    #     condition = {
    #     # 2:[], # bored
    #     # 0:[], # early
    #     3:[], # early
    #     # 5:[], # early
    #     7:[], # early
    #     # 1:[], # late
    #     4:[], # late
    #     # 6:[], # late
    #     8:[] #late
    #     }
    #     for i, pred in zip(X_test, preds):
    #         if pred in condition and len(condition[pred]) < 100 and int(i.obs['leiden']) == cluster:
    #             condition[pred].append(i.raw.X.toarray())
        

    #     for i in condition:
    #         samples = np.squeeze(condition[i])
    #         shap_vals = explainer.shap_values(samples)

    #         with open(f'intersect_shap/intersect_cluster_{cluster}_condition_{i}.pkl', 'wb') as f:
    #             pickle.dump(shap_vals, f)
    #         with open(f'intersect_shap/intersect_cluster_{cluster}_condition_{i}_samples.pkl', 'wb') as f:
    #             pickle.dump(condition[i], f)



    # task 2: generate shap vals for each leiden cluster
    for i in X_test.obs['leiden'].unique():
        cluster_samples = X_test[X_test.obs['leiden'] == i]

        # max 100 samples for each cluster
        sample_size = 250 if len(cluster_samples) > 250 else len(cluster_samples)
        indices = np.random.choice(len(cluster_samples), sample_size)
        cluster_samples = cluster_samples.X.toarray()[indices]
        # cluster_samples = np.expand_dims(cluster_samples, axis = 2)
        shap_vals = explainer.shap_values(cluster_samples)
        with open(f'pickles/{which_model}-data/leiden_cluster_{i}_shap.pkl', 'wb') as f:
            pickle.dump(shap_vals, f)
        with open(f'pickles/{which_model}-data/leiden_cluster_{i}_shap_data_subset.pkl', 'wb') as f:
            pickle.dump(cluster_samples, f)

    # task 3: generate shap vals for each condition...?
    for i in X_test.obs['encoded_condition'].unique():
        cluster_samples = X_test[X_test.obs['encoded_condition'] == i]

        # max 100 samples for each cluster
        sample_size = 250 if len(cluster_samples) > 250 else len(cluster_samples)
        indices = np.random.choice(len(cluster_samples), sample_size)
        cluster_samples = cluster_samples.X.toarray()[indices]
        # cluster_samples = np.expand_dims(cluster_samples, axis = 2)
        shap_vals = explainer.shap_values(cluster_samples)

        with open(f'pickles/{which_model}-data/encoded_condition_{i}_shap.pkl', 'wb') as f:
            pickle.dump(shap_vals, f)
        with open(f'pickles/{which_model}-data/encoded_condition_{i}_shap_data_subset.pkl', 'wb') as f:
            pickle.dump(cluster_samples, f)

    print("successfully dumped")

    # print(X_train_subset.shape)
    # explainer = shap.DeepExplainer(model.architecture, X_train_subset)

    # print("creating shap")
    # X_test = np.squeeze(X_test)
    # X_test = np.expand_dims(X_test, axis=2)
    # print(X_test.shape)
    # shap_vals = explainer.shap_values(X_test)


    # print("pickling shap")
    # with open('shap_vals_compressed.pkl', 'wb') as f:
    #     pickle.dump(shap_vals, f)

    # # print("pickling explainer")
    # # with open('explainer', 'wb') as f:
    # #     pickle.dump(explainer, f)

    # print("pickiling validation values.")
    # with open('validation.pkl', 'wb') as f:
    #     pickle.dump(X_test, f)


['30IN_CGE_NPY', '30IN_CGE_VIP', '30IN_MGE-Pvalb', '30IN_MGE-SST', '30Pyrlayer2a/b', '30Pyrlayer3', '30SL1', '30SL2', '30Vglut2', '90IN_CGE_NPY', '90IN_CGE_VIP', '90IN_MGE-Pvalb', '90IN_MGE-SST', '90Pyrlayer2a/b', '90Pyrlayer3', '90SL1', '90SL2', '90Vglut2']

main("30min", 1024, 0.3)
main("90min", 1024, 0.1)
# interpret()