import scanpy as sc
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import data
import hyperparameters as hp
import numpy as np
import pickle
import pandas as pd
import anndata
import os

"""
    pretty self-explanatory. gets all of the data and preprocess it.
"""

def get_data(TIME_POINT):
    np.random.seed(hp.random_seed)
    adata = sc.read_h5ad("../antonscode_neurons_with_raw.h5ad")
    
    # Filtering out all of the Gm, mt, rik genes.
    var_mask = (~adata.var_names.str.contains('Gm')) & \
            (~adata.var_names.str.contains('mt-')) & \
            (~adata.var_names.str.contains('Rik'))
    
    mask = (adata.obs['condition'] == f"naive30") |\
            (adata.obs['condition'] == f"sucrose30") |\
            (adata.obs['condition'] == f"quinine30") |\
            (adata.obs['condition'] == f"naive90") |\
            (adata.obs['condition'] == f"sucrose90") |\
            (adata.obs['condition'] == f"quinine90")

    adata_masked = adata[mask, var_mask]
    adata_masked.raw = adata.raw.to_adata()[mask, var_mask]
    adata = adata_masked
    del adata_masked

    # Reducing the feature space to top 3000 most variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=hp.num_features)
    subset_adata = adata[:, adata.var["highly_variable"]]
    subset_adata.raw = subset_adata.raw.to_adata()[:, adata.var["highly_variable"]]

    mask = (subset_adata.obs['condition'] == f"naive{TIME_POINT}") |\
           (subset_adata.obs['condition'] == f"sucrose{TIME_POINT}") |\
           (subset_adata.obs['condition'] == f"quinine{TIME_POINT}")
    
    final_adata = subset_adata[mask].copy()

    condition_encoder = LabelEncoder()
    final_adata.obs['encoded_condition'] = condition_encoder.fit_transform(final_adata.obs['condition'])

    type_encoder = LabelEncoder()
    final_adata.obs['encoded_ct'] = type_encoder.fit_transform(final_adata.obs['cell_type_lvl_2'])

    # Extract input data (X) and labels
    X = final_adata
    CONDITIONS = final_adata.obs['encoded_condition']
    TYPES = final_adata.obs['encoded_ct']

    print(TYPES)

    # One-hot encode the 'cell_type_lvl_2'
    ct_onehot = pd.get_dummies(final_adata.obs['cell_type_lvl_2'])

    # Split into train and test sets
    train_idx, test_idx = train_test_split(
        range(final_adata.n_obs),
        test_size=0.20,
        random_state=hp.random_seed,
        stratify=final_adata.obs['encoded_condition']
    )

    # data
    X_train, X_test = X[train_idx], X[test_idx]
    ct_onehot_train, ct_onehot_test = ct_onehot.iloc[train_idx], ct_onehot.iloc[test_idx]

    # labels
    cond_train, cond_test = CONDITIONS[train_idx], CONDITIONS[test_idx]
    ct_train, ct_test = TYPES[train_idx], TYPES[test_idx]



    with open(f'pickles/{TIME_POINT}min-data/train_data.pkl', 'wb') as f:
        pickle.dump([X_train, ct_onehot_train], f)
    with open(f'pickles/{TIME_POINT}min-data/train_labels.pkl', 'wb') as f:
        pickle.dump([cond_train, ct_train], f)

    with open(f'pickles/{TIME_POINT}min-data/test_data.pkl', 'wb') as f:
        pickle.dump([X_test, ct_onehot_test], f)
    with open(f'pickles/{TIME_POINT}min-data/test_labels.pkl', 'wb') as f:
        pickle.dump([cond_test, ct_test], f)

    X_train = X_train.X.toarray()
    X_test = X_test.X.toarray()



    # Convert to TensorFlow Datasets
    # train_input = {"input1": X_train.astype(np.float32), "input2": ct_onehot_train.astype(np.float32)}
    # test_input = {"input1": X_test.astype(np.float32), "input2": ct_onehot_test.astype(np.float32)}
    train_input = X_train.astype(np.float32)
    test_input = X_test.astype(np.float32)

    train_label = {"condition-out": cond_train.astype(np.int64), "cell-type-out": ct_train.astype(np.int64)}
    test_label = {"condition-out": cond_test.astype(np.int64), "cell-type-out": ct_test.astype(np.int64)}

    # Create TensorFlow datasets with shuffling and batching
    train_dataset = data.Dataset.from_tensor_slices((train_input, train_label))
    test_dataset = data.Dataset.from_tensor_slices((test_input, test_label))

    buffer_size = 10000
    train_dataset = train_dataset.shuffle(buffer_size).batch(hp.batch_size)
    test_dataset = test_dataset.batch(hp.batch_size)

    return train_dataset, test_dataset

