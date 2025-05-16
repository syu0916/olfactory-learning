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


def get_data(TIME_POINT, CELL_TYPE = ""):
    np.random.seed(hp.random_seed)
    adata = sc.read_h5ad("../antonscode_neurons_with_raw.h5ad")
    
    # adata_masked_raw = adata.raw.X[:, shuffled_indices]
    # adata_masked_raw = adata_masked_raw[:, mask]
    # adata_masked.raw=anndata.AnnData(X=adata_masked_raw, var=adata.raw.var[mask], obs=adata.obs)


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

    # Apply the mask to adata (both the main and raw data)
    adata_masked = adata[mask, var_mask]
    adata_masked.raw = adata.raw.to_adata()[mask, var_mask]


    # Create a filtered version of adata.raw
    adata = adata_masked
    del adata_masked


    # reducing the feature space to top 3000 most variable genes
    adata.X = adata.raw.X.toarray()
    sc.pp.highly_variable_genes(adata, n_top_genes=hp.num_features, )
    subset_adata = adata[:, adata.var["highly_variable"]]
    subset_adata.raw = subset_adata.raw.to_adata()[:, adata.var["highly_variable"]]

    # now splitting out the 30-min time points.
    mask = (subset_adata.obs['condition'] == f"naive{TIME_POINT}") |\
        (subset_adata.obs['condition'] == f"sucrose{TIME_POINT}") |\
        (subset_adata.obs['condition'] == f"quinine{TIME_POINT}")
    if CELL_TYPE != "":
        mask = mask & (subset_adata.obs['cell_type_lvl_2'] == CELL_TYPE)
    print(mask)
    print(CELL_TYPE)
    print(subset_adata.obs['cell_type_lvl_2'])
    
    
    final_adata = subset_adata[mask].copy()

    label_encoder = LabelEncoder()
    
    # encode the data
    final_adata.obs['encoded_condition'] = label_encoder.fit_transform(final_adata.obs['condition'] )

    del subset_adata
    print(final_adata)

    # # One-hot encode the Leiden clusters
    # leiden_onehot = pd.get_dummies(final_adata.obs['cell_type_lvl_2'], prefix='leiden')
    
    # # Ensure leiden_onehot index matches final_adata.obs
    # leiden_onehot = leiden_onehot.reindex(final_adata.obs.index, fill_value=0)

    # var_names = list(final_adata.var_names) + ["leiden_IN_CGE_NPY", "leiden_IN_CGE_VIP", "leiden_IN_MGE-Pvalb",  "leiden_IN_MGE-SST", "leiden_Pyrlayer2a/b", "leiden_Pyrlayer3", "leiden_SL1",  "leiden_SL2",  "leiden_Vglut2"]  


    # final_adata = anndata.AnnData(X=np.hstack((final_adata.X, leiden_onehot)), 
    #                               var= var_names, 
    #                               obs=final_adata.obs, 
    #                               uns = final_adata.uns, 
    #                               obsm = final_adata.obsm, 
    #                               )

    train_idx, test_idx = train_test_split(
        range(final_adata.n_obs),  # Use index values
        test_size=0.20,
        random_state=hp.random_seed, 
        stratify=final_adata.obs['encoded_condition']
    )

    # Use indices to subset `final_adata`, preserving metadata
    X_train = final_adata[train_idx].copy()
    obs_train = final_adata.obs['encoded_condition'][train_idx]

    X_test = final_adata[test_idx].copy()
    obs_test = final_adata.obs['encoded_condition'][test_idx]

    print("Pickling Split Values:")
    os.makedirs(f'pickles/{TIME_POINT}min-data-{CELL_TYPE}/', exist_ok=True)
    with open(f'pickles/{TIME_POINT}min-data-{CELL_TYPE}/train_data.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open(f'pickles/{TIME_POINT}min-data-{CELL_TYPE}/train_labels.pkl', 'wb') as f:
        pickle.dump(obs_train, f)


    with open(f'pickles/{TIME_POINT}min-data-{CELL_TYPE}/test_data.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open(f'pickles/{TIME_POINT}min-data-{CELL_TYPE}/test_labels.pkl', 'wb') as f:
        pickle.dump(obs_test, f)
    
    # print("successfully dumped")
    
    # print(X_train.var['gene_ids'])
    # converting all to raw data in prep for the model
    X_train = X_train.X
    X_test = X_test.X


    # return X_train, obs_train, X_test, obs_test
    # Create TensorFlow datasets
    train_dataset = data.Dataset.from_tensor_slices((X_train.astype(np.float32), obs_train.astype(np.int64)))
    test_dataset = data.Dataset.from_tensor_slices((X_test.astype(np.float32), obs_test.astype(np.int64)))

    buffer_size = 10000
    train_dataset = train_dataset.shuffle(buffer_size).batch(hp.batch_size)
    test_dataset = test_dataset.batch(hp.batch_size)

    return train_dataset, test_dataset

