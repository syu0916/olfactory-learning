from sklearn.model_selection import train_test_split
import hyperparameters as hp
import numpy as np
import tensorflow as tf
from keras.models import load_model
import hyperparameters as hp
import numpy as np
import shap
import pickle
import pandas as pd
from collections import defaultdict
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
import seaborn as sns

NUM_SHAP_FEATURES = 40
np.random.seed(hp.random_seed)

def generate_values(time, condition_model, celltype_model, X_train, X_test, obs_test):

    X_train = X_train[0].X.toarray()
    X_test = X_test[0]
    print(X_train)
    print(obs_test)
    tf.config.run_functions_eagerly(True)

    X_train_subset = X_train
    if len(X_train) > 1000:
        # sample data
        subset_indices = np.random.choice(len(X_train), size=1000, replace=False)
        X_train_subset = X_train[subset_indices, :]
        print(X_train_subset.shape)
        # X_train_subset = np.array(X_train_subset.X.toarray())


    # instantiating shap explainer
    dummy_input = tf.zeros(shape=(1, hp.num_features))
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough


    print(model.evaluate(X_test.X.toarray(), obs_test))
    cond_explainer = shap.DeepExplainer(condition_model, X_train_subset)
    type_explainer = shap.DeepExplainer(celltype_model, X_train_subset)
    tf.config.run_functions_eagerly(False)  

    if len(X_test) > 300:
        subset_indices = np.random.choice(len(X_test), size=300, replace=False)
        X_test = X_test[subset_indices].copy()
    
    # task 1: generate shap values for the entire test set
    print("shap created: getting shap values")
    cond_shap_vals = cond_explainer.shap_values(X_test.X.toarray())
    type_shap_vals = type_explainer.shap_values(X_test.X.toarray())

    # generate the top shap values. 
    feature_names = X_test.var.index.tolist()
    shap_vals = np.array(cond_shap_vals).reshape((3, len(cond_shap_vals), 3000))
    mean_shap_vals = np.abs(shap_vals).mean(axis = 1)

    total_shap = mean_shap_vals.sum(axis=0)
    sorted_indices = np.argsort(total_shap)[::-1][:NUM_SHAP_FEATURES]  # Top 40 features
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_shap_vals = mean_shap_vals[:, sorted_indices]  # Keep top features

    
    return sorted_features, sorted_shap_vals
    
    



final_df = pd.DataFrame()
for t in ["30", "90"]:

    df = pd.DataFrame()
    shap_vals = {}
    gene_rank_score = defaultdict(float)
    gene_counts = defaultdict(int)


    with open(f'pickles/{t}min-data/train_data.pkl', 'rb') as f:
        X_train = pickle.load(f)

    # reading in the data
    with open(f'pickles/{t}min-data/test_data.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open(f'pickles/{t}min-data/test_labels.pkl', 'rb') as f:
        obs_test = pickle.load(f)

    model = load_model(f"ct-models/{t}min-model.h5")

    condition_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("condition-out").output
    )

    # Get submodel for cell type prediction
    celltype_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("cell-type-out").output
    )

    for i in range(50):
        print(f"trial {i}")
        sorted_features, sorted_shap_vals = generate_values(t, condition_model, celltype_model, X_train, X_test, obs_test)
        sorted_shap_vals = np.array(sorted_shap_vals)
        for feat in range(len(sorted_features)):
            if sorted_features[feat] not in shap_vals:
                shap_vals[sorted_features[feat]] = [sorted_shap_vals[:, feat]]
            else:
                shap_vals[sorted_features[feat]].append(sorted_shap_vals[:, feat])

            # Score genes: higher rank => higher score
            score = 40 - feat if feat < 40 else 0
            gene_rank_score[sorted_features[feat]] += score
            gene_counts[sorted_features[feat]] += 1

    print(gene_rank_score)
    print(gene_counts)
    for feat in shap_vals:
        shap_vals[feat] = np.mean(np.absolute(shap_vals[feat]))

    # Sort by rank score (with count as tiebreaker)
    sorted_genes = sorted(
        gene_rank_score.items(),
        key=lambda x: (-x[1], -gene_counts[x[0]])
    )

    top_40_genes = [gene for gene, _ in sorted_genes[:40]]

    final_df[f'{t}min: aggregate'] = top_40_genes

    top_40_shap_vals = np.stack([shap_vals[feat] for feat in top_40_genes])

    # Define class colors
    class_labels = ["Naive",  "Quinine", "Sucrose",]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    bottom = np.zeros(len(top_40_genes))  # Initialize bottom stack

    for i, (class_name, color) in enumerate(zip(class_labels, colors)):
        ax.barh(top_40_genes, top_40_shap_vals[i], left=bottom, label=class_name, color=color)
        bottom += sorted_shap_vals[i]  # Stack bars

    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_title(f"{t}-min: aggregate")
    ax.legend(title="Condition Class")
    plt.gca().invert_yaxis()  # Highest feature at the top
    plt.show()
    plt.savefig(f"csvs/{t}_min_aggregate_shap.png", dpi=300, bbox_inches='tight')

final_df.to_csv("csvs/top_features.csv")



