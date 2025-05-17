import pandas as pd
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize


"""
    This script is used to generate a heat map of the average gene expression of 
    the top 40 features, identified by shap, on a cell-type x condition basis. 
    Also calculates the variance across such groups to identify which genes are 
    most variable across cell-type x condition.
"""



def generate_variance(data, t):
    top_features = pd.read_csv("csvs/top_features.csv")
    print(top_features)
    average_expressions_df = pd.DataFrame()
    print(data)


    for c in sorted(data.obs['cell_type_lvl_2'].unique()):
        for cond in sorted(data.obs['condition'].unique()):
            cluster_samples = data[(data.obs['cell_type_lvl_2'] == c) & (data.obs['condition'] == cond)]
            cluster_values = []
            
            for gene in top_features[f'{t}min: aggregate']:
                gene_index = None
                if gene in cluster_samples.var.index:
                    gene_index = cluster_samples.var.index.get_loc(gene)
            
                if gene_index is None:
                    cluster_values.append(0)
                    continue
        
                gene_expression_values = cluster_samples.X[:, gene_index]
                mean_expression = np.mean(gene_expression_values)
                
            
                cluster_values.append(mean_expression)


            average_expressions_df[c + cond] = cluster_values
        
    average_expressions_df["genes"] = top_features[f'{t}min: aggregate']
    var = []
    for _, row in average_expressions_df.iterrows():
        var.append(np.var(list(row[0:27])))
    average_expressions_df["var"] = var
    average_expressions_df.to_csv(f"csvs/{t}min_mean_variance_expressions.csv")


# print(average_expressions_df)

def generate_heat_map(t):
    df = pd.read_csv(f"csvs/{t}min_mean_variance_expressions.csv", index_col=0)
    data = df.drop(columns=["genes", "var",])
    # data = df[:, 1:10]
    ii = [f'{i}:  {j:.4f}' for i, j in zip(df["genes"], df["var"])]
    data.index = ii
    print(data)

    zero_mask = (data == 0)
    nonzero_mask = ~zero_mask
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.where(nonzero_mask), cmap="viridis", cbar=True, linewidths=0.1, linecolor='gray')
    
    # zero values using black
    sns.heatmap(data.where(zero_mask), cmap=ListedColormap(["black"]), cbar=False, linewidths=0.1, linecolor='gray')
    
    plt.title(f"{t}mins: Average Gene Expression Heatmap")
    plt.xlabel("Cell Types")
    plt.ylabel("Genes")
    plt.tight_layout()
    plt.savefig(f"csvs/{t}_top40_heatmap.png", dpi=300, bbox_inches='tight')

for t in ["30", "90"]:

    with open(f'pickles/{t}min-data/test_data.pkl', 'rb') as f:
        data= pickle.load(f)

    generate_variance(data[0], t)
    generate_heat_map(t)