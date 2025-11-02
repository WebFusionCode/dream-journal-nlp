# src/clustering.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
# optional: from hdbscan import HDBSCAN
from sklearn.decomposition import PCA

def cluster_with_kmeans(embeddings, n_clusters=6, random_state=42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km

# optional better clustering
# def cluster_with_hdbscan(embeddings, min_cluster_size=5):
#     clusterer = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
#     labels = clusterer.fit_predict(embeddings)
#     return labels, clusterer

def label_clusters_by_top_terms(df, labels, top_n_terms=5):
    # returns a summary dataframe: cluster -> size -> sample texts
    df2 = df.reset_index(drop=True).copy()
    df2["cluster"] = labels
    summary = []
    for c in sorted(df2["cluster"].unique()):
        sub = df2[df2["cluster"]==c]
        sample_texts = sub["text"].head(5).tolist()
        summary.append({
            "cluster": int(c),
            "size": len(sub),
            "samples": sample_texts
        })
    return pd.DataFrame(summary)
