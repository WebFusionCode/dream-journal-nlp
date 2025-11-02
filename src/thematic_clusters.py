from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import pandas as pd

def cluster_dreams(df: pd.DataFrame, n_clusters=5):
    if "text" not in df.columns:
        raise ValueError("DataFrame must contain 'text' column")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=False)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    return df, centers
