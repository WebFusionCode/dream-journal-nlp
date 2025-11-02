# src/semantic.py
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Model (singleton style)
_MODEL = None
def get_model(name="all-MiniLM-L6-v2"):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(name)
    return _MODEL

def embed_texts(texts, model=None):
    model = model or get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def build_embeddings_index(df, text_col="text", model=None):
    model = model or get_model()
    texts = df[text_col].astype(str).tolist()
    embeddings = embed_texts(texts, model=model)
    return embeddings  # numpy array (n x dim)

def semantic_search(query, df, embeddings, top_k=5, model=None):
    model = model or get_model()
    q_emb = model.encode([str(query)], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    idx_sorted = np.argsort(-sims)[:top_k]
    results = df.reset_index().loc[idx_sorted].copy()
    results["score"] = sims[idx_sorted]
    return results
