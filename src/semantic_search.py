from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

def semantic_search(df: pd.DataFrame, query: str, top_k=5):
    if "text" not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column")
    query_emb = embed_texts([query])
    corpus_emb = embed_texts(df["text"].tolist())
    hits = util.semantic_search(query_emb, corpus_emb, top_k=top_k)[0]
    results = df.iloc[[h["corpus_id"] for h in hits]].copy()
    results["score"] = [h["score"] for h in hits]
    return results
