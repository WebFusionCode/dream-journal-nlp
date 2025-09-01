import argparse
import os
import json
import pandas as pd
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from .preprocess import preprocess_text, clean_text

def ensure_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    df = df.copy()
    df["sentiment"] = df["text"].apply(lambda t: sia.polarity_scores(str(t))["compound"])
    return df

def top_keywords(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    tokens = df["text"].apply(preprocess_text)
    all_tokens = [t for row in tokens for t in row if len(t) > 2]
    counts = Counter(all_tokens).most_common(n)
    return pd.DataFrame(counts, columns=["token","count"])

def topic_model(df: pd.DataFrame, n_topics: int = 4, n_top_words: int = 8):
    texts = df["text"].fillna("").astype(str).apply(clean_text)
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")
    X = vectorizer.fit_transform(texts)
    if X.shape[0] == 0 or X.shape[1] == 0:
        return []
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
    lda.fit(X)
    vocab = vectorizer.get_feature_names_out()
    topics = []
    for idx, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[-n_top_words:][::-1]
        topics.append({
            "topic": idx,
            "keywords": [vocab[i] for i in top_idx]
        })
    return topics

def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Dream Journal NLP baseline analysis")
    parser.add_argument("--input", required=True, help="Path to CSV with columns: date,text")
    parser.add_argument("--outdir", default="reports", help="Output directory")
    parser.add_argument("--topics", type=int, default=4, help="Number of LDA topics")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dreams = pd.read_csv(args.input)
    if "date" not in dreams.columns or "text" not in dreams.columns:
        raise ValueError("Input CSV must have columns: date,text")

    dreams["date"] = ensure_datetime(dreams["date"])
    dreams = dreams.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Sentiment
    dreams = compute_sentiment(dreams)
    save_csv(dreams, os.path.join(args.outdir, "dreams_with_sentiment.csv"))

    # Top keywords
    kw = top_keywords(dreams, n=40)
    save_csv(kw, os.path.join(args.outdir, "top_keywords.csv"))

    # Topic modeling
    topics = topic_model(dreams, n_topics=args.topics, n_top_words=8)
    topics_path = os.path.join(args.outdir, "topics.json")
    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)

    # Daily aggregation
    daily = dreams.groupby("date", as_index=False)["sentiment"].mean()
    save_csv(daily, os.path.join(args.outdir, "daily_sentiment.csv"))

    print("Analysis complete.")
    print(f"- Detailed rows: {os.path.join(args.outdir, 'dreams_with_sentiment.csv')}")
    print(f"- Top keywords: {os.path.join(args.outdir, 'top_keywords.csv')}")
    print(f"- Topics JSON:  {topics_path}")
    print(f"- Daily sentiment: {os.path.join(args.outdir, 'daily_sentiment.csv')}")

if __name__ == "__main__":
    main()
