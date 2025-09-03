import argparse
import os
import pandas as pd
from transformers import pipeline

def ensure_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def load_emotion_model():
    # ✅ Use top_k=None instead of return_all_scores=True
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

def analyze_emotions(df: pd.DataFrame, text_col="text"):
    model = load_emotion_model()
    results = model(df[text_col].astype(str).tolist(), top_k=None)

    # Each result is now a list of dicts: [{'label': 'joy', 'score': 0.7}, ...]
    labels = [item["label"] for item in results[0]]
    scores_df = pd.DataFrame([[item["score"] for item in r] for r in results], columns=labels)

    return pd.concat([df.reset_index(drop=True), scores_df], axis=1)

def main():
    ap = argparse.ArgumentParser(description="Emotion classifier")
    ap.add_argument("--input", required=True, help="CSV with date,text")
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)
    if not {"date","text"}.issubset(df.columns):
        raise ValueError("CSV must contain date,text")

    df["date"] = ensure_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    out = analyze_emotions(df)
    out.to_csv(os.path.join(args.outdir, "dreams_with_emotions.csv"), index=False)

    print(f"✅ Saved {os.path.join(args.outdir, 'dreams_with_emotions.csv')}")

if __name__ == "__main__":
    main()
