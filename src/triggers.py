import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
import numpy as np

def detect_emotion_triggers(df, emotion_df):
    """
    Detect words that correlate with sentiment or specific emotions.
    Returns a DataFrame of top positive/negative triggers.
    """
    merged = pd.merge(df[["date", "text", "sentiment"]],
                      emotion_df[["date"]],
                      on="date", how="inner")

    # Text vectorization
    vectorizer = CountVectorizer(max_features=800, stop_words="english")
    X = vectorizer.fit_transform(merged["text"])
    words = vectorizer.get_feature_names_out()

    # Model: sentiment as dependent variable
    y = merged["sentiment"].values
    reg = LinearRegression()
    reg.fit(X.toarray(), y)

    coef_df = pd.DataFrame({
        "word": words,
        "coef": reg.coef_
    }).sort_values("coef", ascending=False)

    top_pos = coef_df.head(15)
    top_neg = coef_df.tail(15).sort_values("coef")

    triggers = pd.concat([top_pos, top_neg]).reset_index(drop=True)
    triggers["impact"] = np.where(triggers["coef"] > 0, "positive", "negative")
    return triggers
