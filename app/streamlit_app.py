import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from wordcloud import WordCloud
from src.analyze import compute_sentiment, top_keywords, topic_model, ensure_datetime
from src.emotions import analyze_emotions

st.set_page_config(page_title="Dream Journal NLP", layout="wide")
st.title("🌙 Dream Journal NLP")
st.caption("Upload a CSV with columns: date, text")

uploaded = st.file_uploader("Upload dream journal CSV", type=["csv"])

def make_wordcloud(freq: dict):
    wc = WordCloud(width=1200, height=600, background_color="white").generate_from_frequencies(freq)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    st.image(buf.getvalue(), use_column_width=True)

if uploaded:
    df = pd.read_csv(uploaded)
    if not {"date","text"}.issubset(df.columns):
        st.error("CSV must contain columns: date, text")
        st.stop()

    df["date"] = ensure_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Sentiment
    st.subheader("Daily Sentiment Trend")
    df_sent = compute_sentiment(df)
    daily = df_sent.groupby("date", as_index=False)["sentiment"].mean()
    st.line_chart(daily.set_index("date"))

    # Emotions
    st.subheader("Emotion Probabilities")
    emo_df = analyze_emotions(df)
    avg = emo_df.drop(columns=["date","text"]).mean().sort_values(ascending=False).reset_index()
    avg.columns = ["emotion","average_score"]
    st.bar_chart(avg.set_index("emotion"))

    # Keywords
    st.subheader("Top Keywords")
    kw_df = top_keywords(df_sent, n=30)
    st.dataframe(kw_df, use_container_width=True)
    if len(kw_df):
        freq = {row.token: int(row["count"]) for _, row in kw_df.iterrows()}
        make_wordcloud(freq)

    # Topics
    st.subheader("Topics (LDA)")
    topics = topic_model(df_sent, n_topics=4, n_top_words=8)
    if topics:
        for t in topics:
            st.write(f"**Topic {t['topic']}**: {', '.join(t['keywords'])}")
    else:
        st.info("Not enough data for topics. Add more entries.")

    # --- Personal Dream Dictionary ---
    st.subheader("📖 Personal Dream Dictionary")

    if os.path.exists("config/symbols.yaml"):
        import yaml, re
        with open("config/symbols.yaml", "r", encoding="utf-8") as f:
            lex_cfg = yaml.safe_load(f).get("groups", {})
        compiled = {g: [re.compile(rf"\b{re.escape(term)}\b", re.I) for term in terms]
                    for g, terms in lex_cfg.items()}

        def count_row(text):
            text = str(text)
            out = {}
            for g, pats in compiled.items():
                out[g] = sum(len(p.findall(text)) for p in pats)
            return out

        counts = df["text"].apply(count_row).apply(pd.Series).fillna(0).astype(int)
        totals = counts.sum().sort_values(ascending=False).rename_axis("symbol_group").reset_index(name="total_count")

        choice = st.selectbox("Select a symbol to explore:", options=totals["symbol_group"].tolist())
        if choice:
            mask = counts[choice] > 0
            matched = df[mask]
            st.write(f"Dreams containing **{choice}** ({len(matched)} entries):")
            st.dataframe(matched[["date", "text"]], use_container_width=True)
    else:
        st.info("Add config/symbols.yaml to use the Dream Dictionary.")

else:
    st.info("Tip: Use your own CSV or create one under data/sample_dreams.csv")
