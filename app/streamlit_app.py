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
from src.reporting import build_pdf

st.set_page_config(page_title="Dream Journal NLP", layout="wide")
st.title("🌙 Dream Journal NLP")
st.caption("Upload a CSV with columns: date, text")

uploaded = st.file_uploader("Upload dream journal CSV", type=["csv"])

def make_wordcloud(freq: dict):
    wc = WordCloud(width=1200, height=600, background_color="white", colormap="plasma").generate_from_frequencies(freq)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    st.image(buf.getvalue(), use_container_width=True)

if uploaded:
    df = pd.read_csv(uploaded)
    if not {"date","text"}.issubset(df.columns):
        st.error("CSV must contain columns: date, text")
        st.stop()

    df["date"] = ensure_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # --- Quick Stats ---
    st.subheader("📊 Dream Journal Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Dreams", len(df))
    with col2:
        st.metric("Date Range", f"{df['date'].min().date()} → {df['date'].max().date()}")
    with col3:
        st.metric("Most Frequent Word", df["text"].str.split().explode().value_counts().index[0])

    st.divider()

    st.subheader("🔍 Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # --- Sentiment & Emotions side by side ---
    df_sent = compute_sentiment(df)
    daily = df_sent.groupby("date", as_index=False)["sentiment"].mean()

    emo_df = analyze_emotions(df)
    avg = emo_df.drop(columns=["date","text"]).mean().sort_values(ascending=False).reset_index()
    avg.columns = ["emotion","average_score"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Daily Sentiment Trend**")
        st.line_chart(daily.set_index("date"))
    with col2:
        st.markdown("**Average Emotion Scores**")
        st.bar_chart(avg.set_index("emotion"))

    st.divider()

    # --- Keywords ---
    st.subheader("💡 Top Keywords")
    kw_df = top_keywords(df_sent, n=30)
    st.dataframe(kw_df, use_container_width=True)
    if len(kw_df):
        freq = {row.token: int(row["count"]) for _, row in kw_df.iterrows()}
        make_wordcloud(freq)

    st.divider()

    # --- Topics ---
    st.subheader("📂 Topics (LDA)")
    topics = topic_model(df_sent, n_topics=4, n_top_words=8)
    if topics:
        for t in topics:
            st.write(f"**Topic {t['topic']}**: {', '.join(t['keywords'])}")
    else:
        st.info("Not enough data for topics. Add more entries.")

    st.divider()

    # --- Dream Dictionary ---
    st.subheader("📖 Personal Dream Dictionary")
    symbol_summary = pd.DataFrame()
    if os.path.exists("config/symbols.yaml"):
        import yaml, re
        with open("config/symbols.yaml", "r", encoding="utf-8") as f:
            lex_cfg = yaml.safe_load(f).get("groups", {})
        compiled = {
    g: [re.compile(rf"\b{re.escape(term)}\b", re.I) for term in terms]
    for g, terms in lex_cfg.items()
}

        def count_row(text):
            text = str(text)
            out = {}
            for g, pats in compiled.items():
                out[g] = sum(len(p.findall(text)) for p in pats)
            return out

        counts = df["text"].apply(count_row).apply(pd.Series).fillna(0).astype(int)
        symbol_summary = counts.sum().sort_values(ascending=False).rename_axis("symbol_group").reset_index(name="total_count")

        choice = st.selectbox("Select a symbol to explore:", options=symbol_summary["symbol_group"].tolist())
        if choice:
            matched = df[counts[choice] > 0]
            st.write(f"Dreams containing **{choice}** ({len(matched)} entries):")
            st.dataframe(matched[["date", "text"]], use_container_width=True)
    else:
        st.info("Add config/symbols.yaml to use the Dream Dictionary.")

    st.divider()

    # --- PDF Export ---
    st.subheader("📄 Export Report")
    if st.button("Generate PDF Report"):
        pdf_buffer = build_pdf(df_sent, daily, avg, kw_df, topics, symbol_summary)
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_buffer,
            file_name="dream_journal_report.pdf",
            mime="application/pdf",
        )

else:
    sample_path = os.path.join("data", "sample_dreams.csv")
    if os.path.exists(sample_path):
        st.info("No file uploaded. Using sample dataset for demo.")
        df = pd.read_csv(sample_path)
        df["date"] = ensure_datetime(df["date"])
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        
        # 👇 Re-run same analysis steps with demo dataset
        st.subheader("Preview (Sample Data)")
        st.dataframe(df.head(10), use_container_width=True)

        df_sent = compute_sentiment(df)
        daily = df_sent.groupby("date", as_index=False)["sentiment"].mean()

        emo_df = analyze_emotions(df)
        avg = emo_df.drop(columns=["date","text"]).mean().sort_values(ascending=False).reset_index()
        avg.columns = ["emotion","average_score"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Daily Sentiment Trend**")
            st.line_chart(daily.set_index("date"))
        with col2:
            st.markdown("**Average Emotion Scores**")
            st.bar_chart(avg.set_index("emotion"))

        # Keywords
        st.subheader("Top Keywords (Sample)")
        kw_df = top_keywords(df_sent, n=30)
        st.dataframe(kw_df, use_container_width=True)
        if len(kw_df):
            freq = {row.token: int(row["count"]) for _, row in kw_df.iterrows()}
            make_wordcloud(freq)

        # Topics
        st.subheader("Topics (Sample)")
        topics = topic_model(df_sent, n_topics=4, n_top_words=8)
        if topics:
            for t in topics:
                st.write(f"**Topic {t['topic']}**: {', '.join(t['keywords'])}")
        else:
            st.info("Not enough data for topics.")

    else:
        st.warning("No CSV uploaded and no sample dataset found. Please upload a file.")

