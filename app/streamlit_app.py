import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
from io import BytesIO
from wordcloud import WordCloud
from src.analyze import compute_sentiment, top_keywords, topic_model, ensure_datetime
from src.emotions import analyze_emotions
from src.reporting import build_pdf

# --------------------
# Streamlit Config
# --------------------
st.set_page_config(page_title="Dream Journal NLP", layout="wide")
st.title("🌙 Dream Journal NLP")
st.caption("Upload a CSV with columns: date, text")

# --------------------
# Helper Functions
# --------------------
def make_wordcloud(freq: dict):
    wc = WordCloud(width=1200, height=600, background_color="white", colormap="plasma").generate_from_frequencies(freq)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    st.image(buf.getvalue(), use_column_width=True)

def run_analysis(df: pd.DataFrame):
    """Run the full analysis pipeline on a dream dataset."""
    df["date"] = ensure_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        # --- Search / Filter Dreams ---
    st.subheader("🔎 Search Dreams")
    search_term = st.text_input("Enter a keyword to filter dreams (leave empty to see all):").strip()

    if search_term:
        filtered = df[df["text"].str.contains(search_term, case=False, na=False)]
        if filtered.empty:
            st.warning(f"No dreams found with keyword: '{search_term}'")
            return
        else:
            st.info(f"Showing {len(filtered)} dreams containing '{search_term}'.")
            df = filtered
    
        # --- Date Range Filter ---
    st.subheader("📅 Date Range Filter")
    min_date, max_date = df["date"].min().date(), df["date"].max().date()
    start_date, end_date = st.date_input(
        "Select date range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Ensure valid range
    if start_date > end_date:
        st.error("Start date must be before end date.")
        return

    df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]
    if df.empty:
        st.warning(f"No dreams found between {start_date} and {end_date}.")
        return



    # --- Summary Metrics ---
    st.subheader("📊 Dream Journal Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Dreams", len(df))
    with col2:
        st.metric("Date Range", f"{df['date'].min().date()} → {df['date'].max().date()}")
    with col3:
        st.metric("Most Frequent Word", df["text"].str.split().explode().value_counts().index[0])

    st.divider()

    # --- Preview ---
    st.subheader("🔍 Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # --- Sentiment & Emotions ---
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

    # --- PDF Export ---
    st.subheader("📄 Export Report")
    if st.button("Generate PDF Report"):
        pdf_buffer = build_pdf(df_sent, daily, avg, kw_df, topics, pd.DataFrame())
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_buffer,
            file_name="dream_journal_report.pdf",
            mime="application/pdf",
        )

# --------------------
# Main Logic
# --------------------
uploaded = st.file_uploader("Upload dream journal CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    if not {"date","text"}.issubset(df.columns):
        st.error("CSV must contain columns: date, text")
        st.stop()
    run_analysis(df)

else:
    sample_path = os.path.join("data", "sample_dreams.csv")
    if os.path.exists(sample_path):
        st.info("No file uploaded. Using sample dataset for demo.")
        df = pd.read_csv(sample_path)
        run_analysis(df)
    else:
        st.warning("No CSV uploaded and no sample dataset found. Please upload a file.")
