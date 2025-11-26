import os
import sys
import streamlit as st
import pandas as pd
from io import BytesIO
from wordcloud import WordCloud

# Add parent dir for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Existing imports
from src.analyze import compute_sentiment, top_keywords, topic_model, ensure_datetime
from src.emotions import analyze_emotions
from src.reporting import build_pdf
from src.summary import generate_summary

# NEW imports for advanced NLP
from src.symbols_ext import load_symbol_lexicon, symbol_summary_for_df
from src.semantic import get_model, build_embeddings_index, semantic_search
from src.clustering import cluster_with_kmeans, label_clusters_by_top_terms

# Visuals
from src.visuals import (
    plot_emotion_trends,
    plot_dream_frequency,
    plot_keyword_emotion_network,
    plot_cluster_projection
)


# Streamlit Config
st.set_page_config(page_title="Dream Journal NLP", layout="wide")
st.title("🌙 Dream Journal NLP")
st.caption("Upload a CSV with columns: date, text")


# --- Helper Functions ---
def make_wordcloud(freq: dict):
    """Generate and display a word cloud."""
    wc = WordCloud(width=1200, height=600, background_color="white", colormap="plasma").generate_from_frequencies(freq)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    st.image(buf.getvalue(), use_container_width=True)  # ✅ fixed deprecated arg


# --- Main Analysis Pipeline ---
def run_analysis(df: pd.DataFrame):
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
        max_value=max_date,
    )

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
    avg = emo_df.drop(columns=["date", "text"]).mean().sort_values(ascending=False).reset_index()
    avg.columns = ["emotion", "average_score"]

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
    st.subheader("📂 Topics")
    topics = topic_model(df_sent, n_topics=4, n_top_words=8)
    if topics:
        for t in topics:
            st.write(f"**Topic {t['topic']}**: {', '.join(t['keywords'])}")
    else:
        st.info("Not enough data for topics. Add more entries.")

    st.divider()

    # --- 🔮 Dream Symbol Analysis ---
    st.subheader("🔮 Dream Symbol Analysis")
    try:
        lexicon = load_symbol_lexicon()
        per_entry_counts, symbol_totals = symbol_summary_for_df(df, lexicon)
    except Exception as e:
        st.error(f"Error loading dream symbols: {e}")
        symbol_totals = pd.DataFrame()

    if not symbol_totals.empty:
        st.dataframe(symbol_totals, use_container_width=True)
    else:
        st.info("No dream symbols detected in your entries.")

    st.divider()

    # --- 🧠 Semantic Search ---
    st.subheader("🧠 Semantic Search (Meaning-based)")
    model = get_model()

    if "embeddings" not in st.session_state or st.session_state.get("embeddings_len") != len(df):
        with st.spinner("Building semantic embeddings..."):
            embeddings = build_embeddings_index(df, model=model)
            st.session_state["embeddings"] = embeddings
            st.session_state["embeddings_len"] = len(df)
    else:
        embeddings = st.session_state["embeddings"]

    query = st.text_input("Enter a phrase to search semantically (e.g., 'fear', 'ocean', 'falling'):")
    if query:
        results = semantic_search(query, df, embeddings, top_k=8, model=model)
        st.write(f"Top semantic matches for **'{query}'**:")
        st.dataframe(results[["date", "text", "score"]], use_container_width=True)

    st.divider()

    # --- 🌌 Thematic Clustering ---
    st.subheader("🌌 Thematic Clustering of Dreams")
    n_clusters = st.slider("Number of clusters (KMeans)", 2, 12, 6)
    labels, km = cluster_with_kmeans(embeddings, n_clusters=n_clusters)
    cluster_summary = label_clusters_by_top_terms(df, labels)

    if not cluster_summary.empty:
        st.dataframe(cluster_summary[["cluster", "size"]], use_container_width=True)
        selected_cluster = st.selectbox("Select cluster to view example dreams:",
                                        options=cluster_summary["cluster"].tolist())
        if selected_cluster is not None:
            examples = cluster_summary.loc[cluster_summary["cluster"] == selected_cluster, "samples"].explode().tolist()
            st.write("**Sample dreams in this cluster:**")
            for e in examples[:8]:
                st.markdown(f"- {e}")
    else:
        st.info("Not enough data to form meaningful clusters.")

    st.divider()

    # --- 📊 Interactive Visual Analytics ---
    st.subheader("📊 Interactive Visual Analytics")
    tabs = st.tabs(["Emotion Trends", "Dream Frequency", "Keyword–Emotion Network", "Cluster Map"])

    with tabs[0]:
        st.plotly_chart(plot_emotion_trends(emo_df), use_container_width=True)
    with tabs[1]:
        buf = plot_dream_frequency(df)
        st.image(buf, use_container_width=True)
    with tabs[2]:
        html_path = plot_keyword_emotion_network(kw_df, emo_df)
        st.components.v1.html(open(html_path).read(), height=520, scrolling=True)
    with tabs[3]:
        st.plotly_chart(plot_cluster_projection(df, embeddings, labels), use_container_width=True)

    st.divider()

    # --- 🔮 Emotional Forecasting ---
    from src.forecast import forecast_emotions

    st.subheader("🔮 Emotional Forecasting")
    try:
        buf, summary = forecast_emotions(daily)
        st.image(buf, use_container_width=True)
        st.success(summary)
    except Exception as e:
        st.error(f"Forecasting failed: {e}")

    st.divider()

    # --- 🎯 Emotional Triggers ---
    from src.triggers import detect_emotion_triggers

    st.subheader("🎯 Emotional Triggers in Dreams")

    try:
        triggers = detect_emotion_triggers(df_sent, emo_df)
        st.dataframe(triggers, use_container_width=True)
        st.markdown("**Interpretation:** Words with higher positive coefficients "
                    "are linked to happier dreams, while negative ones indicate stressors or anxieties.")
    except Exception as e:
        st.error(f"Trigger detection failed: {e}")

    st.divider()

    # --- 🧘 Automated Insight Generation ---
    from src.insights import generate_insights

    st.subheader("🧘 Automated Insights & Recommendations")

    try:
        insights = generate_insights(
            df=df_sent,
            daily=daily,
            avg_emotions=avg,
            keywords=kw_df,
            topics=topics,
            symbol_summary=symbol_totals,
            cluster_summary=cluster_summary
        )
        for ins in insights:
            st.markdown(f"- {ins}")
    except Exception as e:
        st.error(f"⚠️ Insight generation failed: {e}")
        st.write("DEBUG: Symbol Summary Columns ->", list(symbol_totals.columns))
        st.dataframe(symbol_totals.head())



    # --- 📝 Narrative Summary ---
    st.subheader("📝 Narrative Summary")
    summary_text = generate_summary(daily, avg, kw_df, topics)
    st.markdown(summary_text)

    # --- 📄 PDF Export ---
    st.subheader("📄 Export Report")
    if st.button("Generate PDF Report"):
        pdf_buffer = build_pdf(df_sent, daily, avg, kw_df, topics, symbol_totals, cluster_summary)
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_buffer,
            file_name="dream_journal_report.pdf",
            mime="application/pdf",
        )

        # --- 💬 Dream AI Assistant ---
    import src.ai_assistant as assistant  # ✅ make sure this file exists in src/

    st.markdown("---")
    st.header("💬 Dream AI Assistant")
    st.caption("Ask the AI to interpret your dreams, find patterns, or summarize insights.")

    # User input
    user_input = st.text_area(
        "Ask something about your dreams:",
        placeholder="e.g., What does it mean that I keep dreaming about water?"
    )

    # How many recent dreams to include in context
    context_depth = st.slider("Number of recent dreams to include in analysis:", 3, 20, 5)

    # Initialize assistant history in session state
    if "assistant_history" not in st.session_state:
        st.session_state["assistant_history"] = []

    # Create the Ask Assistant button (enabled only when input exists)
    ask_button = st.button("Ask Assistant", disabled=not bool(user_input.strip()))

    if ask_button:
        with st.spinner("✨ The AI is analyzing your dreams..."):
            try:
                # Use the most recent N dreams as context
                if "text" not in df.columns:
                    st.error("❌ The uploaded CSV must have a 'text' column.")
                else:
                    context = "\n\n".join(df["text"].astype(str).tail(context_depth).tolist())
                    response = assistant.get_ai_response(user_input, context)
                    st.session_state["assistant_history"].append((user_input, response))
                    st.success("✅ Response generated successfully!")
            except Exception as e:
                st.error(f"⚠️ AI Assistant failed: {e}")

    # Display the chat history
    if st.session_state["assistant_history"]:
        st.subheader("Conversation History")
        for q, a in st.session_state["assistant_history"]:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**AI Assistant:** {a}")
            st.markdown("---")




import reportlab
st.write()

# --- Main Logic ---
uploaded = st.file_uploader("Upload dream journal CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    if not {"date", "text"}.issubset(df.columns):
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

# --- FOOTER SECTION ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 15px 0; font-size: 15px; color: gray;'>
        © 2025 <b>Dreams Psychology AI</b> — All Rights Reserved.<br>
        Developed by <b>Web Fusion(Harsh Singh)</b> | 
        <a href="mailto:webwithfusion@gmail.com" style="color: #4b9be0; text-decoration: none;">
            webwithfusion@gmail.com
        </a>
    </div>
    """,
    unsafe_allow_html=True
)