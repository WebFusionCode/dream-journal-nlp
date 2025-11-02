import os
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.io as pio
import streamlit as st  # ✅ needed for st.session_state access

# Local imports
from src.summary import generate_summary
from src.visuals import (
    plot_emotion_trends,
    plot_dream_frequency,
    plot_cluster_projection,
)

# --- Utility Functions ---
def save_plotly(fig):
    """Convert a Plotly figure to a PNG buffer."""
    buf = BytesIO()
    pio.write_image(fig, buf, format="png", scale=2)
    buf.seek(0)
    return buf


def save_plot(fig):
    """Save a Matplotlib figure to BytesIO for embedding in PDF."""
    buf = BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def add_table_to_story(story, df, title, color=colors.lightgrey):
    """Helper to add a styled table to the PDF."""
    styles = getSampleStyleSheet()
    if df is not None and not df.empty:
        data = [df.columns.tolist()] + df.values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), color),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(table)
        story.append(Spacer(1, 12))
    else:
        story.append(Paragraph(f"{title}: No data available", styles["Normal"]))
        story.append(Spacer(1, 12))


# --- Core PDF Builder ---
def build_pdf(df, daily, avg_emotions, keywords, topics, symbol_summary, cluster_summary, meta=None):
    """Generates a visual and textual Dream Journal PDF report."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # 🧭 Title Section
    story.append(Paragraph("🌙 Dream Journal NLP Report", styles["Title"]))
    story.append(Paragraph(datetime.now().strftime("%B %d, %Y"), styles["Normal"]))
    story.append(Spacer(1, 20))

    # 📝 Narrative Summary
    story.append(Paragraph("📝 Narrative Summary", styles["Heading2"]))
    summary_text = generate_summary(daily, avg_emotions, keywords, topics)
    story.append(Paragraph(summary_text, styles["Normal"]))
    story.append(Spacer(1, 12))

    # 📈 Sentiment Trend
    try:
        fig, ax = plt.subplots()
        daily.plot(x="date", y="sentiment", ax=ax, legend=False)
        ax.set_title("Daily Sentiment Trend")
        story.append(Image(save_plot(fig), width=400, height=200))
        story.append(Spacer(1, 12))
    except Exception as e:
        story.append(Paragraph(f"Sentiment plot error: {e}", styles["Normal"]))

    # 😊 Emotion Trends (Plotly)
    try:
        emo_plot = plot_emotion_trends(df)
        story.append(Paragraph("Emotion Trends Over Time", styles["Heading2"]))
        story.append(Image(save_plotly(emo_plot), width=400, height=250))
        story.append(Spacer(1, 12))
    except Exception as e:
        story.append(Paragraph(f"Emotion trends error: {e}", styles["Normal"]))

    # 🔥 Dream Frequency Heatmap
    try:
        freq_buf = plot_dream_frequency(df)
        story.append(Paragraph("Dream Frequency Heatmap", styles["Heading2"]))
        story.append(Image(freq_buf, width=400, height=180))
        story.append(Spacer(1, 12))
    except Exception as e:
        story.append(Paragraph(f"Dream frequency heatmap error: {e}", styles["Normal"]))

    # 🧠 Cluster Projection
    try:
        if cluster_summary is not None and "cluster" in cluster_summary.columns:
            if "embeddings" in st.session_state:
                embeddings = st.session_state["embeddings"]
                labels = cluster_summary["cluster"].astype(int).tolist()
                proj_fig = plot_cluster_projection(df, embeddings, labels)
                story.append(Paragraph("Dream Clusters (2D Projection)", styles["Heading2"]))
                story.append(Image(save_plotly(proj_fig), width=400, height=250))
                story.append(Spacer(1, 12))
    except Exception as e:
        story.append(Paragraph(f"Cluster projection error: {e}", styles["Normal"]))

    # 💡 Top Keywords Table
    add_table_to_story(story, keywords, "Top Keywords", color=colors.lightblue)

    # 🧵 Topics
    story.append(Paragraph("Extracted Topics", styles["Heading2"]))
    if topics:
        for t in topics:
            story.append(Paragraph(f"Topic {t['topic']}: {', '.join(t['keywords'])}", styles["Normal"]))
    else:
        story.append(Paragraph("Not enough data for topics.", styles["Normal"]))
    story.append(Spacer(1, 12))

    # 🌙 Dream Dictionary
    add_table_to_story(story, symbol_summary, "Dream Dictionary (Symbol Summary)")

    # 🧭 Cluster Summary
    add_table_to_story(story, cluster_summary[["cluster", "size"]] if cluster_summary is not None else None,
                       "Cluster Summary")

    # ✅ Build & Return
    doc.build(story)
    buffer.seek(0)
    return buffer
