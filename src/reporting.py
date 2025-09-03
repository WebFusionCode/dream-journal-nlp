import os
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
import matplotlib.pyplot as plt

def save_plot(fig):
    buf = BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def build_pdf(df, daily, avg_emotions, keywords, topics, symbol_summary):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("🌙 Dream Journal NLP Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # Sentiment plot
    fig, ax = plt.subplots()
    daily.plot(x="date", y="sentiment", ax=ax, legend=False)
    ax.set_title("Daily Sentiment Trend")
    story.append(Image(save_plot(fig), width=400, height=200))
    story.append(Spacer(1, 12))

    # Emotions chart
    fig, ax = plt.subplots()
    avg_emotions.plot(kind="bar", x="emotion", y="average_score", ax=ax, legend=False)
    ax.set_title("Average Emotion Scores")
    story.append(Image(save_plot(fig), width=400, height=200))
    story.append(Spacer(1, 12))

    # Keywords
    story.append(Paragraph("Top Keywords", styles["Heading2"]))
    if not keywords.empty:
        data = [keywords.columns.tolist()] + keywords.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        story.append(table)
    story.append(Spacer(1, 12))

    # Topics
    story.append(Paragraph("Extracted Topics", styles["Heading2"]))
    if topics:
        for t in topics:
            story.append(Paragraph(f"Topic {t['topic']}: {', '.join(t['keywords'])}", styles["Normal"]))
    else:
        story.append(Paragraph("Not enough data for topics.", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Dream Dictionary
    story.append(Paragraph("Dream Dictionary (Symbol Summary)", styles["Heading2"]))
    if not symbol_summary.empty:
        data = [symbol_summary.columns.tolist()] + symbol_summary.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        story.append(table)

    doc.build(story)
    buffer.seek(0)
    return buffer
