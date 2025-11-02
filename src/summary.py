# src/summary.py
import numpy as np

def generate_summary(daily, avg_emotions, keywords, topics):
    """Generate a narrative summary from analyzed dream data."""
    lines = []

    # Sentiment trend
    if not daily.empty:
        avg_sent = daily["sentiment"].mean()
        if avg_sent > 0.2:
            lines.append("Overall, your dreams have had a mostly positive tone.")
        elif avg_sent < -0.2:
            lines.append("Overall, your dreams have reflected more negative or stressful emotions.")
        else:
            lines.append("Overall, your dreams have been fairly balanced between positive and negative tones.")
    
    # Emotions
    if not avg_emotions.empty:
        top_emotion = avg_emotions.sort_values("average_score", ascending=False).iloc[0]["emotion"]
        lines.append(f"The most prominent emotion detected was **{top_emotion}**.")
    
    # Keywords
    if not keywords.empty:
        top_words = ", ".join(keywords.head(5)["token"].tolist())
        lines.append(f"Frequent dream themes included: {top_words}.")
    
    # Topics
    if topics:
        topic_summaries = []
        for t in topics:
            topic_summaries.append(", ".join(t["keywords"][:3]))
        lines.append(f"Your dreams clustered into topics such as: { '; '.join(topic_summaries) }.")

    return " ".join(lines)
