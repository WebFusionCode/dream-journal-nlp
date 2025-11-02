import pandas as pd


def generate_insights(df, daily, avg_emotions, keywords, topics, symbol_summary, cluster_summary):
    """Generate intelligent textual insights about the user's dreams."""
    insights = []

    try:
        # 1️⃣ Sentiment
        avg_sent = daily["sentiment"].mean()
        if avg_sent > 0.2:
            insights.append("🌞 Your overall dreams lean toward positive or hopeful moods.")
        elif avg_sent < -0.2:
            insights.append("🌧️ Your dreams tend to express anxiety or stress — consider journaling before bed.")
        else:
            insights.append("😐 Your dreams are emotionally balanced, neither strongly positive nor negative.")

        # 2️⃣ Emotions
        top_emotion = avg_emotions.sort_values("average_score", ascending=False).iloc[0]
        insights.append(f"💖 The most dominant emotion across your dreams is **{top_emotion['emotion']}**.")

        # 3️⃣ Keywords
        if not keywords.empty:
            top_kw = keywords.iloc[0]["token"]
            insights.append(f"🗝️ The most recurring theme in your dreams is **'{top_kw}'**.")

        # 4️⃣ Topics
        if topics:
            insights.append(f"📚 {len(topics)} main dream topics were detected — recurring narratives are forming.")

        # 5️⃣ Dream Symbol Summary — now robust
        if symbol_summary is not None and not symbol_summary.empty:
            # Try to detect the correct count column name dynamically
            count_col = None
            for c in symbol_summary.columns:
                if c.lower() in ["count", "frequency", "total", "occurrences", "times"]:
                    count_col = c
                    break

            if count_col and "symbol" in symbol_summary.columns:
                top_symbol = symbol_summary.sort_values(count_col, ascending=False).iloc[0]
                meaning = top_symbol["meaning"] if "meaning" in top_symbol else "varied interpretations"
                insights.append(
                    f"🔮 The most frequent dream symbol is **'{top_symbol['symbol']}'**, representing {meaning}."
                )
            else:
                insights.append("✨ Dream symbols detected, but no frequency data available.")
        else:
            insights.append("🕯️ No recurring dream symbols found in this dataset.")

        # 6️⃣ Clusters
        if cluster_summary is not None and not cluster_summary.empty:
            largest_cluster = cluster_summary.sort_values("size", ascending=False).iloc[0]
            insights.append(
                f"🌌 The largest dream cluster (Cluster {largest_cluster['cluster']}) contains {largest_cluster['size']} dreams."
            )

    except Exception as e:
        insights.append(f"⚠️ Insight generation failed: {e}")

    return insights
