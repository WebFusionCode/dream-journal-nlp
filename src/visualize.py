import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def plot_daily_sentiment(csv_path: str, out_png: str):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    plt.figure()
    plt.plot(df["date"], df["sentiment"], marker="o")
    plt.title("Daily Average Sentiment")
    plt.xlabel("Date")
    plt.ylabel("VADER Compound")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def wordcloud_from_keywords(csv_path: str, out_png: str):
    kw = pd.read_csv(csv_path)
    freq = {row["token"]: int(row["count"]) for _, row in kw.iterrows()}
    wc = WordCloud(width=1200, height=600, background_color="white").generate_from_frequencies(freq)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    wc.to_file(out_png)

if __name__ == "__main__":
    os.makedirs("reports/figures", exist_ok=True)
    plot_daily_sentiment("reports/daily_sentiment.csv", "reports/figures/daily_sentiment.png")
    wordcloud_from_keywords("reports/top_keywords.csv", "reports/figures/wordcloud.png")
    print("Saved figures to reports/figures/")
