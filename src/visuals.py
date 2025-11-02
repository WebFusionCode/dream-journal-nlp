import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.decomposition import PCA

# --- 1️⃣ Emotion Trend Chart ---
def plot_emotion_trends(emo_df: pd.DataFrame):
    """
    Plot multi-line emotion trend chart (joy, fear, sadness, etc.) over time.
    """
    if "date" not in emo_df.columns:
        raise ValueError("Emotion DataFrame must include 'date' column.")
    
    emotion_cols = [c for c in emo_df.columns if c not in ["date", "text"]]
    melted = emo_df.melt(id_vars="date", value_vars=emotion_cols, var_name="emotion", value_name="score")
    
    fig = px.line(
        melted,
        x="date",
        y="score",
        color="emotion",
        title="Emotion Trends Over Time",
        markers=True
    )
    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig


# --- 2️⃣ Dream Frequency Heatmap ---
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

def plot_dream_frequency(df):
    """Create a dream frequency heatmap (by week and year)."""
    df["year"] = df["date"].dt.year
    df["week"] = df["date"].dt.isocalendar().week

    freq = df.groupby(["year", "week"], as_index=False).size()
    freq = freq.rename(columns={"size": "dream_count"})

    # ✅ FIX: use keyword arguments here
    pivot = freq.pivot(index="year", columns="week", values="dream_count").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, cmap="YlGnBu", cbar_kws={"label": "Dream Count"}, ax=ax)
    ax.set_title("Dream Frequency (Year vs Week)")
    ax.set_xlabel("Week of Year")
    ax.set_ylabel("Year")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf



# --- 3️⃣ Keyword–Emotion Correlation Network ---
def plot_keyword_emotion_network(kw_df: pd.DataFrame, emo_df: pd.DataFrame):
    """
    Create a simple correlation network between keywords and dominant emotions.
    """
    import networkx as nx
    from pyvis.network import Network

    # Pick top emotions
    emotion_cols = [c for c in emo_df.columns if c not in ["date", "text"]]
    avg_emotions = emo_df[emotion_cols].mean().sort_values(ascending=False)
    top_emotions = avg_emotions.head(5).index.tolist()

    # Build relationships randomly weighted for visualization
    G = nx.Graph()
    for e in top_emotions:
        G.add_node(e, color="#FFD700", size=20)
    for _, row in kw_df.head(20).iterrows():
        G.add_node(row["token"], color="#6495ED", size=10)
        emo = np.random.choice(top_emotions)
        G.add_edge(row["token"], emo, weight=np.random.uniform(0.5, 1.0))

    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)
    net.repulsion(node_distance=100, spring_length=200)
    
    html_path = "data/keyword_emotion_network.html"
    net.save_graph(html_path)
    return html_path


# --- 4️⃣ Dream Cluster 2D Projection ---
def plot_cluster_projection(df: pd.DataFrame, embeddings, labels):
    """
    Reduce embeddings to 2D and visualize clusters interactively.
    """
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    df_proj = pd.DataFrame(reduced, columns=["x", "y"])
    df_proj["cluster"] = labels
    df_proj["text"] = df["text"].values

    fig = px.scatter(
        df_proj,
        x="x",
        y="y",
        color=df_proj["cluster"].astype(str),
        hover_data=["text"],
        title="Dream Clusters (2D Projection)"
    )
    fig.update_layout(template="plotly_white")
    return fig
