"""NLP-specific visualizations: word clouds, topic charts, sentiment plots."""

import matplotlib.pyplot as plt
import pandas as pd
try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None  # pragma: no cover

from wf_analysis.visualization.theme import Theme


class NLPVisualizer:
    @staticmethod
    def wordcloud(texts: pd.Series, title: str = "", figsize=(10, 6)) -> plt.Figure | None:
        if WordCloud is None:
            return None
        text = " ".join(texts.fillna("").astype(str))
        fig, ax = plt.subplots(figsize=figsize)
        wc = WordCloud(
            width=800, height=400, background_color="white",
            colormap="viridis", max_words=100,
        ).generate(text if text else "no data")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, pad=20)
        plt.tight_layout()
        return fig

    @staticmethod
    def sentiment_distribution(sentiments: pd.DataFrame, figsize=(10, 6)) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        colors = {"Positive": "#2E86AB", "Neutral": "#F18F01", "Negative": "#C73E1D"}
        counts = sentiments["sentiment_label"].value_counts()
        bars = ax.bar(
            counts.index, counts.values,
            color=[colors.get(l, "#888") for l in counts.index],
        )
        ax.set_title("Sentiment Distribution", fontsize=14)
        ax.set_ylabel("Count")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(int(bar.get_height())),
                ha="center", va="bottom",
            )
        plt.tight_layout()
        return fig
