"""VADER sentiment analysis for text data."""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, texts: pd.Series) -> pd.DataFrame:
        results = []
        for text in texts.fillna("").astype(str):
            scores = self.analyzer.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                label = "Positive"
            elif compound <= -0.05:
                label = "Negative"
            else:
                label = "Neutral"
            results.append({
                "sentiment_score": compound,
                "sentiment_label": label,
                "sentiment_magnitude": abs(compound),
                "neg_score": scores["neg"],
                "neu_score": scores["neu"],
                "pos_score": scores["pos"],
            })
        return pd.DataFrame(results)
