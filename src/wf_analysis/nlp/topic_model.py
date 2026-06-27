import pandas as pd
import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import matplotlib.pyplot as plt


class TopicModeler:
    def __init__(
        self,
        method: str = "lda",
        n_topics: int = 5,
        random_state: int = 42,
        max_features: int = 1000,
    ):
        self.method = method
        self.n_topics = n_topics
        self.random_state = random_state
        self.vectorizer = CountVectorizer(
            max_df=0.9, min_df=2, max_features=max_features, stop_words="english"
        )
        self.model = LatentDirichletAllocation(
            n_components=n_topics, random_state=random_state
        )
        self._fitted = False
        self.feature_names: list[str] = []

    def fit(self, texts: pd.Series) -> "TopicModeler":
        cleaned = texts.fillna("").astype(str)
        cleaned = cleaned[cleaned.str.strip() != ""]
        doc_term = self.vectorizer.fit_transform(cleaned)
        self.model.fit(doc_term)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        self._fitted = True
        logger.info(
            f"TopicModeler ({self.method}) fitted with {self.n_topics} topics "
            f"on {len(cleaned)} documents, {len(self.feature_names)} features"
        )
        return self

    def transform(self, texts: pd.Series) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("TopicModeler must be fitted before transform")
        cleaned = texts.fillna("").astype(str)
        doc_term = self.vectorizer.transform(cleaned)
        topic_dist = self.model.transform(doc_term)
        dominant = topic_dist.argmax(axis=1)
        probs = topic_dist.max(axis=1)
        return pd.DataFrame({
            "dominant_topic": dominant,
            "topic_probability": probs,
        })

    def get_topic_info(self, n_words: int = 10) -> list[dict]:
        if not self._fitted:
            raise RuntimeError("TopicModeler must be fitted first")
        info = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[:-n_words - 1:-1]
            top_words = [self.feature_names[i] for i in top_indices]
            info.append({
                "topic_id": topic_idx,
                "top_words": top_words,
                "prevalence": float(topic.sum()),
            })
        return info

    def plot_topics(self, n_words: int = 10) -> plt.Figure | None:
        if not self._fitted:
            return None
        info = self.get_topic_info(n_words)
        fig, axes = plt.subplots(
            self.n_topics, 1, figsize=(12, 3 * self.n_topics)
        )
        if self.n_topics == 1:
            axes = [axes]
        for ax, topic in zip(axes, info):
            words = topic["top_words"]
            scores = list(range(len(words), 0, -1))
            ax.barh(words, scores, color="#2E86AB")
            ax.set_title(f"Topic {topic['topic_id']}")
            ax.invert_yaxis()
        plt.tight_layout()
        return fig
