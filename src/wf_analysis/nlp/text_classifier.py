import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt
import seaborn as sns


class TextClassifier:
    def __init__(
        self,
        vectorizer: str = "tfidf",
        model: str = "logistic",
        max_features: int = 1000,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.model = LogisticRegression(max_iter=1000)
        self._fitted = False
        self.metrics: dict = {}
        self._labels: list[str] = []

    def fit(
        self, texts: pd.Series, labels: pd.Series
    ) -> "TextClassifier":
        X = self.vectorizer.fit_transform(texts.fillna("").astype(str))
        self._labels = sorted(labels.unique())

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted"
        )
        self.metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }
        self._fitted = True
        logger.info(
            f"TextClassifier fitted: accuracy={acc:.3f}, "
            f"precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}"
        )
        return self

    def predict(self, texts: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TextClassifier must be fitted before predict")
        X = self.vectorizer.transform(texts.fillna("").astype(str))
        return self.model.predict(X)

    def evaluate(
        self, texts: pd.Series, labels: pd.Series
    ) -> dict:
        y_pred = self.predict(texts)
        cm = confusion_matrix(labels, y_pred, labels=self._labels)
        return {
            "accuracy": float(accuracy_score(labels, y_pred)),
            "confusion_matrix": cm,
        }

    def plot_confusion_matrix(
        self, cm: np.ndarray | None = None
    ) -> plt.Figure:
        if cm is None:
            cm = self.metrics.get("confusion_matrix", np.zeros((1, 1)))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        return fig
