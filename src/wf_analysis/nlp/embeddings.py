import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class TextEmbedder:
    def __init__(self):
        self._vocab: dict[str, np.ndarray] = {}
        self._dim = 50

    def _simple_embed(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vec = np.zeros(self._dim)
        for i, w in enumerate(words):
            np.random.seed(hash(w) % 2**31)
            rng = np.random.default_rng(hash(w) % 2**31)
            wv = rng.uniform(-0.1, 0.1, size=self._dim)
            vec += wv
        if len(words) > 0:
            vec /= len(words)
        return vec

    def embed(self, texts: pd.Series) -> np.ndarray:
        cleaned = texts.fillna("").astype(str)
        embeddings = np.zeros((len(cleaned), self._dim))
        for i, text in enumerate(cleaned):
            if text.strip():
                embeddings[i] = self._simple_embed(text)
        return embeddings

    def similarity_matrix(
        self, texts_a: pd.Series, texts_b: pd.Series | None = None
    ) -> np.ndarray:
        emb_a = self.embed(texts_a)
        emb_b = self.embed(texts_b) if texts_b is not None else emb_a
        return cosine_similarity(emb_a, emb_b)

    def find_similar(
        self, query: str, texts: pd.Series, top_k: int = 5
    ) -> list[tuple[int, float]]:
        query_emb = self._simple_embed(query).reshape(1, -1)
        all_emb = self.embed(texts)
        sims = cosine_similarity(query_emb, all_emb).flatten()
        top_indices = sims.argsort()[-top_k:][::-1]
        return [(int(i), float(sims[i])) for i in top_indices]
