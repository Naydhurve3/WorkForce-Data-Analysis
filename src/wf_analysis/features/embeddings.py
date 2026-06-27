import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from wf_analysis.features.base import BaseFeatureTransformer


class EmbeddingTransformer(BaseFeatureTransformer):
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.pca_models: dict[str, PCA] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "EmbeddingTransformer":
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        text_cols = {
            "Title": "Title",
            "JobFunctionDescription": "FuncDesc",
            "Division": "Division",
        }
        for src_col, prefix in text_cols.items():
            if src_col in df.columns:
                labels = df[src_col].fillna("").astype(str)
                unique = labels.unique()
                label_to_id = {l: i for i, l in enumerate(unique)}
                id_matrix = np.array([label_to_id[l] for l in labels]).reshape(-1, 1)
                id_matrix = np.hstack(
                    [id_matrix, (id_matrix % 7).astype(float)]
                )
                pca = PCA(n_components=min(self.n_components, 2))
                reduced = pca.fit_transform(id_matrix)
                for i in range(reduced.shape[1]):
                    df[f"{prefix}_Emb_{i}"] = reduced[:, i]
                self.pca_models[src_col] = pca
        return df
