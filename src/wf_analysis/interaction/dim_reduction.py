import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import networkx as nx


class DimReduction:
    def __init__(self, config):
        self.cfg = config

    def _prepare_numeric(self, df):
        numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        numeric = numeric.dropna()
        if len(numeric) == 0:
            raise ValueError("No numeric data available after dropping NaN rows")
        return numeric

    def run_pca(self, df):
        logger.info("Running PCA")
        numeric = self._prepare_numeric(df)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric)

        n = min(self.cfg.dim_reduction.pca_n_components, scaled.shape[1], scaled.shape[0])
        pca = PCA(n_components=n, random_state=self.cfg.random_state)
        transformed = pca.fit_transform(scaled)

        result = {
            "transformed": transformed,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "components": pca.components_.tolist(),
            "feature_names": numeric.columns.tolist(),
            "singular_values": pca.singular_values_.tolist(),
            "n_components": n,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }
        logger.info(f"PCA complete: {n} components, "
                     f"cumulative variance: {result['cumulative_variance'][-1]:.2%}")
        return result

    def run_tsne(self, df):
        logger.info("Running t-SNE with multiple perplexities")
        numeric = self._prepare_numeric(df)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric)

        n_neighbors = min(30, scaled.shape[0] - 1)
        perplexities = [p for p in self.cfg.dim_reduction.tsne_perplexities if p < scaled.shape[0] - 1]
        if not perplexities:
            perplexities = [min(5, n_neighbors)]

        results = {}
        for perp in perplexities:
            tsne = TSNE(
                n_components=2, perplexity=perp,
                random_state=self.cfg.dim_reduction.tsne_random_state,
                max_iter=500, init="pca",
            )
            embedded = tsne.fit_transform(scaled)
            kl_div = float(tsne.kl_divergence_)
            results[f"perp_{perp}"] = {
                "embedding": embedded,
                "kl_divergence": kl_div,
                "perplexity": perp,
            }
            logger.info(f"  t-SNE perplexity={perp}: KL divergence={kl_div:.2f}")

        return results

    def build_correlation_network(self, df):
        logger.info("Building correlation network")
        numeric = df.select_dtypes(include=[np.number])
        corr = numeric.corr()

        threshold = self.cfg.dim_reduction.correlation_threshold
        G = nx.Graph()
        edges = []
        for i, c1 in enumerate(corr.columns):
            for j, c2 in enumerate(corr.columns):
                if i < j:
                    val = corr.loc[c1, c2]
                    if abs(val) >= threshold:
                        edges.append({"source": c1, "target": c2, "weight": round(abs(float(val)), 3), "sign": "pos" if val > 0 else "neg"})
                        G.add_edge(c1, c2, weight=abs(float(val)))

        component_sizes = [len(c) for c in nx.connected_components(G)] if len(G.nodes) > 0 else []

        result = {
            "edges": edges,
            "nodes": list(G.nodes),
            "n_nodes": len(G.nodes),
            "n_edges": len(edges),
            "density": round(float(nx.density(G)), 4) if len(G.nodes) > 0 else 0,
            "component_sizes": sorted(component_sizes, reverse=True) if component_sizes else [],
            "threshold": threshold,
        }
        logger.info(f"Network: {result['n_nodes']} nodes, {result['n_edges']} edges, density={result['density']}")
        return {"graph": G, "correlation_matrix": corr, "result": result}

    def compute_silhouette_scores(self, pca_embedding, tsne_results):
        scores = {}
        try:
            for k in range(2, min(8, pca_embedding.shape[0])):
                labels = KMeans(n_clusters=k, random_state=self.cfg.random_state, n_init=10).fit_predict(pca_embedding)
                if len(set(labels)) > 1:
                    scores[f"pca_k{k}"] = round(float(silhouette_score(pca_embedding, labels)), 4)
        except Exception as e:
            logger.warning(f"Silhouette for PCA failed: {e}")

        for key, tsne_data in tsne_results.items():
            emb = tsne_data["embedding"]
            try:
                for k in range(2, min(8, emb.shape[0])):
                    labels = KMeans(n_clusters=k, random_state=self.cfg.random_state, n_init=10).fit_predict(emb)
                    if len(set(labels)) > 1:
                        scores[f"{key}_k{k}"] = round(float(silhouette_score(emb, labels)), 4)
            except Exception as e:
                logger.warning(f"Silhouette for {key} failed: {e}")

        return scores

    def run_all(self, df):
        logger.info("Running all dimensionality reductions")
        pca = self.run_pca(df)
        tsne = self.run_tsne(df)
        network = self.build_correlation_network(df)
        silhouette = self.compute_silhouette_scores(pca["transformed"], tsne)

        return {
            "pca": pca,
            "tsne": tsne,
            "network": network,
            "silhouette": silhouette,
        }
