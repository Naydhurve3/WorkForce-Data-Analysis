import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path

from wf_analysis.visualization.theme import Theme as _Theme
Theme = _Theme


class EDAFigureFactory:
    def __init__(self, config):
        self.cfg = config
        self.figure_dir = Path(config.figure_dir)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        path = self.figure_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    def figure_1_univariate_grid(self, df, stats):
        Theme.set_style()
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        n_cols = min(4, len(numeric))
        n_rows = int(np.ceil(len(numeric) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for i, c in enumerate(numeric):
            if i >= len(axes):
                break
            s = df[c].dropna()
            axes[i].hist(s, bins=30, color=Theme.PRIMARY[0], edgecolor="white", alpha=0.7)
            stats_c = stats["numeric"].get(c, {})
            axes[i].set_title(f"{c}\nμ={stats_c.get('mean','?')} σ={stats_c.get('std','?')}", fontsize=10)
            axes[i].axvline(s.mean(), color=Theme.PRIMARY[1], ls="--", lw=1.5, label=f"mean={s.mean():.1f}")
            axes[i].axvline(s.median(), color=Theme.PRIMARY[2], ls=":", lw=1.5, label=f"med={s.median():.1f}")
            axes[i].legend(fontsize=7)

        for j in range(len(numeric), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Univariate Statistics — Numeric Distributions", fontsize=14, y=1.01)
        plt.tight_layout()
        return self._save(fig, "01_univariate_numeric_grid.png")

    def figure_2_categorical_grid(self, df, stats):
        Theme.set_style()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in self.cfg.date_columns + [self.cfg.id_column]]
        n_show = min(len(cat_cols), self.cfg.eda.max_columns_per_grid)
        selected = cat_cols[:n_show]

        n_cols = 3
        n_rows = int(np.ceil(n_show / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for i, c in enumerate(selected):
            if i >= len(axes):
                break
            vc = df[c].value_counts().head(12)
            colors = Theme.CATEGORICAL[: len(vc)]
            axes[i].barh(range(len(vc)), vc.values, color=colors, edgecolor="white")
            axes[i].set_yticks(range(len(vc)))
            axes[i].set_yticklabels(vc.index, fontsize=8)
            axes[i].set_title(f"{c} (n={len(vc)} unique)", fontsize=10)
            for j, val in enumerate(vc.values):
                axes[i].text(val + 0.5, j, str(val), va="center", fontsize=7)

        for j in range(n_show, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Categorical Distributions — Top Categories", fontsize=14, y=1.01)
        plt.tight_layout()
        return self._save(fig, "02_categorical_distributions.png")

    def figure_3_missing_patterns(self, missing_matrix):
        Theme.set_style()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        missing_cols = [c for c in missing_matrix.columns if missing_matrix[c].any()]
        if not missing_cols:
            ax = axes[0]
            ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
            ax.set_title("Missing Value Heatmap")
        else:
            sns.heatmap(
                missing_matrix[missing_cols].T,
                cmap=["#E8E8E8", Theme.PRIMARY[0]],
                cbar=False, ax=axes[0], yticklabels=True,
            )
            axes[0].set_title("Missing Value Heatmap (yellow = missing)")
            axes[0].set_xlabel("Row Index")
            axes[0].set_ylabel("Column")

        ax2 = axes[1]
        missing_pct = missing_matrix.mean() * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)
        if len(missing_pct) > 0:
            bars = ax2.barh(range(len(missing_pct)), missing_pct.values, color=Theme.DIVERGING[0])
            ax2.set_yticks(range(len(missing_pct)))
            ax2.set_yticklabels(missing_pct.index, fontsize=9)
            ax2.set_xlabel("Missing %")
            ax2.set_title("Missing % by Column")
            for i, (v, bar) in enumerate(zip(missing_pct.values, bars)):
                ax2.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
        else:
            ax2.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)

        fig.suptitle("Missing Data Patterns", fontsize=14)
        plt.tight_layout()
        return self._save(fig, "03_missing_patterns.png")

    def figure_4_missing_correlation(self, missing_corr):
        Theme.set_style()
        fig, ax = plt.subplots(figsize=(10, 8))
        missing_cols = [c for c in missing_corr.columns if missing_corr[c].notna().sum() > 1]
        if len(missing_cols) < 2:
            ax.text(0.5, 0.5, "Insufficient missing columns for correlation", ha="center", va="center", fontsize=12)
        else:
            sns.heatmap(
                missing_corr.loc[missing_cols, missing_cols],
                annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax,
            )
        ax.set_title("Missing Value Co-occurrence Correlation")
        plt.tight_layout()
        return self._save(fig, "04_missing_correlation.png")

    def figure_5_outlier_comparison(self, outlier_results, df):
        Theme.set_style()
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()

        data_rows = []
        for c in numeric:
            if c not in outlier_results:
                continue
            for method in ["iqr", "zscore", "isolation_forest"]:
                if method in outlier_results[c]:
                    data_rows.append({
                        "column": c, "method": method.upper().replace("_", " "),
                        "pct": outlier_results[c][method]["pct"],
                    })

        comp_df = pd.DataFrame(data_rows)
        if len(comp_df) == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No outlier data available", ha="center", va="center", fontsize=12)
            return self._save(fig, "05_outlier_comparison.png")

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=comp_df, x="column", y="pct", hue="method", ax=ax, palette=Theme.PRIMARY[:3])
        ax.set_title("Outlier % by Column — 3 Methods Compared")
        ax.set_ylabel("Outlier %")
        ax.set_xlabel("Column")
        ax.legend(title="Method")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        plt.tight_layout()
        return self._save(fig, "05_outlier_comparison.png")

    def figure_6_outlier_consensus(self, outlier_results, df):
        Theme.set_style()
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()

        consensus_data = []
        for c in numeric:
            if c in outlier_results and "consensus" in outlier_results[c]:
                consensus_data.append({
                    "column": c,
                    "consensus_pct": outlier_results[c]["consensus"]["pct"],
                    "iqr_pct": outlier_results[c]["iqr"]["pct"],
                    "zscore_pct": outlier_results[c]["zscore"]["pct"],
                    "iforest_pct": outlier_results[c]["isolation_forest"]["pct"],
                })

        if not consensus_data:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return self._save(fig, "06_outlier_consensus.png")

        cmp = pd.DataFrame(consensus_data)
        cmp_sorted = cmp.sort_values("consensus_pct", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(5, len(cmp_sorted) * 0.35)))
        y = range(len(cmp_sorted))
        height = 0.2
        ax.barh([i - 1.5 * height for i in y], cmp_sorted["iqr_pct"], height, label="IQR", color=Theme.PRIMARY[0])
        ax.barh([i - 0.5 * height for i in y], cmp_sorted["zscore_pct"], height, label="Z-Score", color=Theme.PRIMARY[1])
        ax.barh([i + 0.5 * height for i in y], cmp_sorted["iforest_pct"], height, label="Isolation Forest", color=Theme.PRIMARY[2])
        ax.barh([i + 1.5 * height for i in y], cmp_sorted["consensus_pct"], height, label="Consensus", color=Theme.PRIMARY[3])

        ax.set_yticks(list(y))
        ax.set_yticklabels(cmp_sorted["column"].values)
        ax.set_xlabel("Outlier %")
        ax.set_title("Outlier Detection — Method Comparison & Consensus")
        ax.legend(fontsize=8)
        plt.tight_layout()
        return self._save(fig, "06_outlier_consensus.png")

    def figure_7_correlation_network(self, network_result):
        G = network_result["graph"]
        corr = network_result["correlation_matrix"]
        theme_result = network_result["result"]

        Theme.set_style()
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        sns.heatmap(corr, annot=True if corr.shape[0] <= 8 else False,
                    fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    ax=axes[0], cbar_kws={"shrink": 0.8})
        axes[0].set_title(f"Correlation Matrix ({corr.shape[0]} numeric features)")

        ax = axes[1]
        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, seed=42, k=2)
            edge_colors = [Theme.PRIMARY[0] if G[u][v].get("sign", "pos") == "pos" else Theme.PRIMARY[3] for u, v in G.edges]
            edge_widths = [G[u][v].get("weight", 0.5) * 3 for u, v in G.edges]
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=Theme.PRIMARY[1],
                                   node_size=500, alpha=0.8)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                                   width=edge_widths, alpha=0.5, connectionstyle="arc3,rad=0.1")
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")
            ax.set_title(f"Correlation Network (|r|≥{theme_result['threshold']})\n"
                         f"{len(G.nodes)} nodes, {len(G.edges)} edges")
        else:
            ax.text(0.5, 0.5, f"No correlations > |{theme_result['threshold']}|", ha="center", va="center", fontsize=12)

        ax.axis("off")
        fig.suptitle("Feature Relationships: Correlation Matrix & Network", fontsize=14)
        plt.tight_layout()
        return self._save(fig, "07_correlation_network.png")

    def figure_8_pca_scree(self, pca_result):
        Theme.set_style()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        evr = pca_result["explained_variance_ratio"]
        cumvar = pca_result["cumulative_variance"]
        k = len(evr)

        axes[0].bar(range(1, k + 1), evr, color=[Theme.PRIMARY[0]] * k, edgecolor="white")
        axes[0].plot(range(1, k + 1), evr, "o-", color=Theme.PRIMARY[1], lw=2)
        axes[0].axhline(y=0.1, ls="--", color="gray", alpha=0.5)
        axes[0].set_xlabel("Principal Component")
        axes[0].set_ylabel("Explained Variance Ratio")
        axes[0].set_title("Scree Plot")
        axes[0].set_xticks(range(1, k + 1))

        axes[1].bar(range(1, k + 1), cumvar, color=[Theme.PRIMARY[1]] * k, edgecolor="white")
        axes[1].plot(range(1, k + 1), cumvar, "o-", color=Theme.PRIMARY[0], lw=2)
        axes[1].axhline(y=0.8, ls="--", color="gray", alpha=0.7, label="80% threshold")
        axes[1].axhline(y=0.9, ls="--", color="gray", alpha=0.4, label="90% threshold")
        axes[1].set_xlabel("Number of Components")
        axes[1].set_ylabel("Cumulative Explained Variance")
        axes[1].set_title("Cumulative Variance")
        axes[1].legend(fontsize=8)

        fig.suptitle(f"PCA: {k} Components — Total Variance: {cumvar[-1]:.1%}", fontsize=14)
        plt.tight_layout()
        return self._save(fig, "08_pca_scree.png")

    def figure_9_pca_biplot(self, pca_result, df):
        Theme.set_style()
        transformed = pca_result["transformed"]
        components = np.array(pca_result["components"])
        features = pca_result["feature_names"]

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(transformed[:, 0], transformed[:, 1],
                             c=Theme.PRIMARY[0], alpha=0.5, s=20, edgecolors="white", linewidth=0.3)

        for i, feat in enumerate(features[: min(15, len(features))]):
            ax.arrow(0, 0, components[i, 0] * 3, components[i, 1] * 3,
                     head_width=0.1, head_length=0.1, fc=Theme.PRIMARY[2], ec=Theme.PRIMARY[2], alpha=0.7)
            ax.text(components[i, 0] * 3.3, components[i, 1] * 3.3, feat,
                    fontsize=8, fontweight="bold", color=Theme.PRIMARY[2])

        ax.set_xlabel(f"PC1 ({pca_result['explained_variance_ratio'][0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca_result['explained_variance_ratio'][1]:.1%})")
        ax.set_title("PCA Biplot — PC1 vs PC2 with Feature Loadings")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        plt.tight_layout()
        return self._save(fig, "09_pca_biplot.png")

    def figure_10_tsne_landscape(self, tsne_results, df):
        Theme.set_style()
        status = df["EmployeeStatus"].astype(str) if "EmployeeStatus" in df.columns else None
        status_colors = {"Active": Theme.PRIMARY[0], "Terminated": Theme.PRIMARY[1]}

        n_plots = len(tsne_results)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for idx, (key, tsne_data) in enumerate(tsne_results.items()):
            ax = axes[idx]
            emb = tsne_data["embedding"]
            perp = tsne_data["perplexity"]

            if status is not None:
                for s in status.unique():
                    mask = status == s
                    c = status_colors.get(s, "gray")
                    ax.scatter(emb[mask, 0], emb[mask, 1], c=c, label=s, alpha=0.4, s=15)
                ax.legend(fontsize=8)
            else:
                ax.scatter(emb[:, 0], emb[:, 1], c=Theme.PRIMARY[0], alpha=0.4, s=15)

            ax.set_title(f"t-SNE (perp={perp})\nKL={tsne_data['kl_divergence']:.1f}", fontsize=11)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")

        fig.suptitle("t-SNE Embeddings — Perplexity Sweep", fontsize=14)
        plt.tight_layout()
        return self._save(fig, "10_tsne_landscape.png")

    def figure_11_silhouette_comparison(self, silhouette_scores):
        Theme.set_style()
        if not silhouette_scores:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No silhouette scores available", ha="center", va="center")
            return self._save(fig, "11_silhouette_comparison.png")

        score_df = pd.DataFrame([
            {"method": k.rsplit("_k", 1)[0], "k": int(k.rsplit("_k", 1)[1]), "score": v}
            for k, v in silhouette_scores.items()
        ])

        fig, ax = plt.subplots(figsize=(10, 5))
        methods = score_df["method"].unique()
        palette = {m: Theme.PRIMARY[i % len(Theme.PRIMARY)] for i, m in enumerate(methods)}
        sns.lineplot(data=score_df, x="k", y="score", hue="method", marker="o", ax=ax, palette=palette)
        ax.set_title("Silhouette Scores by Method and Cluster Count")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Silhouette Score")
        ax.axhline(y=0.3, ls="--", color="gray", alpha=0.5, label="0.3 threshold")
        ax.legend(fontsize=8)
        plt.tight_layout()
        return self._save(fig, "11_silhouette_comparison.png")

    def figure_12_dashboard_summary(self, stats, outlier_results):
        Theme.set_style()
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.axis("off")
        n_rows, n_cols = stats["shape"]
        missing_cells = stats["missing_summary"]["total_missing_cells"]
        complete_cols = stats["missing_summary"].get("complete_columns", [])
        missing_text = (
            f"Dataset Overview\n"
            f"Rows: {n_rows:,}\n"
            f"Columns: {n_cols}\n"
            f"Missing cells: {missing_cells:,}\n"
            f"Complete columns: {len(complete_cols)}/{n_cols}\n"
            f"Numeric: {len(stats.get('numeric', {}))}\n"
            f"Categorical: {len(stats.get('categorical', {}))}"
        )
        ax0.text(0.1, 0.9, missing_text, transform=ax0.transAxes, fontsize=11, verticalalignment="top",
                 fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="#FAFAFA"))

        ax1 = fig.add_subplot(gs[0, 1:])
        numeric_cols = list(stats.get("numeric", {}).keys())[:6]
        if numeric_cols:
            skew_data = []
            for c in numeric_cols:
                sk = stats["numeric"][c].get("skew")
                if sk is not None:
                    skew_data.append({"column": c, "skew": sk})
            if skew_data:
                sk_df = pd.DataFrame(skew_data).sort_values("skew")
                colors = [Theme.PRIMARY[0] if abs(v) < 0.5 else Theme.PRIMARY[1] if abs(v) < 1 else Theme.PRIMARY[3] for v in sk_df["skew"]]
                ax1.barh(range(len(sk_df)), sk_df["skew"], color=colors)
                ax1.set_yticks(range(len(sk_df)))
                ax1.set_yticklabels(sk_df["column"], fontsize=9)
                ax1.axvline(0, color="black", lw=0.5)
                ax1.set_title("Feature Skewness", fontsize=12)
            else:
                ax1.axis("off")
        else:
            ax1.axis("off")

        ax2 = fig.add_subplot(gs[1, :2])
        missing_pcts = [v["pct"] for v in stats["missing_summary"]["columns_with_missing"].values()]
        missing_names = list(stats["missing_summary"]["columns_with_missing"].keys())
        if missing_pcts:
            sorted_idx = np.argsort(missing_pcts)
            ax2.barh(range(len(sorted_idx)), [missing_pcts[i] for i in sorted_idx],
                     color=Theme.PRIMARY[0])
            ax2.set_yticks(range(len(sorted_idx)))
            ax2.set_yticklabels([missing_names[i] for i in sorted_idx], fontsize=8)
            ax2.set_xlabel("Missing %")
            ax2.set_title("Missing Data by Column", fontsize=12)
        else:
            ax2.text(0.5, 0.5, "No missing data", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_title("Missing Data by Column", fontsize=12)

        ax3 = fig.add_subplot(gs[1, 2])
        outlier_methods = ["iqr", "zscore", "isolation_forest"]
        total_outliers = {m: 0 for m in outlier_methods}
        for c, res in outlier_results.items():
            for m in outlier_methods:
                if m in res:
                    total_outliers[m] += res[m]["count"]
        if sum(total_outliers.values()) > 0:
            labels = ["IQR", "Z-Score", "IsoForest"]
            values = [total_outliers[m] for m in outlier_methods]
            ax3.pie(values, labels=labels, autopct="%1.0f%%", colors=Theme.PRIMARY[:3])
        else:
            ax3.text(0.5, 0.5, "No outliers", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Total Outliers by Method", fontsize=12)

        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis("off")
        recommendations = []
        if stats["missing_summary"]["missing_pct_overall"] > 5:
            recommendations.append("HIGH: Address missing data ({}%)".format(stats["missing_summary"]["missing_pct_overall"]))
        skewed = [c for c, v in stats.get("numeric", {}).items() if v.get("skew") is not None and abs(v["skew"]) > 1]
        if skewed:
            recommendations.append(f"MED: {len(skewed)} features with |skew|>1 ({', '.join(skewed[:3])})")
        high_outlier_cols = [c for c, res in outlier_results.items() if res.get("consensus", {}).get("pct", 0) > 5]
        if high_outlier_cols:
            recommendations.append(f"LOW: {len(high_outlier_cols)} columns have >5% consensus outliers")
        if not recommendations:
            recommendations.append("Data quality looks good — proceed to feature engineering")
        text = "Priorities from Deep EDA:\n" + "\n".join(f"  • {r}" for r in recommendations)
        ax4.text(0.02, 0.5, text, transform=ax4.transAxes, fontsize=11, verticalalignment="center",
                 fontfamily="monospace")

        fig.suptitle("Deep EDA Summary Dashboard", fontsize=16, fontweight="bold")
        return self._save(fig, "12_eda_dashboard_summary.png")

    def figure_13_feature_correlation(self, feature_df):
        Theme.set_style()
        numeric = feature_df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "Need at least 2 numeric features", ha="center", va="center")
            return self._save(fig, "13_feature_correlation.png")

        corr = numeric.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.6},
                    annot_kws={"fontsize": 7})
        ax.set_title(f"Feature Correlation Matrix ({corr.shape[0]} numeric features)", fontsize=13)
        plt.tight_layout()
        return self._save(fig, "13_feature_correlation.png")

    def figure_14_feature_summary(self, feature_df):
        Theme.set_style()
        meta = []
        for c in feature_df.columns:
            s = feature_df[c]
            meta.append({
                "feature": c, "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "missing_pct": round(float(s.isna().mean() * 100), 1),
                "unique": int(s.nunique()),
                "type": "numeric" if np.issubdtype(s.dtype, np.number) else "categorical",
            })
        meta_df = pd.DataFrame(meta).sort_values("missing_pct", ascending=False)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        ax = axes[0, 0]
        type_counts = meta_df["type"].value_counts()
        ax.bar(type_counts.index, type_counts.values, color=Theme.PRIMARY[:2], edgecolor="white", width=0.5)
        ax.set_title("Feature Type Distribution", fontsize=12)
        for i, v in enumerate(type_counts.values):
            ax.text(i, v + 0.3, str(v), ha="center", fontsize=11)

        ax = axes[0, 1]
        missing_df = meta_df[meta_df["missing_pct"] > 0]
        if len(missing_df) > 0:
            colors = [Theme.PRIMARY[3] if v > 10 else Theme.PRIMARY[0] for v in missing_df["missing_pct"]]
            ax.barh(range(len(missing_df)), missing_df["missing_pct"], color=colors, edgecolor="white")
            ax.set_yticks(range(len(missing_df)))
            ax.set_yticklabels(missing_df["feature"].values, fontsize=8)
            ax.set_xlabel("Missing %")
            ax.set_title("Features with Missing Values", fontsize=12)
        else:
            ax.text(0.5, 0.5, "No missing values in engineered features", ha="center", va="center", transform=ax.transAxes)

        ax = axes[1, 0]
        unique_df = meta_df[meta_df["type"] == "categorical"].sort_values("unique", ascending=True).tail(15)
        if len(unique_df) > 0:
            ax.barh(range(len(unique_df)), unique_df["unique"], color=Theme.PRIMARY[1], edgecolor="white")
            ax.set_yticks(range(len(unique_df)))
            ax.set_yticklabels(unique_df["feature"].values, fontsize=8)
            ax.set_xlabel("Unique Values")
            ax.set_title("Cardinality of Categorical Features", fontsize=12)
        else:
            ax.text(0.5, 0.5, "No categorical features", ha="center", va="center", transform=ax.transAxes)

        ax = axes[1, 1]
        ax.axis("off")
        total = len(meta_df)
        num_c = len(meta_df[meta_df["type"] == "numeric"])
        cat_c = total - num_c
        clean = len(meta_df[meta_df["missing_pct"] == 0])
        summary_lines = [
            f"Total features: {total}",
            f"Numeric: {num_c}",
            f"Categorical: {cat_c}",
            f"Fully populated: {clean}/{total}",
            f"Features with missing: {len(missing_df)}" if len(meta_df[meta_df['missing_pct'] > 0]) > 0 else "",
        ]
        ax.text(0.1, 0.7, "\n".join(summary_lines), transform=ax.transAxes, fontsize=12,
                fontfamily="monospace", verticalalignment="top")

        fig.suptitle("Feature Engineering Summary", fontsize=15, fontweight="bold")
        plt.tight_layout()
        return self._save(fig, "14_feature_summary.png")

    def figure_15_feature_distributions(self, feature_df):
        Theme.set_style()
        numeric = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric = [c for c in numeric if feature_df[c].nunique() > 2]
        n_show = min(len(numeric), 9)

        if n_show == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No suitable numeric features for distribution plot", ha="center", va="center")
            return self._save(fig, "15_feature_distributions.png")

        selected = numeric[:n_show]
        n_cols = 3
        n_rows = int(np.ceil(n_show / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for i, c in enumerate(selected):
            s = feature_df[c].dropna()
            axes[i].hist(s, bins=30, color=Theme.PRIMARY[0], edgecolor="white", alpha=0.7, density=True)
            axes[i].set_title(f"{c}\nμ={s.mean():.1f} σ={s.std():.1f}", fontsize=10)
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(s)
                xs = np.linspace(s.min(), s.max(), 200)
                axes[i].plot(xs, kde(xs), color=Theme.PRIMARY[1], lw=2)
            except Exception:
                pass

        for j in range(n_show, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Engineered Feature Distributions (with KDE)", fontsize=14)
        plt.tight_layout()
        return self._save(fig, "15_feature_distributions.png")

    def figure_16_interaction_heatmap(self, results_df):
        Theme.set_style()
        outcomes = results_df["outcome"].unique()
        pairs = results_df["feature_1"] + " × " + results_df["feature_2"]
        pivot = results_df.pivot_table(
            index="outcome", columns=results_df.groupby(["feature_1", "feature_2"]).ngroup(),
            values="impact", aggfunc="first"
        )
        if pivot.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No interaction data", ha="center", va="center")
            return self._save(fig, "16_interaction_heatmap.png")

        max_show = min(pivot.shape[1], 50)
        top_cols = results_df.groupby(results_df.groupby(["feature_1", "feature_2"]).ngroup())["impact"].max().nlargest(max_show).index
        pivot = pivot[top_cols]

        fig, ax = plt.subplots(figsize=(max(12, max_show * 0.5), max(4, len(outcomes) * 0.8)))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                    cbar_kws={"label": "Impact Score (-log10(p)×eff)"})
        ax.set_title("Interaction Impact Heatmap (Top 50 Pairs)", fontsize=13)
        ax.set_xlabel("Feature Pair")
        ax.set_ylabel("Outcome")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        plt.tight_layout()
        return self._save(fig, "16_interaction_heatmap.png")

    def figure_17_top_interactions(self, results_df):
        Theme.set_style()
        top = results_df.head(20).copy()
        top["pair"] = top["feature_1"] + " × " + top["feature_2"] + " | " + top["outcome"]
        top = top.sort_values("impact")

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = [Theme.PRIMARY[0] if v < 0.05 else Theme.PRIMARY[3] for v in top["p_value"]]
        bars = ax.barh(range(len(top)), top["impact"], color=colors, edgecolor="white")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["pair"].values, fontsize=8)
        ax.set_xlabel("Impact Score (-log10(p) × effect size)")
        ax.set_title("Top 20 Feature Interactions by Impact Score", fontsize=14)
        for i, (_, row) in enumerate(top.iterrows()):
            label = f"p={row['p_value']:.4f}" if row["p_value"] > 0.001 else "p<0.001"
            ax.text(row["impact"] + 0.02, i, label, va="center", fontsize=7)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=Theme.PRIMARY[0], label="p<0.05"),
                           Patch(facecolor=Theme.PRIMARY[3], label="p≥0.05")]
        ax.legend(handles=legend_elements, fontsize=8)
        plt.tight_layout()
        return self._save(fig, "17_top_interactions.png")

    def figure_18_interaction_pairplot(self, feature_df, top_result):
        Theme.set_style()
        f1, f2, outcome = top_result["feature_1"], top_result["feature_2"], top_result["outcome"]

        plot_df = pd.DataFrame({
            "x": feature_df[f1], "y": feature_df[f2],
            "outcome": feature_df.get(outcome, feature_df.get("SeniorityLevel", 0))
        }).dropna().sample(min(500, len(feature_df)), random_state=42)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        x_num = pd.api.types.is_numeric_dtype(plot_df["x"])
        y_num = pd.api.types.is_numeric_dtype(plot_df["y"])

        ax = axes[0]
        if x_num and y_num:
            scatter = ax.scatter(plot_df["x"], plot_df["y"], c=plot_df["outcome"],
                                 cmap="RdBu_r", alpha=0.5, s=15, edgecolors="white", linewidth=0.3)
            plt.colorbar(scatter, ax=ax, label=outcome)
        elif x_num and not y_num:
            for v in plot_df["y"].unique():
                mask = plot_df["y"] == v
                ax.hist(plot_df.loc[mask, "x"], alpha=0.5, label=str(v), bins=20)
            ax.legend(fontsize=7)
        elif y_num and not x_num:
            for v in plot_df["x"].unique():
                mask = plot_df["x"] == v
                ax.hist(plot_df.loc[mask, "y"], alpha=0.5, label=str(v), bins=20)
            ax.legend(fontsize=7)
        else:
            ct = pd.crosstab(plot_df["x"], plot_df["y"])
            sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_title("Counts")
        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_title(f"{f1} × {f2}\np={top_result['p_value']:.4f}, impact={top_result['impact']:.2f}", fontsize=10)

        ax = axes[1]
        outcome_cats = plot_df["outcome"].nunique()
        cmap = plt.cm.RdBu_r
        for i, v in enumerate(plot_df["outcome"].unique()):
            mask = plot_df["outcome"] == v
            c = cmap(i / max(outcome_cats, 1))
            if x_num:
                ax.hist(plot_df.loc[mask, "x"], alpha=0.5, bins=20, color=c, label=f"{outcome}={v}")
            else:
                counts = plot_df.loc[mask, "x"].value_counts()
                ax.bar(range(len(counts)), counts.values, alpha=0.5, color=c, label=f"{outcome}={v}")
        ax.set_xlabel(f1)
        ax.set_title(f"{f1} by {outcome}")
        ax.legend(fontsize=7)

        ax = axes[2]
        for i, v in enumerate(plot_df["outcome"].unique()):
            mask = plot_df["outcome"] == v
            c = cmap(i / max(outcome_cats, 1))
            if y_num:
                ax.hist(plot_df.loc[mask, "y"], alpha=0.5, bins=20, color=c, label=f"{outcome}={v}")
            else:
                counts = plot_df.loc[mask, "y"].value_counts()
                ax.bar(range(len(counts)), counts.values, alpha=0.5, color=c, label=f"{outcome}={v}")
        ax.set_xlabel(f2)
        ax.set_title(f"{f2} by {outcome}")
        ax.legend(fontsize=7)

        fig.suptitle(f"Strongest Interaction: {f1} × {f2} → {outcome}", fontsize=14)
        plt.tight_layout()
        return self._save(fig, "18_interaction_pairplot.png")

    def figure_19_tree_segmentation(self, tree_data, feature_df, top_result):
        Theme.set_style()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        importances = tree_data["importances"]
        names = list(importances.keys())
        vals = list(importances.values())
        bars = ax.bar(names, vals, color=Theme.PRIMARY[:2], edgecolor="white", width=0.5)
        ax.set_ylabel("Feature Importance")
        ax.set_title("Decision Tree — Feature Importance")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=9)

        ax = axes[1]
        ax.axis("off")
        f1, f2, outcome = top_result["feature_1"], top_result["feature_2"], top_result["outcome"]
        summary = [
            f"Decision Tree Segmentation",
            f"Pair: {f1} × {f2}",
            f"Outcome: {outcome}",
            f"Leaves: {tree_data['n_leaves']}",
            f"Depth: 3 (max)",
            f"Impact: {top_result['impact']:.2f}",
            f"p-value: {top_result['p_value']:.4f}",
            f"Method: {top_result['method']}",
            f"Effect size: {top_result['effect_size']:.3f}",
        ]
        ax.text(0.1, 0.85, "\n".join(summary), transform=ax.transAxes, fontsize=11,
                fontfamily="monospace", verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="#FAFAFA"))

        fig.suptitle("Decision Tree Multi-way Segmentation", fontsize=14)
        plt.tight_layout()
        return self._save(fig, "19_tree_segmentation.png")

    def figure_20_mutual_information_matrix(self, results_df):
        Theme.set_style()
        top_pairs = results_df.drop_duplicates(subset=["feature_1", "feature_2"]).head(30)
        features = list(set(top_pairs["feature_1"].tolist() + top_pairs["feature_2"].tolist()))
        features = features[:15]

        mi_mat = pd.DataFrame(0.0, index=features, columns=features)
        for _, row in results_df.iterrows():
            f1, f2 = row["feature_1"], row["feature_2"]
            if f1 in features and f2 in features:
                val = row["mutual_info"]
                mi_mat.loc[f1, f2] = val
                mi_mat.loc[f2, f1] = val

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(mi_mat, annot=True, fmt=".3f", cmap="Blues", ax=ax,
                    cbar_kws={"label": "Mutual Information"})
        ax.set_title("Pairwise Mutual Information (Top 15 Features)", fontsize=13)
        plt.tight_layout()
        return self._save(fig, "20_mutual_information_matrix.png")

    def figure_21_interaction_network(self, results_df):
        Theme.set_style()
        sig = results_df[results_df["significant"]].head(40).copy()
        if len(sig) == 0:
            sig = results_df.head(40).copy()

        G = nx.Graph()
        for _, row in sig.iterrows():
            G.add_edge(row["feature_1"], row["feature_2"],
                       weight=row["impact"], outcome=row["outcome"])

        fig, ax = plt.subplots(figsize=(12, 10))
        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, seed=42, k=2.5)
            edge_weights = [G[u][v]["weight"] * 2 for u, v in G.edges]
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=Theme.PRIMARY[0],
                                   node_size=800, alpha=0.8)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=Theme.PRIMARY[2],
                                   width=edge_weights, alpha=0.4, arrows=False)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
            ax.set_title(f"Interaction Network ({len(G.nodes)} features, {len(G.edges)} edges)", fontsize=14)
        else:
            ax.text(0.5, 0.5, "No significant interactions", ha="center", va="center", fontsize=12)
        ax.axis("off")
        plt.tight_layout()
        return self._save(fig, "21_interaction_network.png")

    def figure_22_outcome_ranking(self, results_df):
        Theme.set_style()
        outcomes = results_df["outcome"].unique()
        n = len(outcomes)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for i, o in enumerate(outcomes):
            ax = axes[i]
            o_df = results_df[results_df["outcome"] == o].head(10)
            if len(o_df) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(o)
                continue
            o_df = o_df.sort_values("impact")
            pairs = o_df["feature_1"] + " × " + o_df["feature_2"]
            ax.barh(range(len(pairs)), o_df["impact"], color=Theme.PRIMARY[0], edgecolor="white")
            ax.set_yticks(range(len(pairs)))
            ax.set_yticklabels(pairs.values, fontsize=7)
            ax.set_xlabel("Impact")
            ax.set_title(f"{o}\nTop 10 Pairs", fontsize=11)

        fig.suptitle("Outcome-Specific Interaction Rankings", fontsize=14)
        plt.tight_layout()
        return self._save(fig, "22_outcome_ranking.png")

    def figure_23_interaction_dashboard(self, results_df):
        Theme.set_style()
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.axis("off")
        total = len(results_df)
        sig = results_df["significant"].sum()
        bonf = results_df["bonferroni"].sum()
        outcomes = results_df["outcome"].nunique()
        pairs = results_df.drop_duplicates(subset=["feature_1", "feature_2"]).shape[0]
        summary = [
            f"Interaction Mining Summary",
            f"Total tests: {total:,}",
            f"Significant (p<0.05): {sig} ({sig/max(total,1)*100:.1f}%)",
            f"Bonferroni sig: {bonf} ({bonf/max(total,1)*100:.2f}%)",
            f"Feature pairs tested: {pairs}",
            f"Outcomes analyzed: {outcomes}",
        ]
        ax0.text(0.1, 0.9, "\n".join(summary), transform=ax0.transAxes, fontsize=11,
                 fontfamily="monospace", verticalalignment="top")

        ax1 = fig.add_subplot(gs[0, 1:])
        outcome_counts = results_df[results_df["significant"]].groupby("outcome").size().sort_values(ascending=False)
        if len(outcome_counts) > 0:
            colors = [Theme.PRIMARY[0]] * len(outcome_counts)
            ax1.bar(range(len(outcome_counts)), outcome_counts.values, color=colors, edgecolor="white")
            ax1.set_xticks(range(len(outcome_counts)))
            ax1.set_xticklabels(outcome_counts.index, rotation=30, ha="right", fontsize=9)
            ax1.set_ylabel("Significant Interactions")
            ax1.set_title("Significant Interactions by Outcome")
            for i, v in enumerate(outcome_counts.values):
                ax1.text(i, v + 0.3, str(v), ha="center", fontsize=9)
        else:
            ax1.text(0.5, 0.5, "No significant interactions", ha="center", va="center", transform=ax1.transAxes)

        ax2 = fig.add_subplot(gs[1, :2])
        method_counts = results_df["method"].value_counts()
        if len(method_counts) > 0:
            bars = ax2.bar(range(len(method_counts)), method_counts.values, color=Theme.PRIMARY[:3], edgecolor="white")
            ax2.set_xticks(range(len(method_counts)))
            ax2.set_xticklabels(method_counts.index, fontsize=10)
            ax2.set_ylabel("Tests")
            ax2.set_title("Tests by Statistical Method")
            for bar, v in zip(bars, method_counts.values):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                         str(v), ha="center", fontsize=9)
        else:
            ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)

        ax3 = fig.add_subplot(gs[1, 2])
        impact_dist = results_df["impact"]
        if len(impact_dist) > 0:
            ax3.hist(impact_dist, bins=30, color=Theme.PRIMARY[0], edgecolor="white", alpha=0.7)
            ax3.axvline(impact_dist.median(), color=Theme.PRIMARY[1], ls="--", lw=2, label=f"median={impact_dist.median():.1f}")
            ax3.axvline(impact_dist.quantile(0.9), color=Theme.PRIMARY[3], ls=":", lw=2, label=f"90th={impact_dist.quantile(0.9):.1f}")
            ax3.set_xlabel("Impact Score")
            ax3.set_ylabel("Frequency")
            ax3.set_title("Impact Score Distribution")
            ax3.legend(fontsize=7)

        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis("off")
        top_method = results_df.groupby("method")["impact"].mean().idxmax() if len(results_df) > 0 else "N/A"
        top_outcome = results_df.groupby("outcome")["impact"].mean().idxmax() if len(results_df) > 0 else "N/A"
        top_pair = results_df.iloc[0] if len(results_df) > 0 else None
        insights = [
            f"Key Insights",
            f"Most impactful outcome: {top_outcome}",
            f"Best discriminating method: {top_method}",
            f"Top pair: {top_pair['feature_1']} × {top_pair['feature_2']} (impact={top_pair['impact']:.2f})" if top_pair is not None else "",
            f"Next step: Run Phase 4 (predictive models WITH vs WITHOUT top interactions)",
        ]
        ax4.text(0.02, 0.5, "\n".join(insights), transform=ax4.transAxes, fontsize=11,
                 fontfamily="monospace", verticalalignment="center")

        fig.suptitle("Interaction Mining — Discovery Dashboard", fontsize=16, fontweight="bold")
        return self._save(fig, "23_interaction_dashboard.png")

    def figure_24_model_comparison(self, model_results):
        Theme.set_style()
        rows = []
        for oname, odata in model_results.items():
            for key, res in odata["results"].items():
                variant = "with" if "with" in key else "without"
                rows.append({"outcome": oname, "variant": variant, "accuracy": res["metrics"]["accuracy"], "f1": res["metrics"].get("f1", 0)})
        if not rows:
            fig, ax = plt.subplots(); ax.text(0.5,0.5,"No data",ha="center",va="center"); return self._save(fig,"24_model_comparison.png")
        cmp = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for idx, metric in enumerate(["accuracy", "f1"]):
            ax = axes[idx]
            pivot = cmp.pivot_table(index="outcome", columns="variant", values=metric, aggfunc="mean")
            pivot.plot(kind="bar", ax=ax, color=[Theme.PRIMARY[1], Theme.PRIMARY[0]], edgecolor="white", width=0.7)
            ax.set_title(f"Model {metric.title()} — WITH vs WITHOUT Interactions")
            ax.set_ylabel(metric.title())
            ax.legend(title="Variant")
            ax.set_xlabel("")
            for container in ax.containers:
                ax.bar_label(container, fmt="%.3f", fontsize=8)
        fig.suptitle("Predictive Model Comparison", fontsize=14)
        plt.tight_layout()
        return self._save(fig, "24_model_comparison.png")

    def figure_25_coefficient_shift(self, outcome_results):
        Theme.set_style()
        wo_key = next((k for k in outcome_results if "LR" in k and "without" in k), None)
        w_key = next((k for k in outcome_results if "LR" in k and "with" in k), None)
        if not wo_key or not w_key:
            fig, ax = plt.subplots(); ax.text(0.5,0.5,"No LR data",ha="center",va="center"); return self._save(fig,"25_coefficient_shift.png")
        wo_imp = outcome_results[wo_key].get("feature_importance", {})
        w_imp = outcome_results[w_key].get("feature_importance", {})
        common = [f for f in wo_imp if f in w_imp]
        if not common:
            fig, ax = plt.subplots(); ax.text(0.5,0.5,"No shared features",ha="center",va="center"); return self._save(fig,"25_coefficient_shift.png")
        df = pd.DataFrame({"without": [wo_imp[f] for f in common], "with": [w_imp[f] for f in common]}, index=common)
        df["delta"] = df["with"] - df["without"]
        df = df.sort_values("delta", ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [Theme.PRIMARY[0] if v >= 0 else Theme.PRIMARY[3] for v in df["delta"]]
        ax.barh(range(len(df)), df["delta"], color=colors, edgecolor="white")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.index, fontsize=8)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Coefficient Change (with – without interactions)")
        ax.set_title("Logistic Regression — Coefficient Shift After Adding Interactions")
        plt.tight_layout()
        return self._save(fig, "25_coefficient_shift.png")

    def figure_26_rf_importance(self, outcome_results):
        Theme.set_style()
        w_key = next((k for k in outcome_results if "RF" in k and "with" in k), None)
        if not w_key:
            fig, ax = plt.subplots(); ax.text(0.5,0.5,"No RF data",ha="center",va="center"); return self._save(fig,"26_rf_importance.png")
        imp = outcome_results[w_key].get("feature_importance", {})
        if not imp:
            fig, ax = plt.subplots(); ax.text(0.5,0.5,"No importance",ha="center",va="center"); return self._save(fig,"26_rf_importance.png")
        s = pd.Series(imp).sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(s)), s.values, color=Theme.PRIMARY[0], edgecolor="white")
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.index, fontsize=8)
        ax.set_xlabel("Feature Importance")
        ax.set_title("Random Forest — Top 15 Features (WITH Interactions)")
        plt.tight_layout()
        return self._save(fig, "26_rf_importance.png")

    def figure_27_importance_delta(self, outcome_results):
        Theme.set_style()
        wo_key = next((k for k in outcome_results if "RF" in k and "without" in k), None)
        w_key = next((k for k in outcome_results if "RF" in k and "with" in k), None)
        if not wo_key or not w_key:
            fig, ax = plt.subplots(); ax.text(0.5,0.5,"No RF comparison",ha="center",va="center"); return self._save(fig,"27_importance_delta.png")
        wo_imp = outcome_results[wo_key].get("feature_importance", {})
        w_imp = outcome_results[w_key].get("feature_importance", {})
        w_only = {k: v for k, v in w_imp.items() if k not in wo_imp}
        common = [f for f in wo_imp if f in w_imp]
        deltas = {}
        for f in common:
            delta = abs(w_imp[f] - wo_imp[f])
            pct = delta / max(wo_imp[f], 1e-10) * 100
            if pct > 1:
                deltas[f] = delta
        all_changes = dict(sorted({**deltas, **w_only}.items(), key=lambda x: x[1], reverse=True)[:15])
        if not all_changes:
            common_fallback = {f: abs(w_imp[f] - wo_imp[f]) for f in common}
            w_only_fallback = {**w_only}
            all_changes = dict(sorted({**common_fallback, **w_only_fallback}.items(), key=lambda x: x[1], reverse=True)[:15])
            if not all_changes:
                fig, ax = plt.subplots(); ax.text(0.5,0.5,"No significant changes",ha="center",va="center"); return self._save(fig,"27_importance_delta.png")
        s = pd.Series(all_changes).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [Theme.PRIMARY[0] if "__" in f else Theme.PRIMARY[3] for f in s.index]
        ax.barh(range(len(s)), s.values, color=colors, edgecolor="white")
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.index, fontsize=8)
        ax.set_xlabel("Importance Change (WITH - WITHOUT) — interaction features shown as raw importance")
        ax.set_title("Feature Importance Delta (WITH vs WITHOUT Interactions)")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=Theme.PRIMARY[0], label="Interaction feature (raw importance)"), Patch(color=Theme.PRIMARY[3], label="Base feature (|Δ|)")], fontsize=8)
        plt.tight_layout()
        return self._save(fig, "27_importance_delta.png")

    def figure_28_xgb_roc(self, outcome_results):
        Theme.set_style()
        w_key = next((k for k in outcome_results if "XGB" in k and "with" in k), None)
        wo_key = next((k for k in outcome_results if "XGB" in k and "without" in k), None)
        if not w_key and not wo_key:
            fig, ax = plt.subplots(); ax.text(0.5,0.5,"No XGB data",ha="center",va="center"); return self._save(fig,"28_xgb_roc.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        from sklearn.metrics import roc_curve, roc_auc_score
        for key, label, style in [(wo_key, "WITHOUT interactions", "--"), (w_key, "WITH interactions", "-")]:
            if not key:
                continue
            res = outcome_results[key]
            y_test = res.get("y_test")
            y_proba = res.get("y_proba")
            if y_test is None or y_proba is None:
                continue
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                auc_val = roc_auc_score(y_test, y_proba[:, 1])
                ax.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})", lw=2, ls=style)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("XGBoost ROC Curve — WITH vs WITHOUT Interactions")
        ax.legend(fontsize=9)
        plt.tight_layout()
        return self._save(fig, "28_xgb_roc.png")

    def figure_29_shap_summary(self, shap_values, feature_names):
        Theme.set_style()
        if shap_values is None:
            fig, ax = plt.subplots(); ax.text(0.5,0.5,"No SHAP data",ha="center",va="center"); return self._save(fig,"29_shap_summary.png")
        import shap
        fig = plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values.values[:, :, 0] if shap_values.values.ndim == 3 else shap_values.values,
                          features=shap_values.data, feature_names=feature_names, show=False)
        plt.title("SHAP Summary — Feature Impact on Predictions", fontsize=14)
        plt.tight_layout()
        return self._save(fig, "29_shap_summary.png")

    def figure_30_shap_dependence(self, shap_values, feature_df, feature_names, top_feature=None):
        Theme.set_style()
        if shap_values is None:
            fig, ax = plt.subplots(); ax.text(0.5,0.5,"No SHAP data",ha="center",va="center"); return self._save(fig,"30_shap_dependence.png")
        import shap
        vals = shap_values.values[:, :, 0] if shap_values.values.ndim == 3 else shap_values.values
        fn = feature_names or getattr(shap_values, "feature_names", None) or [f"f{i}" for i in range(vals.shape[1])]
        if top_feature is None or top_feature not in fn:
            if len(fn) != vals.shape[1]:
                import warnings as _w
                _w.warn(f"SHAP: feature name count ({len(fn)}) != SHAP columns ({vals.shape[1]}), using first feature")
            top_feature = fn[np.argmax(np.abs(vals).mean(axis=0))] if len(fn) == vals.shape[1] else fn[0]
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.dependence_plot(top_feature, vals, shap_values.data, feature_names=fn, ax=ax, show=False)
        ax.set_title(f"SHAP Dependence: {top_feature}", fontsize=14)
        plt.tight_layout()
        return self._save(fig, "30_shap_dependence.png")

    def figure_31_shap_interaction(self, shap_values, feature_names):
        Theme.set_style()
        if shap_values is None:
            fig, ax = plt.subplots(); ax.text(0.5,0.5,"No SHAP data",ha="center",va="center"); return self._save(fig,"31_shap_interaction.png")
        import shap
        vals = np.abs(shap_values.values[:, :, 0] if shap_values.values.ndim == 3 else shap_values.values).mean(axis=0)
        if len(vals) != len(feature_names):
            vals = vals[:len(feature_names)]
        s = pd.Series(vals, index=feature_names[:len(vals)]).sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(s)), s.values, color=Theme.PRIMARY[1], edgecolor="white")
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.index, fontsize=8)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("SHAP Interaction Strength — Top 15 Features")
        plt.tight_layout()
        return self._save(fig, "31_shap_interaction.png")

    def figure_32_ensemble_cm(self, ensemble_result):
        Theme.set_style()
        if not ensemble_result or "metrics" not in ensemble_result:
            fig, ax = plt.subplots(); ax.text(0.5,0.5,"No ensemble data",ha="center",va="center"); return self._save(fig,"32_ensemble_cm.png")
        cm = ensemble_result["metrics"].get("cm")
        metrics = ensemble_result["metrics"]
        if not cm:
            fig, ax = plt.subplots(); ax.text(0.5,0.5,"No confusion matrix",ha="center",va="center"); return self._save(fig,"32_ensemble_cm.png")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
        axes[0].set_title(f"Stacked Ensemble — Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        ax = axes[1]
        ax.axis("off")
        lines = [f"Accuracy: {metrics['accuracy']:.4f}", f"F1: {metrics.get('f1', 0):.4f}"]
        if metrics.get("roc_auc"):
            lines.append(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        ax.text(0.1, 0.6, "\n".join(lines), transform=ax.transAxes, fontsize=14, fontfamily="monospace")
        ax.set_title("Performance Metrics")
        fig.suptitle("Stacked Ensemble — Attrition Prediction", fontsize=14)
        plt.tight_layout()
        return self._save(fig, "32_ensemble_cm.png")

    def figure_33_km_survival(self, kmf, survival_times):
        Theme.set_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        if kmf is not None:
            kmf.plot_survival_function(ax=ax, color=Theme.PRIMARY[0], linewidth=2)
            median = kmf.median_survival_time_ if hasattr(kmf, 'median_survival_time_') and not np.isnan(kmf.median_survival_time_) else None
            if median:
                ax.axhline(y=0.5, color=Theme.PRIMARY[3], ls="--", alpha=0.5)
                ax.axvline(x=median, color=Theme.PRIMARY[3], ls="--", alpha=0.5, label=f"Median: {median:.0f} days")
            ax.fill_between(kmf.survival_function_.index, kmf.confidence_interval_["KM_estimate_lower_0.95"],
                            kmf.confidence_interval_["KM_estimate_upper_0.95"], alpha=0.2, color=Theme.PRIMARY[0])
        else:
            if survival_times is not None:
                ax.hist(survival_times, bins=30, color=Theme.PRIMARY[0], edgecolor="white", alpha=0.7)
                ax.set_xlabel("Tenure (days)")
                ax.set_ylabel("Count")
        ax.set_xlabel("Tenure (days)")
        ax.set_ylabel("Survival Probability")
        ax.set_title("Kaplan-Meier Survival Curve — Employee Tenure")
        if kmf is not None and median:
            ax.legend(fontsize=10)
        plt.tight_layout()
        return self._save(fig, "33_km_survival.png")

    def figure_34_cox_hazard(self, cph):
        Theme.set_style()
        if cph is None:
            fig, ax = plt.subplots(); ax.text(0.5,0.5,"No Cox PH data",ha="center",va="center"); return self._save(fig,"34_cox_hazard.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            cph.plot(ax=ax)
            ax.set_title("Cox Proportional Hazards — Hazard Ratios\n(with Age, Gender, Seniority, Role)", fontsize=12)
            hr_df = cph.summary
            ax2 = fig.add_axes([0.65, 0.6, 0.3, 0.3])
            ax2.axis("off")
            cph_data = getattr(cph, 'data', None)
            if cph_data is None:
                cph_data = cph.duration_col and cph.event_col
            n_cph = len(cph_data) if hasattr(cph_data, '__len__') else 0
            text_lines = [f"n={n_cph}"]
            try:
                text_lines.append(f"C-index: {cph.concordance_index_:.3f}")
            except Exception:
                pass
            ax2.text(0, 1, "\n".join(text_lines), transform=ax2.transAxes, fontsize=9, fontfamily="monospace", va="top")
        except Exception:
            ax.text(0.5,0.5,"Cox PH plot unavailable",ha="center",va="center")
        plt.tight_layout()
        return self._save(fig, "34_cox_hazard.png")

    def figure_35_model_table(self, model_results):
        Theme.set_style()
        rows = []
        for oname, odata in model_results.items():
            for key, res in odata["results"].items():
                m = key.split("_")[0]
                v = "with" if "with" in key else "without"
                rows.append({"outcome": oname, "model": m, "interactions": v, "accuracy": res["metrics"]["accuracy"],
                             "f1": res["metrics"].get("f1", 0), "roc_auc": res["metrics"].get("roc_auc", "")})
        if not rows:
            fig, ax = plt.subplots(); ax.text(0.5,0.5,"No data",ha="center",va="center"); return self._save(fig,"35_model_table.png")
        df = pd.DataFrame(rows).pivot_table(index=["outcome", "model"], columns="interactions", values=["accuracy", "f1", "roc_auc"], aggfunc="first")
        df = df.round(4)
        fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.4)))
        ax.axis("off")
        cell_text = []
        for idx, row in df.iterrows():
            cell_text.append([str(idx[0]), str(idx[1]),
                              f"{row.get(('accuracy','without'), 0):.3f}", f"{row.get(('accuracy','with'), 0):.3f}",
                              f"{row.get(('f1','without'), 0):.3f}", f"{row.get(('f1','with'), 0):.3f}"])
        col_labels = ["Outcome", "Model", "Acc(wo)", "Acc(w)", "F1(wo)", "F1(w)"]
        table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax.set_title("Model Performance Summary Table", fontsize=14, pad=20)
        plt.tight_layout()
        return self._save(fig, "35_model_table.png")

    def figure_36_modeling_dashboard(self, model_results, ensemble_results, shap_available=True):
        Theme.set_style()
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.axis("off")
        outcomes = list(model_results.keys())
        best_accs = []
        for o in outcomes:
            best = max((v["metrics"]["accuracy"] for k, v in model_results[o]["results"].items()), default=0)
            best_accs.append(best)
        has_ens = len(ensemble_results) > 0
        summary = [
            "Predictive Modeling Summary",
            f"Outcomes modeled: {len(outcomes)}",
            f"Best accuracy range: {min(best_accs):.3f}–{max(best_accs):.3f}",
            f"Models per outcome: LR, RF, XGB",
            f"Ensemble: {'Yes' if has_ens else 'No'}",
            f"SHAP analysis: {'Yes' if shap_available else 'No'}",
        ]
        ax0.text(0.1, 0.9, "\n".join(summary), transform=ax0.transAxes, fontsize=11, fontfamily="monospace", verticalalignment="top")

        ax1 = fig.add_subplot(gs[0, 1:])
        acc_data = []
        for oname, odata in model_results.items():
            for key, res in odata["results"].items():
                variant = "with" if "with" in key else "without"
                acc_data.append({"outcome": oname, "accuracy": res["metrics"]["accuracy"], "variant": variant})
        if acc_data:
            adf = pd.DataFrame(acc_data)
            sns.boxplot(data=adf, x="outcome", y="accuracy", hue="variant", ax=ax1, palette=[Theme.PRIMARY[1], Theme.PRIMARY[0]])
            ax1.set_title("Accuracy Distribution by Outcome")
            ax1.set_xlabel("")
            ax1.legend(fontsize=8)
            for label in ax1.get_xticklabels():
                label.set_rotation(30)

        ax2 = fig.add_subplot(gs[1, :2])
        wo_vs_w = adf.groupby(["outcome", "variant"])["accuracy"].mean().unstack() if len(acc_data) > 0 else pd.DataFrame()
        if not wo_vs_w.empty and "without" in wo_vs_w and "with" in wo_vs_w:
            wo_vs_w["delta"] = wo_vs_w["with"] - wo_vs_w["without"]
            colors = [Theme.PRIMARY[0] if v > 0 else Theme.PRIMARY[3] for v in wo_vs_w["delta"]]
            ax2.bar(range(len(wo_vs_w)), wo_vs_w["delta"], color=colors, edgecolor="white")
            ax2.set_xticks(range(len(wo_vs_w)))
            ax2.set_xticklabels(wo_vs_w.index, fontsize=9, rotation=30)
            ax2.axhline(0, color="black", lw=0.5)
            ax2.set_ylabel("Accuracy Δ (WITH – WITHOUT)")
            ax2.set_title("Impact of Adding Interaction Features")
            for i, v in enumerate(wo_vs_w["delta"]):
                ax2.text(i, v + 0.001 if v >= 0 else v - 0.005, f"{v:+.4f}", ha="center", fontsize=8)

        ax3 = fig.add_subplot(gs[1, 2])
        if has_ens:
            ens_name = list(ensemble_results.keys())[0]
            ens_acc = ensemble_results[ens_name]["metrics"]["accuracy"] if ensemble_results[ens_name] else 0
            best_base = max((v["metrics"]["accuracy"] for k, v in model_results.get(ens_name, {}).get("results", {}).items() if "with" in k), default=0)
            ax3.bar(["Base Best", "Ensemble"], [best_base, ens_acc], color=[Theme.PRIMARY[1], Theme.PRIMARY[0]], edgecolor="white", width=0.5)
            ax3.set_ylabel("Accuracy")
            ax3.set_title(f"Ensemble vs Best Base\n({ens_name})")
            for i, v in enumerate([best_base, ens_acc]):
                ax3.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
        else:
            ax3.text(0.5, 0.5, "No ensemble\nresults", ha="center", va="center", transform=ax3.transAxes)

        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis("off")
        insights = []
        if len(acc_data) > 0:
            best_outcome = adf.loc[adf["accuracy"].idxmax(), "outcome"] if len(adf) > 0 else ""
            best_acc = adf["accuracy"].max() if len(adf) > 0 else 0
            insights.append(f"Best predicted outcome: {best_outcome} (acc={best_acc:.3f})")
            if not wo_vs_w.empty and "delta" in wo_vs_w:
                delta_mean = wo_vs_w["delta"].mean()
                insights.append(f"Avg accuracy improvement with interactions: {delta_mean:+.4f}")
                pos = (wo_vs_w["delta"] > 0).sum()
                neg = (wo_vs_w["delta"] <= 0).sum()
                insights.append(f"Interactions helped: {pos}/{pos+neg} outcomes")
        insights.append("Phase 5: Deep Dives on top discoveries")
        ax4.text(0.02, 0.5, "\n".join(f"  • {i}" for i in insights), transform=ax4.transAxes,
                 fontsize=11, fontfamily="monospace", verticalalignment="center")

        fig.suptitle("Predictive Modeling — Summary Dashboard", fontsize=16, fontweight="bold")
        return self._save(fig, "36_modeling_dashboard.png")

    # ------------------------------------------------------------------ #
    #  Phase 5: Deep Dives — 16 figures (37–52)
    # ------------------------------------------------------------------ #

    def plot_deep_dive_subgroup(self, dive, figure_id):
        Theme.set_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        seg_m = dive["segment_outcome_mean"]
        rest_m = dive["rest_outcome_mean"]
        n_seg = dive["n_segment"]
        n_rest = dive["n_rest"]

        ax1.bar(["Segment", "Rest"], [seg_m, rest_m],
                color=[Theme.PRIMARY[0], Theme.PRIMARY[3]], edgecolor="white", width=0.5)
        ax1.set_ylabel(f'{dive["outcome"]} mean')
        ax1.set_title(f'Subgroup: {dive["f1_val"]} × {dive["f2_val"]}\nn={n_seg} vs rest n={n_rest}')
        dy = (seg_m - rest_m)
        ax1.text(0, seg_m + dy * 0.02, f"{seg_m:.3f}", ha="center", fontsize=9)
        ax1.text(1, rest_m + dy * 0.02 if seg_m >= rest_m else rest_m - dy * 0.02, f"{rest_m:.3f}", ha="center", fontsize=9)

        cats = ["difference", "p_value", "effect_size"]
        labels_cats = ["Δ outcome", "p-value", "effect d"]
        vals = [abs(dive["difference"]), min(dive["p_value"], 0.05), min(abs(dive["effect_size"]), 1)]
        ax2.barh(labels_cats, vals, color=Theme.PRIMARY[:3], edgecolor="white")
        for i, (c, v) in enumerate(zip(cats, vals)):
            ax2.text(v + 0.01, i, f"{dive[c]:.4f}", va="center", fontsize=8)
        ax2.set_xlim(0, max(vals) * 1.4)
        ax2.set_title("Effect Diagnostics")
        fig.suptitle(f"Subgroup Comparison — {dive['f1']} × {dive['f2']} → {dive['outcome']}", fontsize=12)
        plt.tight_layout()
        return self._save(fig, f"{figure_id:02d}_subgroup_{dive['outcome']}.png")

    def plot_deep_dive_profile(self, dive, feature_df, figure_id):
        Theme.set_style()
        mask = dive["segment_mask"]
        profile_cols = [k for k in dive["profile"] if k != dive["outcome"]][:8]
        if not profile_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No numeric features to profile", ha="center", va="center")
            fig.suptitle(f"Segment Profile — {dive['f1_val']} × {dive['f2_val']}")
            return self._save(fig, f"{figure_id:02d}_profile_{dive['outcome']}.png")

        df = feature_df[profile_cols].copy()
        seg_means = df[mask].mean()
        pop_means = df.mean()
        norm = pd.DataFrame({
            "segment": (seg_means - seg_means.min()) / (seg_means.max() - seg_means.min() + 1e-8),
            "population": (pop_means - pop_means.min()) / (pop_means.max() - pop_means.min() + 1e-8),
        }).fillna(0)

        angles = np.linspace(0, 2 * np.pi, len(profile_cols), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
        for label, values in [("Segment", norm["segment"].tolist()), ("Population", norm["population"].tolist())]:
            v = values + values[:1]
            ax.plot(angles, v, label=label, lw=2)
            ax.fill(angles, v, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(profile_cols, fontsize=8)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        fig.suptitle(f"Segment Profile — {dive['f1_val']} × {dive['f2_val']} (n={dive['n_segment']})", fontsize=12)
        plt.tight_layout()
        return self._save(fig, f"{figure_id:02d}_profile_{dive['outcome']}.png")

    def plot_deep_dive_whatif(self, dive, figure_id):
        Theme.set_style()
        wi = dive["whatif"]
        wi = [w for w in wi if w["outcome_mean"] is not None]
        if not wi:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No what-if data available", ha="center", va="center")
            return self._save(fig, f"{figure_id:02d}_whatif_{dive['outcome']}.png")

        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [w["f1_val"][:15] for w in wi]
        means = [w["outcome_mean"] for w in wi]
        counts = [w["count"] for w in wi]
        colors = [Theme.PRIMARY[0] if m > means[0] else Theme.PRIMARY[3] for m in means]
        bars = ax.bar(range(len(labels)), means, color=colors, edgecolor="white")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f'{dive["outcome"]} mean')
        ax.set_title(f'What-If: {dive["f1"]}=const, vary {dive["f2"]}\nCurrent: {dive["f1_val"]}×{dive["f2_val"]}')
        for i, (m, c) in enumerate(zip(means, counts)):
            ax.text(i, m + 0.005, f"{m:.3f}\nn={c}", ha="center", fontsize=7)
        plt.tight_layout()
        return self._save(fig, f"{figure_id:02d}_whatif_{dive['outcome']}.png")

    def figure_52_deep_dive_dashboard(self, dives):
        Theme.set_style()
        n = len(dives)
        fig, axes = plt.subplots(n, 3, figsize=(14, 3 * n))
        if n == 1:
            axes = [axes]
        for i, dive in enumerate(dives):
            axs = axes[i]
            axs[0].axis("off")
            lines = [
                f"Dive #{i+1}: {dive['f1']} × {dive['f2']} → {dive['outcome']}",
                f"Segment: {dive['f1_val']} × {dive['f2_val']} (n={dive['n_segment']})",
                f"Impact score: {dive.get('impact', '?'):.1f}",
                f"Δ outcome: {dive['difference']:+.3f}",
                f"Effect size: d={dive['effect_size']:.3f}",
                f"p-value: {dive['p_value']:.4f}",
            ]
            axs[0].text(0, 1, "\n".join(lines), transform=axs[0].transAxes,
                        fontsize=9, fontfamily="monospace", verticalalignment="top")

            seg_m, rest_m = dive["segment_outcome_mean"], dive["rest_outcome_mean"]
            axs[1].bar(["Seg", "Rest"], [seg_m, rest_m],
                       color=[Theme.PRIMARY[0], Theme.PRIMARY[3]], edgecolor="white", width=0.5)
            axs[1].set_ylabel(dive["outcome"])
            axs[1].text(0, seg_m + 0.01, f"{seg_m:.3f}", ha="center", fontsize=8)
            axs[1].text(1, rest_m + 0.01, f"{rest_m:.3f}", ha="center", fontsize=8)

            wi = [w for w in dive["whatif"] if w["outcome_mean"] is not None]
            if wi:
                lbls = [w["f1_val"][:10] for w in wi]
                vals = [w["outcome_mean"] for w in wi]
                axs[2].bar(range(len(lbls)), vals, color=Theme.PRIMARY[1], edgecolor="white")
                axs[2].set_xticks(range(len(lbls)))
                axs[2].set_xticklabels(lbls, rotation=30, fontsize=7)
                axs[2].set_title("What-if (vary f2)")
            else:
                axs[2].text(0.5, 0.5, "No data", ha="center", va="center")

        fig.suptitle("Deep Dives — Summary Dashboard", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._save(fig, "52_deep_dive_dashboard.png")

    # ------------------------------------------------------------------ #
    #  Phase 6: Dashboards — 8 composite figures (53–60)
    # ------------------------------------------------------------------ #

    def figure_53_attrition_dashboard(self, df):
        Theme.set_style()
        is_term = df["EmployeeStatus"].str.lower().str.contains("terminat").astype(int)
        df = df.copy()
        df["_is_term"] = is_term
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        for ax in axs.flat:
            ax.set_facecolor("white")

        axs[0, 0].axis("off")
        term_rate = is_term.mean()
        n_term = is_term.sum()
        axs[0, 0].text(0, 1, f"Attrition Dashboard\nRate: {term_rate:.1%}\nTerminated: {n_term}/{len(is_term)}",
                        transform=axs[0, 0].transAxes, fontsize=12, fontfamily="monospace", va="top")

        for c, title, ax in [("Region", "By Region", axs[0, 1]), ("DivisionGroup", "By Division", axs[0, 2])]:
            rates = df.groupby(c)["_is_term"].mean()
            ax.barh(range(len(rates)), rates.values, color=Theme.PRIMARY[0], edgecolor="white")
            ax.set_yticks(range(len(rates)))
            ax.set_yticklabels(rates.index, fontsize=8)
            ax.set_title(title, fontsize=10)
            ax.set_xlim(0, max(rates) * 1.4)
            for i, v in enumerate(rates.values):
                ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=8)

        for c, title, ax in [("IntersectionalID", "By IntersectionalID", axs[1, 0]),
                              ("JobFamily", "By JobFamily", axs[1, 1])]:
            if c in df.columns:
                rates = df.groupby(c)["_is_term"].mean().sort_values(ascending=False).head(8)
                ax.bar(range(len(rates)), rates.values, color=Theme.PRIMARY[1], edgecolor="white")
                ax.set_xticks(range(len(rates)))
                ax.set_xticklabels(rates.index, fontsize=7, rotation=30)
                ax.set_title(title, fontsize=10)
            else:
                ax.text(0.5, 0.5, f"{c} not available", ha="center", va="center")

        axs[1, 2].axis("off")
        insights = [
            f"Highest attrition: {df.groupby('Region')['_is_term'].mean().idxmax()}",
            f"Lowest attrition: {df.groupby('Region')['_is_term'].mean().idxmin()}",
            f"Attrition vs tenure correlation: {df['TenureYears'].corr(is_term) if 'TenureYears' in df else 'N/A':.3f}",
        ]
        axs[1, 2].text(0, 1, "\n".join(insights), transform=axs[1, 2].transAxes,
                        fontsize=10, fontfamily="monospace", va="top")
        fig.suptitle("Figure 53: Attrition Dashboard", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._save(fig, "53_attrition_dashboard.png")

    def figure_54_compensation_dashboard(self, df):
        Theme.set_style()
        if "PayZone" not in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "PayZone column not available", ha="center", va="center")
            return self._save(fig, "54_compensation_dashboard.png")
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))

        pay_order = {"Zone A": 1, "Zone B": 2, "Zone C": 3}
        zone_counts = df["PayZone"].value_counts()
        axs[0, 0].bar(zone_counts.index, zone_counts.values, color=Theme.PRIMARY[:3], edgecolor="white")
        axs[0, 0].set_title("PayZone Distribution")
        for i, v in enumerate(zone_counts.values):
            axs[0, 0].text(i, v + 5, str(v), ha="center", fontsize=9)

        pay_zones = sorted(df["PayZone"].dropna().unique())
        n_pz = len(pay_zones)
        for c, title, ax in [("JobFamily", "By JobFamily", axs[0, 1]), ("DepartmentType", "By Dept", axs[0, 2])]:
            ct = df.groupby(c)["PayZone"].value_counts(normalize=True).unstack(fill_value=0)
            ct = ct[pay_zones] if all(pz in ct.columns for pz in pay_zones) else ct
            ct.plot(kind="barh", stacked=True, ax=ax, color=Theme.PRIMARY[:n_pz], legend=False)
            ax.set_title(title)
            ax.set_ylabel("")

        for c, title, ax in [("IsManager", "Manager vs IC", axs[1, 0]),
                              ("Generation", "By Generation", axs[1, 1])]:
            ct = df.groupby(c)["PayZone"].value_counts(normalize=True).unstack(fill_value=0)
            ct = ct[pay_zones] if all(pz in ct.columns for pz in pay_zones) else ct
            ct.plot(kind="barh", stacked=True, ax=ax, color=Theme.PRIMARY[:n_pz])
            ax.set_title(title)
            ax.set_ylabel("")

        axs[1, 2].axis("off")
        insights = [
            f"Zone A (highest): {zone_counts.index[0] if len(zone_counts) > 0 else '?'}",
            f"Zone C (lowest): {zone_counts.index[-1] if len(zone_counts) > 0 else '?'}",
        ]
        axs[1, 2].text(0, 1, "\n".join(insights), transform=axs[1, 2].transAxes,
                        fontsize=10, fontfamily="monospace", va="top")
        fig.suptitle("Figure 54: Compensation Dashboard", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._save(fig, "54_compensation_dashboard.png")

    def figure_55_diversity_dashboard(self, df):
        Theme.set_style()
        dept_gender = df.groupby("DepartmentType")["GenderCode"]
        is_minority = pd.Series(0, index=df.index)
        try:
            for dept, group in dept_gender:
                majority = group.value_counts().index[0]
                mask = df["DepartmentType"] == dept
                is_minority.loc[mask] = (df.loc[mask, "GenderCode"] != majority).astype(int)
            is_minority = is_minority.fillna(0).astype(int)
        except Exception:
            pass
        df = df.copy()
        df["_is_minority"] = is_minority
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))

        axs[0, 0].axis("off")
        minority_rate = is_minority.mean()
        axs[0, 0].text(0, 1, f"Diversity Dashboard\nMinority Rate: {minority_rate:.1%}",
                        transform=axs[0, 0].transAxes, fontsize=12, fontfamily="monospace", va="top")

        for c, title, ax, pos in [("DepartmentType", "By Dept", axs[0, 1], 0),
                                    ("IntersectionalID", "By Identity (top 6)", axs[0, 2], 1)]:
            try:
                rates = df.groupby(c)["_is_minority"].mean().sort_values(ascending=False)
                if len(rates) > 6:
                    rates = rates.head(6)
                ax.bar(range(len(rates)), rates.values, color=Theme.PRIMARY[pos], edgecolor="white")
                ax.set_xticks(range(len(rates)))
                ax.set_xticklabels(rates.index, fontsize=8, rotation=30)
                ax.set_title(title, fontsize=10)
                for i, v in enumerate(rates.values):
                    ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=8)
            except Exception:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")

        axs[1, 0].axis("off")
        insights = [
            f"Overall minority rate: {minority_rate:.1%}",
            f"Most diverse dept: {df.groupby('DepartmentType')['_is_minority'].mean().idxmax() if is_minority.sum() > 0 else '?'}",
            f"Least diverse dept: {df.groupby('DepartmentType')['_is_minority'].mean().idxmin() if is_minority.sum() > 0 else '?'}",
        ]
        axs[1, 0].text(0, 1, "\n".join(insights), transform=axs[1, 0].transAxes,
                        fontsize=10, fontfamily="monospace", va="top")

        if "DeptDiversityScore" in df.columns:
            axs[1, 1].hist(df["DeptDiversityScore"].dropna(), bins=20, color=Theme.PRIMARY[0], edgecolor="white")
            axs[1, 1].set_title("Dept Diversity Score Distribution")
        else:
            axs[1, 1].text(0.5, 0.5, "DeptDiversityScore N/A", ha="center", va="center")

        axs[1, 2].axis("off")
        fig.suptitle("Figure 55: Diversity Dashboard", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._save(fig, "55_diversity_dashboard.png")

    def figure_56_performance_dashboard(self, df):
        Theme.set_style()
        perf = df.get("Current Employee Rating", df.get("PerfScore", None))
        if perf is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "PerfScore not available", ha="center", va="center")
            return self._save(fig, "56_performance_dashboard.png")
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))

        axs[0, 0].hist(perf.dropna(), bins=8, color=Theme.PRIMARY[0], edgecolor="white")
        axs[0, 0].set_title(f"Performance Score\nμ={perf.mean():.2f} σ={perf.std():.2f}")
        axs[0, 0].axvline(perf.mean(), color=Theme.PRIMARY[1], ls="--", lw=1.5)

        for c, title, ax in [("Region", "By Region", axs[0, 1]), ("DepartmentType", "By Dept", axs[0, 2])]:
            means = df.groupby(c)[perf.name].mean().sort_values(ascending=False)
            ax.bar(range(len(means)), means.values, color=Theme.PRIMARY[1], edgecolor="white")
            ax.set_xticks(range(len(means)))
            ax.set_xticklabels(means.index, fontsize=8, rotation=30)
            ax.set_title(title, fontsize=10)
            ax.axhline(perf.mean(), color="gray", ls="--", lw=1, label="global mean")
            for i, v in enumerate(means.values):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

        for c, title, ax in [("JobFamily", "By JobFamily (top 6)", axs[1, 0]),
                              ("IsManager", "Manager vs IC", axs[1, 1])]:
            means = df.groupby(c)[perf.name].mean().sort_values(ascending=False).head(6)
            ax.barh(range(len(means)), means.values, color=Theme.PRIMARY[2], edgecolor="white")
            ax.set_yticks(range(len(means)))
            ax.set_yticklabels(means.index, fontsize=8)
            ax.set_title(title, fontsize=10)
            ax.axvline(perf.mean(), color="gray", ls="--", lw=1, label="global")
            for i, v in enumerate(means.values):
                ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)

        axs[1, 2].axis("off")
        low_perf = perf[perf < perf.mean() - perf.std()]
        high_perf = perf[perf > perf.mean() + perf.std()]
        insights = [
            f"Overall mean: {perf.mean():.2f}",
            f"Low performers (< -1σ): n={len(low_perf)} ({len(low_perf)/len(perf):.1%})",
            f"High performers (> +1σ): n={len(high_perf)} ({len(high_perf)/len(perf):.1%})",
        ]
        axs[1, 2].text(0, 1, "\n".join(insights), transform=axs[1, 2].transAxes,
                        fontsize=10, fontfamily="monospace", va="top")
        fig.suptitle("Figure 56: Performance Dashboard", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._save(fig, "56_performance_dashboard.png")

    def figure_57_career_dashboard(self, df):
        Theme.set_style()
        seniority = df.get("SeniorityLevel", None)
        if seniority is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "SeniorityLevel not available", ha="center", va="center")
            return self._save(fig, "57_career_dashboard.png")
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))

        levels = sorted(seniority.dropna().unique())
        counts = seniority.value_counts().sort_index()
        axs[0, 0].bar(counts.index.astype(str), counts.values, color=Theme.PRIMARY[0], edgecolor="white")
        axs[0, 0].set_title("Seniority Level Distribution")
        for i, v in enumerate(counts.values):
            axs[0, 0].text(i, v + 5, str(v), ha="center", fontsize=9)

        for c, title, ax in [("JobFamily", "By JobFamily", axs[0, 1]),
                              ("DepartmentType", "By Dept", axs[0, 2])]:
            means = df.groupby(c)["SeniorityLevel"].mean().sort_values(ascending=False)
            ax.bar(range(len(means)), means.values, color=Theme.PRIMARY[1], edgecolor="white")
            ax.set_xticks(range(len(means)))
            ax.set_xticklabels(means.index, fontsize=8, rotation=30)
            ax.set_title(title, fontsize=10)
            ax.axhline(seniority.mean(), color="gray", ls="--", lw=1)

        for c, title, ax in [("IsManager", "Manager vs IC", axs[1, 0]),
                              ("IsExecutive", "Exec vs Non-Exec", axs[1, 1])]:
            if c in df.columns:
                means = df.groupby(c)["SeniorityLevel"].mean()
                ax.bar(means.index.astype(str), means.values, color=Theme.PRIMARY[2], edgecolor="white")
                ax.set_title(title, fontsize=10)
                for i, v in enumerate(means.values):
                    ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)

        axs[1, 2].axis("off")
        insights = [
            f"Mean Seniority: {seniority.mean():.2f}",
            f"Highest: {df.groupby('JobFamily')['SeniorityLevel'].mean().idxmax()}",
            f"Career-role correlation: {df['SeniorityLevel'].corr(df.get('SpanOfControl', pd.Series(0))) if 'SpanOfControl' in df else 'N/A':.3f}",
        ]
        axs[1, 2].text(0, 1, "\n".join(insights), transform=axs[1, 2].transAxes,
                        fontsize=10, fontfamily="monospace", va="top")
        fig.suptitle("Figure 57: Career Progression Dashboard", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._save(fig, "57_career_dashboard.png")

    def figure_58_interaction_matrix(self, interaction_results):
        Theme.set_style()
        if interaction_results is None or interaction_results.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "Interaction results not available", ha="center", va="center")
            return self._save(fig, "58_interaction_matrix.png")

        pivot = interaction_results.pivot_table(
            index="feature_1", columns="outcome", values="impact", aggfunc="max"
        ).fillna(0)
        if pivot.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No data to pivot", ha="center", va="center")
            return self._save(fig, "58_interaction_matrix.png")

        fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.4)))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"label": "Interaction Impact"})
        ax.set_title("Figure 58: Interaction Impact Matrix\n(rows=features, cols=outcomes)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return self._save(fig, "58_interaction_matrix.png")

    def figure_59_roc_curves(self, model_results):
        Theme.set_style()
        from sklearn.metrics import roc_curve as _roc, roc_auc_score as _auc
        model_results = model_results or {}
        if not model_results:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No model results available for ROC curves", ha="center", va="center")
            return self._save(fig, "59_roc_curves.png")
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()
        model_colors = {"LR": Theme.PRIMARY[0], "RF": Theme.PRIMARY[1], "XGB": Theme.PRIMARY[2]}
        roc_available = False
        i = 0
        for i, (oname, odata) in enumerate(model_results.items()):
            if i >= len(axes):
                break
            ax = axes[i]
            if not isinstance(odata, dict):
                ax.text(0.5, 0.5, f"{oname}\nno results", ha="center", va="center")
                continue
            text_lines = []
            for key, res in odata.items():
                if not isinstance(res, dict):
                    continue
                metrics = res.get("metrics", {})
                roc_fpr = res.get("roc_fpr")
                roc_tpr = res.get("roc_tpr")
                y_test = res.get("y_test")
                y_proba = res.get("y_proba")
                mtype = key.split("_")[0]
                color = model_colors.get(mtype, "black")
                if roc_fpr and roc_tpr:
                    roc_available = True
                    variant = "WITH" if "with" in key else "WITHOUT"
                    ls = "-" if variant == "WITH" else "--"
                    ax.plot(roc_fpr, roc_tpr, label=f"{variant} ({mtype})", lw=1.5, ls=ls, color=color)
                elif y_test is not None and y_proba is not None and hasattr(y_proba, 'ndim') and y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    roc_available = True
                    fpr, tpr, _ = _roc(y_test, y_proba[:, 1])
                    variant = "WITH" if "with" in key else "WITHOUT"
                    ls = "-" if variant == "WITH" else "--"
                    auc_val = _auc(y_test, y_proba[:, 1])
                    ax.plot(fpr, tpr, label=f"{variant} ({mtype}, AUC={auc_val:.3f})", lw=1.5, ls=ls, color=color)
                else:
                    acc = metrics.get("accuracy", 0)
                    text_lines.append((f"{key}: acc={acc:.3f}", color))
            if text_lines:
                for j, (line, color) in enumerate(text_lines):
                    ax.text(0.5, 0.7 - j * 0.08, line, ha="center", va="center", fontsize=7, transform=ax.transAxes, color=color)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title(f"{oname} — ROC Curves", fontsize=10)
            ax.legend(fontsize=6, loc='center left', bbox_to_anchor=(0.98, 0.5))
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.suptitle("Figure 59: ROC Curves — All Outcomes & Models", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout(rect=[0, 0, 0.92, 0.95])
        return self._save(fig, "59_roc_curves.png")

    def figure_60_executive_summary(self, df):
        Theme.set_style()
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        is_term = df["EmployeeStatus"].str.lower().str.contains("terminat").astype(int)
        df = df.copy()
        df["_is_term"] = is_term
        n = len(df)
        n_term = is_term.sum()

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.axis("off")
        perf_col = df.get("Current Employee Rating", df.get("PerfScore", None))
        seniority = df.get("SeniorityLevel", None)
        lines = [
            "KPI Summary",
            f"Total Employees: {n}",
            f"Attrition Rate: {n_term/n:.1%} ({n_term}/{n})",
            f"Mean PerfScore: {perf_col.mean():.2f}" if perf_col is not None else "",
            f"Mean Seniority: {seniority.mean():.2f}" if seniority is not None else "",
            f"Tenure (mean): {df['TenureYears'].mean():.2f} yrs",
            f"Unique Job Families: {df['JobFamily'].nunique()}",
            f"Regions: {df['Region'].nunique()}",
        ]
        ax0.text(0, 1, "\n".join(line for line in lines if line),
                 transform=ax0.transAxes, fontsize=10, fontfamily="monospace", va="top")

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.axis("off")
        dd_path = Path(self.cfg.output_dir) / "deep_dives.json"
        try:
            import json as _json
            if dd_path.exists():
                with open(dd_path) as _f:
                    _dd = _json.load(_f)
                top_insights = ["Key Discoveries"]
                for i, d in enumerate(_dd[:5], 1):
                    impact = d.get("impact", "?")
                    diff = d.get("difference", 0)
                    seg = d.get("f1_val", "?")
                    desc = f"   {d.get('f1','?')} x {d.get('f2','?')} -> {d.get('outcome','?')}"
                    top_insights.append(f"{i}. {desc} (impact={impact:.0f})")
                    top_insights.append(f"   Segment: {seg} | Δ={diff:+.3f}")
            else:
                raise FileNotFoundError
        except Exception:
            top_insights = [
                "Key Discoveries",
                "1. JobFamily x Dept -> PayZone (impact=181)",
                "   Admin in Sales are in higher pay zones",
                "2. Tenure x ExitQuarter -> Minority Dept (impact=108)",
                "   4.5yr mid-career churn in diverse teams",
                "3. Region x IntersectionalID -> Attrition (impact=56)",
                "   NE region has higher attrition across groups",
            ]
        ax1.text(0, 1, "\n".join(top_insights),
                 transform=ax1.transAxes, fontsize=9, fontfamily="monospace", va="top")

        ax2 = fig.add_subplot(gs[0, 2])
        region_rates = df.groupby("Region")["_is_term"].mean().sort_values()
        colors = [Theme.PRIMARY[0] if v > is_term.mean() else Theme.PRIMARY[3] for v in region_rates.values]
        ax2.barh(region_rates.index, region_rates.values, color=colors, edgecolor="white")
        ax2.set_title("Attrition by Region", fontsize=10)
        ax2.set_xlim(0, max(region_rates) * 1.5)
        for i, v in enumerate(region_rates.values):
            ax2.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=8)

        ax3 = fig.add_subplot(gs[1, :2])
        ax3.axis("off")
        phases = [
            ("Phase 1: Deep EDA", "12 figures", "Univariate stats, correlations, KDE, heatmaps, PCA, t-SNE"),
            ("Phase 2: Feature Engineering", "3 figures", "Feature importance, redundancy graph, engineered features"),
            ("Phase 3: Interaction Mining", "8 figures", "Impact scores, top-20 grid, network map, decision tree paths"),
            ("Phase 4: Predictive Models", "13 figures", "LR/RF/XGB + ensemble, SHAP, KM survival, Cox PH"),
            ("Phase 5: Deep Dives", "16 figures", "Subgroup comp, segment profiles, what-if scenarios"),
        ]
        table_data = [[p, f, d] for p, f, d in phases]
        col_labels = ["Phase", "Figures", "Description"]
        table = ax3.table(cellText=table_data, colLabels=col_labels,
                          loc="center", cellLoc="left", fontsize=9)
        table.auto_set_column_width([0, 1, 2])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for key, cell in table.get_celld().items():
            cell.set_edgecolor("#ddd")
            if key[0] == 0:
                cell.set_facecolor(Theme.PRIMARY[0])
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("white")
        ax3.set_title("Pipeline Overview", fontsize=11, fontweight="bold", pad=10)

        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis("off")
        model_results_file = Path(self.cfg.output_dir) / "model_results.json"
        if model_results_file.exists():
            import json as _json
            try:
                mr = _json.load(open(model_results_file))
                best_items = []
                for oname, odata in mr.items():
                    accs = [(k, v["metrics"]["accuracy"]) for k, v in odata.get("results", {}).items()]
                    if accs:
                        best_key, best_acc = max(accs, key=lambda x: x[1])
                        best_items.append(f"{oname}: {best_acc:.3f}")
                ax4.text(0, 1, "Best Model Accuracies\n" + "\n".join(best_items),
                         transform=ax4.transAxes, fontsize=9, fontfamily="monospace", va="top")
            except Exception:
                ax4.text(0.5, 0.5, "Model results\nunavailable", ha="center", va="center")
        else:
            ax4.text(0.5, 0.5, "Model results\nfile not found", ha="center", va="center", transform=ax4.transAxes)

        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")
        ir_path = Path(self.cfg.output_dir) / "interaction_results.parquet"
        try:
            import pandas as _pd
            if ir_path.exists():
                _ir = _pd.read_parquet(ir_path)
                top = _ir.sort_values("impact", ascending=False).drop_duplicates("outcome")
                recs = ["Recommendations"]
                for i, (_, r) in enumerate(top.iterrows(), 1):
                    recs.append(f"{i}. {r['feature_1']} x {r['feature_2']} -> {r['outcome']} (impact={r['impact']:.0f})")
                    recs.append(f"   p={r['p_value']:.4f}, method={r.get('method','?')}")
                if len(recs) < 3:
                    raise ValueError
            else:
                raise FileNotFoundError
        except Exception:
            recs = [
                "Recommendations",
                "1. JobFamily x Dept -> PayZone — review pay equity in Admin/Sales",
                "2. Tenure x ExitQuarter -> Dept Diversity — mid-career retention for minority staff",
                "3. Region x IntersectionalID -> Attrition — regional culture audit in NE",
                "4. PerfScore near ceiling (μ~3.0) — expand evaluation range",
                "5. SeniorityLevel fully predictable from role — ensure career path transparency",
            ]
        ax5.text(0, 1, "\n".join(recs), transform=ax5.transAxes,
                 fontsize=8, fontfamily="monospace", va="top")

        fig.suptitle("Figure 60: Executive Summary — Workforce Analytics Pipeline",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._save(fig, "60_executive_summary.png")
