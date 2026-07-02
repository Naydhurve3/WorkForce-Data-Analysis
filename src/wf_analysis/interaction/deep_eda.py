import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest


class DeepEDA:
    def __init__(self, config):
        self.cfg = config

    def compute_univariate_stats(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        stats = {"shape": list(df.shape), "total_columns": len(df.columns), "total_rows": len(df)}

        stats["numeric"] = {}
        for c in numeric_cols:
            s = df[c].dropna()
            stats["numeric"][c] = {
                "count": int(len(s)),
                "missing": int(df[c].isna().sum()),
                "missing_pct": round(float(df[c].isna().mean() * 100), 2),
                "min": round(float(s.min()), 2) if len(s) > 0 else None,
                "max": round(float(s.max()), 2) if len(s) > 0 else None,
                "mean": round(float(s.mean()), 2) if len(s) > 0 else None,
                "median": round(float(s.median()), 2) if len(s) > 0 else None,
                "std": round(float(s.std()), 2) if len(s) > 0 else None,
                "skew": round(float(s.skew()), 3) if len(s) > 1 else None,
                "kurtosis": round(float(s.kurtosis()), 3) if len(s) > 1 else None,
                "q1": round(float(s.quantile(0.25)), 2) if len(s) > 0 else None,
                "q3": round(float(s.quantile(0.75)), 2) if len(s) > 0 else None,
                "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 2) if len(s) > 0 else None,
            }

        stats["categorical"] = {}
        for c in categorical_cols:
            vc = df[c].value_counts()
            stats["categorical"][c] = {
                "count": int(df[c].notna().sum()),
                "missing": int(df[c].isna().sum()),
                "missing_pct": round(float(df[c].isna().mean() * 100), 2),
                "unique": int(df[c].nunique()),
                "top_values": {str(k): int(v) for k, v in vc.head(10).items()},
                "top_pct": round(float(vc.iloc[0] / df[c].notna().sum() * 100), 1) if len(vc) > 0 else None,
            }

        stats["missing_summary"] = {
            "total_missing_cells": int(df.isna().sum().sum()),
            "missing_pct_overall": round(float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100), 2),
            "columns_with_missing": {c: {"count": int(v), "pct": round(float(v / len(df) * 100), 2)}
                                      for c, v in df.isna().sum().items() if v > 0},
        }

        return stats

    def detect_outliers(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Detecting outliers across {len(numeric_cols)} numeric columns using 3 methods")

        results = {}
        for c in numeric_cols:
            s = df[c].dropna()
            if len(s) < 10:
                continue

            n = len(s)
            outliers = {}

            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            iqr_mask = (s < q1 - self.cfg.eda.outlier_iqr_multiplier * iqr) | \
                       (s > q3 + self.cfg.eda.outlier_iqr_multiplier * iqr)
            outliers["iqr"] = {
                "count": int(iqr_mask.sum()),
                "pct": round(float(iqr_mask.mean() * 100), 2),
                "indices": s.index[iqr_mask].tolist() if iqr_mask.sum() <= 20 else s.index[iqr_mask][:20].tolist(),
            }

            z_scores = np.abs((s - s.mean()) / s.std())
            z_mask = z_scores > self.cfg.eda.outlier_zscore_threshold
            outliers["zscore"] = {
                "count": int(z_mask.sum()),
                "pct": round(float(z_mask.mean() * 100), 2),
                "indices": s.index[z_mask].tolist() if z_mask.sum() <= 20 else s.index[z_mask][:20].tolist(),
            }

            ifo = IsolationForest(
                contamination=self.cfg.eda.isolation_forest_contamination,
                random_state=self.cfg.random_state, n_estimators=100,
            )
            ifo_preds = ifo.fit_predict(s.values.reshape(-1, 1))
            ifo_mask = ifo_preds == -1
            outliers["isolation_forest"] = {
                "count": int(ifo_mask.sum()),
                "pct": round(float(ifo_mask.mean() * 100), 2),
                "indices": s.index[ifo_mask].tolist() if ifo_mask.sum() <= 20 else s.index[ifo_mask][:20].tolist(),
            }

            iqr_idx = set(outliers["iqr"]["indices"])
            z_idx = set(outliers["zscore"]["indices"])
            ifo_idx = set(outliers["isolation_forest"]["indices"])
            consensus = iqr_idx & z_idx & ifo_idx

            outliers["consensus"] = {
                "count": len(consensus),
                "pct": round(float(len(consensus) / n * 100), 2) if n > 0 else 0,
            }

            results[c] = outliers

        logger.info(f"Outlier detection complete for {len(results)} columns")
        return results

    def compute_missing_patterns(self, df):
        logger.info("Computing missing value patterns")
        missing_matrix = df.isna()

        missing_summary = {
            "rows_with_any_missing": int(missing_matrix.any(axis=1).sum()),
            "rows_with_all_missing": int(missing_matrix.all(axis=1).sum()),
            "pct_rows_with_missing": round(float(missing_matrix.any(axis=1).mean() * 100), 2),
        }

        missing_corr = missing_matrix.corr()
        missing_corr_dict = {}
        for c1 in missing_corr.columns:
            for c2 in missing_corr.columns:
                if c1 < c2 and abs(missing_corr.loc[c1, c2]) > 0.3:
                    key = f"{c1}___{c2}"
                    missing_corr_dict[key] = round(float(missing_corr.loc[c1, c2]), 3)

        missing_summary["co_missing_correlations"] = missing_corr_dict
        missing_summary["missing_columns"] = list(missing_matrix.columns[missing_matrix.any()])
        missing_summary["complete_columns"] = list(missing_matrix.columns[~missing_matrix.any()])

        pattern_counts = (
            missing_matrix
            .loc[:, missing_matrix.any()]
            .drop_duplicates()
            .value_counts()
            .head(15)
        )
        missing_summary["top_missing_patterns"] = {
            str(k): int(v) for k, v in pattern_counts.items()
        }

        return {"missing_matrix": missing_matrix, "missing_corr": missing_corr, "summary": missing_summary}

    def save_report(self, stats, outliers, missing, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "univariate": stats,
            "outliers": outliers,
            "missing": missing["summary"],
            "config": {
                "outlier_iqr_multiplier": self.cfg.eda.outlier_iqr_multiplier,
                "outlier_zscore_threshold": self.cfg.eda.outlier_zscore_threshold,
                "isolation_forest_contamination": self.cfg.eda.isolation_forest_contamination,
                "random_state": self.cfg.random_state,
            },
        }

        path = output_dir / "eda_summary.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"EDA report saved to {path}")

    def run(self, df):
        logger.info("Starting Deep EDA")
        stats = self.compute_univariate_stats(df)
        outliers = self.detect_outliers(df)
        missing = self.compute_missing_patterns(df)
        self.save_report(stats, outliers, missing, self.cfg.output_dir)
        logger.info("Deep EDA complete")
        return {"stats": stats, "outliers": outliers, "missing_matrix": missing["missing_matrix"], "missing_corr": missing["missing_corr"]}
