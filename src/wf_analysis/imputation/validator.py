import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import ks_2samp, wasserstein_distance

import matplotlib.pyplot as plt
from wf_analysis.visualization.theme import Theme


class ImputationValidator:
    @staticmethod
    def compare_distributions(
        original: pd.Series, imputed: pd.Series, plot: bool = True
    ) -> dict:
        orig_clean = original.dropna()
        imp_clean = imputed.dropna()

        if len(orig_clean) == 0 or len(imp_clean) == 0:
            return {"error": "Empty series for comparison"}

        ks_stat, ks_pval = ks_2samp(orig_clean, imp_clean)
        wass_dist = wasserstein_distance(orig_clean.values, imp_clean.values)

        result = {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "ks_same_distribution": bool(ks_pval > 0.05),
            "wasserstein_distance": float(wass_dist),
            "mean_original": float(orig_clean.mean()),
            "mean_imputed": float(imp_clean.mean()),
            "std_original": float(orig_clean.std()),
            "std_imputed": float(imp_clean.std()),
        }

        if plot:
            Theme.set_style()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(
                orig_clean,
                bins=30,
                alpha=0.5,
                label="Original",
                color="#2E86AB",
                density=True,
            )
            ax.hist(
                imp_clean,
                bins=30,
                alpha=0.5,
                label="Imputed",
                color="#F18F01",
                density=True,
            )
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.set_title(
                f"Original vs Imputed Distribution (KS p={ks_pval:.3f})"
            )
            ax.legend()
            plt.tight_layout()
            result["plot"] = fig

        logger.info(
            f"Imputation validation: KS stat={ks_stat:.3f}, "
            f"p-value={ks_pval:.3f}, Wasserstein={wass_dist:.3f}"
        )
        return result

    @staticmethod
    def full_report(
        df_original: pd.DataFrame,
        df_imputed: pd.DataFrame,
        target_columns: list[str],
    ) -> dict:
        report = {}
        for col in target_columns:
            if col in df_original.columns and col in df_imputed.columns:
                report[col] = ImputationValidator.compare_distributions(
                    df_original[col], df_imputed[col], plot=True
                )
        return report
