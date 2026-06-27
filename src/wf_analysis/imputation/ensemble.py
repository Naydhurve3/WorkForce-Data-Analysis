import numpy as np
import pandas as pd
from loguru import logger

from wf_analysis.imputation.base import ImputerStrategy
from wf_analysis.imputation.statistical import StatisticalImputer
from wf_analysis.imputation.predictive import PredictiveImputer


class EnsembleImputer(ImputerStrategy):
    def __init__(
        self,
        models: list[tuple[str, dict]] | None = None,
        weights: list[float] | None = None,
        jitter_range: list[float] | None = None,
        use_distribution_match: bool = True,
    ):
        self.models = models or [
            ("statistical", {"method": "median"}),
            ("predictive", {"model_type": "rf"}),
            ("predictive", {"model_type": "gbm"}),
        ]
        self.weights = weights
        self.jitter_range = jitter_range
        self.use_distribution_match = use_distribution_match
        self._imputers: list[ImputerStrategy] = []
        self.target_column: str = ""
        self.feature_columns: list[str] = []
        self._known_values: np.ndarray = np.array([])
        self._known_std: float = 1.0

    def fit(
        self, df: pd.DataFrame, target_column: str, feature_columns: list[str]
    ) -> "EnsembleImputer":
        self.target_column = target_column
        self.feature_columns = feature_columns

        for model_type, params in self.models:
            if model_type == "statistical":
                imp = StatisticalImputer(**params)
            elif model_type == "predictive":
                imp = PredictiveImputer(**params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            imp.fit(df, target_column, feature_columns)
            self._imputers.append(imp)

        n_models = len(self._imputers)
        if self.weights is None:
            self.weights = [1.0 / n_models] * n_models
        assert len(self.weights) == n_models, (
            f"Weights count ({len(self.weights)}) != model count ({n_models})"
        )

        if self.use_distribution_match:
            known = df[df[target_column].notna()][target_column].values
            if len(known) > 0:
                self._known_values = known
                self._known_std = float(np.std(known)) if len(known) > 1 else 1.0

        if self.jitter_range is None:
            self.jitter_range = [-self._known_std * 0.15, self._known_std * 0.15]

        logger.info(
            f"EnsembleImputer fitted with {n_models} models, "
            f"weights={self.weights}, jitter={self.jitter_range}"
        )
        return self

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        missing_mask = df[self.target_column].isna()
        if missing_mask.sum() == 0:
            return df

        predictions = np.zeros(len(df))
        weight_sum = sum(self.weights)

        for imp, weight in zip(self._imputers, self.weights):
            result = imp.impute(df.copy())
            predictions += weight * result[self.target_column].fillna(0).values

        predictions = predictions / weight_sum

        if self.use_distribution_match and len(self._known_values) > 0:
            n_missing = missing_mask.sum()
            sampled = np.random.choice(self._known_values, size=n_missing)
            blend = 1.0
            predictions[missing_mask] = (
                blend * sampled + (1 - blend) * predictions[missing_mask]
            )

        jitter = np.random.uniform(
            self.jitter_range[0],
            self.jitter_range[1],
            size=len(df),
        )
        predictions += jitter
        predictions = np.maximum(predictions, 0)

        df.loc[missing_mask, self.target_column] = predictions[missing_mask]
        logger.info(
            f"EnsembleImputer: imputed {missing_mask.sum()} values for {self.target_column}"
        )
        return df
