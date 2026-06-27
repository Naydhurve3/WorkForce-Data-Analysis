import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from wf_analysis.imputation.base import ImputerStrategy


class PredictiveImputer(ImputerStrategy):
    def __init__(
        self,
        model_type: str = "gbm",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []
        self.target_column: str = ""
        self.metrics: dict = {}
        self._feature_names: list[str] = []

    def _build_model(self):
        if self.model_type == "rf":
            return RandomForestRegressor(
                n_estimators=300, max_depth=10, random_state=self.random_state
            )
        return GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, random_state=self.random_state
        )

    def fit(
        self, df: pd.DataFrame, target_column: str, feature_columns: list[str]
    ) -> "PredictiveImputer":
        self.target_column = target_column
        self.feature_columns = feature_columns
        known = df[df[target_column].notna()].copy()
        X = known[feature_columns]
        y = known[target_column].astype(float)

        cat_cols = [c for c in feature_columns if X[c].dtype == "object"]
        num_cols = [c for c in feature_columns if c not in cat_cols]

        X_encoded = []
        self._feature_names = []
        if cat_cols:
            encoded = self.encoder.fit_transform(X[cat_cols])
            X_encoded.append(encoded)
            self._feature_names.extend(
                self.encoder.get_feature_names_out(cat_cols)
            )
        if num_cols:
            scaled = self.scaler.fit_transform(X[num_cols])
            X_encoded.append(scaled)
            self._feature_names.extend(num_cols)

        X_processed = np.hstack(X_encoded) if X_encoded else X.values

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=self.test_size, random_state=self.random_state
        )

        self.model = self._build_model()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }
        logger.info(
            f"PredictiveImputer ({self.model_type}) fit on {target_column}: "
            f"R²={self.metrics['r2']:.3f}, MAE={self.metrics['mae']:.2f}"
        )
        return self

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        missing = df[df[self.target_column].isna()]
        if len(missing) == 0:
            return df

        X_miss = missing[self.feature_columns]
        cat_cols = self.encoder.feature_names_in_.tolist() if hasattr(self.encoder, "feature_names_in_") else []
        num_cols = [c for c in self.feature_columns if c not in cat_cols]

        X_parts = []
        if cat_cols:
            X_parts.append(self.encoder.transform(X_miss[cat_cols]))
        if num_cols:
            X_parts.append(self.scaler.transform(X_miss[num_cols]))
        X_processed = np.hstack(X_parts)

        predictions = self.model.predict(X_processed)
        df.loc[df[self.target_column].isna(), self.target_column] = predictions
        logger.info(
            f"PredictiveImputer: imputed {len(predictions)} values for {self.target_column}"
        )
        return df
