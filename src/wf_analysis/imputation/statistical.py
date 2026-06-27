"""Statistical imputation using group median/mode."""

import pandas as pd
from loguru import logger

from wf_analysis.imputation.base import ImputerStrategy


class StatisticalImputer(ImputerStrategy):
    def __init__(self, method: str = "median"):
        self.method = method
        self.group_stats: pd.DataFrame | None = None
        self.group_columns: list[str] = []
        self.target_column: str = ""

    def fit(
        self, df: pd.DataFrame, target_column: str, feature_columns: list[str]
    ) -> "StatisticalImputer":
        self.target_column = target_column
        self.group_columns = feature_columns

        known = df[df[target_column].notna()]
        agg_func = "median" if self.method == "median" else "mean"

        if feature_columns:
            self.group_stats = known.groupby(feature_columns)[target_column].agg(agg_func).reset_index()
        else:
            self.group_stats = pd.DataFrame({target_column: [known[target_column].agg(agg_func)]})

        logger.info(f"Statistical imputer fitted: method={self.method}, target={target_column}")
        return self

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        missing = df[df[self.target_column].isna()]
        if len(missing) == 0:
            return df

        if self.group_columns:
            for _, row in self.group_stats.iterrows():
                mask = df[self.target_column].isna()
                for col in self.group_columns:
                    mask &= df[col] == row[col]
                df.loc[mask, self.target_column] = row[self.target_column]
        else:
            global_val = self.group_stats[self.target_column].iloc[0]
            df.loc[df[self.target_column].isna(), self.target_column] = global_val

        filled = missing.shape[0] - df[self.target_column].isna().sum()
        logger.info(f"StatisticalImputer: filled {filled} missing values in {self.target_column}")
        return df
