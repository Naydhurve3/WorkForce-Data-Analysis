"""Abstract base class for imputation strategies."""

from abc import ABC, abstractmethod

import pandas as pd


class ImputerStrategy(ABC):
    @abstractmethod
    def fit(
        self, df: pd.DataFrame, target_column: str, feature_columns: list[str]
    ) -> "ImputerStrategy":
        ...

    @abstractmethod
    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        ...
