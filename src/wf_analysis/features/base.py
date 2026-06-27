from abc import ABC, abstractmethod

import pandas as pd


class BaseFeatureTransformer(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseFeatureTransformer":
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...
