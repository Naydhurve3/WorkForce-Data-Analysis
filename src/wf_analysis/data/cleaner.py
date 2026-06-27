"""Data cleaning operations: PII removal, date standardization, dedup."""

import pandas as pd
from loguru import logger


class DataCleaner:
    @staticmethod
    def remove_pii(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        removed = []
        for col in columns:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
                removed.append(col)
        if removed:
            logger.info(f"Removed PII columns: {removed}")
        return df

    @staticmethod
    def standardize_dates(
        df: pd.DataFrame, columns: list[str], errors: str = "coerce"
    ) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors=errors)
                logger.info(f"Converted {col} to datetime")
        return df

    @staticmethod
    def remove_duplicates(
        df: pd.DataFrame, subset: list[str] | None = None
    ) -> pd.DataFrame:
        before = len(df)
        df = df.copy().drop_duplicates(subset=subset)
        after = len(df)
        if before > after:
            logger.info(f"Removed {before - after} duplicate rows")
        return df
