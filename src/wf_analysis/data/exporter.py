"""Multi-format data export (CSV, Parquet, Excel)."""

from pathlib import Path

import pandas as pd
from loguru import logger


class DataExporter:
    @staticmethod
    def to_csv(df: pd.DataFrame, path: str | Path, index: bool = False, **kwargs) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index, **kwargs)
        logger.info(f"Exported {df.shape[0]} rows to {path}")
        return str(path)

    @staticmethod
    def to_parquet(
        df: pd.DataFrame, path: str | Path, compression: str = "snappy", **kwargs
    ) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression=compression, **kwargs)
        logger.info(f"Exported {df.shape[0]} rows to {path}")
        return str(path)

    @staticmethod
    def to_excel(
        df: pd.DataFrame, path: str | Path, sheet_name: str = "Sheet1", **kwargs
    ) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(path, sheet_name=sheet_name, index=False, **kwargs)
        logger.info(f"Exported {df.shape[0]} rows to {path}")
        return str(path)
