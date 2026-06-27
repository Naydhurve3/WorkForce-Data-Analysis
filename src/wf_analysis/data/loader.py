"""Data loading with caching, format detection, and validation."""

import os
from pathlib import Path

import pandas as pd
from loguru import logger

from wf_analysis.data.validator import DataValidator


class DataLoader:
    _cache: dict[str, pd.DataFrame] = {}

    @classmethod
    def load(
        cls,
        path: str | Path,
        format: str | None = None,
        validate: bool = True,
        schema_path: str | None = None,
        cache: bool = True,
    ) -> pd.DataFrame:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at: {path}")

        cache_key = str(path.absolute())
        if cache and cache_key in cls._cache:
            logger.info(f"Using cached dataset: {path.name}")
            return cls._cache[cache_key].copy()

        fmt = format or path.suffix.lstrip(".")
        logger.info(f"Loading dataset from: {path}")

        loaders = {
            "csv": pd.read_csv,
            "parquet": pd.read_parquet,
            "xlsx": pd.read_excel,
            "json": pd.read_json,
        }

        loader = loaders.get(fmt)
        if loader is None:
            raise ValueError(f"Unsupported format: {fmt}. Use: {list(loaders.keys())}")

        df = loader(path)
        logger.info(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

        if validate:
            report = DataValidator.validate_schema(df, schema_path)
            if not report.passed:
                logger.warning(f"Schema validation had {len(report.errors)} errors, {len(report.warnings)} warnings")
                for err in report.errors:
                    logger.warning(f"  Schema error: {err}")

        if cache:
            cls._cache[cache_key] = df.copy()

        return df

    @classmethod
    def load_sample(cls, path: str | Path, n: int = 100, random_state: int = 42) -> pd.DataFrame:
        df = cls.load(path, validate=False, cache=False)
        sample = df.sample(n=min(n, len(df)), random_state=random_state)
        logger.info(f"Loaded sample of {len(sample)} rows")
        return sample
