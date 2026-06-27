"""Schema validation and data quality reporting."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger


@dataclass
class ValidationReport:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_summary: pd.Series = field(default_factory=pd.Series)
    dtypes: pd.Series = field(default_factory=pd.Series)
    shape: tuple = (0, 0)


class DataValidator:
    @staticmethod
    def validate_schema(
        df: pd.DataFrame, schema_path: str | None = None
    ) -> ValidationReport:
        report = ValidationReport(shape=df.shape, dtypes=df.dtypes)

        if schema_path and Path(schema_path).exists():
            with open(schema_path) as f:
                schema = yaml.safe_load(f)

            col_defs = schema.get("columns", {})
            for col_name, col_spec in col_defs.items():
                if col_spec.get("required", False) and col_name not in df.columns:
                    report.errors.append(f"Required column '{col_name}' missing")
                    report.passed = False

                if col_name in df.columns:
                    expected_dtype = col_spec.get("dtype")
                    actual_dtype = str(df[col_name].dtype)
                    if expected_dtype and expected_dtype != actual_dtype:
                        report.warnings.append(
                            f"Column '{col_name}': expected {expected_dtype}, got {actual_dtype}"
                        )

                    enum_vals = col_spec.get("enum")
                    if enum_vals:
                        actual_vals = df[col_name].dropna().unique()
                        invalid = set(actual_vals) - set(enum_vals)
                        if invalid:
                            report.warnings.append(
                                f"Column '{col_name}': unexpected values: {list(invalid)[:5]}"
                            )

        missing = df.isnull().sum()
        missing = missing[missing > 0]
        report.missing_summary = missing

        if len(missing) > 0:
            logger.info(f"Missing values detected in {len(missing)} columns")
            for col, count in missing.items():
                pct = count / len(df) * 100
                logger.info(f"  {col}: {count} ({pct:.1f}%)")

        return report

    @staticmethod
    def generate_report(df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport(shape=df.shape, dtypes=df.dtypes)

        missing = df.isnull().sum()
        missing = missing[missing > 0]
        report.missing_summary = missing

        for col in df.columns:
            if df[col].dtype == "object":
                n_unique = df[col].nunique()
                if n_unique < 20:
                    report.warnings.append(
                        f"Column '{col}' has {n_unique} unique values"
                    )

        return report
