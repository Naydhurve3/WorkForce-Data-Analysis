from pathlib import Path
import streamlit as st
import pandas as pd


@st.cache_data
def load_processed_data(
    path: str = "data/processed/workforce_clean_base.parquet",
) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        csv_path = p.with_suffix(".csv")
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data
def load_raw_sample(
    path: str = "data/sample/employee_data_sample.csv",
) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def format_number(value: float, precision: int = 1) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{precision}f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.{precision}f}K"
    return f"{value:.{precision}f}"
