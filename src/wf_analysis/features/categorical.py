from pathlib import Path

import pandas as pd
import yaml

from wf_analysis.features.base import BaseFeatureTransformer


class CategoricalTransformer(BaseFeatureTransformer):
    def __init__(self, mappings_path: str = "config/categorical_mappings.yaml"):
        self.mappings_path = mappings_path
        self.mappings = self._load_mappings()
        self._fitted = False

    def _load_mappings(self) -> dict:
        path = Path(self.mappings_path)
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def fit(self, df: pd.DataFrame) -> "CategoricalTransformer":
        self._fitted = True
        return self

    def _map_by_keywords(self, value: str, category_map: dict) -> str:
        if not isinstance(value, str):
            return "Other"
        v_lower = value.lower()
        def _max_kw_len(item):
            return max((len(k) for k in item[1]), default=0)

        sorted_cats = sorted(
            category_map.items(),
            key=_max_kw_len,
            reverse=True,
        )
        for category, keywords in sorted_cats:
            for kw in keywords:
                if kw and kw.lower() in v_lower:
                    return category
        return "Other"

    def map_job_family(self, title: str) -> str:
        return self._map_by_keywords(title, self.mappings.get("job_family", {}))

    def map_seniority_level(self, title: str) -> str:
        return self._map_by_keywords(title, self.mappings.get("seniority_level", {}))

    def map_division_group(self, division: str) -> str:
        return self._map_by_keywords(division, self.mappings.get("division_group", {}))

    def map_region(self, state: str) -> str:
        return self._map_by_keywords(state, self.mappings.get("region_mapping", {}))

    def map_job_function_group(self, func_desc: str) -> str:
        return self._map_by_keywords(
            func_desc, self.mappings.get("job_function_group", {})
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Title" in df.columns:
            df["JobFamily"] = df["Title"].apply(self.map_job_family)
            df["SeniorityLevel"] = df["Title"].apply(self.map_seniority_level)
        if "Division" in df.columns:
            df["DivisionGroup"] = df["Division"].apply(self.map_division_group)
        if "State" in df.columns:
            df["Region"] = df["State"].apply(self.map_region)
        if "JobFunctionDescription" in df.columns:
            df["JobFunctionGroup"] = df["JobFunctionDescription"].apply(
                self.map_job_function_group
            )
        return df
