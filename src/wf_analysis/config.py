"""Configuration management using pydantic models loaded from YAML."""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    raw_path: str = "data/raw/employee_data.csv"
    interim_dir: str = "data/interim/"
    processed_dir: str = "data/processed/"
    schema_path: str = "config/schema.yaml"
    sample_size: Optional[int] = None
    pii_columns: list[str] = ["FirstName", "LastName", "ADEmail"]
    date_columns: list[str] = ["StartDate", "ExitDate", "DOB"]


class FeatureConfig(BaseModel):
    age_bins: list[float] = [0, 29, 39, 49, 59, 69, 120]
    age_labels: list[str] = ["<30", "30s", "40s", "50s", "60s", "70+"]
    mappings_path: str = "config/categorical_mappings.yaml"
    generate_embeddings: bool = True


class ColumnNLPConfig(BaseModel):
    enabled: bool = True
    sentiment: bool = False
    topic_modeling: bool = False
    classification: bool = False
    keywords: bool = False
    embeddings: bool = False
    similarity: bool = False
    clustering: bool = False
    network_analysis: bool = False


class NLPConfig(BaseModel):
    columns: dict[str, ColumnNLPConfig] = Field(default_factory=dict)
    sentiment_model: str = "vader"
    topic_model: str = "lda"
    n_topics: int = 6
    embedding_model: str = "all-MiniLM-L6-v2"
    classifier: str = "logistic"


class ImputationModelConfig(BaseModel):
    n_estimators: int = 200
    max_depth: int = 10
    learning_rate: float = 0.05


class ImputationConfig(BaseModel):
    strategy: Literal["statistical", "predictive", "ensemble"] = "ensemble"
    target_columns: list[str] = ["Age", "DOB"]
    feature_columns: list[str] = [
        "JobFamily", "SeniorityLevel", "DivisionGroup", "Region",
        "GenderCode", "Performance Score", "Current Employee Rating",
    ]
    test_size: float = 0.2
    models: dict[str, ImputationModelConfig] = Field(default_factory=lambda: {
        "rf": ImputationModelConfig(n_estimators=300, max_depth=10),
        "gbm": ImputationModelConfig(n_estimators=200, learning_rate=0.05),
    })
    jitter_range: list[float] = [-1.5, 1.5]
    random_state: int = 42


class AttritionConfig(BaseModel):
    survival_method: str = "kaplan_meier"
    confidence_interval: float = 0.95


class DiversityConfig(BaseModel):
    significance_level: float = 0.05
    benchmark_source: Optional[str] = None


class ForecastingConfig(BaseModel):
    method: str = "prophet"
    forecast_periods: int = 12
    seasonality: list[str] = ["yearly", "quarterly"]


class AnalysisConfig(BaseModel):
    attrition: AttritionConfig = Field(default_factory=AttritionConfig)
    diversity: DiversityConfig = Field(default_factory=DiversityConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)


class OutputConfig(BaseModel):
    formats: list[str] = ["csv", "parquet"]
    compression: str = "snappy"
    clean_versions: int = 3
    export_visualizations: bool = True
    visualization_dir: str = "reports/figures/"


class PipelineConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    imputation: ImputationConfig = Field(default_factory=ImputationConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw.get("pipeline", raw))
