from pydantic import BaseModel, Field


class EDAConfig(BaseModel):
    outlier_iqr_multiplier: float = 1.5
    outlier_zscore_threshold: float = 3.0
    isolation_forest_contamination: float = 0.05
    max_unique_categorical: int = 30
    max_columns_per_grid: int = 8


class DimReductionConfig(BaseModel):
    pca_n_components: int = 6
    tsne_perplexities: list[int] = Field(default_factory=lambda: [5, 30, 50])
    tsne_random_state: int = 42
    correlation_threshold: float = 0.3
    network_min_edges: int = 1


class InteractionConfig(BaseModel):
    raw_path: str = "data/raw/employee_data.csv"
    output_dir: str = "data/interaction"
    figure_dir: str = "reports/figures/interaction"
    random_state: int = 42
    date_columns: list[str] = Field(default_factory=lambda: ["StartDate", "ExitDate", "DOB"])
    id_column: str = "EmpID"
    eda: EDAConfig = Field(default_factory=EDAConfig)
    dim_reduction: DimReductionConfig = Field(default_factory=DimReductionConfig)
