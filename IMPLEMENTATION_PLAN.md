# WorkForce Data Analysis v2.0 - AI Agent Implementation Plan

> **Version**: 2.0.0  
> **Status**: Ready for Implementation  
> **Priority**: Analytical depth + HR Manager usability  
> **Source Dataset**: `data/raw/employee_data.csv` (3,000 records, 26 columns)  
> **Total Phases**: 8 | **Estimated Effort**: 8 weeks

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Python Package Architecture](#3-python-package-architecture)
4. [Configuration System](#4-configuration-system)
5. [Pipeline Orchestrator](#5-pipeline-orchestrator)
6. [Data Module](#6-data-module)
7. [Feature Engineering Module](#7-feature-engineering-module)
8. [NLP Module](#8-nlp-module)
9. [Imputation Module](#9-imputation-module)
10. [Analysis Modules](#10-analysis-modules)
11. [Visualization Module](#11-visualization-module)
12. [Streamlit Dashboard](#12-streamlit-dashboard)
13. [Testing Strategy](#13-testing-strategy)
14. [Implementation Phases - Step by Step](#14-implementation-phases---step-by-step)
15. [Acceptance Criteria](#15-acceptance-criteria)

---

## 1. Project Overview

This project transforms raw employee data (3,000 records, 26 columns) into a professional workforce analytics platform. The system is a modular Python package with:

- **Data Pipeline**: Loading, validation, cleaning, export
- **Feature Engineering**: Categorical, demographic, temporal, embedding-based features
- **NLP Pipeline**: Sentiment analysis, topic modeling, text classification, key phrase extraction, text embeddings on `TerminationDescription`, `Title`, `JobFunctionDescription`, `Supervisor`
- **ML Imputation**: Ensemble (RF + GBM) for Age/DOB with statistical distribution validation
- **7 Analysis Modules**: Attrition, Diversity, Performance, Compensation, Network, Career Path, Forecasting
- **Interactive Dashboard**: Streamlit with 7 pages + global filters
- **Testing**: pytest with 80%+ coverage

### Dataset Schema (employee_data.csv)

| Column | Type | Description | Missing |
|--------|------|-------------|---------|
| EmpID | int64 | Unique ID | 0 |
| FirstName | object | First name (PII) | 0 |
| LastName | object | Last name (PII) | 0 |
| StartDate | object | Start date | 0 |
| ExitDate | object | Exit date | 1,467 |
| Title | object | Job title (32 unique) | 0 |
| Supervisor | object | Manager name | 0 |
| ADEmail | object | Email (PII) | 0 |
| BusinessUnit | object | BU code (10 unique) | 0 |
| EmployeeStatus | object | Status | 0 |
| EmployeeType | object | Type | 0 |
| PayZone | object | Comp zone | 0 |
| EmployeeClassificationType | object | Classification | 0 |
| TerminationType | object | Term type (5 values) | 0 |
| TerminationDescription | object | Term reason text (1,533 non-empty) | 1,467 |
| DepartmentType | object | Department | 0 |
| Division | object | Division (25 unique) | 0 |
| DOB | object | Date of birth (1,823 invalid) | 0* |
| State | object | US state code | 0 |
| JobFunctionDescription | object | Function (83 unique) | 0 |
| GenderCode | object | Gender | 0 |
| LocationCode | int64 | Location ID | 0 |
| RaceDesc | object | Race/ethnicity | 0 |
| MaritalDesc | object | Marital status | 0 |
| Performance Score | object | Rating (Exceeds, Fully Meets, Needs Improvement, PIP) | 0 |
| Current Employee Rating | int64 | Numeric rating 1-5 | 0 |

> *DOB has 1,823 unparseable dates (60.8%) that must be imputed

---

## 2. Folder Structure

```
WorkForce-Data-Analysis/
│
├── IMPLEMENTATION_PLAN.md          # This file - AI agent instructions
├── pyproject.toml                   # Modern Python packaging (PEP 621)
├── setup.cfg                        # pytest, flake8, mypy config
├── Makefile                         # make install, make test, make run
├── .gitignore
├── .env.example
├── requirements.txt                 # Production deps only
├── requirements-dev.txt             # Dev/test deps
├── README.md                        # Project readme
│
├── src/
│   └── wf_analysis/
│       ├── __init__.py              # Package metadata, __version__ = "2.0.0"
│       ├── config.py                # pydantic config from YAML files
│       ├── pipeline.py              # DAG-based pipeline orchestrator
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py            # CSV/Parquet loader with caching
│       │   ├── cleaner.py           # PII removal, date standardization
│       │   ├── validator.py         # Schema validation engine
│       │   └── exporter.py          # CSV, Parquet, Excel, JSON export
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── base.py              # Base FeatureTransformer (ABC)
│       │   ├── demographic.py       # Age, AgeGroup, Tenure, IsActive, Generation
│       │   ├── categorical.py       # JobFamily, SeniorityLevel, DivisionGroup, Region
│       │   ├── temporal.py          # JoinYear, BirthYear, Month/Quarter/Season
│       │   └── embeddings.py        # Sentence-BERT + PCA-reduced features
│       │
│       ├── nlp/
│       │   ├── __init__.py
│       │   ├── preprocessor.py      # Text cleaning, tokenization, lemmatization
│       │   ├── sentiment.py         # VADER sentiment analysis
│       │   ├── topic_model.py       # LDA topic modeling
│       │   ├── text_classifier.py   # TF-IDF + LogisticRegression
│       │   ├── keywords.py          # RAKE key phrase extraction
│       │   ├── embeddings.py        # Sentence-BERT text embeddings
│       │   └── visualizer.py        # Word clouds, topic bubbles, sentiment trends
│       │
│       ├── imputation/
│       │   ├── __init__.py
│       │   ├── base.py              # Base ImputerStrategy (ABC)
│       │   ├── statistical.py       # Median/mode/mean strategies
│       │   ├── predictive.py        # RF/GBM regression imputation
│       │   ├── ensemble.py          # Ensemble blend + natural jitter
│       │   └── validator.py         # KS test, Wasserstein distance, KDE overlap
│       │
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── base.py              # Base AnalysisModule (ABC)
│       │   ├── attrition.py         # Kaplan-Meier, attrition rates, cohort analysis
│       │   ├── diversity.py         # Simpson index, intersectional, chi-square
│       │   ├── performance.py       # Score dist, rating trends, PIP
│       │   ├── compensation.py      # PayZone, Gini coefficient, pay equity
│       │   ├── network.py           # NetworkX graph, centrality, span of control
│       │   ├── career_path.py       # Role similarity matrix, mobility paths
│       │   └── forecasting.py       # Prophet headcount forecasting
│       │
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── theme.py             # Consistent color palette, styling
│       │   ├── plots.py             # Reusable chart functions
│       │   ├── dashboards.py        # Multi-plot grid layouts
│       │   └── reports.py           # HTML report generator
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logging.py           # loguru structured logging
│           ├── helpers.py           # timer, memoize, chunk utilities
│           └── decorators.py        # @log_call, @timer, @cache_to_disk
│
├── config/
│   ├── config.yaml                  # Main pipeline configuration
│   ├── schema.yaml                  # Data schema definitions
│   ├── logging.yaml                 # Logging configuration
│   ├── categorical_mappings.yaml    # JobFamily, SenorityLevel, Region mappings
│   └── nlp_config.yaml              # NLP model parameters
│
├── data/
│   ├── raw/                         # Original employee_data.csv (read-only)
│   │   └── employee_data.csv
│   ├── interim/                     # Phase-by-phase intermediate outputs
│   ├── processed/                   # Final ready-to-analyze datasets
│   ├── external/                    # External reference data
│   ├── metadata/                    # Auto-generated data dictionaries
│   └── sample/                      # Small sample for testing (100 rows)
│       └── employee_data_sample.csv
│
├── notebooks/                       # Exploration notebooks
│   ├── 01-data-exploration.ipynb
│   ├── 02-data-cleaning.ipynb
│   ├── 03-feature-engineering.ipynb
│   ├── 04-nlp-analysis.ipynb
│   ├── 05-imputation-strategies.ipynb
│   ├── 06-attrition-analysis.ipynb
│   ├── 07-diversity-analysis.ipynb
│   ├── 08-performance-analysis.ipynb
│   ├── 09-org-network-analysis.ipynb
│   └── 10-forecasting-dashboard.ipynb
│
├── dashboard/
│   ├── app.py                       # Main entry + sidebar navigation
│   ├── pages/
│   │   ├── 01_overview.py           # Executive KPIs
│   │   ├── 02_attrition.py          # Attrition explorer
│   │   ├── 03_diversity.py          # Diversity & inclusion
│   │   ├── 04_performance.py        # Performance analytics
│   │   ├── 05_nlp_insights.py       # NLP termination analysis
│   │   ├── 06_org_network.py        # Org structure explorer
│   │   └── 07_workforce_planning.py # Forecasting & scenarios
│   ├── components/
│   │   ├── kpi_card.py
│   │   ├── filter_sidebar.py
│   │   └── chart_wrapper.py
│   ├── assets/
│   │   ├── logo.png
│   │   └── style.css
│   └── utils.py
│
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── test_data/
│   │   ├── test_loader.py
│   │   ├── test_cleaner.py
│   │   └── test_validator.py
│   ├── test_features/
│   │   ├── test_demographic.py
│   │   └── test_categorical.py
│   ├── test_nlp/
│   │   ├── test_preprocessor.py
│   │   ├── test_sentiment.py
│   │   └── test_topic_model.py
│   ├── test_imputation/
│   │   ├── test_statistical.py
│   │   └── test_predictive.py
│   ├── test_analysis/
│   │   ├── test_attrition.py
│   │   └── test_diversity.py
│   └── test_pipeline.py
│
├── models/
│   ├── imputation/
│   ├── nlp/
│   └── forecasting/
│
├── scripts/
│   ├── run_pipeline.py              # Full pipeline runner
│   ├── run_analysis.py              # Run specific analysis modules
│   └── run_dashboard.py             # Launch Streamlit
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
│
├── docs/
│   ├── user_guide.md                # HR manager user guide
│   └── developer_guide.md           # Developer setup guide
│
└── reports/
    └── figures/                     # Generated visualization files
```

---

## 3. Python Package Architecture

### 3.1 Package Configuration

```python
# pyproject.toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "wf_analysis"
version = "2.0.0"
description = "Workforce Data Analytics Pipeline"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "scikit-learn>=1.2",
    "scipy>=1.10",
    "python-dateutil>=2.8",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "loguru>=0.7",
    "sentence-transformers>=2.2",
    "vaderSentiment>=3.3",
    "nltk>=3.8",
    "gensim>=4.3",
    "networkx>=3.0",
    "lifelines>=0.27",
    "prophet>=1.1",
    "streamlit>=1.25",
    "plotly>=5.15",
    "graphviz>=0.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "flake8>=6.1",
    "mypy>=1.5",
    "black>=23.7",
    "pre-commit>=3.3",
]

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-v --cov=wf_analysis --cov-report=term-missing"
testpaths = ["tests"]

[tool.mypy]
strict = true
ignore_missing_imports = true
```

### 3.2 Class Hierarchy

```
wf_analysis/
│
├── Config (pydantic.BaseSettings)
│   ├── PipelineConfig
│   ├── DataConfig
│   ├── ImputationConfig
│   └── NLPConfig
│
├── Pipeline
│   ├── +add_stage(name, func, dependencies) -> Pipeline
│   ├── +run(config, df) -> Dict[str, DataFrame]
│   └── -_validate_dag()
│
├── data/
│   ├── DataLoader
│   │   ├── +load(path, format, validate, schema_path, cache) -> DataFrame
│   │   └── +load_sample(path, n, random_state) -> DataFrame
│   ├── DataCleaner
│   │   ├── +remove_pii(df, columns) -> DataFrame
│   │   ├── +standardize_dates(df, columns) -> DataFrame
│   │   └── +remove_duplicates(df, keys) -> DataFrame
│   ├── DataValidator
│   │   ├── +validate_schema(df, schema_path) -> ValidationReport
│   │   └── +generate_report(df) -> ValidationReport
│   └── DataExporter
│       ├── +to_csv(df, path)
│       ├── +to_parquet(df, path)
│       └── +to_excel(df, path)
│
├── features/
│   └── FeatureTransformer (ABC)
│       ├── +fit(df) -> FeatureTransformer
│       └── +transform(df) -> DataFrame
│       ├── DemographicFeatures
│       ├── CategoricalFeatures
│       ├── TemporalFeatures
│       └── EmbeddingFeatures
│
├── nlp/
│   ├── TextPreprocessor
│   │   └── +transform(text_series) -> Series
│   ├── SentimentAnalyzer
│   │   └── +analyze(text_series) -> DataFrame[scores, labels]
│   ├── TopicModeler
│   │   ├── +fit(texts) -> TopicModeler
│   │   └── +transform(texts) -> DataFrame[topics]
│   ├── TextClassifier
│   │   ├── +fit(X_text, y) -> TextClassifier
│   │   ├── +predict(texts) -> array
│   │   └── +evaluate(texts, labels) -> dict
│   ├── KeywordExtractor
│   │   └── +extract(text_series) -> Dict[id, phrases]
│   ├── TextEmbedder
│   │   └── +embed(text_series) -> ndarray
│   └── NLPVisualizer
│       ├── +wordcloud(texts) -> Figure
│       ├── +topics_bubble(model) -> Figure
│       └── +sentiment_trend(sentiments, dates) -> Figure
│
├── imputation/
│   └── ImputerStrategy (ABC)
│       ├── +fit(df, target, features) -> ImputerStrategy
│       └── +impute(df) -> DataFrame
│       ├── StatisticalImputer
│       ├── PredictiveImputer
│       ├── EnsembleImputer
│       └── ImputationValidator
│           └── +compare_distributions(original, imputed) -> dict
│
├── analysis/
│   └── AnalysisModule (ABC)
│       ├── +run(df) -> AnalysisResult
│       ├── +plot(result) -> Figure
│       └── +summarize(result) -> str
│       ├── AttritionAnalysis
│       ├── DiversityAnalysis
│       ├── PerformanceAnalysis
│       ├── CompensationAnalysis
│       ├── NetworkAnalysis
│       ├── CareerPathAnalysis
│       └── ForecastingAnalysis
│
├── visualization/
│   ├── Theme
│   │   ├── +set_style()
│   │   └── +color_palette(name) -> list
│   ├── PlotFactory
│   │   ├── +bar_chart(data, x, y, ...) -> Figure
│   │   ├── +box_plot(data, x, y, ...) -> Figure
│   │   ├── +kde_plot(data, x, ...) -> Figure
│   │   ├── +heatmap(data, ...) -> Figure
│   │   └── +wordcloud(texts, ...) -> Figure
│   ├── DashboardBuilder
│   │   └── +build_grid(plots, layout) -> Figure
│   └── ReportGenerator
│       └── +generate_html(analyses, figures, output_path) -> str
│
└── utils/
    ├── LoggerSetup
    │   └── +configure(config_path) -> logger
    ├── Timer
    │   └── +__enter__ / +__exit__ -> float
    └── decorators
        ├── @log_call
        ├── @timer
        └── @cache_to_disk(path)
```

### 3.3 Key Function Signatures

```python
# config.py
class PipelineConfig(BaseSettings):
    data: DataConfig
    features: FeatureConfig
    nlp: NLPConfig
    imputation: ImputationConfig
    analysis: AnalysisConfig
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig": ...

class DataConfig(BaseModel):
    raw_path: str = "data/raw/employee_data.csv"
    interim_dir: str = "data/interim/"
    processed_dir: str = "data/processed/"
    schema_path: str = "config/schema.yaml"
    sample_size: int | None = None
    pii_columns: list[str] = ["FirstName", "LastName", "ADEmail"]
    date_columns: list[str] = ["StartDate", "ExitDate", "DOB"]

class ImputationConfig(BaseModel):
    strategy: Literal["statistical", "predictive", "ensemble"] = "ensemble"
    target_columns: list[str] = ["Age", "DOB"]
    feature_columns: list[str] = [
        "JobFamily", "SeniorityLevel", "DivisionGroup", "Region",
        "GenderCode", "Performance Score", "Current Employee Rating"
    ]
    test_size: float = 0.2
    models: dict = {"rf": {"n_estimators": 300, "max_depth": 10},
                    "gbm": {"n_estimators": 200, "learning_rate": 0.05}}
    jitter_range: list[float] = [-1.5, 1.5]
    random_state: int = 42

class NLPConfig(BaseModel):
    columns: dict = {
        "TerminationDescription": {"sentiment": True, "topics": True, "classify": True, "keywords": True, "embeddings": True},
        "Title": {"embeddings": True, "similarity": True},
        "JobFunctionDescription": {"embeddings": True, "clustering": True},
        "Supervisor": {"network": True}
    }
    sentiment_model: str = "vader"
    topic_model: str = "lda"
    n_topics: int = 6
    embedding_model: str = "all-MiniLM-L6-v2"
    classifier: str = "logistic"

# pipeline.py
class Pipeline:
    def add_stage(self, name: str, func: Callable, dependencies: list[str] | None = None, output_key: str | None = None) -> "Pipeline": ...
    def run(self, config: PipelineConfig, df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]: ...
    def visualize_dag(self) -> graphviz.Digraph: ...

# data/loader.py
class DataLoader:
    @staticmethod
    def load(path: str | Path, format: str | None = None, validate: bool = True, schema_path: str | None = None, cache: bool = True) -> pd.DataFrame: ...
    @staticmethod
    def load_sample(path: str | Path, n: int = 100, random_state: int = 42) -> pd.DataFrame: ...

# data/cleaner.py
class DataCleaner:
    @staticmethod
    def remove_pii(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame: ...
    @staticmethod
    def standardize_dates(df: pd.DataFrame, columns: list[str], errors: str = "coerce") -> pd.DataFrame: ...
    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame: ...

# data/validator.py
class DataValidator:
    @staticmethod
    def validate_schema(df: pd.DataFrame, schema_path: str | None = None) -> "ValidationReport": ...
    @staticmethod
    def generate_report(df: pd.DataFrame) -> "ValidationReport": ...

class ValidationReport:
    passed: bool
    errors: list[str]
    warnings: list[str]
    missing_summary: pd.Series
    dtypes: pd.Series
    shape: tuple

# features/demographic.py
class DemographicFeatures(FeatureTransformer):
    def __init__(self, age_bins: list = [0, 29, 39, 49, 59, 69, 120], age_labels: list = ["<30", "30s", "40s", "50s", "60s", "70+"]): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds Age, AgeGroup, IsActive, TenureDays, TenureYears, TenureBucket, Generation""" ...

# features/categorical.py
class CategoricalFeatures(FeatureTransformer):
    def __init__(self, mappings_path: str = "config/categorical_mappings.yaml", use_embeddings: bool = True): ...
    def map_job_family(self, title: str) -> str: ...
    def map_seniority_level(self, title: str) -> str: ...
    def map_division_group(self, division: str) -> str: ...
    def map_region(self, state: str) -> str: ...
    def map_job_function_group(self, func_desc: str) -> str: ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

# features/embeddings.py
class EmbeddingFeatures(FeatureTransformer):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", n_components: int = 10): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds PCA-reduced embeddings for Title, JobFunctionDescription, Division""" ...

# nlp/sentiment.py
class SentimentAnalyzer:
    def __init__(self, method: Literal["vader", "textblob"] = "vader"): ...
    def analyze(self, texts: pd.Series) -> pd.DataFrame:
        """Returns DataFrame[sentiment_score, sentiment_label, magnitude, neg/neu/pos scores]""" ...
    def plot_distribution(self, sentiments: pd.DataFrame) -> plt.Figure: ...
    def plot_trend(self, sentiments: pd.DataFrame, dates: pd.Series) -> plt.Figure: ...

# nlp/topic_model.py
class TopicModeler:
    def __init__(self, method: Literal["lda", "nmf"] = "lda", n_topics: int = 5, random_state: int = 42): ...
    def fit(self, texts: pd.Series) -> "TopicModeler": ...
    def transform(self, texts: pd.Series) -> pd.DataFrame:
        """Returns DataFrame[dominant_topic, topic_probability, top_words]""" ...
    def plot_topics(self, n_words: int = 10) -> plt.Figure: ...

# nlp/text_classifier.py
class TextClassifier:
    def __init__(self, vectorizer: str = "tfidf", model: Literal["logistic", "rf", "svm"] = "logistic"): ...
    def fit(self, texts: pd.Series, labels: pd.Series) -> "TextClassifier": ...
    def predict(self, texts: pd.Series) -> np.ndarray: ...
    def evaluate(self, texts: pd.Series, labels: pd.Series) -> dict:
        """Returns dict[accuracy, precision, recall, f1, confusion_matrix]""" ...
    def plot_confusion_matrix(self, texts, labels) -> plt.Figure: ...

# nlp/keywords.py
class KeywordExtractor:
    def __init__(self, method: Literal["rake", "textrank"] = "rake"): ...
    def extract(self, texts: pd.Series) -> dict[int, list[tuple[str, float]]]:
        """Returns {doc_index: [(phrase, score), ...]}""" ...

# imputation/predictive.py
class PredictiveImputer(ImputerStrategy):
    def __init__(self, model_type: Literal["rf", "gbm"] = "gbm", test_size: float = 0.2, random_state: int = 42): ...
    def fit(self, df: pd.DataFrame, target_column: str, feature_columns: list[str]) -> "PredictiveImputer":
        """Trains model on known values, reports R², MAE, RMSE""" ...
    def impute(self, df: pd.DataFrame) -> pd.DataFrame: ...

# imputation/validator.py
class ImputationValidator:
    @staticmethod
    def compare_distributions(original: pd.Series, imputed: pd.Series, plot: bool = True) -> dict:
        """Returns dict[ks_statistic, ks_pvalue, wasserstein_distance, kde_overlap]""" ...

# analysis/attrition.py
class AttritionAnalysis(AnalysisModule):
    def run(self, df: pd.DataFrame) -> "AttritionResult":
        """Overall rate, Kaplan-Meier, attrition by group, seasonal patterns, termination breakdown""" ...
    def plot_survival_curves(self, result: "AttritionResult", group_col: str) -> plt.Figure: ...
    def plot_attrition_waterfall(self, result: "AttritionResult") -> plt.Figure: ...

# analysis/diversity.py
class DiversityAnalysis(AnalysisModule):
    def run(self, df: pd.DataFrame) -> "DiversityResult":
        """Gender/race dist, intersectional, Simpson index, chi-square, representation ratios""" ...
    def plot_diversity_heatmap(self, result: "DiversityResult", dim1: str, dim2: str) -> plt.Figure: ...
    def plot_intersectional_treemap(self, result: "DiversityResult") -> plt.Figure: ...

# analysis/network.py
class NetworkAnalysis(AnalysisModule):
    def run(self, df: pd.DataFrame) -> "NetworkResult":
        """Build DiGraph from Supervisor, centrality, span of control, org depth""" ...
    def plot_org_chart(self, result: "NetworkResult", max_depth: int = 5) -> plt.Figure: ...
    def plot_span_of_control(self, result: "NetworkResult") -> plt.Figure: ...

# analysis/forecasting.py
class ForecastingAnalysis(AnalysisModule):
    def run(self, df: pd.DataFrame) -> "ForecastingResult":
        """Monthly headcount, Prophet forecast, scenario modeling""" ...
    def plot_headcount_forecast(self, result: "ForecastingResult", periods: int = 12) -> plt.Figure: ...

# visualization/theme.py
class Theme:
    PRIMARY = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
    CATEGORICAL = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#6AB547", "#E58C6A", "#8B5CF6", "#06B6D4", "#84CC16"]
    @classmethod
    def set_style(cls): ...
    @classmethod
    def get_palette(cls, name: str = "categorical") -> list: ...
```

---

## 4. Configuration System

### 4.1 config.yaml

```yaml
project:
  name: "WorkForce Data Analysis v2"
  version: "2.0.0"
  seed: 42

data:
  raw_path: "data/raw/employee_data.csv"
  interim_dir: "data/interim/"
  processed_dir: "data/processed/"
  schema_path: "config/schema.yaml"
  pii_columns:
    - "FirstName"
    - "LastName"
    - "ADEmail"
  date_columns:
    - "StartDate"
    - "ExitDate"
    - "DOB"

features:
  age_bins: [0, 29, 39, 49, 59, 69, 120]
  age_labels: ["<30", "30s", "40s", "50s", "60s", "70+"]
  mappings_path: "config/categorical_mappings.yaml"
  generate_embeddings: true

nlp:
  columns:
    TerminationDescription:
      enabled: true
      sentiment: true
      topic_modeling: true
      classification: true
      keywords: true
      embeddings: true
    Title:
      enabled: true
      embeddings: true
      similarity: true
    JobFunctionDescription:
      enabled: true
      embeddings: true
      clustering: true
    Supervisor:
      enabled: true
      network_analysis: true
  sentiment_model: "vader"
  topic_model: "lda"
  n_topics: 6
  embedding_model: "all-MiniLM-L6-v2"
  classifier: "logistic"

imputation:
  strategy: "ensemble"
  target_columns: ["Age", "DOB"]
  feature_columns:
    - "JobFamily"
    - "SeniorityLevel"
    - "DivisionGroup"
    - "Region"
    - "GenderCode"
    - "Performance Score"
    - "Current Employee Rating"
  test_size: 0.2
  models:
    rf:
      n_estimators: 300
      max_depth: 10
    gbm:
      n_estimators: 200
      learning_rate: 0.05
  jitter_range: [-1.5, 1.5]
  random_state: 42

analysis:
  attrition:
    survival_method: "kaplan_meier"
    confidence_interval: 0.95
  diversity:
    significance_level: 0.05
    benchmark_source: null
  forecasting:
    method: "prophet"
    forecast_periods: 12
    seasonality: ["yearly", "quarterly"]

output:
  formats: ["csv", "parquet"]
  compression: "snappy"
  clean_versions: 3
  export_visualizations: true
  visualization_dir: "reports/figures/"
```

### 4.2 categorical_mappings.yaml

```yaml
job_family:
  Data & Analytics: ["data", "bi", "dba", "analyst"]
  IT & Infrastructure: ["it", "network", "infra", "support"]
  Sales: ["sales"]
  Finance & Accounting: ["accountant"]
  Executive & Leadership: ["president", "ceo", "cio", "director"]
  Operations / Shared Services: ["manager"]
  Production: ["production"]
  Admin & Support: ["admin"]
  Other: []

seniority_level:
  Executive: ["president", "ceo", "cio"]
  Director: ["director"]
  Manager: ["manager"]
  Senior: ["sr", "senior", "principal"]
  Entry: ["i"]
  Mid: []

division_group:
  Finance: ["finance"]
  IT: ["technology", "it"]
  Sales & Marketing: ["sales", "marketing"]
  Executive: ["executive"]
  HR / People Services: ["people"]
  Project Management: ["project management"]
  General Operations: ["general"]
  Field Engineering / Operations: ["aerial", "splicing", "field", "wireline", "catv", "isp"]
  Consulting / Client Delivery: ["consultant"]
  Logistics / Maintenance: ["yard", "fleet", "shop"]
  Safety & Compliance: ["safety"]
  Corporate Operations: ["corp"]
  Unknown: ["unknown"]
  Other: []

region_mapping:
  West: ["CA", "OR", "WA", "NV", "AZ", "UT", "ID", "MT"]
  Midwest: ["ND", "IN", "OH"]
  South: ["TX", "FL", "GA", "AL", "KY", "TN", "NC", "VA"]
  Northeast: ["MA", "CT", "RI", "VT", "NY", "PA", "NH", "ME"]
  Other: []
```

---

## 5. Pipeline Orchestrator

The `Pipeline` class manages a DAG of processing stages. Each stage is a function that takes a DataFrame and returns a DataFrame.

```python
# Usage:
pipeline = Pipeline()

pipeline.add_stage("load_data", load_raw_data, output_key="raw")
pipeline.add_stage("validate_schema", validate_data, ["load_data"], output_key="validated")
pipeline.add_stage("clean_pii", remove_pii, ["validate_schema"], output_key="no_pii")
pipeline.add_stage("standardize_dates", convert_dates, ["clean_pii"], output_key="dates_ready")
pipeline.add_stage("categorical_features", engineer_categorical, ["dates_ready"], output_key="with_cat_feats")
# ... etc

results = pipeline.run(config, df)
```

The `run()` method:
1. Validates the DAG (no circular dependencies, all dependencies exist)
2. Topologically sorts stages
3. Executes each stage, passing cached results from dependencies
4. Returns dict of `{output_key: DataFrame}`

---

## 6. Data Module

### 6.1 DataLoader

```python
class DataLoader:
    _cache: dict[str, pd.DataFrame] = {}
    
    @staticmethod
    def load(path, format=None, validate=True, schema_path=None, cache=True):
        """
        - Detects format from extension if not provided
        - Validates schema if validate=True
        - Caches in memory if cache=True (avoid reloading same path)
        - Handles: CSV, Parquet, Excel, JSON
        """
    
    @staticmethod
    def load_sample(path, n=100, random_state=42):
        """Load random sample of n rows for testing."""
```

### 6.2 DataCleaner

```python
class DataCleaner:
    @staticmethod
    def remove_pii(df, columns):
        """Drop specified columns, log which were removed."""
    
    @staticmethod
    def standardize_dates(df, columns, errors="coerce"):
        """Convert columns to datetime, coerce errors to NaT."""
    
    @staticmethod
    def remove_duplicates(df, subset=None):
        """Remove duplicate rows, log count removed."""
```

### 6.3 DataValidator

```python
class DataValidator:
    @staticmethod
    def validate_schema(df, schema_path=None):
        """Check column presence, dtypes, required fields, enum values."""
    
    @staticmethod
    def generate_report(df):
        """Generate comprehensive quality report: missing, dtypes, unique values, sample values."""
```

### 6.4 DataExporter

```python
class DataExporter:
    @staticmethod
    def to_csv(df, path, index=False, **kwargs): ...
    @staticmethod
    def to_parquet(df, path, compression="snappy", **kwargs): ...
    @staticmethod
    def to_excel(df, path, sheet_name="Sheet1", **kwargs): ...
```

---

## 7. Feature Engineering Module

### 7.1 DemographicFeatures

Derived features to create:

| Feature | Source | Logic |
|---------|--------|-------|
| Age | DOB | `(today - DOB).days / 365.25` |
| AgeGroup | Age | `pd.cut(Age, bins=[0,29,39,49,59,69,120], labels=["<30","30s","40s","50s","60s","70+"])` |
| IsActive | ExitDate | `ExitDate.isna()` |
| TenureDays | StartDate, ExitDate | `(ExitDate.fillna(today) - StartDate).days` |
| TenureYears | TenureDays | `TenureDays / 365.25` |
| TenureBucket | TenureYears | `<1yr, 1-3yr, 3-5yr, 5-10yr, 10+yr` |
| Generation | BirthYear | Silent(≤1945), Boomer(1946-1964), GenX(1965-1980), Millennial(1981-1996), GenZ(≥1997) |

### 7.2 CategoricalFeatures

Mapping functions:

- `map_job_family(title)`: Keyword-matches title to one of 9 JobFamilies
- `map_seniority_level(title)`: Keyword-matches title to one of 6 levels
- `map_division_group(division)`: Keyword-matches division to one of 14 groups
- `map_region(state)`: Maps US state code to 4 regions + Other
- `map_job_function_group(func_desc)`: Keyword-matches to one of 11 groups

Use YAML-driven mappings from `config/categorical_mappings.yaml`.

### 7.3 TemporalFeatures

| Feature | Source | Logic |
|---------|--------|-------|
| JoinYear | StartDate | `StartDate.dt.year` |
| BirthYear | DOB | `DOB.dt.year` |
| JoinMonth | StartDate | `StartDate.dt.month` |
| JoinQuarter | StartDate | `StartDate.dt.quarter` |
| JoinSeason | JoinMonth | Dec-Feb=Winter, Mar-May=Spring, Jun-Aug=Summer, Sep-Nov=Fall |
| ExitMonth | ExitDate | For terminated employees only |

### 7.4 EmbeddingFeatures

```python
class EmbeddingFeatures(FeatureTransformer):
    def __init__(self, model_name="all-MiniLM-L6-v2", n_components=10):
        self.embedder = SentenceTransformer(model_name)
        self.n_components = n_components
        self.pca_models = {}  # One PCA per column
    
    def transform(self, df):
        text_columns = {"Title": "Title", "JobFunctionDescription": "FuncDesc", "Division": "Division"}
        for src_col, prefix in text_columns.items():
            if src_col in df.columns:
                embeddings = self.embedder.encode(df[src_col].fillna("").astype(str))
                pca = PCA(n_components=self.n_components)
                reduced = pca.fit_transform(embeddings)
                for i in range(self.n_components):
                    df[f"{prefix}_Emb_{i}"] = reduced[:, i]
                self.pca_models[src_col] = pca
        return df
```

---

## 8. NLP Module

### 8.1 TextPreprocessor

```python
class TextPreprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
    
    def transform(self, texts: pd.Series) -> pd.Series:
        """Apply: lowercase → remove punctuation → tokenize → remove stopwords → lemmatize → rejoin"""
```

### 8.2 SentimentAnalyzer

```python
class SentimentAnalyzer:
    def __init__(self, method="vader"):
        self.analyzer = SentimentIntensityAnalyzer() if method == "vader" else None
    
    def analyze(self, texts: pd.Series) -> pd.DataFrame:
        """
        For each text, compute VADER compound score (-1 to 1):
        - sentiment_score: compound score
        - sentiment_label: Positive (>0.05), Neutral (-0.05 to 0.05), Negative (<-0.05)
        - sentiment_magnitude: abs(compound score)
        - neg_score, neu_score, pos_score: sub-scores
        """
```

### 8.3 TopicModeler

```python
class TopicModeler:
    def __init__(self, method="lda", n_topics=5, random_state=42):
        self.vectorizer = CountVectorizer(max_df=0.9, min_df=2)
        self.model = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
    
    def fit(self, texts: pd.Series) -> "TopicModeler":
        """Vectorize texts and fit LDA model."""
    
    def transform(self, texts: pd.Series) -> pd.DataFrame:
        """Return: dominant_topic (int), topic_probability (float), top_words per doc."""
    
    def get_topic_info(self, n_words=10) -> list[dict]:
        """Return list of {topic_id, top_words, interpretation, prevalence}."""
```

### 8.4 TextClassifier

```python
class TextClassifier:
    def __init__(self, vectorizer="tfidf", model="logistic"):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.model = LogisticRegression(multi_class="multinomial", max_iter=1000)
    
    def fit(self, texts, labels):
        """Vectorize, split train/test, train model, report metrics."""
    
    def predict(self, texts):
        """Predict labels for new texts."""
    
    def evaluate(self, texts, labels) -> dict:
        """Return accuracy, precision, recall, f1, confusion_matrix, classification_report."""
```

### 8.5 KeyWordExtractor

```python
class KeywordExtractor:
    def __init__(self, method="rake"):
        self.extractor = RAKE([nltk.corpus.stopwords.words("english")])
    
    def extract(self, texts: pd.Series) -> dict:
        """Return {index: [(phrase, score), ...]} sorted by score descending."""
```

### 8.6 TextEmbedder

```python
class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts: pd.Series) -> np.ndarray:
        """Return (n_docs, 384) embedding array."""
    
    def similarity_matrix(self, texts_a, texts_b=None):
        """Compute cosine similarity between all pairs."""
    
    def find_similar(self, query, texts, top_k=5):
        """Find most similar texts to query string."""
```

### 8.7 NLP Visualizer

```python
class NLPVisualizer:
    @staticmethod
    def wordcloud(texts: pd.Series, title="") -> plt.Figure: ...
    @staticmethod
    def topics_bubble(model: TopicModeler) -> plt.Figure: ...
    @staticmethod
    def sentiment_distribution(sentiments: pd.DataFrame) -> plt.Figure: ...
    @staticmethod
    def sentiment_by_group(sentiments: pd.DataFrame, df: pd.DataFrame, group_col: str) -> plt.Figure: ...
    @staticmethod
    def sentiment_trend(sentiments: pd.DataFrame, dates: pd.Series) -> plt.Figure: ...
```

---

## 9. Imputation Module

### 9.1 Imputation Strategy (5 Levels)

```
Level 1: StatisticalFallback → Group median by SeniorityLevel
Level 2: Predictive (RF) → RandomForestRegressor(300 estimators, max_depth=10)
Level 3: Predictive (GBM) → GradientBoostingRegressor(200 est, lr=0.05)
Level 4: Ensemble → Weighted blend of RF + GBM predictions
Level 5: Jitter → Add uniform noise [-1.5, 1.5] for natural distribution
```

### 9.2 StatisticalImputer

```python
class StatisticalImputer(ImputerStrategy):
    def __init__(self, method="median"):
        self.method = method
    
    def fit(self, df, target_column, feature_columns=None):
        """Compute group statistics."""
    
    def impute(self, df):
        """Fill missing values with group statistics."""
```

### 9.3 PredictiveImputer

```python
class PredictiveImputer(ImputerStrategy):
    def __init__(self, model_type="gbm", test_size=0.2, random_state=42):
        self.model_type = model_type
        self.test_size = test_size
    
    def fit(self, df, target_column, feature_columns):
        """Encode features, train on known values, evaluate on held-out set."""
        # 1. Split into known/missing
        # 2. OneHot encode categoricals, StandardScale numerics
        # 3. Train model on known values
        # 4. Report R², MAE, RMSE on test set
    
    def impute(self, df):
        """Predict missing values."""
```

### 9.4 EnsembleImputer

```python
class EnsembleImputer(ImputerStrategy):
    def __init__(self, models=None, weights=None, jitter_range=None):
        """Blend multiple imputers with optional jitter for natural distribution."""
    
    def fit(self, df, target_column, feature_columns):
        """Fit all sub-models."""
    
    def impute(self, df):
        """Combine predictions with weighted average + jitter."""
```

### 9.5 ImputationValidator

```python
class ImputationValidator:
    @staticmethod
    def compare_distributions(original, imputed, plot=True):
        """
        - KS test: p-value > 0.05 means same distribution
        - Wasserstein distance: lower is better
        - KDE overlap: 1.0 = identical
        """
    
    @staticmethod
    def full_report(df_original, df_imputed, target_columns):
        """Generate comprehensive validation report with all metrics + plots."""
```

---

## 10. Analysis Modules

### 10.1 AttritionAnalysis

```python
class AttritionAnalysis(AnalysisModule):
    def run(self, df):
        """Compute:
        1. Overall attrition rate (terminated / total)
        2. Kaplan-Meier survival curves (overall + by group)
        3. Attrition by JobFamily, DepartmentType, Region, GenderCode
        4. Termination type breakdown (Involuntary, Voluntary, Resignation, Retirement)
        5. Tenure comparison: leavers vs stayers
        6. Seasonal attrition patterns by month/quarter
        """
```

### 10.2 DiversityAnalysis

```python
class DiversityAnalysis(AnalysisModule):
    def run(self, df):
        """Compute:
        1. Gender distribution overall + by JobFamily/SeniorityLevel/Dept
        2. Race distribution overall + by JobFamily/Region
        3. Intersectional analysis: Gender×Race×JobFamily
        4. Simpson Diversity Index per department (for Gender and Race)
        5. Chi-square test for Gender×JobFamily independence
        6. Representation ratios (group proportion vs overall proportion)
        """
```

### 10.3 PerformanceAnalysis

```python
class PerformanceAnalysis(AnalysisModule):
    def run(self, df):
        """Compute:
        1. Performance Score distribution + by JobFamily/SeniorityLevel/Dept
        2. Current Employee Rating distribution + statistics by group
        3. PIP (Performance Improvement Plan) flag detection
        4. Performance vs Tenure correlation
        5. Performance vs Age correlation
        """
```

### 10.4 CompensationAnalysis

```python
class CompensationAnalysis(AnalysisModule):
    def run(self, df):
        """Compute:
        1. PayZone distribution overall + by JobFamily/SeniorityLevel
        2. Pay equity stats: gender breakdown within PayZone×JobFamily
        3. Gini coefficient of pay distribution
        4. PayZone vs Performance correlation
        """
```

### 10.5 NetworkAnalysis

```python
class NetworkAnalysis(AnalysisModule):
    def run(self, df):
        """Build org graph from Supervisor column, compute:
        1. Directed graph (NetworkX DiGraph)
        2. Degree/betweenness/closeness centrality
        3. Span of control (direct reports per manager)
        4. Organizational depth (levels from root)
        5. Top 10 influencers by betweenness centrality
        """
```

### 10.6 CareerPathAnalysis

```python
class CareerPathAnalysis(AnalysisModule):
    def run(self, df):
        """Compute:
        1. Role similarity matrix using Title embeddings (cosine similarity)
        2. For each role, top-5 most similar roles
        3. Career mobility: distribution of employees across JobFamilies
        4. Seniority level progression patterns
        """
```

### 10.7 ForecastingAnalysis

```python
class ForecastingAnalysis(AnalysisModule):
    def run(self, df):
        """Compute:
        1. Monthly/quarterly headcount time series
        2. Prophet forecast for next 12 months
        3. Scenario modeling (low/medium/high growth)
        4. Attrition rate trend and forecast
        """
```

---

## 11. Visualization Module

### 11.1 Theme

```python
class Theme:
    PRIMARY = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
    DIVERGING = ["#2E86AB", "#F18F01", "#C73E1D"]
    CATEGORICAL = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
                   "#6AB547", "#E58C6A", "#8B5CF6", "#06B6D4", "#84CC16"]
    
    @classmethod
    def set_style(cls):
        sns.set_theme(style="whitegrid", palette=cls.CATEGORICAL)
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        })
```

### 11.2 PlotFactory

```python
class PlotFactory:
    @staticmethod
    def bar_chart(data, x, y, title="", palette=None, figsize=(10, 6)) -> plt.Figure: ...
    @staticmethod
    def box_plot(data, x, y, hue=None, title="", figsize=(12, 6)) -> plt.Figure: ...
    @staticmethod
    def kde_plot(data, x, hue=None, title="", figsize=(10, 5)) -> plt.Figure: ...
    @staticmethod
    def heatmap(data, annot=True, title="", figsize=(10, 8)) -> plt.Figure: ...
    @staticmethod
    def pie_chart(data, title="", figsize=(8, 8)) -> plt.Figure: ...
    @staticmethod
    def wordcloud(texts, title="", figsize=(10, 6)) -> plt.Figure: ...
    @staticmethod
    def survival_curves(kmf, group=None, title="", figsize=(10, 6)) -> plt.Figure: ...
    @staticmethod
    def confusion_matrix(cm, labels, title="", figsize=(8, 6)) -> plt.Figure: ...
```

---

## 12. Streamlit Dashboard

### 12.1 App Structure

```python
# dashboard/app.py
def main():
    st.set_page_config(page_title="WorkForce Analytics v2.0", layout="wide")
    
    @st.cache_data
    def load_data():
        return pd.read_parquet("data/processed/workforce_clean_v3.parquet")
    
    df = load_data()
    
    # Sidebar: global filters + navigation
    with st.sidebar:
        st.image("dashboard/assets/logo.png", width=200)
        selected_job_families = st.multiselect("Job Family", df["JobFamily"].unique())
        selected_departments = st.multiselect("Department", df["DepartmentType"].unique())
        selected_regions = st.multiselect("Region", df["Region"].unique())
        st.page_link("pages/01_overview.py", label="Executive Overview")
        st.page_link("pages/02_attrition.py", label="Attrition Explorer")
        st.page_link("pages/03_diversity.py", label="Diversity & Inclusion")
        st.page_link("pages/04_performance.py", label="Performance Analytics")
        st.page_link("pages/05_nlp_insights.py", label="NLP Insights")
        st.page_link("pages/06_org_network.py", label="Org Network")
        st.page_link("pages/07_workforce_planning.py", label="Workforce Planning")
```

### 12.2 Page Summaries

| Page | Key Elements |
|------|-------------|
| **Overview** | KPI cards (headcount, attrition, tenure, diversity), headcount trend, dept distribution |
| **Attrition** | Survival curves by group, attrition rate by dept/job/region, seasonal patterns, termination breakdown |
| **Diversity** | Gender/race dist, intersectional treemap, Simpson index by dept, representation ratios |
| **Performance** | Score distribution, rating by group, PIP analysis, performance vs tenure scatter |
| **NLP Insights** | Sentiment distribution + trend, topic explorer with examples, classifier metrics, key phrase word clouds |
| **Org Network** | Interactive org chart, span of control histogram, centrality ranking table |
| **Workforce Planning** | Headcount forecast chart, scenario sliders (growth rate, attrition rate), projection table |

---

## 13. Testing Strategy

### 13.1 Test Organization

```
tests/
├── conftest.py           # sample_df(), config(), temp_dir() fixtures
├── test_data/
│   ├── test_loader.py    # CSV load, caching, error handling
│   ├── test_cleaner.py   # PII removal, date parsing, dedup
│   └── test_validator.py # Schema pass/fail, report generation
├── test_features/
│   ├── test_demographic.py  # Age calc, AgeGroup, TenureDays
│   └── test_categorical.py  # All mapping functions
├── test_nlp/
│   ├── test_preprocessor.py # Cleaning pipeline
│   ├── test_sentiment.py    # VADER results
│   └── test_topic_model.py  # LDA fit/transform
├── test_imputation/
│   ├── test_statistical.py  # Median imputation
│   └── test_predictive.py   # RF/GBM imputation + metrics
├── test_analysis/
│   ├── test_attrition.py    # Rate calc, survival
│   └── test_diversity.py    # Simpson index, chi-square
└── test_pipeline.py         # Full DAG execution
```

### 13.2 Key Test Cases

```python
# test_features/test_categorical.py
def test_job_family_director():
    assert map_job_family("Director of Sales") == "Executive & Leadership"

def test_job_family_manager():
    assert map_job_family("Production Manager") == "Production"

def test_job_family_analyst():
    assert map_job_family("Data Analyst") == "Data & Analytics"

def test_seniority_entry():
    assert map_seniority_level("Production Technician I") == "Entry"

def test_seniority_senior():
    assert map_seniority_level("Senior BI Developer") == "Senior"

def test_region_ca():
    assert map_region("CA") == "West"

def test_region_tx():
    assert map_region("TX") == "South"

# test_imputation/test_predictive.py
def test_imputation_reduces_missing(sample_df):
    imputer = PredictiveImputer()
    imputer.fit(sample_df, "Age", ["JobFamily", "SeniorityLevel"])
    result = imputer.impute(sample_df)
    assert result["Age"].isna().sum() < sample_df["Age"].isna().sum()

def test_r2_above_threshold(sample_df):
    imputer = PredictiveImputer()
    imputer.fit(sample_df, "Age", ["JobFamily", "SeniorityLevel", "Region"])
    assert imputer.metrics["r2"] > 0.3

# test_nlp/test_sentiment.py
def test_positive_sentiment():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(pd.Series(["Great opportunity for growth"]))
    assert result["sentiment_label"].iloc[0] == "Positive"

def test_negative_sentiment():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(pd.Series(["Terrible experience, unfair treatment"]))
    assert result["sentiment_label"].iloc[0] == "Negative"

# test_pipeline.py
def test_full_pipeline_runs(sample_df, config):
    pipeline = Pipeline()
    # Add stages...
    results = pipeline.run(config, sample_df)
    assert "imputed" in results
    assert results["imputed"]["Age"].isna().sum() == 0

def test_dag_validation():
    pipeline = Pipeline()
    pipeline.add_stage("a", lambda x: x)
    pipeline.add_stage("b", lambda x: x, ["nonexistent"])
    with pytest.raises(ValueError, match="nonexistent"):
        pipeline._validate_dag()
```

---

## 14. Implementation Phases - Step by Step

### Phase 1: Foundation (Build First)

**Goal**: Package structure, config system, data I/O, pipeline orchestrator

**Steps**:
1. Create `pyproject.toml`, `setup.cfg`, `Makefile`, `.gitignore`, `.env.example`
2. Create `requirements.txt`, `requirements-dev.txt`
3. Create `src/wf_analysis/__init__.py` with `__version__ = "2.0.0"`
4. Create `config/config.yaml`, `config/schema.yaml`, `config/logging.yaml`, `config/categorical_mappings.yaml`, `config/nlp_config.yaml`
5. Implement `src/wf_analysis/config.py` with pydantic models
6. Implement `src/wf_analysis/utils/logging.py` (loguru setup)
7. Implement `src/wf_analysis/utils/helpers.py` (timer, memoize, chunk)
8. Implement `src/wf_analysis/utils/decorators.py` (@log_call, @timer, @cache_to_disk)
9. Implement `src/wf_analysis/data/loader.py` (DataLoader with caching)
10. Implement `src/wf_analysis/data/cleaner.py` (DataCleaner)
11. Implement `src/wf_analysis/data/validator.py` (DataValidator)
12. Implement `src/wf_analysis/data/exporter.py` (DataExporter)
13. Implement `src/wf_analysis/pipeline.py` (Pipeline with DAG)
14. Implement `scripts/run_pipeline.py` (CLI entry point)
15. Write tests: `test_data/test_loader.py`, `test_data/test_cleaner.py`, `test_data/test_validator.py`, `test_pipeline.py`

**Verification**: Run `pytest tests/test_data/` — all pass. Run `python scripts/run_pipeline.py --config config/config.yaml --stage load` — loads dataset successfully.

---

### Phase 2: Feature Engineering (Build Second)

**Goal**: All feature transformers operational

**Steps**:
1. Implement `src/wf_analysis/features/base.py` (FeatureTransformer ABC)
2. Implement `src/wf_analysis/features/demographic.py` (Age, AgeGroup, IsActive, TenureDays, TenureYears, TenureBucket, Generation)
3. Implement `src/wf_analysis/features/categorical.py` (JobFamily, SeniorityLevel, DivisionGroup, Region, JobFunctionGroup — YAML-driven)
4. Implement `src/wf_analysis/features/temporal.py` (JoinYear, BirthYear, JoinMonth, JoinQuarter, JoinSeason, ExitMonth)
5. Implement `src/wf_analysis/features/embeddings.py` (Sentence-BERT + PCA)
6. Add feature stages to pipeline
7. Write tests: `test_features/test_demographic.py`, `test_features/test_categorical.py`

**Verification**: `pytest tests/test_features/` — all pass. Pipeline produces enriched DataFrame with all derived features.

---

### Phase 3: Age/DOB Imputation (Build Third)

**Goal**: Zero missing values with validated distribution preservation

**Steps**:
1. Implement `src/wf_analysis/imputation/base.py` (ImputerStrategy ABC)
2. Implement `src/wf_analysis/imputation/statistical.py` (median/mode imputation)
3. Implement `src/wf_analysis/imputation/predictive.py` (RF + GBM)
4. Implement `src/wf_analysis/imputation/ensemble.py` (blend + jitter)
5. Implement `src/wf_analysis/imputation/validator.py` (KS test, Wasserstein, KDE overlap)
6. Add imputation stages to pipeline
7. Cross-validate all strategies and select best
8. Write tests: `test_imputation/test_statistical.py`, `test_imputation/test_predictive.py`

**Verification**: Zero missing values in Age/DOB. KDE plots show imputed distribution matches original. KS test p-value > 0.05.

---

### Phase 4: NLP Pipeline (Build Fourth)

**Goal**: Full NLP analysis of TerminationDescription, Title, JobFunctionDescription

**Steps**:
1. Implement `src/wf_analysis/nlp/preprocessor.py` (text cleaning)
2. Implement `src/wf_analysis/nlp/sentiment.py` (VADER)
3. Implement `src/wf_analysis/nlp/topic_model.py` (LDA)
4. Implement `src/wf_analysis/nlp/text_classifier.py` (TF-IDF + LogisticRegression)
5. Implement `src/wf_analysis/nlp/keywords.py` (RAKE)
6. Implement `src/wf_analysis/nlp/embeddings.py` (sentence-transformers)
7. Implement `src/wf_analysis/nlp/visualizer.py` (word clouds, topic bubbles, sentiment charts)
8. Add NLP stages to pipeline
9. Write tests: `test_nlp/test_preprocessor.py`, `test_nlp/test_sentiment.py`, `test_nlp/test_topic_model.py`

**Verification**: NLP produces sentiment scores (6 categories), topic model (6 topics), text classifier (75%+ accuracy), key phrases, and embeddings for all text columns.

---

### Phase 5: Analysis Modules (Build Fifth)

**Goal**: 7 analysis modules producing structured results + plots

**Steps**:
1. Implement `src/wf_analysis/analysis/base.py` (AnalysisModule ABC)
2. Implement `src/wf_analysis/analysis/attrition.py` (Kaplan-Meier with lifelines)
3. Implement `src/wf_analysis/analysis/diversity.py` (Simpson, intersectional, chi-square)
4. Implement `src/wf_analysis/analysis/performance.py` (score dist, PIP)
5. Implement `src/wf_analysis/analysis/compensation.py` (PayZone, Gini)
6. Implement `src/wf_analysis/analysis/network.py` (NetworkX graph)
7. Implement `src/wf_analysis/analysis/career_path.py` (role similarity)
8. Implement `src/wf_analysis/analysis/forecasting.py` (Prophet)
9. Add analysis stages to pipeline
10. Write tests: `test_analysis/test_attrition.py`, `test_analysis/test_diversity.py`

**Verification**: All 7 modules produce structured AnalysisResult objects. Plots are generated for each. Metrics are reasonable.

---

### Phase 6: Visualization & Reports (Build Sixth)

**Goal**: Consistent visual styling + auto-generated HTML reports

**Steps**:
1. Implement `src/wf_analysis/visualization/theme.py` (palettes, style)
2. Implement `src/wf_analysis/visualization/plots.py` (reusable chart functions)
3. Implement `src/wf_analysis/visualization/dashboards.py` (multi-plot layouts)
4. Implement `src/wf_analysis/visualization/reports.py` (HTML generator)
5. Generate complete set of figures for all analyses
6. Generate comprehensive HTML report

**Verification**: All figures generated in `reports/figures/`. HTML report opens in browser with executive summary, analysis sections, and interactive elements.

---

### Phase 7: Streamlit Dashboard (Build Seventh)

**Goal**: 7-page interactive dashboard for HR managers

**Steps**:
1. Create `dashboard/app.py` (main entry + sidebar)
2. Create `dashboard/components/kpi_card.py`, `filter_sidebar.py`, `chart_wrapper.py`
3. Create `dashboard/assets/style.css`, `logo.png` (placeholder)
4. Create `dashboard/utils.py` (data loading with caching)
5. Build `pages/01_overview.py` (KPI cards, trend charts)
6. Build `pages/02_attrition.py` (survival curves, attrition explorer)
7. Build `pages/03_diversity.py` (diversity dashboards)
8. Build `pages/04_performance.py` (performance analytics)
9. Build `pages/05_nlp_insights.py` (NLP analysis with tabs)
10. Build `pages/06_org_network.py` (org chart, span of control)
11. Build `pages/07_workforce_planning.py` (forecast, scenarios)
12. Create `scripts/run_dashboard.py` (streamlit launcher)

**Verification**: `streamlit run dashboard/app.py` launches with all pages functional. Global filters work across pages. Charts are interactive.

---

### Phase 8: Hardening (Build Last)

**Goal**: Production-ready with tests, Docker, CI/CD, documentation

**Steps**:
1. Write remaining tests to reach 80%+ coverage
2. Create `docker/Dockerfile` (Python 3.10-slim)
3. Create `docker/docker-compose.yml` (app + optional Jupyter)
4. Configure GitHub Actions: `lint → test → build`
5. Write `docs/user_guide.md` (HR manager focused)
6. Write `docs/developer_guide.md` (setup, contribution)
7. Create `data/sample/employee_data_sample.csv` (100 rows)
8. Final end-to-end integration test
9. Tag release v2.0.0

**Verification**: `docker-compose up` launches full app. `pytest --cov=wf_analysis` shows 80%+ coverage. GitHub Actions passes all checks.

---

## 15. Acceptance Criteria

### Must Have (v2.0 Release)

- [ ] **Data Pipeline**: Load raw CSV, validate schema, clean PII/dates, export in 3 formats
- [ ] **Feature Engineering**: All derived features from v1.0 replicated and improved
- [ ] **Age/DOB Imputation**: Zero missing values, ML-based (R² > 0.5), distribution validated (KS p > 0.05)
- [ ] **NLP - Sentiment**: VADER scores + labels + trend for TerminationDescription
- [ ] **NLP - Topics**: LDA identifies 5-6 exit reason themes with interpretation
- [ ] **NLP - Classifier**: Text→TerminationType classifier with accuracy > 70%
- [ ] **NLP - Keywords**: Extracted key phrases per termination type
- [ ] **Attrition Analysis**: Kaplan-Meier survival curves, attrition rates by group
- [ ] **Diversity Analysis**: Gender/race distribution, intersectional view, Simpson index, chi-square
- [ ] **Network Analysis**: Org graph, centrality, span of control
- [ ] **Streamlit Dashboard**: 7 pages with interactive filtering
- [ ] **Tests**: pytest with 80%+ coverage
- [ ] **Docker**: Containerized one-command startup

### Nice to Have (Post-v2.0)

- [ ] Career path similarity matrix
- [ ] Prophet workforce forecasting
- [ ] Performance analysis (PIP detection)
- [ ] Compensation/Gini analysis
- [ ] CI/CD pipeline
- [ ] HTML report generation

---

## Implementation Notes

### Coding Standards
- All code must pass `flake8` and `mypy --strict`
- Google-style docstrings on all public methods
- Type hints on all function signatures
- Logging via `loguru` (`logger.info()`, `logger.error()`) — never `print()`
- Configuration via YAML — never hardcoded paths

### Data Integrity Rules
- Raw data is **read-only** — never modify `data/raw/employee_data.csv`
- Each pipeline stage creates a **new** DataFrame (immutable pattern)
- Always validate after cleaning, after feature engineering, and after imputation
- Track data lineage: every output file has metadata about how it was created

### Error Handling
- All file operations wrapped in try/except with meaningful error messages
- Schema validation failures are logged and raise clear errors
- Missing columns produce warnings + graceful fallback
- Pipeline stages can be run individually for debugging

---

This plan is self-contained. Begin implementation with Phase 1 and proceed sequentially.
