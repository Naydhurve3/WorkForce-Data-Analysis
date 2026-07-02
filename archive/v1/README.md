# v1.0 — Dashboard-based Workforce Analytics (Legacy)

> **Navigate**: [← Back to v2.0 (current)](../../README.md) |
> [View on GitHub tag](https://github.com/Naydhurve3/WorkForce-Data-Analysis/tree/v1.0)

## Overview

v1.0 is the original version of this workforce analytics project. It consists of a
production-grade Python pipeline with ML-based imputation, NLP text analysis, and an
interactive [Streamlit](https://streamlit.io) dashboard for exploring workforce data.

The analysis covers **7 domains**: attrition, diversity, compensation equity,
performance trends, organizational network, career progression, and headcount
forecasting.

## Architecture

```
Raw Data (CSV)
    │
    ▼
┌─────────────────────┐
│  Data Pipeline       │
│  ├─ Load & Validate  │
│  ├─ Clean & Impute   │
│  ├─ Feature Engineer  │
│  └─ Export           │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Analysis Modules    │
│  ├─ Attrition        │
│  ├─ Diversity        │
│  ├─ Compensation     │
│  ├─ Performance      │
│  ├─ Org Network      │
│  ├─ Career Path      │
│  └─ Forecasting      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Streamlit Dashboard │
│  └─ 7 interactive    │
│     pages + filters  │
└─────────────────────┘
```

## Modules

| Module | Purpose |
|--------|---------|
| `data/` | CSV loading, schema validation, PII removal, date standardization, export |
| `features/` | Demographic, categorical, temporal, and embedding feature transformers |
| `nlp/` | Sentiment analysis, topic modeling (LDA), keyword extraction, text classification |
| `imputation/` | ML-based Age/DOB imputation with distribution validation |
| `analysis/` | 7 sub-modules: attrition, diversity, compensation, performance, network, career path, forecasting |
| `visualization/` | Consistent theme, reusable seaborn/matplotlib charts, HTML report generation |

## How to Run (v1.0)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python scripts/run_pipeline.py --config config/config.yaml

# Launch the dashboard
streamlit run dashboard/app.py
```

## Dataset

**Source**: `data/raw/employee_data.csv` — 3,000 employee records, 26 columns covering
demographics, employment history, performance ratings, and termination data.

**Sample**: An archive copy of the sample dataset is preserved at
[`data/sample/employee_data_sample.csv`](data/sample/employee_data_sample.csv).

## Archived Config Files

Old configuration files are preserved in [`config/`](config/) for reference:

| File | Purpose |
|------|---------|
| `schema.yaml` | Column schemas, data types, validation rules |
| `config.yaml` | Pipeline configuration (paths, model params, thresholds) |
| `categorical_mappings.yaml` | Encoding mappings for categorical variables |
| `logging.yaml` | Logging configuration |
| `nlp_config.yaml` | NLP model parameters (topic count, keyword limits) |

## Differences from v2.0

| Aspect | v1.0 (Legacy) | v2.0 (Current) |
|--------|---------------|-----------------|
| **Approach** | Dashboard-based exploration | Statistical interaction mining |
| **Output** | Streamlit dashboard + charts | 60 publication-ready figures + HTML report |
| **Modeling** | Basic attrition prediction | LR/RF/XGB with/without interactions, SHAP, Survival |
| **Key Technique** | Standard EDA | Interaction impact scoring (1,755 pair tests) |
| **Deep Dives** | None | Subgroup / Profile / What-if analysis |
| **Architecture** | 12 modules | 9 interaction-mining modules |
| **Formulas** | Implicit | Explicit (mutual info, chi-square, Cohen's d, etc.) |

---

**[← Back to v2.0 (current)](../../README.md)**
