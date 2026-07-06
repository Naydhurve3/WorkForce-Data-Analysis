# Reproducibility Guide

## Environment Setup

```bash
# Create conda environment
conda create -n workforce python=3.10
conda activate workforce

# Install from source
cd WorkForce-Data-Analysis
pip install -e .

# Optional: development dependencies
pip install -e ".[dev]"
```

## Dependencies

See `pyproject.toml` for full list. Key packages:
- pandas >= 2.0
- numpy >= 1.24
- scikit-learn >= 1.2
- xgboost >= 1.7
- shap >= 0.41
- lifelines >= 0.27
- matplotlib >= 3.7
- seaborn >= 0.12
- networkx >= 3.0
- jinja2 >= 3.1

## Reproduction Steps

### Full Pipeline (7 Phases + Report)

```bash
# Phase 1: Deep EDA
python scripts/phase1_deep_eda.py

# Phase 2: Feature Engineering
python scripts/phase2_feature_engineering.py

# Phase 3: Interaction Mining
python scripts/phase3_interaction_mining.py

# Phase 4: Predictive Models
python scripts/phase4_predictive_models.py

# Phase 5: Deep Dives
python scripts/phase5_deep_dives.py

# Phase 6: Dashboards
python scripts/phase6_dashboards.py

# Phase 7: HTML Report
python scripts/phase7_html_report.py
```

### Individual Notebooks

```bash
jupyter notebook notebooks/01_data_profile.ipynb
# ... through ...
jupyter notebook notebooks/10_integrated.ipynb
```

### Tests

```bash
python -m pytest tests/ --tb=short
```

## Expected Outputs

| Run | Output | Location |
|-----|--------|----------|
| Phase 1 | 12 figures + eda_summary.json | reports/figures/interaction/ + data/interaction/ |
| Phase 2 | 3 figures + feature_matrix.parquet | reports/figures/interaction/ + data/interaction/ |
| Phase 3 | 8 figures + interaction_results.parquet | reports/figures/interaction/ + data/interaction/ |
| Phase 4 | 13 figures + model_results.json | reports/figures/interaction/ + data/interaction/ |
| Phase 5 | 16 figures + deep_dives.json | reports/figures/interaction/ + data/interaction/ |
| Phase 6 | 8 figures | reports/figures/interaction/ |
| Phase 7 | report.html | reports/ |

## Data

Source: `data/raw/employee_data.csv` — 3,000 synthetic employee records.
This data is simulated and does not contain real employee information.

## Tested Platform

- OS: Windows 11 / Ubuntu 22.04
- Python: 3.10 - 3.13
- Tests verified: 58 passing (as of v2.0.0)
