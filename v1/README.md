# v1.0 — Legacy Workforce Analytics (Notebooks + Power BI + Streamlit)

> **Navigate**: [← Back to v2.0 (current)](../README.md) |
> [View git tag](https://github.com/Naydhurve3/WorkForce-Data-Analysis/tree/v1.0)

---

## Overview

This directory contains the complete v1.0 legacy project — **48 files (20+ MB)** spanning
two phases of the original analysis:

| Phase | Tools | Location |
|-------|-------|----------|
| **Notebook Analysis** | Jupyter, Python, Pandas, Matplotlib | [`notebooks/`](notebooks/) |
| **Power BI Dashboard** | Power BI Desktop | [`dashboard/PowerBi.pbix`](dashboard/PowerBi.pbix) |
| **Streamlit Dashboard** (git tag) | Streamlit, Python modules | `v1.0` git tag |

---

## Repository Structure

```
v1/
│
├── notebooks/                          # Jupyter pipeline (8 notebooks)
│   ├── 00-complete-pipeline.ipynb      # End-to-end pipeline
│   ├── 01-data-loading-cleaning.ipynb  # Data ingestion & cleaning
│   ├── 02-imputation.ipynb             # Age/DOB imputation
│   ├── 03-verification.ipynb           # Data quality verification
│   ├── 04-eda-univariate.ipynb         # Univariate EDA
│   ├── 05-eda-bivariate.ipynb          # Bivariate EDA
│   ├── 06-time-analysis.ipynb          # Time series analysis
│   ├── 07-segmentation.ipynb           # Employee segmentation
│   └── analysis/                       # 15 deep-dive notebooks
│       ├── 01-attrition-retention.ipynb
│       ├── 02-diversity-inclusion.ipynb
│       ├── 03-compensation-equity.ipynb
│       ├── 04-performance-management.ipynb
│       ├── 05-career-progression.ipynb
│       ├── 06-hipo-identification.ipynb
│       ├── 07-workforce-planning.ipynb
│       ├── 08-training-roi.ipynb
│       ├── 09-recruitment-analysis.ipynb
│       ├── 10-departmental-health.ipynb
│       ├── 11-employee-engagement.ipynb
│       ├── 12-workforce-tenure.ipynb
│       ├── 13-workforce-composition.ipynb
│       ├── 14-location-distribution.ipynb
│       └── 15-org-structure.ipynb
│
├── dashboard/
│   └── PowerBi.pbix                    # Power BI interactive dashboard
│
├── config/                             # Config files (from git tag)
│   ├── schema.yaml
│   ├── config.yaml
│   ├── categorical_mappings.yaml
│   ├── logging.yaml
│   └── nlp_config.yaml
│
├── data/
│   ├── processed/                      # Cleaned datasets
│   │   ├── cleaned1.csv – cleaned3.csv
│   │   ├── cleaned_converted_employee_data.csv
│   │   └── cleaned_mapped_employee_data.csv
│   ├── outputs/                        # Analysis output CSVs
│   │   ├── High_Potential_Employees.csv
│   │   ├── Region_Workforce_Summary.csv
│   │   ├── Supervisor_Mapping.csv
│   │   └── dataset_cell_level_mismatches.csv
│   └── sample/
│       └── employee_data_sample.csv
│
├── reports/
│   ├── Report.pdf                      # Comprehensive PDF report
│   └── figures/                        # 10 report figures
│       ├── workforce_headcount_dashboard.png
│       ├── diversity_inclusion_dashboard.png
│       ├── payzone_analysis_*.png
│       └── ...
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## How to Use

### Jupyter Notebooks

```bash
pip install -r requirements.txt
jupyter notebook notebooks/00-complete-pipeline.ipynb
```

### Power BI Dashboard

Open [`dashboard/PowerBi.pbix`](dashboard/PowerBi.pbix) with Power BI Desktop
(requires data source connection to `data/processed/`).

### Streamlit Dashboard (Git Tag)

The Streamlit dashboard version is preserved in the git history:

```bash
git checkout v1.0
pip install -r requirements.txt
python scripts/run_pipeline.py --config config/config.yaml
streamlit run dashboard/app.py
```

---

## Dataset

**Source**: `data/raw/employee_data.csv` — 3,000 employee records, 26 columns.
(The raw file is at the repo root: [`data/raw/employee_data.csv`](../../data/raw/employee_data.csv))

---

## Key Analyses (15 Topics)

| # | Analysis | Notebook |
|---|----------|----------|
| 1 | Attrition & Retention | [`notebooks/analysis/01-attrition-retention.ipynb`](notebooks/analysis/01-attrition-retention.ipynb) |
| 2 | Diversity & Inclusion | [`notebooks/analysis/02-diversity-inclusion.ipynb`](notebooks/analysis/02-diversity-inclusion.ipynb) |
| 3 | Compensation Equity | [`notebooks/analysis/03-compensation-equity.ipynb`](notebooks/analysis/03-compensation-equity.ipynb) |
| 4 | Performance Management | [`notebooks/analysis/04-performance-management.ipynb`](notebooks/analysis/04-performance-management.ipynb) |
| 5 | Career Progression | [`notebooks/analysis/05-career-progression.ipynb`](notebooks/analysis/05-career-progression.ipynb) |
| 6 | HiPo Identification | [`notebooks/analysis/06-hipo-identification.ipynb`](notebooks/analysis/06-hipo-identification.ipynb) |
| 7 | Workforce Planning | [`notebooks/analysis/07-workforce-planning.ipynb`](notebooks/analysis/07-workforce-planning.ipynb) |
| 8 | Training ROI | [`notebooks/analysis/08-training-roi.ipynb`](notebooks/analysis/08-training-roi.ipynb) |
| 9 | Recruitment Analysis | [`notebooks/analysis/09-recruitment-analysis.ipynb`](notebooks/analysis/09-recruitment-analysis.ipynb) |
| 10 | Departmental Health | [`notebooks/analysis/10-departmental-health.ipynb`](notebooks/analysis/10-departmental-health.ipynb) |
| 11 | Employee Engagement | [`notebooks/analysis/11-employee-engagement.ipynb`](notebooks/analysis/11-employee-engagement.ipynb) |
| 12 | Workforce Tenure | [`notebooks/analysis/12-workforce-tenure.ipynb`](notebooks/analysis/12-workforce-tenure.ipynb) |
| 13 | Workforce Composition | [`notebooks/analysis/13-workforce-composition.ipynb`](notebooks/analysis/13-workforce-composition.ipynb) |
| 14 | Location Distribution | [`notebooks/analysis/14-location-distribution.ipynb`](notebooks/analysis/14-location-distribution.ipynb) |
| 15 | Org Structure | [`notebooks/analysis/15-org-structure.ipynb`](notebooks/analysis/15-org-structure.ipynb) |

---

## Differences from v2.0

| Aspect | v1.0 (Legacy) | v2.0 (Current) |
|--------|---------------|-----------------|
| **Approach** | Notebook exploration + BI dashboards | Statistical interaction mining pipeline |
| **Output** | 15 topical notebooks + Power BI + PDF | 60 figures + HTML report |
| **Modeling** | Basic churn prediction | LR/RF/XGB + SHAP + Survival (KM + Cox PH) |
| **Interaction Effects** | Not measured | 1,755 pair-wise statistical tests |
| **Deep Dives** | 15 topical analyses | Subgroup / Profile / What-if methodology |
| **Automation** | Manual notebook execution | 7-phase scripted pipeline |
| **Code** | Notebooks + Python scripts | 9 interaction modules + 8 scripts |

---

**[← Back to v2.0 (current)](../../README.md)**
