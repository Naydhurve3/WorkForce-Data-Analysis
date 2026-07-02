# WorkForce Data Analysis v2.0 — Implementation Plan

> **Status**: Analysis-First Approach (Dashboard Last)  
> **Source**: `data/raw/employee_data.csv` (3,000 records, 26 columns)  
> **Phases**: 12 | **ML Progression**: Statistical → Classic → Advanced → Hybrid

---

## Table of Contents

1. [Philosophy & Approach](#1-philosophy--approach)
2. [10 Analysis Topics Overview](#2-10-analysis-topics-overview)
3. [Dataset Derivation Strategy](#3-dataset-derivation-strategy)
4. [ML Model Progression](#4-ml-model-progression)
5. [Phase-by-Phase Execution](#5-phase-by-phase-execution)
6. [Data Dictionary](#6-data-dictionary)
7. [Acceptance Criteria](#7-acceptance-criteria)

---

## 1. Philosophy & Approach

### Why Analysis-First?

Instead of building a monolithic pipeline and dashboard upfront, we:

1. **Start with raw data** — Load `employee_data.csv` as-is
2. **Derive 10 specialized datasets** — Each analysis gets its own tailored dataset optimized for its questions
3. **Apply progressive ML** — Each analysis uses a ladder of models (statistical → classic → advanced → hybrid)
4. **Synthesize findings** — After all 10 analyses, cross-reference insights
5. **Build dashboard last** — Use insights to design an informed, impactful dashboard

### Key Principle: HR Value + Employee Value

Every analysis must answer:
- **What does this mean for HR leaders?** (retention, cost, compliance, planning)
- **What does this mean for employees?** (fairness, growth, transparency, wellbeing)

---

## 2. 10 Analysis Topics Overview

| # | Analysis | Dataset Focus | Target Audience | Primary Techniques |
|---|----------|---------------|-----------------|-------------------|
| 01 | **Data Profile & Quality** | Full raw dataset | Data team | Profiling, missing patterns, distributions, outlier detection |
| 02 | **Attrition Risk & Prediction** | Active + terminated employees | HR leaders | Logistic Regression → RF → XGBoost → **Stacked Ensemble** + SHAP |
| 03 | **Compensation Equity** | Active employees | HR + Finance | Linear Regression → **Isolation Forest (anomaly)** → Gender/Race pay gap modeling |
| 04 | **Performance Drivers** | Rated active employees | Talent management | Decision Tree → **Gradient Boosting** → Feature importance + Partial Dependence |
| 05 | **Career Path & Mobility** | All employees | Employee development | Markov transitions → **K-Means clustering (archetypes)** → Tenure/role modeling |
| 06 | **Diversity & Inclusion** | Full dataset | DEI + Compliance | Chi-square → **Simpson Index** → Representation ratios by dept/level |
| 07 | **Org Network & Span** | Employees with supervisor | Org design | NetworkX metrics → **Betweenness centrality** → Span-of-control analysis |
| 08 | **Workforce Forecasting** | Active employees | Strategic planning | **Survival Analysis (Kaplan-Meier)** → ARIMA → Retirement risk scoring |
| 09 | **Exit & Engagement (NLP)** | Terminated employees | Employee experience | LDA Topic Model → **Sentiment Analysis** → Termination pattern clustering |
| 10 | **Integrated Strategy** | All analysis outputs | Leadership | Cross-analysis correlation → **Insight scoring** → Actionable recommendations |

---

## 3. Dataset Derivation Strategy

Each analysis derives a focused dataset from the raw source. All derived datasets are saved to `data/analysis/{nn}_{name}/` for reproducibility.

### Derivation Details

| # | Dataset Name | Filter | Added Columns | Saved Path |
|---|--------------|--------|---------------|------------|
| 01 | `raw_profile` | None | `missing_pct`, `col_type` (metadata only) | `data/analysis/01_data_profile/` |
| 02 | `attrition_model` | All employees | `is_terminated`, `tenure_days`, `tenure_years`, `age_group`, `job_family` | `data/analysis/02_attrition/` |
| 03 | `compensation_equity` | Active employees | `pay_equity_flag`, `comp_to_market`, `gender_salary_diff` | `data/analysis/03_compensation/` |
| 04 | `performance_drivers` | Active with rating | `perf_score_encoded`, `high_performer_flag` | `data/analysis/04_performance/` |
| 05 | `career_path` | All employees | `job_transitions`, `time_in_role`, `career_archetype` | `data/analysis/05_career/` |
| 06 | `diversity_metrics` | Full dataset | `gender_code`, `race_ethnicity`, `dept_gender_ratio` | `data/analysis/06_diversity/` |
| 07 | `org_network` | With supervisor | `span_of_control`, `org_level`, `centrality_score` | `data/analysis/07_network/` |
| 08 | `workforce_forecast` | Active employees | `retirement_risk_score`, `tenure_bucket`, `exit_probability` | `data/analysis/08_forecast/` |
| 09 | `exit_nlp` | Terminated only | `cleaned_description`, `sentiment_label`, `topic_id`, `keyword_phrases` | `data/analysis/09_exit_nlp/` |
| 10 | `integrated` | Merged from all 9 | `cross_analysis_flags`, `risk_composite_score` | `data/analysis/10_integrated/` |

### Code Pattern for Dataset Derivation

```python
# Each analysis notebook follows this pattern
import pandas as pd
import numpy as np

# 1. Load raw
raw = pd.read_csv("data/raw/employee_data.csv")

# 2. Derive dataset (example: attrition)
df = raw.copy()
df["is_terminated"] = df["EmployeeStatus"].str.lower().isin(
    ["terminated", "voluntary termination", "involuntary termination"]
).astype(int)
df["tenure_days"] = ...  # derived from StartDate / ExitDate

# 3. Save derived dataset
os.makedirs("data/analysis/02_attrition/", exist_ok=True)
df.to_parquet("data/analysis/02_attrition/dataset.parquet")

# 4. ML models executed within notebook
# 5. Results saved as JSON + visualizations
```

---

## 4. ML Model Progression

Each analysis uses a **progressive ML ladder** — starting simple, then increasing complexity:

### The ML Ladder (by analysis)

| Analysis | Level 1: Statistical | Level 2: Classic | Level 3: Advanced | Level 4: Hybrid |
|----------|---------------------|------------------|-------------------|-----------------|
| **02 — Attrition** | Descriptive rates, crosstabs | Logistic Regression (baseline accuracy) | Random Forest → XGBoost (feature importance) | **Stacked Ensemble** (RF + GBM + LR) + SHAP |
| **03 — Compensation** | Mean/median by group, Gini | Linear Regression (pay predictors) | **Isolation Forest** (anomaly detection) | Pay equity model + statistical testing |
| **04 — Performance** | Score distribution, correlations | Decision Tree (interpretable rules) | **Gradient Boosting** (partial dependence plots) | Feature importance + SHAP values |
| **05 — Career** | Tenure distributions, transition counts | **Markov Chain** (transition probabilities) | **K-Means** clustering (career archetypes) | Cluster profiling + mobility scoring |
| **06 — Diversity** | Proportions, percentages | **Chi-square** independence tests | Simpson Index, representation ratios | Intersectional analysis heatmap |
| **07 — Network** | Degree counts, path lengths | NetworkX centrality metrics | **Betweenness + Closeness** centrality | Span-of-control optimization analysis |
| **08 — Forecasting** | Headcount trends, moving averages | **Kaplan-Meier** survival curves | **ARIMA** time series | Retirement risk scoring model |
| **09 — Exit NLP** | Word frequencies, term counts | **LDA** topic modeling | **VADER** sentiment analysis | Topic + Sentiment combined clustering |
| **10 — Integrated** | Correlation matrix | Cross-analysis flag aggregation | **Composite risk scoring** | Strategic recommendation engine |

### Model Evaluation Standards

| Model Type | Metric | Target |
|------------|--------|--------|
| Classification | Accuracy, Precision, Recall, F1, ROC-AUC | > 0.70 baseline |
| Regression | R², MAE, RMSE | Context-dependent |
| Clustering | Silhouette Score, Inertia | > 0.3 |
| Survival | Log-rank test p-value, Median survival time | p < 0.05 |
| NLP | Topic coherence, Sentiment accuracy | Human-interpretable |

---

## 5. Phase-by-Phase Execution

### Phase 0: Environment & Structure Setup

**Goal**: Create project scaffolding for analysis-first workflow

**Steps**:
1. Create directory structure: `data/analysis/{01..10}_*/`
2. Create `notebooks/` with template structure
3. Create `scripts/derive_datasets.py` — generates all 10 derived datasets from raw
4. Update `.gitignore` for new analysis data directories
5. Verify raw data loads correctly

**Output**: Ready-to-use analysis environment

---

### Phase 1: Data Profile & Quality (Notebook 01)

**Goal**: Understand raw data thoroughly before any analysis

**Derived Dataset**: Metadata tables (no transformation of raw data)

**Analysis**:
- Basic profiling: shape, dtypes, missing %, unique values
- Distribution plots for every column (histograms, bar charts)
- Missing value patterns (UpSet plot, heatmap)
- Outlier detection (IQR, Z-score)
- Date parsing success rates (DOB, StartDate, ExitDate)
- Correlation heatmap of numeric columns

**ML**: Minimal — statistical profiling only

**Deliverables**:
- `reports/figures/01_*_profile.png` — distribution plots
- `data/analysis/01_data_profile/profile_report.json` — quality metrics
- Insights summary: key data quality issues to address

**HR Value**: Data trust and transparency  
**Employee Value**: Confidence in data-driven decisions

---

### Phase 2: Attrition Risk & Prediction (Notebook 02)

**Goal**: Predict which employees are at risk of leaving and why

**Derived Dataset**: `data/analysis/02_attrition/dataset.parquet` (3000 rows, ~20 cols)

**Columns added**: `is_terminated`, `tenure_days`, `tenure_years`, `age_group`, `job_family`, `seniority_level`, `region`, `division_group`, `exit_month`, `exit_quarter`

**ML Ladder**:

| Step | Model | Purpose | Output |
|------|-------|---------|--------|
| 1 | Descriptive statistics | Understand base attrition rates | Rate: ~48.9% |
| 2 | Crosstab analysis | Attrition by department, gender, tenure | Group rates |
| 3 | **Logistic Regression** | Baseline prediction, interpretable coefficients | OR, p-values, accuracy |
| 4 | **Random Forest** | Non-linear patterns, feature importance | Top 10 features |
| 5 | **XGBoost** | Gradient boosting for better accuracy | ROC-AUC, confusion matrix |
| 6 | **Stacked Ensemble** | RF + XGBoost + LR combined via LogisticRegression meta-model | Best accuracy |
| 7 | **SHAP** | Model interpretation | SHAP summary plot, dependence plots |

**HR Value**: Early warning system for retention intervention  
**Employee Value**: Understanding which factors affect job stability

---

### Phase 3: Compensation Equity (Notebook 03)

**Goal**: Analyze pay fairness across gender, race, departments

**Derived Dataset**: `data/analysis/03_compensation/dataset.parquet` (active employees only, ~1500 rows)

**Columns added**: `comp_score` , `gender_code`, `race_desc`, `pay_zone_encoded`, `department_type`

**ML Ladder**:

| Step | Model | Purpose | Output |
|------|-------|---------|--------|
| 1 | Summary statistics | Mean/median salary by gender, race, department | Pay gaps table |
| 2 | Gini coefficient | Overall pay inequality | Gini index |
| 3 | **Linear Regression** | Salary ~ Gender + Race + Dept + Tenure (controlling for factors) | Coefficients, p-values |
| 4 | **Isolation Forest** | Anomaly detection — identify outliers in compensation | Anomaly scores |
| 5 | **Statistical tests** | T-test / Mann-Whitney for gender pay gap within roles | Significance |
| 6 | **Visualization** | Salary distribution by group, box plots, residual analysis | Charts |

**HR Value**: Pay equity compliance, budget fairness  
**Employee Value**: Transparency on fair compensation

---

### Phase 4: Performance Drivers (Notebook 04)

**Goal**: Identify what drives high performance and predict performance outcomes

**Derived Dataset**: `data/analysis/04_performance/dataset.parquet` (employees with ratings)

**Columns added**: `perf_encoded`, `high_performer`, `tenure_years`, `age_group`, `job_family`

**ML Ladder**:

| Step | Model | Purpose | Output |
|------|-------|---------|--------|
| 1 | Distribution analysis | Performance score distribution | Bar chart, percentages |
| 2 | Correlation analysis | Rating vs Tenure, Age, Salary | Correlation matrix |
| 3 | **Decision Tree** | Interpretable rules for high performance | Tree visualization |
| 4 | **Gradient Boosting** | Accurate prediction of performance | Feature importance |
| 5 | **Partial Dependence Plots** | How features affect predicted performance | PDP charts |
| 6 | **SHAP** | Explain individual predictions | Force plots |

**HR Value**: Talent development strategy, promotion criteria  
**Employee Value**: Clear growth path and success factors

---

### Phase 5: Career Path & Mobility (Notebook 05)

**Goal**: Map career trajectories and identify mobility patterns

**Derived Dataset**: `data/analysis/05_career/dataset.parquet` (all employees)

**Columns added**: `job_family`, `seniority_level`, `tenure_years`, `title_cluster`

**ML Ladder**:

| Step | Model | Purpose | Output |
|------|-------|---------|--------|
| 1 | Job function distribution | Current role landscape | Function counts |
| 2 | Tenure analysis by role | Average time in role | Tenure table |
| 3 | **Markov Chain** | Transition probabilities between job families | Transition matrix |
| 4 | **K-Means Clustering** | Identify career archetypes | Cluster profiles |
| 5 | **Elbow + Silhouette** | Optimal number of clusters | Cluster metrics |

**HR Value**: Succession planning, internal mobility programs  
**Employee Value**: Visible career roadmap and growth opportunities

---

### Phase 6: Diversity & Inclusion (Notebook 06)

**Goal**: Measure and visualize DEI metrics across the organization

**Derived Dataset**: `data/analysis/06_diversity/dataset.parquet` (full dataset)

**Columns added**: `gender_code`, `race_desc`, `marital_desc`, `department_type`

**ML Ladder**:

| Step | Model | Purpose | Output |
|------|-------|---------|--------|
| 1 | Demographic proportions | Gender, race, marital status overall | Pie/bar charts |
| 2 | **Chi-square test** | Gender × Department independence | p-value, Cramer's V |
| 3 | **Simpson Diversity Index** | Diversity score per department | Index values |
| 4 | Representation ratios | Group proportion vs overall | Ratio chart |
| 5 | **Intersectional analysis** | Gender × Race × Department | Treemap, heatmap |

**HR Value**: DEI compliance, targeted hiring strategies  
**Employee Value**: Inclusive workplace awareness

---

### Phase 7: Organizational Network & Span of Control (Notebook 07)

**Goal**: Analyze reporting structure, manager effectiveness, org design

**Derived Dataset**: `data/analysis/07_network/dataset.parquet` (employees with supervisor)

**Columns added**: `span_of_control`, `org_level`, `centrality_score`, `org_depth`

**ML Ladder**:

| Step | Model | Purpose | Output |
|------|-------|---------|--------|
| 1 | Network construction | Build DiGraph from Supervisor column | G (nodes, edges) |
| 2 | **Betweenness Centrality** | Key influencers in the org | Top 10 ranking |
| 3 | **Closeness Centrality** | How quickly info flows | Centrality table |
| 4 | Span of control analysis | Direct reports per manager | Histogram, stats |
| 5 | Organizational depth | Levels from top to bottom | Depth distribution |

**Notes**: With 2,952 unique supervisors for 3,000 employees, the graph is sparse — centrality metrics will highlight structural positions rather than social influence.

**HR Value**: Org design optimization, manager effectiveness  
**Employee Value**: Understanding team structure and reporting

---

### Phase 8: Workforce Forecasting (Notebook 08)

**Goal**: Forecast headcount needs, identify retirement risk, plan for future

**Derived Dataset**: `data/analysis/08_forecast/dataset.parquet` (active employees)

**Columns added**: `tenure_years`, `age_group`, `retirement_risk`, `exit_risk_score`

**ML Ladder**:

| Step | Model | Purpose | Output |
|------|-------|---------|--------|
| 1 | Headcount trends | Monthly/quarterly headcount over time | Trend line |
| 2 | Age distribution | Current age pyramid | Histogram |
| 3 | **Kaplan-Meier Survival** | Employee retention curves | Survival curves |
| 4 | **Retirement risk scoring** | Age ≥ 55 + tenure > 20 | Risk categories |
| 5 | **ARIMA** | Headcount forecasting for next 12 months | Forecast + CI |

**HR Value**: Strategic workforce planning, hiring budget  
**Employee Value**: Job security context, retirement planning

---

### Phase 9: Exit Analysis & NLP (Notebook 09)

**Goal**: Understand why employees leave through termination text analysis

**Derived Dataset**: `data/analysis/09_exit_nlp/dataset.parquet` (terminated employees only, ~1500 rows)

**Columns added**: `cleaned_desc`, `sentiment_score`, `sentiment_label`, `topic_id`, `keyword_phrases`

**ML Ladder**:

| Step | Model | Purpose | Output |
|------|-------|---------|--------|
| 1 | Text preprocessing | Clean termination descriptions | Cleaned text |
| 2 | **LDA Topic Modeling** | Discover 5 termination themes | Topics + top words |
| 3 | **VADER Sentiment** | Sentiment of termination descriptions | Score distribution |
| 4 | **Keyword Extraction (RAKE)** | Key phrases per topic | Keywords table |
| 5 | **Topic + Sentiment combined** | Cluster exits by topic + sentiment | Cluster profiles |

**HR Value**: Deeper exit interview insights, retention program targeting  
**Employee Value**: Voice is heard and analyzed for change

> **Note**: The TerminationDescription is synthetic random text — topic model results will show structural patterns rather than meaningful themes. Real data would produce actionable insights.

---

### Phase 10: Integrated Strategy & Recommendations (Notebook 10)

**Goal**: Combine all 9 analysis outputs into a coherent strategic picture

**Derived Dataset**: `data/analysis/10_integrated/dataset.parquet` (merged from all 9)

**ML Ladder**:

| Step | Model | Purpose | Output |
|------|-------|---------|--------|
| 1 | Cross-analysis correlation | How attrition relates to diversity, performance, comp | Correlation matrix |
| 2 | **Composite risk scoring** | Combine attrition risk, retirement risk, performance | Risk heatmap |
| 3 | **Strategic quadrant** | Attrition rate × Diversity score by department | Quadrant chart |
| 4 | **Insight scoring** | Rate each finding by impact + feasibility | Priority matrix |
| 5 | **Recommendation engine** | Actionable items from all findings | Prioritized list |

**HR Value**: Strategic roadmap with prioritized actions  
**Employee Value**: Clear company direction and improvement plans

---

## 6. Data Dictionary

### raw/employee_data.csv

| Column | Type | Description | Non-Null | Unique |
|--------|------|-------------|----------|--------|
| EmpID | int64 | Unique employee ID | 3000 | 3000 |
| FirstName | object | First name (PII) | 3000 | 2985 |
| LastName | object | Last name (PII) | 3000 | 2992 |
| StartDate | object | Employment start date | 3000 | 2519 |
| ExitDate | object | Termination/exit date | 1533 | 1467 |
| Title | object | Job title | 3000 | 32 |
| Supervisor | object | Manager name | 3000 | 2952 |
| ADEmail | object | Email address (PII) | 3000 | 3000 |
| BusinessUnit | object | Business unit code | 3000 | 10 |
| EmployeeStatus | object | Active/Terminated/LOA | 3000 | 3 |
| EmployeeType | object | Full-time/Part-time/Contract | 3000 | 2 |
| PayZone | object | Compensation zone (A/B/C/D) | 3000 | 4 |
| EmployeeClassificationType | object | Classification | 3000 | 2 |
| TerminationType | object | Type of termination | 3000 | 5 |
| TerminationDescription | object | Termination reason text | 1533 | 1467 |
| DepartmentType | object | Department name | 3000 | 5 |
| Division | object | Division name | 3000 | 25 |
| DOB | object | Date of birth (60.8% unparseable) | 3000* | 2997 |
| State | object | US state code | 3000 | 28 |
| JobFunctionDescription | object | Job function | 3000 | 83 |
| GenderCode | object | Gender | 3000 | 2 |
| LocationCode | int64 | Location ID | 3000 | 4 |
| RaceDesc | object | Race/Ethnicity | 3000 | 5 |
| MaritalDesc | object | Marital status | 3000 | 4 |
| Performance Score | object | Performance rating | 3000 | 4 |
| Current Employee Rating | int64 | Numeric rating 1-5 | 3000 | 5 |

*\*DOB has 1,823 unparseable dates (60.8%)*

### Derived Columns (across all analysis datasets)

| Column | Source | Type | Used In |
|--------|--------|------|---------|
| `is_terminated` | EmployeeStatus | binary | 02, 05, 09 |
| `tenure_days` | StartDate, ExitDate | int | 02, 04, 05, 08 |
| `tenure_years` | tenure_days / 365.25 | float | 02, 04, 05, 08 |
| `age_group` | Age / DOB | categorical | 02, 04, 06, 08 |
| `job_family` | Keyword-mapped Title | categorical | 02, 03, 04, 05, 06 |
| `seniority_level` | Keyword-mapped Title | categorical | 02, 03, 05, 07 |
| `region` | State → Region map | categorical | 02, 06 |
| `division_group` | Division → Group map | categorical | 02, 06 |
| `span_of_control` | Supervisor → direct reports | int | 07 |
| `org_level` | Graph depth from root | int | 07 |
| `retirement_risk` | Age ≥ 55 | binary | 08 |
| `sentiment_label` | VADER on TerminationDescription | categorical | 09 |
| `topic_id` | LDA topic assignment | int | 09 |

---

## 7. Acceptance Criteria

### Must Pass (All Phases)

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | All 10 notebooks execute without errors | `jupyter nbconvert --to notebook --execute` |
| 2 | Each notebook produces its derived dataset | File exists in `data/analysis/{nn}_*/` |
| 3 | ML models produce > 0.60 baseline accuracy | Printed in notebook output |
| 4 | Visualizations generated for each analysis | PNG saved in `reports/figures/` |
| 5 | Cross-analysis synthesis notebook references all 9 prior | Import checks |
| 6 | All derived datasets have < 5% missing critical columns | Validation script |

### Stretch Goals

| # | Criterion | Target |
|---|-----------|--------|
| 1 | Stacked ensemble beats best single model | +3-5% accuracy |
| 2 | SHAP interpretation for at least 3 ML models | Attrition, Performance, Comp |
| 3 | Survival analysis produces statistically significant curves | Log-rank p < 0.05 |
| 4 | NLP topics are human-interpretable | > 3 coherent topics |
| 5 | Dashboard uses ALL analysis insights | 10 data sources |

---

## Appendix: Notebook Template

Each notebook follows this exact structure:

```python
# %% [markdown]
# # {NN} - {Analysis Name}
# Goal: ...
# 
# Derived Dataset: data/analysis/{NN}_{name}/dataset.parquet
# ML Progression: ... → ... → ...
# HR Value: ...
# Employee Value: ...

# %% [markdown]
# ## 1. Load Raw & Derive Dataset

# %% code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

raw = pd.read_csv("data/raw/employee_data.csv")
# ... derive dataset ...
df.to_parquet(f"data/analysis/{NN}_{name}/dataset.parquet")

# %% [markdown]
# ## 2. Exploratory Data Analysis
# - Distributions, correlations, patterns

# %% code
# ... plots ...

# %% [markdown]
# ## 3. ML Model 1: Statistical Baseline

# %% code
# ... model + results ...

# %% [markdown]
# ## 4. ML Model 2: Classic ML

# %% code
# ... model + results ...

# %% [markdown]
# ## 5. ML Model 3: Advanced ML

# %% code
# ... model + results ...

# %% [markdown]
# ## 6. ML Model 4: Hybrid / Ensemble (if applicable)

# %% code
# ... model + results ...

# %% [markdown]
# ## 7. Interpretation & Insights

# %% code
# ... SHAP, feature importance, conclusions ...

# %% [markdown]
# ## 8. Key Takeaways
# - HR Action Items
# - Employee Impact
# - Data Quality Notes
```

---

## Summary Execution Order

```
Phase 0: Environment & structure
Phase 1: 01_data_profile.ipynb        (Statistical profiling)
Phase 2: 02_attrition.ipynb           (Stats → LR → RF → XGBoost → Ensemble + SHAP)
Phase 3: 03_compensation.ipynb        (Stats → LinReg → Isolation Forest)
Phase 4: 04_performance.ipynb         (Stats → DecisionTree → GradientBoost + SHAP)
Phase 5: 05_career_path.ipynb         (Stats → Markov → K-Means)
Phase 6: 06_diversity.ipynb           (Stats → Chi-square → Simpson Index)
Phase 7: 07_org_network.ipynb         (Stats → NetworkX centrality)
Phase 8: 08_forecasting.ipynb         (Stats → Kaplan-Meier → ARIMA)
Phase 9: 09_exit_nlp.ipynb            (Stats → LDA → VADER → Cluster)
Phase 10: 10_integrated.ipynb        (All 9 combined → Strategy → Dashboard blueprint)
```

After all 10 phases: Rebuild the Streamlit dashboard informed by analysis insights.
