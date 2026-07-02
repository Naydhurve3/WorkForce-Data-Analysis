# WorkForce Data Analysis v2.0 — Interaction Mining & Predictive Modeling

> **Navigate**: [v1.0 (Legacy Dashboard) →](archive/v1/README.md) |
> [GitHub tag: v1.0](https://github.com/Naydhurve3/WorkForce-Data-Analysis/tree/v1.0)

---

## What's New in v2.0

v2.0 replaces the dashboard-based exploration of v1.0 with a **statistical interaction
mining pipeline** that systematically discovers how employee attributes combine to drive
workforce outcomes. Key advances:

| Capability | v1.0 | v2.0 |
|------------|------|------|
| Interaction effects | Not measured | 1,755 pair-wise tests |
| Predictive models | Basic attrition | LR / RF / XGB with/without interactions |
| Model interpretability | Feature importance only | SHAP values + dependence plots |
| Survival analysis | None | Kaplan-Meier + Cox PH |
| Deep dives | None | Subgroup / Profile / What-if |
| Figures | ~20 charts | **60 publication-ready figures** |
| Statistical rigor | Implicit | Explicit formulas (MI, χ², Cohen's d) |

---

## Dataset

**Source**: [`data/raw/employee_data.csv`](data/raw/employee_data.csv) — 3,000 employee
records, 26 columns covering demographics, employment history, performance, and
termination.

**Engineered features**: 27 additional features derived from the raw data (seniority
level, intersectional identity, tenure bins, performance scores, etc.).

---

## Architecture: 7-Phase Pipeline

```
Raw Data ──► Phase 1: Deep EDA ──► Phase 2: Feature Engineering ──► Phase 3: Interaction Mining ──► Phase 4: Predictive Models ──► Phase 5: Deep Dives ──► Phase 6: Dashboards ──► Phase 7: HTML Report
                  │                       │                           │                           │                       │                       │
                  ▼                       ▼                           ▼                           ▼                       ▼                       ▼
           12 figures               3 figures                   8 figures                  13 figures              16 figures              8 figures
         (01–12)                  (13–15)                     (16–23)                     (24–36)                  (37–52)                 (53–60)
```

---

## Phase-by-Phase Breakdown

### Phase 1: Deep EDA ([`phase1_deep_eda.py`](scripts/phase1_deep_eda.py))

**Goal**: Understand data structure, distributions, missingness, outliers, and latent
structure.

**Figures**: 01–12

| Method | Formula | Description |
|--------|---------|-------------|
| Univariate distributions | Histograms + KDE | Distribution shape for each numeric column |
| Missing pattern | `missingno` matrix | Visualize missing value patterns and MCAR/MAR structure |
| Outlier detection | IQR: Q1 − 1.5×IQR, Q3 + 1.5×IQR + Z-score | Consensus outlier flagging across methods |
| Correlation network | Pearson ρ = cov(X,Y) / σ_X σ_Y | Graph of correlated feature pairs (|ρ| > 0.5) |
| PCA | Explained variance = λ_k / Σ λ_i | Dimensionality reduction to 2–5 components |
| t-SNE | Perplexity-based embedding | Non-linear 2D projection of feature space |
| Silhouette Score | s(i) = (b(i) − a(i)) / max(a(i), b(i)) | Cluster quality for k=2..10 |

**Output**: [`data/interaction/eda_summary.json`](data/interaction/eda_summary.json)

---

### Phase 2: Feature Engineering ([`phase2_feature_engineering.py`](scripts/phase2_feature_engineering.py))

**Goal**: Engineer 27 features from raw columns for downstream modeling.

**Figures**: 13–15

| Feature Group | Features | Logic |
|---------------|----------|-------|
| SeniorityLevel | 1 ordinal | Tenure-based seniority bucketing (1–5) |
| IsManager, IsIC | 2 binary | Job title → role classification |
| JobFamily | 1 categorical (7 labels) | Title-based job family grouping |
| IntersectionalID | 1 categorical | Concat(Gender, Race, Department) → 60+ groups |
| RecruitCost | 1 numeric | Estimated cost = salary × recruitment_multiple |
| CostPerHire | 1 numeric | RecruitCost / department_size |
| TenureBin, TenureYears | 2 numeric | TenureDays → years + bin label |
| ExitQuarter | 1 categorical | ExitDate → quarter (0–4, 0 = active) |
| PayEquityRatio | 1 numeric | Salary / department_median_salary |
| PromotionRate, PromotionLag | 2 numeric | Promotion frequency and recency |
| PerfScore | 1 numeric | Mean of Current Employee Rating (3-year) |
| EngagementFlag, EarlyTenureFlag | 2 binary | Low performance + low satisfaction triggers |

---

### Phase 3: Interaction Mining ([`phase3_interaction_mining.py`](scripts/phase3_interaction_mining.py))

**Goal**: Systematically test all pair-wise feature interactions against each outcome.

**Figures**: 16–23

**Methodology**: For each of 5 outcomes × all feature pairs × encoding variants:

```
Interaction Impact Score = Mutual Information × |ΔOutcome| × log(n_segment)
```

Where:
- **Mutual Information**: I(X;Y) = Σ Σ p(x,y) × log(p(x,y) / p(x)p(y))
- **|ΔOutcome|**: Absolute difference in outcome mean between the interaction segment
  and the population baseline
- **log(n_segment)**: Logarithmic weight for segment size (penalizes very small groups)

**Statistical tests applied**:

| Pair Type | Test | Statistic |
|-----------|------|-----------|
| Categorical × Categorical | Chi-square | χ² = Σ (O−E)² / E |
| Categorical × Numeric | ANOVA F-test | F = MS_between / MS_within |
| Numeric × Numeric | Pearson correlation | r = cov(X,Y) / σ_X σ_Y |

**Total pairs evaluated**: 1,755

**Output**: [`data/interaction/interaction_results.parquet`](data/interaction/interaction_results.parquet) |
[`data/interaction/interaction_top50.json`](data/interaction/interaction_top50.json)

**Top discoveries** (one per outcome):

| Feature 1 | Feature 2 | Outcome | Impact |
|-----------|-----------|---------|--------|
| JobFamily | DepartmentType | PayZone | 181.3 |
| TenureYears | ExitQuarter | Minority Dept | 107.7 |
| JobFamily | IsManager | Seniority | 68.2 |
| Region | DepartmentType | PerfScore | 62.8 |
| Region | IntersectionalID | Termination | 55.8 |

---

### Phase 4: Predictive Models ([`phase4_predictive_models.py`](scripts/phase4_predictive_models.py))

**Goal**: Compare model performance WITH vs WITHOUT interaction features across 5
outcomes.

**Figures**: 24–36

**Models trained**:

| Model | Config | Outcome Types |
|-------|--------|---------------|
| Logistic Regression | max_iter=2000, class_weight=balanced, L2 penalty | Binary + Multi-class |
| Random Forest | 200 trees, balanced class_weight | Binary + Multi-class |
| XGBoost | 200 estimators, max_depth=4, subsample=0.8 | Binary only |

**Evaluation Metrics**:

```
Accuracy          = (TP + TN) / (TP + TN + FP + FN)
F1 Score          = 2 × Precision × Recall / (Precision + Recall)
ROC-AUC           = ∫₀¹ TPR(FPR) d(FPR)
```

**Feature importance extraction**:
- **Tree models** (RF, XGB): Mean decrease in impurity (`feature_importances_`)
- **Linear models** (LR): |Coefficient| magnitude

**SHAP Analysis** ([SHAP](https://shap.readthedocs.io/)):
```
SHAP(xᵢ) = Σ_S⊆F\{xᵢ} |S|!(|F|−|S|−1)! / |F|! × [fₓ(S∪{xᵢ}) − fₓ(S)]
```
Uses `shap.Explainer` on the XGBoost model with 200 background samples.

**Survival Analysis**:

- **Kaplan-Meier**: S(t) = Π_{tᵢ ≤ t} (1 − dᵢ / nᵢ)
- **Cox Proportional Hazards**: h(t) = h₀(t) × exp(β₁x₁ + β₂x₂ + ... + βₚxₚ)

**Key finding**: Interaction features provide marginal improvement for most outcomes.
RF achieves best F1 for minority-dept prediction; XGB leads in AUC for termination.

**Output**: [`data/interaction/model_results.json`](data/interaction/model_results.json) |
[`data/interaction/modeling_summary.json`](data/interaction/modeling_summary.json)

---

### Phase 5: Deep Dives ([`phase5_deep_dives.py`](scripts/phase5_deep_dives.py))

**Goal**: For each top discovery, analyze the specific driver segment with subgroup
comparison, profile characterization, and what-if simulation.

**Figures**: 37–52 (3 per discovery + 1 dashboard = 16)

**Each dive includes**:

1. **Subgroup Analysis** (Cohen's d effect size):
   ```
   d = (μ_segment − μ_rest) / s_pooled
   ```
   Statistical significance via two-sample t-test.

2. **Profile Characterization**: Mean feature values for the segment vs population,
   identifying what makes this group unique.

3. **What-If Simulation** (for numeric features): For f1 in [p5, p25, p50, p75, p95]:
   - Sample employees with f1 ≈ target value
   - Compute expected outcome for the segment
   - Plot: Feature value → Expected outcome (with confidence band)

**Sample findings**:
- Administration + Sales segment shows 0.33σ higher PayZone (p=0.057)
- Leadership non-managers have SeniorityLevel 4.0 vs 2.2 population mean (p<0.001)
- Northeast + M_Whi_Soft segment has 0.72σ higher termination rate (p=0.048)

**Output**: [`data/interaction/deep_dives.json`](data/interaction/deep_dives.json)

---

### Phase 6: Dashboards ([`phase6_dashboards.py`](scripts/phase6_dashboards.py))

**Goal**: Synthesize all prior phases into 8 composite dashboard figures.

**Figures**: 53–60

| Figure | Title | Content |
|--------|-------|---------|
| 53 | Attrition Dashboard | Termination rates by department, gender, tenure + survival curve |
| 54 | Compensation Dashboard | Pay zone distribution, equity ratio by gender, cost per hire |
| 55 | Diversity Dashboard | Gender/race representation, intersectional breakdowns |
| 56 | Performance Dashboard | PerfScore distribution, rating trends, early-tenure flags |
| 57 | Career Dashboard | Promotion rates by role, seniority distribution, career paths |
| 58 | Interaction Matrix | Heatmap of all pair-wise interaction impacts per outcome |
| 59 | ROC Curves | Side-by-side ROC curves for all outcomes × all models |
| 60 | Executive Summary | Key metrics, top interactions, model comparison in one view |

---

### Phase 7: HTML Report ([`phase7_html_report.py`](scripts/phase7_html_report.py))

**Goal**: Generate a standalone HTML report embedding all 60 figures with structured
sections.

**Output**: [`reports/report.html`](reports/report.html)

Built with Jinja2 templating. Each section maps to one pipeline phase with the
corresponding figures inline.

---

## Key Findings Summary

1. **Interaction features provide ~1–3% accuracy improvement** for most outcomes, with
   the largest gains in PayZone prediction (JobFamily × DepartmentType).
2. **Tenure + Exit Quarter** is the strongest predictor of minority-department status
   (Cohen's d = 0.60, p = 0.005).
3. **SHAP analysis** identifies JobFamily, SeniorityLevel, and Region as the top
   drivers of termination risk.
4. **Logistic Regression with balanced class weights** achieves AUC=0.85 on termination
   prediction, comparable to XGBoost while being fully interpretable.
5. **Leadership non-managers** are the most extreme segment (Seniority 4.0 vs 2.2
   average) — a critical group for career-path analysis.

---

## Repository Structure

```
WorkForce-Data-Analysis/
│
├── archive/v1/                          # Legacy v1.0 (see README there)
│   ├── README.md
│   ├── config/                          # Archived v1.0 configs
│   └── data/sample/                     # Archived sample data
│
├── scripts/                             # v2.0 pipeline scripts
│   ├── phase1_deep_eda.py
│   ├── phase2_feature_engineering.py
│   ├── phase3_interaction_mining.py
│   ├── phase4_predictive_models.py
│   ├── phase5_deep_dives.py
│   ├── phase6_dashboards.py
│   ├── phase7_html_report.py
│   └── derive_datasets.py
│
├── src/wf_analysis/interaction/         # Core library modules
│   ├── config.py                        # InteractionConfig
│   ├── deep_eda.py                      # DeepEDA (univariate, missing, outliers)
│   ├── dim_reduction.py                 # PCA, t-SNE, silhouette
│   ├── features.py                      # FeatureEngineering (27 features)
│   ├── mining.py                        # InteractionMining (1,755 tests)
│   ├── models.py                        # InteractionModeler (LR/RF/XGB/SHAP/Survival)
│   ├── dives.py                         # Deep dive analysis
│   └── figures.py                       # EDAFigureFactory (all 60 figures)
│
├── notebooks/                           # Jupyter exploration notebooks
│   ├── 01_data_profile.ipynb
│   ├── 02_attrition.ipynb
│   ├── 03_compensation.ipynb
│   ├── 04_performance.ipynb
│   ├── 05_career_path.ipynb
│   ├── 06_diversity.ipynb
│   ├── 07_org_network.ipynb
│   ├── 08_forecasting.ipynb
│   ├── 09_exit_nlp.ipynb
│   └── 10_integrated.ipynb
│
├── data/                                # v2.0 data artifacts
│   ├── raw/employee_data.csv            # Original dataset
│   ├── analysis/                        # 10 analysis datasets
│   └── interaction/                     # Pipeline outputs (JSON + Parquet)
│       ├── model_results.json
│       ├── interaction_results.parquet
│       ├── deep_dives.json
│       └── ...
│
├── reports/                             # v2.0 outputs
│   ├── report.html                      # Standalone HTML report
│   └── figures/interaction/             # 60 PNG figures (01–60)
│
├── tests/                               # 58 passing tests
│
├── ROADMAP.md                           # Future development plans
├── IMPLEMENTATION_PLAN.md               # Original implementation plan
└── pyproject.toml                       # Project metadata + dependencies
```

---

## How to Run (v2.0)

```bash
# Install
pip install -e .

# Run phases sequentially (each generates its figures independently)
python scripts/phase1_deep_eda.py
python scripts/phase2_feature_engineering.py
python scripts/phase3_interaction_mining.py
python scripts/phase4_predictive_models.py
python scripts/phase5_deep_dives.py
python scripts/phase6_dashboards.py
python scripts/phase7_html_report.py

# Run tests
python -m pytest tests/
```

---

## Tests

58 tests covering: data loading/cleaning/validation, NLP pipeline (text preprocessing,
sentiment, topic modeling, classification, keyword extraction, embeddings), utility
functions (chunking, timing, caching), and visualization (themes, charts, dashboards,
report generation).

```bash
python -m pytest tests/ --tb=short
```

---

## Legacy: v1.0

The original dashboard-based analysis is preserved in two forms:

| Access | Link |
|--------|------|
| **Archived files** | [`archive/v1/`](archive/v1/README.md) — restored configs, sample data, and v1.0 README |
| **Git tag** | [`v1.0`](https://github.com/Naydhurve3/WorkForce-Data-Analysis/tree/v1.0) — complete original source code as committed |

v1.0 featured a Streamlit dashboard with 7 pages covering attrition, diversity,
compensation, performance, organizational network, career paths, and forecasting.
See the [v1.0 README](archive/v1/README.md) for full details.

---

## License

This project is for research and educational purposes.
