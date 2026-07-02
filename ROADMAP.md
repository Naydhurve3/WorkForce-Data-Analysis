# WorkForce Data Analysis — Feature-Interaction Roadmap

**Goal:** Systematically combine all 27 analytical features across 5 outcomes to discover hidden interaction patterns — demonstrating skills from Data Analyst (deep EDA) → Data Scientist (predictive models + SHAP) → ML Engineer (modular pipeline).

---

## Architecture Overview

```
raw/employee_data.csv
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Deep EDA          12 figures, 3-method outliers, PCA   │
│ Phase 2: Feature Engineering 27 features from 26 raw columns    │
│ Phase 3: Interaction Mining  351×5=1,755 tests, impact scores   │
│ Phase 4: Predictive Models  LR→RF→XGB→Ensemble, WITH/WITHOUT   │
│ Phase 5: Deep Dives         Top-10 discoveries × 3 panels each  │
│ Phase 6: Dashboards         6 composite + interaction matrix    │
│ Phase 7: HTML Report        Jinja2, KPI cards, recommendations  │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
  Final HTML Report (70+ figures, 20+ models, ~1,755 tests)
```

---

## Phase 1: Deep EDA (12 figures)

| # | File | What |
|---|------|------|
| 1 | `src/wf_analysis/interaction/config.py` | EDAConfig, DimReductionConfig, InteractionConfig |
| 2 | `src/wf_analysis/interaction/deep_eda.py` | Univariate stats, 3-method outlier detection, missing patterns |
| 3 | `src/wf_analysis/interaction/dim_reduction.py` | PCA, t-SNE (3 perplexities), correlation network (NetworkX), silhouette scores |
| 4 | `src/wf_analysis/interaction/figures.py` | 12 figure generators using Theme + PlotFactory patterns |
| 5 | `scripts/phase1_deep_eda.py` | Orchestrator — loads data, runs EDA + dim reduction, saves 12 figures |

**Figures produced:**
- Univariate numeric grid (histograms + mean/median lines)
- Categorical distribution grid (top-12 barh per column)
- Missing pattern heatmap + missing% bar chart
- Missing co-occurrence correlation heatmap
- Outlier % by column (IQR vs Z-score vs Isolation Forest)
- Outlier consensus horizontal bar (all 3 methods + overlap)
- Correlation matrix + network graph (NetworkX, |r|≥0.3)
- PCA scree plot + cumulative variance
- PCA biplot (PC1 vs PC2 with feature loadings)
- t-SNE landscape (perplexity sweep: 5, 30, 50)
- Silhouette score comparison (PCA vs t-SNE, k=2..7)
- EDA dashboard summary (overview + skew + missing + outliers + priorities)

---

## Phase 2: Feature Engineering (27 features, 3 figures)

Derives from 26 raw columns:

**Numeric:** Age, TenureDays, TenureYears, CareerStage, TenureCategory, SpanOfControl, OrgLevel, DistanceFromRoot, PromotionReadiness, TenureVsAvg, IsLongTenure

**Categorical:** JobFamily, SeniorityLevel, DivisionGroup, Region, Generation, CareerStageLabel, IsExecutive, IsManager, IsIC, IntersectionalID (Gender×Race×Dept), DeptGenderRatio, DeptDiversityScore

**Date-based:** StartYear, StartQuarter, ExitYear, ExitQuarter, TenureGroup

---

## Phase 3: Interaction Mining (1,755 tests, 8 figures)

- 27 features → 351 pairwise combinations
- Each combination tested against 5 outcomes: Attrition, Performance, Compensation, DiversityImpact, CareerMobility
- Methods: Chi-square (cat×cat), ANOVA (num×cat), Pearson (num×num), Mutual Information
- Decision-tree multi-way segmentation
- Impact-scored ranking

---

## Phase 4: Predictive Models (13 figures)

| Model | Outcomes | WITH vs WITHOUT interactions |
|-------|----------|------------------------------|
| Logistic Regression | All 5 | Coefficient comparison |
| Random Forest | All 5 | Feature importance delta |
| XGBoost | All 5 | SHAP with/without interactions |
| Stacked Ensemble | Attrition, Performance | Best accuracy |
| Survival (KM + Cox PH) | Attrition tenure | Log-rank test |

---

## Phase 5: Deep Dives (~25 figures)

Top-10 discoveries each with:
- Subgroup comparison (discovery group vs rest)
- Segment profile radar chart
- What-if counterfactual analysis

---

## Phase 6: Dashboards (8 figures)

- Attrition Risk Composite
- Compensation Equity
- Diversity & Inclusion
- Performance Drivers
- Career Mobility
- Strategic Overview
- Interaction Master Matrix (all 351 pairs × 5 outcomes)
- Executive Summary (KPI cards)

---

## Phase 7: HTML Report (Jinja2)

- Jinja2 template with all 70+ figures
- KPI cards with color coding
- Prioritized recommendations (HIGH/MED/LOW)
- Navigation: Phase 1 → 2 → 3 → 4 → 5 → 6

---

## Execution Order

```
Phase 1: scripts/phase1_deep_eda.py
Phase 2: scripts/phase2_feature_engineering.py
Phase 3: scripts/phase3_interaction_mining.py
Phase 4: scripts/phase4_predictive_models.py
Phase 5: scripts/phase5_deep_dives.py
Phase 6: scripts/phase6_dashboards.py
Phase 7: scripts/phase7_report.py
```

---

## Skill Demonstration Matrix

| Skill | Where Demonstrated |
|-------|-------------------|
| **Data Analyst** | Phase 1 (3-method outliers, missing patterns, PCA/t-SNE), Phase 2 (27 features) |
| **Statistician** | Phase 3 (1,755 tests, chi-square/ANOVA/MI, Bonferroni correction) |
| **Data Scientist** | Phase 4 (LR→RF→XGB→Ensemble, SHAP, Kaplan-Meier, Cox PH) |
| **ML Engineer** | Modular package (interaction/), config-driven, logging, type hints |
| **Visualization** | 70+ figures, 12-panel EDA dashboard, radar charts, network graphs |
| **Storytelling** | Phase 5 (what-if scenarios), Phase 7 (KPI cards, prioritized recs) |
