"""Phase 7: HTML Report — single-file report with all 60 figures and findings."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from loguru import logger
from jinja2 import Template

from wf_analysis.data.loader import DataLoader
from wf_analysis.interaction.config import InteractionConfig
from wf_analysis.interaction.features import FeatureEngineer


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Workforce Analytics Report</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f7fa; color: #1a1a2e; padding: 0; }
.header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 40px 60px; }
.header h1 { font-size: 28px; margin-bottom: 6px; }
.header p { opacity: 0.8; font-size: 14px; }
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; padding: 24px 60px; background: white; border-bottom: 1px solid #e0e0e0; }
.kpi-card { text-align: center; }
.kpi-card .val { font-size: 28px; font-weight: 700; color: #16213e; }
.kpi-card .lbl { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
.section { padding: 24px 60px; }
.section h2 { font-size: 20px; margin-bottom: 16px; color: #16213e; border-bottom: 2px solid #e94560; padding-bottom: 6px; display: inline-block; }
.section p { margin-bottom: 12px; line-height: 1.6; color: #444; }
.section-desc { margin-bottom: 20px; font-size: 14px; color: #555; }
.figure-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; }
.figure-card { background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); overflow: hidden; }
.figure-card img { width: 100%; height: auto; display: block; }
.figure-card .fig-label { padding: 10px 14px; font-size: 12px; color: #333; background: #fafafa; border-top: 1px solid #eee; }
.figure-card .fig-label strong { color: #16213e; }
.full-width { grid-column: 1 / -1; }
.insights { background: #fff8f0; border-left: 4px solid #e94560; padding: 14px 18px; margin: 16px 0; border-radius: 0 6px 6px 0; font-size: 13px; line-height: 1.5; }
.rec-list { list-style: none; }
.rec-list li { padding: 10px 14px; margin-bottom: 8px; background: white; border-radius: 6px; border-left: 3px solid #e94560; font-size: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.rec-list li strong { color: #16213e; }
.footer { text-align: center; padding: 30px; font-size: 12px; color: #999; border-top: 1px solid #e0e0e0; margin-top: 40px; }
@media (max-width: 768px) { .header, .section, .kpi-row { padding: 20px; } .figure-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<div class="header">
  <h1>Workforce Analytics — Complete Report</h1>
  <p>{{ n }} employees · {{ n_term }} terminated ({{ "{:.1%}".format(rate) }}) · {{ phases|length }} phases · {{ total_figures }} figures</p>
</div>

<div class="kpi-row">
  <div class="kpi-card"><div class="val">{{ n }}</div><div class="lbl">Employees</div></div>
  <div class="kpi-card"><div class="val">{{ "{:.1%}".format(rate) }}</div><div class="lbl">Attrition Rate</div></div>
  <div class="kpi-card"><div class="val">{{ "{:.2f}".format(mean_perf) }}</div><div class="lbl">Mean PerfScore</div></div>
  <div class="kpi-card"><div class="val">{{ "{:.1f}".format(mean_tenure) }}</div><div class="lbl">Mean Tenure (yr)</div></div>
  <div class="kpi-card"><div class="val">{{ "{:.2f}".format(mean_seniority) }}</div><div class="lbl">Mean Seniority</div></div>
  <div class="kpi-card"><div class="val">{{ n_jobfam }}</div><div class="lbl">Job Families</div></div>
  <div class="kpi-card"><div class="val">{{ n_regions }}</div><div class="lbl">Regions</div></div>
</div>

{% for phase in phases %}
<div class="section">
  <h2>{{ phase.title }}</h2>
  <div class="section-desc">{{ phase.desc }}</div>
  <div class="figure-grid">
  {% for fig in phase.figures %}
    <div class="figure-card{% if fig.fullwidth %} full-width{% endif %}">
      <img src="figures/interaction/{{ fig.file }}" alt="{{ fig.label }}">
      <div class="fig-label"><strong>{{ fig.label }}</strong></div>
    </div>
  {% endfor %}
  </div>
</div>
{% endfor %}

<div class="section">
  <h2>Recommendations</h2>
  <div class="section-desc">Actionable insights from the complete analysis pipeline.</div>
  <ol class="rec-list">
    <li><strong>Pay Equity Audit</strong> — Administration staff in Sales show higher PayZone (2.29 vs 1.96). Investigate compensation policies for this segment.</li>
    <li><strong>Mid-Career Retention</strong> — TenureYears (4.5 yr) × ExitQuarter interaction (impact=108) shows minority employees churning at mid-career. Implement targeted retention at Year 4.</li>
    <li><strong>Regional Attrition</strong> — Northeast region has 2.9× higher attrition across multiple identity groups (impact=56). Conduct regional culture audit.</li>
    <li><strong>Seniority Path Transparency</strong> — Leadership non-managers (d=4.28) have drastically higher SeniorityLevel. Clarify IC career progression paths.</li>
    <li><strong>Performance Rating Ceiling</strong> — PerfScore (μ≈3.0) saturates. Consider expanding evaluation range to better differentiate high performers.</li>
  </ol>
</div>

<div class="footer">
  Generated {{ date }} · WorkForce Data Analysis Pipeline
</div>

</body>
</html>"""


def main():
    logger.info("=" * 60)
    logger.info("  Phase 7: HTML Report")
    logger.info("=" * 60)

    cfg = InteractionConfig()
    raw_df = DataLoader.load(cfg.raw_path, validate=False)
    engineer = FeatureEngineer(cfg)
    feature_df = engineer.compute_all(raw_df)

    is_term = raw_df["EmployeeStatus"].str.lower().str.contains("terminat").astype(int)
    n = len(raw_df)
    n_term = int(is_term.sum())
    rate = is_term.mean()

    perf_col = raw_df.get("Current Employee Rating", feature_df.get("PerfScore"))
    mean_perf = perf_col.mean() if perf_col is not None else 0
    mean_tenure = feature_df["TenureYears"].mean() if "TenureYears" in feature_df else 0
    mean_seniority = feature_df["SeniorityLevel"].mean() if "SeniorityLevel" in feature_df else 0
    n_jobfam = feature_df["JobFamily"].nunique() if "JobFamily" in feature_df else 0
    n_regions = raw_df["Region"].nunique() if "Region" in raw_df else 0

    fig_dir = Path(cfg.figure_dir)
    fig_files = {f.stem.split("_")[0]: f.name for f in fig_dir.iterdir() if f.stem.split("_")[0].isdigit()}
    p1 = [f"{i:02d}" for i in range(1, 13)]
    p2 = [f"{i:02d}" for i in range(13, 16)]
    p3 = [f"{i:02d}" for i in range(16, 24)]
    p4 = [f"{i:02d}" for i in range(24, 37)]
    p5 = [f"{i:02d}" for i in range(37, 53)]
    p6 = [f"{i:02d}" for i in range(53, 61)]

    phase_configs = [
        {
            "title": "Phase 1: Exploratory Data Analysis",
            "desc": "Univariate statistics, correlation heatmap, missing patterns, PCA, t-SNE, clustering silhouette, and dashboard summary.",
            "figures": [{"file": fig_files[k], "label": f"Fig {int(k)}: {fig_files[k].split('_',1)[1].rsplit('.',1)[0].replace('_',' ').title()}"} for k in p1 if k in fig_files],
        },
        {
            "title": "Phase 2: Feature Engineering",
            "desc": "Feature correlation, summary stats, and distribution analysis across engineered features.",
            "figures": [{"file": fig_files[k], "label": f"Fig {int(k)}: {fig_files[k].split('_',1)[1].rsplit('.',1)[0].replace('_',' ').title()}"} for k in p2 if k in fig_files],
        },
        {
            "title": "Phase 3: Interaction Mining",
            "desc": "Interaction heatmaps, mutual information, tree segmentation, pair plots, network graph, and outcome ranking.",
            "figures": [{"file": fig_files[k], "label": f"Fig {int(k)}: {fig_files[k].split('_',1)[1].rsplit('.',1)[0].replace('_',' ').title()}"} for k in p3 if k in fig_files],
        },
        {
            "title": "Phase 4: Predictive Models",
            "desc": "Model comparison, coefficient shift, RF/XGB importance, SHAP, ensemble, survival analysis, and modeling dashboard.",
            "figures": [{"file": fig_files[k], "label": f"Fig {int(k)}: {fig_files[k].split('_',1)[1].rsplit('.',1)[0].replace('_',' ').title()}"} for k in p4 if k in fig_files],
        },
        {
            "title": "Phase 5: Deep Dives",
            "desc": "Subgroup comparisons, segment profiles, and what-if scenarios for the top-5 interaction discoveries.",
            "figures": [{"file": fig_files[k], "label": f"Fig {int(k)}: {fig_files[k].split('_',1)[1].rsplit('.',1)[0].replace('_',' ').title()}"} for k in p5 if k in fig_files],
        },
        {
            "title": "Phase 6: Dashboards",
            "desc": "Composite dashboards for attrition, compensation, diversity, performance, career, interaction matrix, ROC curves, and executive summary.",
            "figures": [{"file": fig_files[k], "label": f"Fig {int(k)}: {fig_files[k].split('_',1)[1].rsplit('.',1)[0].replace('_',' ').title()}"} for k in p6 if k in fig_files],
        },
    ]

    from datetime import datetime

    html = Template(TEMPLATE).render(
        n=n, n_term=n_term, rate=rate,
        mean_perf=mean_perf, mean_tenure=mean_tenure,
        mean_seniority=mean_seniority,
        n_jobfam=n_jobfam, n_regions=n_regions,
        phases=phase_configs,
        total_figures=sum(len(p["figures"]) for p in phase_configs),
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    report_path = Path(cfg.output_dir).parent.parent / "reports" / "report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"Report saved to {report_path} ({report_path.stat().st_size / 1024:.0f} KB)")

    logger.info("=" * 60)
    logger.info("  Phase 7 complete — HTML report generated")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
