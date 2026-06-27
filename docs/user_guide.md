# WorkForce Analytics - User Guide

## Overview
WorkForce Analytics v2.0 is an interactive dashboard for HR managers to explore employee data, analyze attrition patterns, assess diversity, and forecast workforce trends.

## Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Running the Dashboard
```bash
streamlit run dashboard/app.py
```
Opens at `http://localhost:8501`

### Running the Full Pipeline
```bash
python scripts/run_pipeline.py
```

## Dashboard Pages

| Page | Description |
|------|-------------|
| Executive Overview | Key KPIs: headcount, attrition rate, avg tenure |
| Attrition Explorer | Attrition rates by department, demographics, trends |
| Diversity & Inclusion | Gender/race distribution, intersectional views |
| Performance Analytics | Score distribution, rating analysis |
| NLP Insights | Sentiment analysis of termination descriptions |
| Org Network | Span of control, organizational hierarchy |
| Workforce Planning | Headcount forecasting, scenario modeling |

## Global Filters
Use the sidebar to filter by:
- Job Family
- Department
- Region
- Gender
- Age Range

## Keyboard Shortcuts
- `r` - Rerun dashboard
- `c` - Clear cache
