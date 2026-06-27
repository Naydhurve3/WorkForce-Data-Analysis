# WorkForce Data Analysis v2.0

A production-grade workforce analytics pipeline with modular Python architecture, ML-based imputation, NLP text analysis, and interactive dashboards.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run full pipeline
python scripts/run_pipeline.py --config config/config.yaml

# Launch dashboard
streamlit run dashboard/app.py
```

## Project Structure

```
src/wf_analysis/     # Main Python package (12 modules)
config/              # YAML configuration files
data/                # raw/ → interim/ → processed/
notebooks/           # Jupyter exploration notebooks
dashboard/           # Streamlit interactive dashboard
tests/               # pytest test suite
```

## Modules

- **data/**: Loading, validation, cleaning, export
- **features/**: Demographic, categorical, temporal, embedding features
- **nlp/**: Sentiment analysis, topic modeling, text classification, key phrases
- **imputation/**: ML-based Age/DOB imputation with distribution validation
- **analysis/**: Attrition, diversity, performance, compensation, network, forecasting
- **visualization/**: Consistent theme, reusable charts, report generation

## Dataset

Source: `data/raw/employee_data.csv` — 3,000 employee records, 26 columns covering demographics, employment, performance, and termination data.

See `IMPLEMENTATION_PLAN.md` for full implementation details.
