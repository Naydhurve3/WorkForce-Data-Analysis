# Developer Guide

## Setup
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Running Tests
```bash
pytest tests/ -v
pytest tests/ --cov=wf_analysis --cov-report=term-missing
```

## Project Structure
```
src/wf_analysis/          # Main package
  config.py               # Pydantic config models
  pipeline.py             # DAG-based orchestrator
  data/                   # Data loading, cleaning, validation
  features/               # Feature transformers
  nlp/                    # NLP pipeline
  imputation/             # ML imputation
  analysis/               # Analysis modules
  visualization/          # Theme, plots, reports
  utils/                  # Logging, decorators, helpers
dashboard/                # Streamlit app
tests/                    # Test suite
config/                   # YAML configuration
```

## Adding a New Analysis Module
1. Create `src/wf_analysis/analysis/your_module.py`
2. Extend `BaseAnalysis` and implement `run(df) -> AnalysisResult`
3. Add to `src/wf_analysis/analysis/__init__.py`
4. Add a pipeline stage if needed
5. Create a dashboard page in `dashboard/pages/`

## Code Standards
- Type hints on all functions
- Google-style docstrings
- Logging via `loguru`
- Configuration via YAML, never hardcoded
- Each pipeline stage creates a new DataFrame (immutable pattern)

## CI/CD
GitHub Actions workflow runs lint → test → build on push/PR.
