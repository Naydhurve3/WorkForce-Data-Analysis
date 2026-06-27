.PHONY: install install-dev test lint mypy clean run-pipeline run-dashboard docker-build docker-up

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=wf_analysis --cov-report=term-missing

lint:
	flake8 src/wf_analysis/ tests/

mypy:
	mypy src/wf_analysis/ --strict

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf data/interim/*
	rm -rf reports/figures/*
	find . -name "*.pyc" -delete

run-pipeline:
	python scripts/run_pipeline.py --config config/config.yaml

run-dashboard:
	streamlit run dashboard/app.py --server.port 8501

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up
