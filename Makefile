
# Makefile for Apairo Project

.PHONY: env install test coverage lint format typecheck soak clean help

help:
	@echo "Available commands:"
	@echo "  make env       : Create virtual environment"
	@echo "  make install   : Install dependencies in editable mode"
	@echo "  make test      : Run tests with pytest"
	@echo "  make coverage  : Run tests with coverage report"
	@echo "  make lint      : Run linting with ruff"
	@echo "  make format    : Format the codebase with ruff"
	@echo "  make typecheck : Run mypy (CI gate -- must stay at zero errors)"
	@echo "  make soak      : Run the synthetic intensive-usage soak (benchmarks/soak.py)"
	@echo "  make clean     : Remove build artifacts and cache"

env:
	python3 -m venv .venv
	@echo "Activate with: source .venv/bin/activate"

install:
	pip install --upgrade pip
	pip install -e ".[dev]"

test:
	python3 -m pytest

coverage:
	python3 -m pytest --cov=apairo --cov-report=term-missing --cov-fail-under=85

lint:
	ruff check apairo test
	ruff format --check apairo test benchmarks examples

format:
	ruff format apairo test benchmarks examples

typecheck:
	mypy apairo

soak:
	python3 benchmarks/soak.py

clean:
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache
