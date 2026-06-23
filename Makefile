
# Makefile for Apairo Project

.PHONY: env install test lint clean help

help:
	@echo "Available commands:"
	@echo "  make env      : Create virtual environment"
	@echo "  make install  : Install dependencies in editable mode"
	@echo "  make test     : Run tests with pytest"
	@echo "  make lint     : Run linting with ruff"
	@echo "  make clean    : Remove build artifacts and cache"

env:
	python3 -m venv .venv
	@echo "Activate with: source .venv/bin/activate"

install:
	pip install --upgrade pip
	pip install -e ".[dev]"

test:
	python3 -m pytest

lint:
	ruff check apairo test

clean:
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache
