.PHONY: help install test build clean format lint check calibration-check modeling-docs-check publication-check audit docs docs-preview

PYTHON ?= python
# Quarto's pre-render hook invokes `python`; use the same environment as tests.
PYTHON_EXECUTABLE = $(shell $(PYTHON) -c "import sys; print(sys.executable)")

# The default goal if no target is specified
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "MLSys·im Development Commands"
	@echo "-----------------------------"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package in editable mode with all dev and cli dependencies
	$(PYTHON) -m pip install -e ".[dev,viz,labs]"

test: ## Run the full test suite
	$(PYTHON) -m pytest tests/ -v

audit: ## Run the package provenance audit
	$(PYTHON) -m mlsysim.tools.audit_provenance --scope all --strict

format: ## Format the code using ruff
	$(PYTHON) -m ruff format mlsysim scripts
	$(PYTHON) -m ruff check --fix mlsysim scripts

lint: ## Run the linter (ruff) to check for style violations
	$(PYTHON) -m ruff check mlsysim scripts

build: ## Build the package for distribution (creates .tar.gz and .whl)
	$(PYTHON) -m build

docs: ## Render the Quarto documentation site locally
	cd docs && PATH="$(dir $(PYTHON_EXECUTABLE)):$$PATH" QUARTO_PYTHON="$(PYTHON_EXECUTABLE)" quarto render

docs-preview: ## Preview the Quarto documentation site with hot-reloading
	cd docs && PATH="$(dir $(PYTHON_EXECUTABLE)):$$PATH" QUARTO_PYTHON="$(PYTHON_EXECUTABLE)" quarto preview

clean: ## Remove build artifacts, cache directories, and .pyc files
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

calibration-check: ## Check the generated calibration report
	$(PYTHON) scripts/calibration_report.py --check

modeling-docs-check: ## Check NumPy docstrings, model contracts, examples, and scoped formatting
	$(PYTHON) scripts/check_modeling_docs.py

publication-check: ## Verify displayed paper values, timing receipts, and benchmark anchors (repository checkout)
	PYTHONPATH=. $(PYTHON) paper/scripts/generate_paper_values.py --check
	PYTHONPATH=. $(PYTHON) paper/scripts/validate_anchors.py
	PYTHONPATH=. $(PYTHON) paper/scripts/analyze_gpu_energy.py --check
	PYTHONPATH=. $(PYTHON) paper/scripts/build_measured_gpu_energy.py --check

check: lint test calibration-check modeling-docs-check ## Run package release checks
