.PHONY: help install test build clean format lint

# The default goal if no target is specified
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "MLSys·im Development Commands"
	@echo "-----------------------------"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s
", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package in editable mode with all dev and cli dependencies
	pip install -e ".[dev,cli,viz]"

test: ## Run the full test suite
	python3 -m pytest tests/ -v

format: ## Format the code using ruff
	ruff format .
	ruff check --fix .

lint: ## Run the linter (ruff) to check for style violations
	ruff check .

build: ## Build the package for distribution (creates .tar.gz and .whl)
	pip install build
	python3 -m build

clean: ## Remove build artifacts, cache directories, and .pyc files
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
