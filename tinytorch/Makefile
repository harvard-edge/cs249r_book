# TinyTorch Makefile
# ==================
# Simple commands for common development tasks.
#
# Usage:
#   make help        # Show all commands
#   make test        # Run all tests
#   make preflight   # Quick verification before work
#   make release     # Full release validation
#

.PHONY: help test preflight release clean lint

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

# Default target
help:
	@echo ""
	@echo "🔥 TinyTorch Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Quick Commands:"
	@echo "  make preflight     Quick check (~1 min) - run before starting work"
	@echo "  make test          Run main test suite"
	@echo "  make test-quick    Fast smoke tests only (~30s)"
	@echo ""
	@echo "Release Validation:"
	@echo "  make release       Full release validation (~10 min)"
	@echo "  make release-check Pre-release checklist"
	@echo ""
	@echo "Development:"
	@echo "  make lint          Check code style"
	@echo "  make clean         Remove generated files"
	@echo "  make setup         Install development dependencies"
	@echo ""
	@echo "Testing Levels:"
	@echo "  make test-e2e-quick      E2E quick tests (~30s)"
	@echo "  make test-e2e-module     E2E module flow tests (~2min)"
	@echo "  make test-e2e-full       E2E complete journey (~10min)"
	@echo "  make test-milestones     Milestone smoke tests"
	@echo ""

# ============================================================================
# QUICK COMMANDS (daily use)
# ============================================================================

# Quick preflight check - run this before starting work
preflight:
	$(PYTHON) -m tito.main dev preflight

# Quick preflight (faster)
preflight-quick:
	$(PYTHON) -m tito.main dev preflight --quick

# Standard test suite
test:
	$(PYTHON) -m pytest tests/ -v --ignore=tests/e2e --ignore=tests/milestones -q

# Fast smoke tests only
test-quick:
	$(PYTHON) -m pytest tests/e2e/test_user_journey.py -k quick -v

# ============================================================================
# E2E TESTING (by level)
# ============================================================================

# E2E quick verification (~30 seconds)
test-e2e-quick:
	$(PYTHON) -m pytest tests/e2e/test_user_journey.py -k quick -v

# E2E module workflow tests (~2 minutes)
test-e2e-module:
	$(PYTHON) -m pytest tests/e2e/test_user_journey.py -k module_flow -v

# E2E milestone tests
test-e2e-milestone:
	$(PYTHON) -m pytest tests/e2e/test_user_journey.py -k milestone_flow -v

# E2E complete journey (~10 minutes)
test-e2e-full:
	$(PYTHON) -m pytest tests/e2e/test_user_journey.py -v

# ============================================================================
# SPECIALIZED TESTS
# ============================================================================

# Milestone smoke tests (fast import/model construction checks)
test-milestones:
	$(PYTHON) -m pytest tests/milestones/test_milestones_smoke.py -v

# CLI tests
test-cli:
	$(PYTHON) -m pytest tests/cli/ -v

# Module-specific tests
test-module-%:
	$(PYTHON) -m pytest tests/$*/ -v

# ============================================================================
# RELEASE VALIDATION
# ============================================================================

# Full release validation - run this before any release
release:
	$(PYTHON) -m tito.main dev preflight --release

# Full release validation with all tests
release-full:
	$(PYTHON) -m tito.main dev preflight --release
	$(PYTHON) -m pytest tests/ -v --tb=short

# Pre-release checklist (manual verification)
release-check:
	@echo ""
	@echo "📋 Pre-Release Checklist"
	@echo "========================"
	@echo ""
	@echo "Run each of these commands and verify they pass:"
	@echo ""
	@echo "  1. make preflight              # Quick sanity check"
	@echo "  2. make test-e2e-full          # E2E user journey"
	@echo "  3. make test-milestones        # Milestone smoke tests"
	@echo "  4. make test                   # Full test suite"
	@echo ""
	@echo "Manual checks:"
	@echo "  □ README.md is up to date"
	@echo "  □ Version number bumped in pyproject.toml"
	@echo "  □ CHANGELOG updated"
	@echo "  □ Git status is clean"
	@echo ""
	@echo "Then run: make release"
	@echo ""

# ============================================================================
# DEVELOPMENT UTILITIES
# ============================================================================

# Install development dependencies
setup:
	$(PIP) install -e ".[dev]"
	$(PIP) install pytest pytest-cov rich

# Lint code
lint:
	$(PYTHON) -m py_compile tito/main.py
	@echo "✓ No syntax errors"

# Clean generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned generated files"

# ============================================================================
# CI/CD TARGETS (used by GitHub Actions)
# ============================================================================

# CI smoke test (fast, for every commit)
ci-smoke:
	$(PYTHON) -m tito.main dev preflight --quick --ci

# CI standard test (for PRs)
ci-standard:
	$(PYTHON) -m tito.main dev preflight --ci
	$(PYTHON) -m pytest tests/e2e/ -k quick --tb=short -q

# CI full test (for releases)
ci-full:
	$(PYTHON) -m tito.main dev preflight --full --ci
	$(PYTHON) -m pytest tests/ -v --ignore=tests/milestones --tb=short

# CI release validation (comprehensive)
ci-release:
	$(PYTHON) -m tito.main dev preflight --release --ci

# CI JSON output (for automation/parsing)
ci-json:
	$(PYTHON) -m tito.main dev preflight --json
