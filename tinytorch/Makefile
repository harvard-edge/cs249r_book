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
	@echo "  make test-milestones     Milestone learning tests (~90s)"
	@echo ""

# ============================================================================
# QUICK COMMANDS (daily use)
# ============================================================================

# Quick preflight check - run this before starting work
preflight:
	python -m tito.main dev preflight

# Quick preflight (faster)
preflight-quick:
	python -m tito.main dev preflight --quick

# Standard test suite
test:
	python -m pytest tests/ -v --ignore=tests/e2e --ignore=tests/milestones -q

# Fast smoke tests only
test-quick:
	python -m pytest tests/e2e/test_user_journey.py -k quick -v

# ============================================================================
# E2E TESTING (by level)
# ============================================================================

# E2E quick verification (~30 seconds)
test-e2e-quick:
	python -m pytest tests/e2e/test_user_journey.py -k quick -v

# E2E module workflow tests (~2 minutes)
test-e2e-module:
	python -m pytest tests/e2e/test_user_journey.py -k module_flow -v

# E2E milestone tests
test-e2e-milestone:
	python -m pytest tests/e2e/test_user_journey.py -k milestone_flow -v

# E2E complete journey (~10 minutes)
test-e2e-full:
	python -m pytest tests/e2e/test_user_journey.py -v

# ============================================================================
# SPECIALIZED TESTS
# ============================================================================

# Milestone learning verification (actually trains models, ~90 seconds)
test-milestones:
	python -m pytest tests/milestones/test_learning_verification.py -v

# CLI tests
test-cli:
	python -m pytest tests/cli/ -v

# Module-specific tests
test-module-%:
	python -m pytest tests/$*/ -v

# ============================================================================
# RELEASE VALIDATION
# ============================================================================

# Full release validation - run this before any release
release:
	python -m tito.main dev preflight --release

# Full release validation with all tests
release-full:
	python -m tito.main dev preflight --release
	python -m pytest tests/ -v --tb=short

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
	@echo "  3. make test-milestones        # ML actually learns"
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
	pip install -e ".[dev]"
	pip install pytest pytest-cov rich

# Lint code
lint:
	python -m py_compile tito/main.py
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
	python -m tito.main dev preflight --quick --ci

# CI standard test (for PRs)
ci-standard:
	python -m tito.main dev preflight --ci
	python -m pytest tests/e2e/ -k quick --tb=short -q

# CI full test (for releases)
ci-full:
	python -m tito.main dev preflight --full --ci
	python -m pytest tests/ -v --ignore=tests/milestones --tb=short

# CI release validation (comprehensive)
ci-release:
	python -m tito.main dev preflight --release --ci

# CI JSON output (for automation/parsing)
ci-json:
	python -m tito.main dev preflight --json
