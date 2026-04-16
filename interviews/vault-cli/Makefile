# Makefile for vault-cli — convenience wrappers over CLI and tests (B.2).

PKG_DIR := $(shell pwd)
# `git rev-parse --git-path hooks` works correctly in both regular clones
# and worktrees (where .git is a file pointing at the main repo's hooks dir).
HOOKS_DIR := $(shell git rev-parse --git-path hooks 2>/dev/null)
HOOK_SRC := $(PKG_DIR)/scripts/pre_commit_corpus_guard.py
HOOK_DST := $(HOOKS_DIR)/pre-commit

.PHONY: install test lint hooks hooks-uninstall help

help:
	@echo "Targets:"
	@echo "  install          pip install -e with dev extras"
	@echo "  test             run pytest"
	@echo "  lint             ruff check (mypy is non-blocking at Phase 0)"
	@echo "  hooks            symlink pre-commit-corpus-guard into .git/hooks/"
	@echo "  hooks-uninstall  remove the hook symlink"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src tests
	@mypy src || echo "[mypy] strict is non-blocking at Phase 0"

hooks:
	@if [ -z "$(HOOKS_DIR)" ]; then \
		echo "could not resolve git hooks dir; run make from a git checkout"; exit 1; \
	fi
	@mkdir -p "$(HOOKS_DIR)"
	@if [ -e "$(HOOK_DST)" ] && [ ! -L "$(HOOK_DST)" ]; then \
		echo "refusing to overwrite non-symlink at $(HOOK_DST); remove it first"; \
		exit 1; \
	fi
	@ln -sf "$(HOOK_SRC)" "$(HOOK_DST)"
	@chmod +x "$(HOOK_SRC)"
	@echo "installed hook: $(HOOK_DST) -> $(HOOK_SRC)"

hooks-uninstall:
	@if [ -L "$(HOOK_DST)" ]; then \
		rm "$(HOOK_DST)"; \
		echo "removed $(HOOK_DST)"; \
	else \
		echo "no symlink at $(HOOK_DST)"; \
	fi
