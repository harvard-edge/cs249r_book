# =============================================================================
# TinyTorch Book build
# =============================================================================
# The book is the narrative artifact: The Problem / The Idea / The Code /
# A Worked Trace / From TinyTorch to Production, per chapter.
#
# This Makefile carries the LaTeX toolchain setup that used to live in
# guide/Makefile, which served the retired Lab Guide PDF. The book is now the
# only PDF built from this tree, so it owns its own dependencies.
#
# Entry points:
#   make install-deps — apt + TeX packages needed for the PDF
#   make pdf          — render the book (HTML + PDF per _quarto.yml)
#   make help
#
# Diagram PDFs: the book borrows guide/tools/diagrams/svg-to-pdf.lua, which
# rewrites .svg -> .pdf for the PDF format only, so sibling PDFs must exist.
# `make -C ../guide figures` produces them.
# =============================================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
QUARTO := quarto

.PHONY: help install-deps pdf

help: ## Show this help.
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install-deps: ## Install system + LaTeX packages needed to build the PDF.
	@# apt packages (requires sudo on CI / Linux; no-op elsewhere)
	@if [ -f apt-requirements.txt ] && command -v apt-get >/dev/null 2>&1; then \
	  echo "  installing apt packages from apt-requirements.txt"; \
	  sudo apt-get update -y; \
	  grep -v '^#' apt-requirements.txt | grep -v '^$$' | xargs -r sudo apt-get install -y; \
	fi
	@# TinyTeX / LaTeX packages (tlmgr from the Quarto TinyTeX install)
	@if [ -f tex-requirements.txt ]; then \
	  TLMGR="$$(command -v tlmgr 2>/dev/null || true)"; \
	  if [ -z "$$TLMGR" ]; then \
	    for dir in "$$HOME/bin" "$$HOME/.TinyTeX" "$$HOME/Library/TinyTeX" "$${APPDATA:-}/TinyTeX"; do \
	      [ -d "$$dir" ] || continue; \
	      TLMGR="$$(find "$$dir" \( -name tlmgr -o -name tlmgr.bat \) -type f 2>/dev/null | head -n 1)"; \
	      [ -n "$$TLMGR" ] && break; \
	    done; \
	  fi; \
	  if [ -n "$$TLMGR" ]; then \
	    echo "  installing TeX packages from tex-requirements.txt using $$TLMGR"; \
	    "$$TLMGR" update --self || true; \
	    grep -v '^#' tex-requirements.txt | grep -v '^$$' | xargs -r "$$TLMGR" install; \
	  else \
	    echo "  tlmgr not found; skipping TeX package install"; \
	  fi; \
	fi
	@echo "  dependencies installed"

pdf: ## Render the book.
	@$(MAKE) -C ../guide figures
	@$(QUARTO) render
