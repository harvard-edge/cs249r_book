# =============================================================================
# mlsysim/tutorial/Makefile — Build ISCA tutorial slides
# =============================================================================
# Usage:
#   make              Build the full tutorial deck (PDF)
#   make svgs         Convert all SVGs to PDFs
#   make slides       Build slides only (assumes SVGs already converted)
#   make clean        Remove build artifacts
#   make check        Verify all SVG source files exist
#
# Prerequisites:
#   - xelatex (TeX Live or MacTeX)
#   - inkscape (for SVG→PDF conversion)
#   OR
#   - rsvg-convert (alternative SVG→PDF, lighter than inkscape)
# =============================================================================

SHELL := /bin/bash

# Directories
SLIDES_DIR  := slides
SVG_DIR     := $(SLIDES_DIR)/images/svg
PDF_IMG_DIR := $(SLIDES_DIR)/images/pdf
BUILD_DIR   := _build

# Source files
PART1   := $(SLIDES_DIR)/tutorial_part1.tex
PART2   := $(SLIDES_DIR)/tutorial_part2.tex
RELATED := $(SLIDES_DIR)/related_work.tex

# All SVG sources (auto-detected)
SVGS := $(wildcard $(SVG_DIR)/*.svg)
PDFS := $(patsubst $(SVG_DIR)/%.svg,$(PDF_IMG_DIR)/%.pdf,$(SVGS))

# LaTeX engine (xelatex for font support, matching slides/Makefile)
LATEX := xelatex
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD_DIR)

# SVG converter (try inkscape first, fall back to rsvg-convert)
SVG2PDF := $(shell command -v inkscape >/dev/null 2>&1 && echo "inkscape --export-type=pdf" || echo "rsvg-convert -f pdf -o")

# =============================================================================
# TARGETS
# =============================================================================

.PHONY: all slides svgs clean check help

all: svgs slides  ## Build everything (SVGs → PDFs → slide deck)

help:  ## Show this help
	@echo "mlsysim Tutorial Build System"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

# --- SVG → PDF conversion ---
svgs: $(PDFS)  ## Convert all SVGs to PDF

$(PDF_IMG_DIR)/%.pdf: $(SVG_DIR)/%.svg | $(PDF_IMG_DIR)
	@echo "SVG → PDF: $<"
	@if command -v inkscape >/dev/null 2>&1; then \
		inkscape --export-type=pdf --export-filename=$@ $<; \
	elif command -v rsvg-convert >/dev/null 2>&1; then \
		rsvg-convert -f pdf -o $@ $<; \
	else \
		echo "ERROR: Neither inkscape nor rsvg-convert found. Install one."; \
		exit 1; \
	fi

$(PDF_IMG_DIR):
	@mkdir -p $@

# --- Slide compilation ---
slides: $(BUILD_DIR)/tutorial.pdf  ## Build the slide deck PDF

$(BUILD_DIR)/tutorial.pdf: $(PART1) $(PART2) $(RELATED) $(PDFS) | $(BUILD_DIR)
	@echo "Building tutorial slides..."
	@cd $(SLIDES_DIR) && $(LATEX) $(LATEX_FLAGS) tutorial_part1.tex
	@cd $(SLIDES_DIR) && $(LATEX) $(LATEX_FLAGS) tutorial_part1.tex
	@echo "Tutorial PDF: $(BUILD_DIR)/tutorial_part1.pdf"
	@echo ""
	@echo "NOTE: Part 1 and Part 2 are separate decks."
	@echo "  Part 1 (Parts 0-4): $(BUILD_DIR)/tutorial_part1.pdf"
	@echo "  Part 2 (Parts 5-9): Build with: cd $(SLIDES_DIR) && $(LATEX) $(LATEX_FLAGS) tutorial_part2.tex"

$(BUILD_DIR):
	@mkdir -p $@

# --- Utilities ---
check:  ## Verify SVG sources and dependencies
	@echo "SVG figures found: $(words $(SVGS))"
	@for svg in $(SVGS); do echo "  ✓ $$svg"; done
	@echo ""
	@echo "Checking tools..."
	@command -v $(LATEX) >/dev/null 2>&1 && echo "  ✓ $(LATEX)" || echo "  ✗ $(LATEX) NOT FOUND"
	@command -v inkscape >/dev/null 2>&1 && echo "  ✓ inkscape" || echo "  ⚠ inkscape not found (trying rsvg-convert)"
	@command -v rsvg-convert >/dev/null 2>&1 && echo "  ✓ rsvg-convert" || echo "  ⚠ rsvg-convert not found"
	@echo ""
	@echo "Slide sources:"
	@wc -l $(PART1) $(PART2) $(RELATED) 2>/dev/null || echo "  Some files missing"

clean:  ## Remove all build artifacts
	@rm -rf $(BUILD_DIR)
	@rm -f $(PDF_IMG_DIR)/*.pdf
	@rm -f $(SLIDES_DIR)/*.aux $(SLIDES_DIR)/*.log $(SLIDES_DIR)/*.nav
	@rm -f $(SLIDES_DIR)/*.out $(SLIDES_DIR)/*.snm $(SLIDES_DIR)/*.toc
	@rm -f $(SLIDES_DIR)/*.vrb $(SLIDES_DIR)/*.fls $(SLIDES_DIR)/*.fdb_latexmk
	@echo "Clean."
