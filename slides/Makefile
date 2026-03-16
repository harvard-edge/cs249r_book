# =============================================================================
# slides/Makefile — Build ML Systems lecture slides
# =============================================================================
# Usage:
#   make ch01              Build Chapter 1
#   make all               Build all chapters
#   make svgs              Convert all SVGs to PDFs
#   make clean             Remove build artifacts from source dirs
#
# Build outputs go to _build/vol1/ (source dirs stay clean).
# SVG→PDF conversion happens automatically before LaTeX compilation.
# =============================================================================

SHELL := /bin/bash

# Directories
BUILD_DIR := _build
VOL1_SRC  := vol1
VOL1_OUT  := $(BUILD_DIR)/vol1

# All chapter directories (auto-detected)
VOL1_CHAPTERS := $(sort $(notdir $(wildcard $(VOL1_SRC)/*_*)))

# LaTeX engine
LATEX := xelatex
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error

# SVG converter
SVG2PDF := rsvg-convert -f pdf -d 300 -p 300

# =============================================================================
# TARGETS
# =============================================================================

.PHONY: all clean svgs help

help:
	@echo "Usage:"
	@echo "  make ch01        Build Chapter 1 (Introduction)"
	@echo "  make ch02        Build Chapter 2 (ML Systems)"
	@echo "  make all         Build all chapters"
	@echo "  make check       Build all + fail if overflows > 10pt"
	@echo "  make svgs        Convert all SVGs to PDFs"
	@echo "  make clean       Remove build artifacts from source dirs"
	@echo ""
	@echo "Available chapters:"
	@for ch in $(VOL1_CHAPTERS); do echo "  $$ch"; done

all: svgs $(addprefix build-,$(VOL1_CHAPTERS))

# =============================================================================
# SVG → PDF conversion (run before any LaTeX build)
# =============================================================================

svgs:
	@echo "Converting SVGs to PDFs..."
	@find $(VOL1_SRC) -name "*.svg" | while read svg; do \
		pdf="$${svg%.svg}.pdf"; \
		if [ ! -f "$$pdf" ] || [ "$$svg" -nt "$$pdf" ]; then \
			$(SVG2PDF) "$$svg" -o "$$pdf" && echo "  ✓ $$pdf" || echo "  ✗ $$svg"; \
		fi; \
	done
	@echo "Done."

# =============================================================================
# Per-chapter build rules
# =============================================================================

# Generic build rule for any chapter
build-%: svgs
	@mkdir -p $(VOL1_OUT)
	@ch_dir="$(VOL1_SRC)/$*"; \
	tex_file=$$(ls $$ch_dir/*.tex 2>/dev/null | head -1); \
	if [ -z "$$tex_file" ]; then \
		echo "No .tex file found in $$ch_dir"; exit 1; \
	fi; \
	tex_name=$$(basename $$tex_file .tex); \
	echo "Building $$tex_name..."; \
	cd $$ch_dir && \
	$(LATEX) $(LATEX_FLAGS) $$tex_name.tex > /dev/null 2>&1 && \
	$(LATEX) $(LATEX_FLAGS) $$tex_name.tex > /dev/null 2>&1 && \
	echo "  ✓ $$tex_name.pdf (pass 2)" || \
	{ echo "  ✗ Build failed. Check $$ch_dir/$$tex_name.log"; exit 1; }; \
	cd - > /dev/null; \
	cp $$ch_dir/$$tex_name.pdf $(VOL1_OUT)/$$tex_name.pdf; \
	echo "  → $(VOL1_OUT)/$$tex_name.pdf"; \
	echo ""; \
	overfull=$$(grep -c "Overfull" $$ch_dir/$$tex_name.log 2>/dev/null || echo 0); \
	underfull=$$(grep -c "Underfull" $$ch_dir/$$tex_name.log 2>/dev/null || echo 0); \
	if [ "$$overfull" -gt 0 ]; then \
		echo "  ⚠ $$overfull overfull boxes:"; \
		grep "Overfull" $$ch_dir/$$tex_name.log | sed 's/^/    /' | head -10; \
		echo ""; \
	fi; \
	if [ "$$overfull" -eq 0 ] && [ "$$underfull" -eq 0 ]; then \
		echo "  ✓ No overflow warnings"; \
	fi; \
	cd $$ch_dir && rm -f *.aux *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.synctex.gz $$tex_name.pdf && cd - > /dev/null; \
	if [ "$$overfull" -gt 5 ]; then \
		echo "  ⛔ Too many overflows ($$overfull) — review slide content"; \
	fi

# Convenience aliases: ch01, ch02, ..., ch16
ch01: build-01_introduction
ch02: build-02_ml_systems
ch03: build-03_ml_workflow
ch04: build-04_data_engineering
ch05: build-05_nn_computation
ch06: build-06_nn_architectures
ch07: build-07_frameworks
ch08: build-08_training
ch09: build-09_data_selection
ch10: build-10_model_compression
ch11: build-11_hw_acceleration
ch12: build-12_benchmarking
ch13: build-13_model_serving
ch14: build-14_ml_ops
ch15: build-15_responsible_engr
ch16: build-16_conclusion

# =============================================================================
# Clean build artifacts from source directories (keep them pristine)
# =============================================================================

clean:
	@echo "Cleaning build artifacts from source directories..."
	@find $(VOL1_SRC) -type f \( \
		-name "*.aux" -o -name "*.log" -o -name "*.nav" -o \
		-name "*.out" -o -name "*.snm" -o -name "*.toc" -o \
		-name "*.vrb" -o -name "*.fls" -o -name "*.fdb_latexmk" -o \
		-name "*.synctex.gz" -o -name "*.bbl" -o -name "*.blg" \
	\) -delete
	@find $(VOL1_SRC) -name "*.pdf" ! -path "*/images/*" -delete
	@echo "Done. Source directories are clean."

# =============================================================================
# Quality check: build + fail on significant overflows (> 10pt)
# =============================================================================

MAX_OVERFLOW := 10

check: svgs
	@echo "Building with overflow check (max $(MAX_OVERFLOW)pt)..."
	@fail=0; \
	for ch_dir in $(VOL1_SRC)/*_*/; do \
		tex_file=$$(ls $$ch_dir/*.tex 2>/dev/null | head -1); \
		[ -z "$$tex_file" ] && continue; \
		tex_name=$$(basename $$tex_file .tex); \
		cd $$ch_dir && \
		$(LATEX) $(LATEX_FLAGS) $$tex_name.tex > /dev/null 2>&1 && \
		$(LATEX) $(LATEX_FLAGS) $$tex_name.tex > /dev/null 2>&1; \
		bad=$$(grep "Overfull" $$tex_name.log 2>/dev/null | \
			sed 's/.*(\([0-9]*\)\..*/\1/' | \
			awk '{ if ($$1 > $(MAX_OVERFLOW)) print }' | wc -l | tr -d ' '); \
		if [ "$$bad" -gt 0 ]; then \
			echo "  ✗ $$tex_name: $$bad overflows > $(MAX_OVERFLOW)pt"; \
			grep "Overfull" $$tex_name.log | sed 's/^/    /'; \
			fail=1; \
		else \
			echo "  ✓ $$tex_name"; \
		fi; \
		rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.synctex.gz $$tex_name.pdf; \
		cd - > /dev/null; \
	done; \
	if [ "$$fail" -eq 1 ]; then \
		echo ""; echo "⛔ FAILED: Some slides have overflows > $(MAX_OVERFLOW)pt"; exit 1; \
	else \
		echo ""; echo "✓ All slides pass overflow check"; \
	fi
