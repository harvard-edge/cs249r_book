# =============================================================================
# slides/Makefile — Build ML Systems lecture slides
# =============================================================================
# Usage:
#   make ch01              Build Chapter 1 PDF
#   make all               Build all chapters (PDF)
#   make pptx              Build all PPTX from PDFs
#   make pptx-vol1         Build Vol1 PPTX only
#   make pptx-ch01         Build Chapter 1 PPTX only
#   make svgs              Convert all SVGs to PDFs
#   make clean             Remove build artifacts from source dirs
#
# Build outputs go to _build/vol{1,2}/ (source dirs stay clean).
# PPTX outputs go to _build/vol{1,2}/ alongside PDFs.
# SVG→PDF conversion happens automatically before LaTeX compilation.
# =============================================================================

SHELL := /bin/bash

# Directories
BUILD_DIR := _build
VOL1_SRC  := vol1
VOL1_OUT  := $(BUILD_DIR)/vol1
VOL2_SRC  := vol2
VOL2_OUT  := $(BUILD_DIR)/vol2
SCRIPTS   := scripts

# All chapter directories (auto-detected)
VOL1_CHAPTERS := $(sort $(notdir $(wildcard $(VOL1_SRC)/*_*)))
VOL2_CHAPTERS := $(sort $(notdir $(wildcard $(VOL2_SRC)/*_*)))

# LaTeX engine
LATEX := xelatex
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error

# SVG converter
SVG2PDF := inkscape --export-type=pdf

# PPTX converter
PDF2PPTX := python3 $(SCRIPTS)/pdf2pptx.py
PPTX_DPI := 300

# =============================================================================
# TARGETS
# =============================================================================

.PHONY: all clean svgs help pptx pptx-vol1 pptx-vol2

help:
	@echo "Usage:"
	@echo "  make ch01        Build Vol1 Chapter 1 (PDF)"
	@echo "  make v2ch01      Build Vol2 Chapter 1 (PDF)"
	@echo "  make vol1        Build all Vol1 chapters (PDF)"
	@echo "  make vol2        Build all Vol2 chapters (PDF)"
	@echo "  make all         Build all chapters (PDF, both volumes)"
	@echo "  make pptx        Build all PPTX from PDFs (both volumes)"
	@echo "  make pptx-vol1   Build Vol1 PPTX only"
	@echo "  make pptx-vol2   Build Vol2 PPTX only"
	@echo "  make pptx-ch01   Build Chapter 1 PPTX only"
	@echo "  make check       Build all + fail if overflows > 10pt"
	@echo "  make svgs        Convert all SVGs to PDFs"
	@echo "  make clean       Remove build artifacts from source dirs"
	@echo ""
	@echo "Volume I chapters:"
	@for ch in $(VOL1_CHAPTERS); do echo "  $$ch"; done
	@echo ""
	@echo "Volume II chapters:"
	@for ch in $(VOL2_CHAPTERS); do echo "  $$ch"; done

vol1: svgs $(addprefix build-,$(VOL1_CHAPTERS))
vol2: v2ch01 v2ch02 v2ch03 v2ch04 v2ch05 v2ch06 v2ch07 v2ch08 v2ch09 v2ch10 v2ch11 v2ch12 v2ch13 v2ch14 v2ch15 v2ch16 v2ch17
all: vol1 vol2

# =============================================================================
# SVG → PDF conversion (run before any LaTeX build)
# =============================================================================

svgs:
	@echo "Converting SVGs to PDFs..."
	@find $(VOL1_SRC) $(VOL2_SRC) -name "*.svg" 2>/dev/null | while read svg; do \
		pdf="$${svg%.svg}.pdf"; \
		if [ ! -f "$$pdf" ] || [ "$$svg" -nt "$$pdf" ]; then \
			$(SVG2PDF) --export-filename="$$pdf" "$$svg" 2>/dev/null && echo "  ✓ $$pdf" || echo "  ✗ $$svg"; \
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
# Volume II: Per-chapter build rules
# =============================================================================

# Helper function for building any chapter from any volume source dir
# Usage: $(call build-chapter,SOURCE_DIR,OUTPUT_DIR,CHAPTER_DIR)
define build-chapter-impl
	@mkdir -p $(2)
	@ch_dir="$(1)/$(3)"; \
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
	cp $$ch_dir/$$tex_name.pdf $(2)/$$tex_name.pdf; \
	echo "  → $(2)/$$tex_name.pdf"; \
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
endef

# Convenience aliases: v2ch01, v2ch02, ..., v2ch17
# Each target explicitly calls the build function with vol2 paths
v2ch01: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),01_introduction)
v2ch02: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),02_compute_infrastructure)
v2ch03: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),03_network_fabrics)
v2ch04: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),04_data_storage)
v2ch05: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),05_distributed_training)
v2ch06: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),06_collective_communication)
v2ch07: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),07_fault_tolerance)
v2ch08: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),08_fleet_orchestration)
v2ch09: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),09_inference)
v2ch10: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),10_performance_engineering)
v2ch11: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),11_edge_intelligence)
v2ch12: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),12_ops_scale)
v2ch13: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),13_security_privacy)
v2ch14: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),14_robust_ai)
v2ch15: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),15_sustainable_ai)
v2ch16: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),16_responsible_ai)
v2ch17: svgs ; $(call build-chapter-impl,$(VOL2_SRC),$(VOL2_OUT),17_conclusion)

# =============================================================================
# PPTX conversion (PDF → PowerPoint, image-based)
# =============================================================================

pptx: pptx-vol1 pptx-vol2

pptx-vol1: vol1
	@echo "Converting Vol1 PDFs to PPTX..."
	@$(PDF2PPTX) $(VOL1_OUT)/*.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)

pptx-vol2: vol2
	@echo "Converting Vol2 PDFs to PPTX..."
	@$(PDF2PPTX) $(VOL2_OUT)/*.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)

# Per-chapter PPTX: pptx-ch01, pptx-ch02, ..., pptx-v2ch01, ...
pptx-ch01: ch01 ; @$(PDF2PPTX) $(VOL1_OUT)/01_introduction.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)
pptx-ch02: ch02 ; @$(PDF2PPTX) $(VOL1_OUT)/02_ml_systems.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)
pptx-ch03: ch03 ; @$(PDF2PPTX) $(VOL1_OUT)/03_ml_workflow.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)
pptx-ch04: ch04 ; @$(PDF2PPTX) $(VOL1_OUT)/04_data_engineering.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)
pptx-ch05: ch05 ; @$(PDF2PPTX) $(VOL1_OUT)/05_nn_computation.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)
pptx-ch06: ch06 ; @$(PDF2PPTX) $(VOL1_OUT)/06_nn_architectures.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)
pptx-ch07: ch07 ; @$(PDF2PPTX) $(VOL1_OUT)/07_frameworks.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)
pptx-ch08: ch08 ; @$(PDF2PPTX) $(VOL1_OUT)/08_training.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)
pptx-ch09: ch09 ; @$(PDF2PPTX) $(VOL1_OUT)/09_data_selection.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)
pptx-ch10: ch10 ; @$(PDF2PPTX) $(VOL1_OUT)/10_model_compression.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)
pptx-ch11: ch11 ; @$(PDF2PPTX) $(VOL1_OUT)/11_hw_acceleration.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)
pptx-ch12: ch12 ; @$(PDF2PPTX) $(VOL1_OUT)/12_benchmarking.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)
pptx-ch13: ch13 ; @$(PDF2PPTX) $(VOL1_OUT)/13_model_serving.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)
pptx-ch14: ch14 ; @$(PDF2PPTX) $(VOL1_OUT)/14_ml_ops.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)
pptx-ch15: ch15 ; @$(PDF2PPTX) $(VOL1_OUT)/15_responsible_engr.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)
pptx-ch16: ch16 ; @$(PDF2PPTX) $(VOL1_OUT)/16_conclusion.pdf --output-dir $(VOL1_OUT) --dpi $(PPTX_DPI)

pptx-v2ch01: v2ch01 ; @$(PDF2PPTX) $(VOL2_OUT)/01_introduction.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch02: v2ch02 ; @$(PDF2PPTX) $(VOL2_OUT)/02_compute_infrastructure.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch03: v2ch03 ; @$(PDF2PPTX) $(VOL2_OUT)/03_network_fabrics.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch04: v2ch04 ; @$(PDF2PPTX) $(VOL2_OUT)/04_data_storage.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch05: v2ch05 ; @$(PDF2PPTX) $(VOL2_OUT)/05_distributed_training.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch06: v2ch06 ; @$(PDF2PPTX) $(VOL2_OUT)/06_collective_communication.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch07: v2ch07 ; @$(PDF2PPTX) $(VOL2_OUT)/07_fault_tolerance.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch08: v2ch08 ; @$(PDF2PPTX) $(VOL2_OUT)/08_fleet_orchestration.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch09: v2ch09 ; @$(PDF2PPTX) $(VOL2_OUT)/09_inference.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch10: v2ch10 ; @$(PDF2PPTX) $(VOL2_OUT)/10_performance_engineering.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch11: v2ch11 ; @$(PDF2PPTX) $(VOL2_OUT)/11_edge_intelligence.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch12: v2ch12 ; @$(PDF2PPTX) $(VOL2_OUT)/12_ops_scale.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch13: v2ch13 ; @$(PDF2PPTX) $(VOL2_OUT)/13_security_privacy.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch14: v2ch14 ; @$(PDF2PPTX) $(VOL2_OUT)/14_robust_ai.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch15: v2ch15 ; @$(PDF2PPTX) $(VOL2_OUT)/15_sustainable_ai.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch16: v2ch16 ; @$(PDF2PPTX) $(VOL2_OUT)/16_responsible_ai.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)
pptx-v2ch17: v2ch17 ; @$(PDF2PPTX) $(VOL2_OUT)/17_conclusion.pdf --output-dir $(VOL2_OUT) --dpi $(PPTX_DPI)

# =============================================================================
# Clean build artifacts from source directories (keep them pristine)
# =============================================================================

clean:
	@echo "Cleaning build artifacts from source directories..."
	@find $(VOL1_SRC) $(VOL2_SRC) -type f \( \
		-name "*.aux" -o -name "*.log" -o -name "*.nav" -o \
		-name "*.out" -o -name "*.snm" -o -name "*.toc" -o \
		-name "*.vrb" -o -name "*.fls" -o -name "*.fdb_latexmk" -o \
		-name "*.synctex.gz" -o -name "*.bbl" -o -name "*.blg" \
	\) -delete 2>/dev/null
	@find $(VOL1_SRC) $(VOL2_SRC) -name "*.pdf" ! -path "*/images/*" -delete 2>/dev/null
	@rm -rf $(BUILD_DIR)
	@echo "Done. Source directories and build output are clean."

# =============================================================================
# Quality check: build + fail on significant overflows (> 10pt)
# =============================================================================

MAX_OVERFLOW := 10

check: svgs
	@echo "Building with overflow check (max $(MAX_OVERFLOW)pt)..."
	@fail=0; \
	for ch_dir in $(VOL1_SRC)/*_*/ $(VOL2_SRC)/*_*/; do \
		[ -d "$$ch_dir" ] || continue; \
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
