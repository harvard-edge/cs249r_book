PYTHON ?= python3
LATEX ?= pdflatex
BIBTEX ?= bibtex

PAPER := paper
BUILD_DIR := build
OUTPUT_DIR := output/pdf
OUTPUT_PDF := $(OUTPUT_DIR)/mlperf-edu-review-draft.pdf
REGISTRY_INPUTS := $(shell find ../registry -type f -name '*.yaml' 2>/dev/null) ../datasets.yaml
REFERENCE_INPUTS := ../provisional_results/index.json $(shell find ../provisional_results -type f -name '*.json' ! -name 'index.json' 2>/dev/null)
GENERATED := generated_registry.tex

.PHONY: all check clean FORCE

all: $(OUTPUT_PDF)

$(GENERATED): FORCE generate_registry_snapshot.py $(REGISTRY_INPUTS) $(REFERENCE_INPUTS)
	PYTHONPATH=../src $(PYTHON) generate_registry_snapshot.py

FORCE:

$(OUTPUT_PDF): $(PAPER).tex refs.bib $(GENERATED) check_paper_pdf.py
	mkdir -p $(BUILD_DIR) $(OUTPUT_DIR)
	$(LATEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(BUILD_DIR) $(PAPER).tex
	$(BIBTEX) $(BUILD_DIR)/$(PAPER)
	$(LATEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(BUILD_DIR) $(PAPER).tex
	$(LATEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(BUILD_DIR) $(PAPER).tex
	cp $(BUILD_DIR)/$(PAPER).pdf $(OUTPUT_PDF)
	$(PYTHON) check_paper_pdf.py $(OUTPUT_PDF) $(BUILD_DIR)/$(PAPER).log

check: $(OUTPUT_PDF)
	$(PYTHON) check_paper_pdf.py $(OUTPUT_PDF) $(BUILD_DIR)/$(PAPER).log

clean:
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR)
