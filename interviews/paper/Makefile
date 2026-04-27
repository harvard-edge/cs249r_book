# MLSys interview paper: https://github.com/MLSysBook/MLSysBook
#
# default   make / make paper  — SVG→PDF, then LaTeX (works without interviews/vault/)
# with vault: make paper-full  — recompute stats, macros, data figures, then paper
# -----------------------------------------------------------------------------
.PHONY: all paper paper-full help paper-clean stats macros figures svgs layout-review clean watch

# paper stats: `analyze_corpus` prefers ../vault/vault.db (D1/ship_d1 lineage), else generated corpus.json
PAPER_CORPUS := $(firstword $(wildcard ../staffml/src/data/corpus.json) $(wildcard ../vault/corpus.json))
PAPER_VAULT_DB := $(wildcard ../vault/vault.db)
# Rebuild from DB alone, or from JSON, or both (make prefers the DB wildcard first when both exist)
PAPER_DATA_OK := $(or $(PAPER_VAULT_DB),$(PAPER_CORPUS))
VAULT := $(PAPER_DATA_OK)

# ── Default: convert SVGs → PDF, compile (uses committed macros.tex + data fig PDFs) ─
all: paper

paper: paper.pdf

# ── With vault: regenerate everything that depends on the corpus, then the paper ─
ifdef VAULT
paper-full: stats macros figures svgs paper
	@:
else
paper-full:
	@echo "paper-full needs a stats input: ../vault/vault.db (after vault build) and/or generated corpus.json, plus release tooling for macros."
	@echo "  (1) vault build  → interviews/vault/vault.db  (optional: vault build --legacy-json for JSON)"
	@echo "  (2) Or: make paper  using committed macros + corpus_stats + figures (no local build)"
	@false
endif

# ── Step 1: Analyze corpus → stats JSON (vault only) ─────────────────────────
stats: corpus_stats.json

# ── Step 1b: LaTeX macros from stats (vault only) ────────────────────────────
macros: macros.tex

# ── Step 2: Data figures from stats (vault only) ──────────────────────────────
# Only PDFs referenced by paper.tex via \includegraphics (see scripts/generate_figures.py).
DATA_FIGURES = figures/fig-corpus-distribution.pdf \
               figures/fig-format-balance.pdf \
               figures/fig-zone-distribution.pdf \
               figures/fig-zone-level-heatmap.pdf

figures: $(DATA_FIGURES)

ifdef VAULT
# Chains for JSON path live in ../vault/chains.json; when using only vault.db, chain rows are inside the DB
corpus_stats.json: $(PAPER_CORPUS) $(PAPER_VAULT_DB) scripts/analyze_corpus.py
	python3 scripts/analyze_corpus.py

# macros: SQL over vault.db via export-paper; corpus.json not read here (only stats/figures do)
macros.tex: corpus_stats.json scripts/generate_macros.py
	PYTHONPATH=../vault-cli/src python3 scripts/generate_macros.py
endif

# Data figures regenerate whenever corpus_stats.json or the script changes,
# regardless of VAULT — corpus_stats.json is committed, so this rule fires
# even on a clean checkout. Side effect: every `make paper` builds these
# figures from source rather than relying on a possibly-stale committed PDF.
$(DATA_FIGURES): corpus_stats.json scripts/generate_figures.py
	python3 scripts/generate_figures.py

# ── Step 3: Convert SVGs to PDFs ─────────────────────────────────────────────
# Filter out data-figure stems: those PDFs are produced by generate_figures.py,
# and a stale .svg of the same name would otherwise win via the pattern rule
# below (with FORCE) and overwrite the matplotlib output on every build.
DATA_FIGURE_SVG_SHADOWS = $(DATA_FIGURES:.pdf=.svg)
SVG_SOURCES = $(filter-out $(DATA_FIGURE_SVG_SHADOWS),$(wildcard figures/*.svg))

SVG_PDFS = $(SVG_SOURCES:.svg=.pdf)

svgs: $(SVG_PDFS)

figures/%.pdf: figures/%.svg FORCE
	rm -f $@
	rsvg-convert -f pdf $< -o $@

FORCE:

# ── Step 4: Compile paper ────────────────────────────────────────────────────
ALL_FIGURES = $(DATA_FIGURES) $(SVG_PDFS)

paper.pdf: paper.tex references.bib macros.tex tables/app_matrix.tex $(ALL_FIGURES)
	pdflatex --shell-escape -interaction=nonstopmode paper.tex
	bibtex paper
	pdflatex --shell-escape -interaction=nonstopmode paper.tex
	pdflatex --shell-escape -interaction=nonstopmode paper.tex

# LaTeX aux only (keeps macros.tex, figures, and corpus_stats.json)
paper-clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.pdf paper.toc
	rm -rf layout-review

# Rasterize each PDF page to PNG (layout / float debugging)
layout-review: paper.pdf
	mkdir -p layout-review
	pdftoppm -png -r 110 paper.pdf layout-review/p

# Remove generated build artifacts, including stats-backed outputs (see README if you use vault)
clean: paper-clean
	rm -f corpus_stats.json macros.tex
	rm -f $(DATA_FIGURES) $(SVG_PDFS)
	rm -rf layout-review

# Rebuild on changes (path to corpus optional — omit if you do not have a local vault)
watch:
	fswatch -o paper.tex references.bib tables/app_matrix.tex figures/*.svg \
	  $(if $(VAULT),$(PAPER_CORPUS) $(PAPER_VAULT_DB) ../vault/chains.json) | xargs -n1 -I{} make all

help:
	@echo "  make, make paper   — rsvg-convert figures/*.svg → .pdf, then pdflatex + bibtex"
	@echo "  make paper-full    — recompute stats/macros/data figures (../vault/vault.db or corpus JSON + chains.json)"
	@echo "  Data figures: prefer vault.db (D1/ship_d1 lineage) → analyze_corpus → corpus_stats → generate_figures"
	@echo "  Text macros:    vault export-paper (release vault.db) → macros.tex (same build = matching counts)"
	@echo "  make svgs          — refresh PDFs from SVGs only"
	@echo "  make paper-clean   — remove paper.pdf and LaTeX aux; keep figures and macros.tex"
	@echo "  make clean         — also remove generated macros, data figures, and corpus_stats.json"
