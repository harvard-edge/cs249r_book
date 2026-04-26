# MLSys interview paper: https://github.com/MLSysBook/MLSysBook
#
# default   make / make paper  — SVG→PDF, then LaTeX (works without interviews/vault/)
# with vault: make paper-full  — recompute stats, macros, data figures, then paper
# -----------------------------------------------------------------------------
.PHONY: all paper paper-full help paper-clean stats macros figures svgs layout-review clean watch

# Set when the vault is present: enables corpus_stats, generated macros, and data plot PDFs.
VAULT := $(and $(wildcard ../vault/corpus.json),$(wildcard ../vault/chains.json))

# ── Default: convert SVGs → PDF, compile (uses committed macros.tex + data fig PDFs) ─
all: paper

paper: paper.pdf

# ── With vault: regenerate everything that depends on the corpus, then the paper ─
ifdef VAULT
paper-full: stats macros figures svgs paper
	@:
else
paper-full:
	@echo "paper-full requires ../vault/corpus.json and ../vault/chains.json (use 'make paper' with committed macros and figures)"
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
corpus_stats.json: ../vault/corpus.json ../vault/chains.json scripts/analyze_corpus.py
	python3 scripts/analyze_corpus.py

macros.tex: corpus_stats.json ../vault/corpus.json scripts/generate_macros.py
	PYTHONPATH=../vault-cli/src python3 scripts/generate_macros.py

$(DATA_FIGURES): corpus_stats.json scripts/generate_figures.py
	python3 scripts/generate_figures.py
endif

# ── Step 3: Convert SVGs to PDFs ─────────────────────────────────────────────
SVG_SOURCES = $(wildcard figures/*.svg)

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
	  $(if $(VAULT),../vault/corpus.json ../vault/chains.json) | xargs -n1 -I{} make all

help:
	@echo "  make, make paper   — rsvg-convert figures/*.svg → .pdf, then pdflatex + bibtex"
	@echo "  make paper-full    — recompute stats/macros/data figures (needs interviews/vault/corpus.json + chains.json)"
	@echo "  make svgs          — refresh PDFs from SVGs only"
	@echo "  make paper-clean   — remove paper.pdf and LaTeX aux; keep figures and macros.tex"
	@echo "  make clean         — also remove generated macros, data figures, and corpus_stats.json"
