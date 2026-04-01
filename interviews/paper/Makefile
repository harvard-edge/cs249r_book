.PHONY: all paper figures stats svgs clean watch

# ── Default: analyze → plot → convert SVGs → compile ─────────
all: paper

# ── Step 1: Analyze corpus → stats JSON ──────────────────────
stats: corpus_stats.json

corpus_stats.json: ../vault/corpus.json ../vault/chains.json analyze_corpus.py
	python3 analyze_corpus.py

# ── Step 2: Generate data figures from stats ─────────────────
DATA_FIGURES = fig-corpus-distribution.pdf \
               fig-format-balance.pdf \
               fig-depth-chain.pdf \
               fig-quality-summary.pdf \
               fig-zone-distribution.pdf \
               fig-zone-level-heatmap.pdf

figures: $(DATA_FIGURES)

$(DATA_FIGURES): corpus_stats.json generate_figures.py
	python3 generate_figures.py

# ── Step 3: Convert SVGs to PDFs via inkscape ────────────────
SVG_SOURCES = fig-competency-model.svg \
              fig-four-axes.svg \
              fig-generation-pipeline.svg \
              fig-quality-pipeline.svg

SVG_PDFS = $(SVG_SOURCES:.svg=.pdf)

svgs: $(SVG_PDFS)

%.pdf: %.svg
	inkscape $< --export-type=pdf --export-filename=$@ 2>/dev/null

# ── Step 4: Compile paper (depends on all figures) ───────────
ALL_FIGURES = $(DATA_FIGURES) $(SVG_PDFS) fig-prereq-graph.pdf

paper: paper.pdf

paper.pdf: paper.tex references.bib $(ALL_FIGURES)
	pdflatex --shell-escape -interaction=nonstopmode paper.tex
	bibtex paper
	pdflatex --shell-escape -interaction=nonstopmode paper.tex
	pdflatex --shell-escape -interaction=nonstopmode paper.tex

# ── Clean all generated artifacts ────────────────────────────
clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.pdf paper.toc
	rm -f corpus_stats.json
	rm -f $(DATA_FIGURES) $(SVG_PDFS)

# ── Watch for changes and rebuild ────────────────────────────
watch:
	fswatch -o paper.tex references.bib ../vault/corpus.json ../vault/chains.json *.svg | xargs -n1 -I{} make all
