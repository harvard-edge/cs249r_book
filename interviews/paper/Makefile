.PHONY: all paper figures stats macros svgs layout-review clean watch

# ── Default: analyze → macros → plot → convert SVGs → compile ─
all: paper

# ── Step 1: Analyze corpus → stats JSON ──────────────────────
stats: corpus_stats.json

corpus_stats.json: ../vault/corpus.json ../vault/chains.json scripts/analyze_corpus.py
	python3 scripts/analyze_corpus.py

# ── Step 1b: Generate LaTeX macros from stats ────────────────
macros: macros.tex

macros.tex: corpus_stats.json ../vault/corpus.json scripts/generate_macros.py
	PYTHONPATH=../vault-cli/src python3 scripts/generate_macros.py

# ── Step 2: Generate data figures from stats ─────────────────
# Only PDFs referenced by paper.tex via \\includegraphics.
DATA_FIGURES = figures/fig-corpus-distribution.pdf \
               figures/fig-format-balance.pdf \
               figures/fig-zone-distribution.pdf \
               figures/fig-zone-level-heatmap.pdf

figures: $(DATA_FIGURES)

$(DATA_FIGURES): corpus_stats.json scripts/generate_figures.py
	python3 scripts/generate_figures.py

# ── Step 3: Convert SVGs to PDFs ─────────────────────────────
SVG_SOURCES = $(wildcard figures/*.svg)

SVG_PDFS = $(SVG_SOURCES:.svg=.pdf)

svgs: $(SVG_PDFS)

figures/%.pdf: figures/%.svg FORCE
	rm -f $@
	rsvg-convert -f pdf $< -o $@

FORCE:

# ── Step 4: Compile paper (depends on all figures + macros) ──
ALL_FIGURES = $(DATA_FIGURES) $(SVG_PDFS)

paper: paper.pdf

paper.pdf: paper.tex references.bib macros.tex tables/app_matrix.tex $(ALL_FIGURES)
	pdflatex --shell-escape -interaction=nonstopmode paper.tex
	bibtex paper
	pdflatex --shell-escape -interaction=nonstopmode paper.tex
	pdflatex --shell-escape -interaction=nonstopmode paper.tex

# ── Rasterize each PDF page to PNG (layout / float debugging) ─
layout-review: paper.pdf
	mkdir -p layout-review
	pdftoppm -png -r 110 paper.pdf layout-review/p

# ── Clean all generated artifacts ────────────────────────────
clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.pdf paper.toc
	rm -f corpus_stats.json macros.tex
	rm -f $(DATA_FIGURES) $(SVG_PDFS)
	rm -rf layout-review

# ── Watch for changes and rebuild ────────────────────────────
watch:
	fswatch -o paper.tex references.bib ../vault/corpus.json ../vault/chains.json figures/*.svg | xargs -n1 -I{} make all
