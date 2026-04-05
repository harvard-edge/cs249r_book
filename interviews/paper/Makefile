.PHONY: all paper figures stats macros svgs layout-review clean watch

# ── Default: analyze → macros → plot → convert SVGs → compile ─
all: paper

# ── Step 1: Analyze corpus → stats JSON ──────────────────────
stats: corpus_stats.json

corpus_stats.json: ../vault/corpus.json ../vault/chains.json analyze_corpus.py
	python3 analyze_corpus.py

# ── Step 1b: Generate LaTeX macros from stats ────────────────
macros: macros.tex

macros.tex: corpus_stats.json ../vault/corpus.json generate_macros.py
	python3 generate_macros.py

# ── Step 2: Generate data figures from stats ─────────────────
# Only PDFs referenced by paper.tex via \\includegraphics.
DATA_FIGURES = fig-corpus-distribution.pdf \
               fig-format-balance.pdf \
               fig-zone-distribution.pdf \
               fig-zone-level-heatmap.pdf

figures: $(DATA_FIGURES)

$(DATA_FIGURES): corpus_stats.json generate_figures.py
	python3 generate_figures.py

# ── Step 3: Convert SVGs to PDFs ─────────────────────────────
# Applicability matrix: canonical data is applicability_matrix.yaml → app_matrix.tex.
app_matrix.tex: applicability_matrix.yaml generate_app_matrix_from_yaml.py
	python3 generate_app_matrix_from_yaml.py

SVG_SOURCES = fig-competency-model.svg \
              fig-four-axes.svg \
              fig-generation-pipeline.svg \
              fig-quality-pipeline.svg \
              fig-backward-design.svg \
              fig-prereq-subset.svg

SVG_PDFS = $(SVG_SOURCES:.svg=.pdf)

svgs: $(SVG_PDFS)

%.pdf: %.svg
	rsvg-convert -f pdf $< -o $@

# ── Step 4: Compile paper (depends on all figures + macros) ──
ALL_FIGURES = $(DATA_FIGURES) $(SVG_PDFS)

paper: paper.pdf

paper.pdf: paper.tex references.bib macros.tex app_matrix.tex $(ALL_FIGURES)
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
	fswatch -o paper.tex references.bib ../vault/corpus.json ../vault/chains.json *.svg | xargs -n1 -I{} make all
