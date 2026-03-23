.PHONY: all paper figures stats clean watch

# ── Default: analyze → plot → compile ───────────────────────────
all: paper

# ── Step 1: Analyze corpus → stats JSON ─────────────────────────
stats: corpus_stats.json

corpus_stats.json: ../corpus.json ../chains.json analyze_corpus.py
	python3 analyze_corpus.py

# ── Step 2: Generate data figures from stats ────────────────────
FIGURES = fig-corpus-distribution.pdf \
          fig-format-balance.pdf \
          fig-depth-chain.pdf \
          fig-quality-summary.pdf

figures: $(FIGURES)

$(FIGURES): corpus_stats.json generate_figures.py
	python3 generate_figures.py

# ── Step 3: Compile paper (depends on figures) ──────────────────
paper: paper.pdf

paper.pdf: paper.tex references.bib $(FIGURES)
	pdflatex -interaction=nonstopmode paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode paper.tex
	pdflatex -interaction=nonstopmode paper.tex

# ── Clean all generated artifacts ───────────────────────────────
clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.pdf paper.toc
	rm -f corpus_stats.json
	rm -f $(FIGURES)

# ── Watch for changes and rebuild ───────────────────────────────
watch:
	fswatch -o paper.tex references.bib ../corpus.json ../chains.json | xargs -n1 -I{} make all
