.PHONY: all paper svgs clean distclean

# ── Default: convert SVGs → compile paper ────────────────────
all: paper

# ── Convert SVG figures to PDF ───────────────────────────────
SVG_SOURCES = $(wildcard figures/*.svg)
SVG_PDFS    = $(SVG_SOURCES:.svg=.pdf)

svgs: $(SVG_PDFS)

figures/%.pdf: figures/%.svg
	rsvg-convert -f pdf $< -o $@

# ── Compile paper ────────────────────────────────────────────
paper: paper.pdf

paper.pdf: paper.tex references.bib $(SVG_PDFS)
	pdflatex -interaction=nonstopmode paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode paper.tex
	pdflatex -interaction=nonstopmode paper.tex

# ── Clean auxiliary files ────────────────────────────────────
clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.toc
	rm -f paper.fls paper.fdb_latexmk paper.synctex.gz paper.lof paper.lot

# ── Clean everything including PDF and generated figures ─────
distclean: clean
	rm -f paper.pdf
	rm -f $(SVG_PDFS)
