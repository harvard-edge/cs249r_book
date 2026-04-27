.PHONY: all paper svgs layout-review clean distclean

# ── Default: convert SVGs → compile paper ────────────────────
all: paper

# ── Convert SVG figures to PDF ───────────────────────────────
SVG_SOURCES = $(wildcard figures/*.svg)
SVG_PDFS    = $(SVG_SOURCES:.svg=.pdf)

svgs: $(SVG_PDFS)

# Always regenerate figure PDFs from their SVG source so a stale committed
# PDF cannot mask a freshly edited SVG.
figures/%.pdf: figures/%.svg FORCE
	rm -f $@
	rsvg-convert -f pdf $< -o $@

FORCE:

# ── Compile paper ────────────────────────────────────────────
paper: paper.pdf

paper.pdf: paper.tex references.bib $(SVG_PDFS)
	pdflatex -interaction=nonstopmode paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode paper.tex
	pdflatex -interaction=nonstopmode paper.tex

# ── Rasterize each PDF page to PNG (layout / float debugging) ─
layout-review: paper.pdf
	mkdir -p layout-review
	pdftoppm -png -r 110 paper.pdf layout-review/p

# ── Clean auxiliary files ────────────────────────────────────
clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.toc
	rm -f paper.fls paper.fdb_latexmk paper.synctex.gz paper.lof paper.lot
	rm -rf layout-review

# ── Clean everything including PDF and generated figures ─────
distclean: clean
	rm -f paper.pdf
	rm -f $(SVG_PDFS)
