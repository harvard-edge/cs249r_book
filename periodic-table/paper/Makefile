.PHONY: all paper svgs hero clean distclean

# ── Default: convert SVGs → compile paper ────────────────────
all: paper

# ── Generate the periodic-table hero SVG from the YAML source of truth ──
# Re-runs whenever the YAML or the generator script changes.
figures/periodic_table_hero.svg: ../table.yml scripts/generate_periodic_svg.py
	python3 scripts/generate_periodic_svg.py

hero: figures/periodic_table_hero.svg

# ── Convert SVG figures to PDF ───────────────────────────────
# Include the generated hero SVG explicitly so it's part of SVG_SOURCES
# even before the wildcard sees it.
SVG_SOURCES = $(sort $(wildcard figures/*.svg) figures/periodic_table_hero.svg)
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