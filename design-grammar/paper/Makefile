.PHONY: all paper svgs catalog clean distclean

# ── Default: convert SVGs → compile paper ────────────────────
all: paper

# ── Generate the primitive-catalog SVG from the YAML source of truth ──
# Re-runs whenever the YAML or the generator script changes.
figures/primitive_catalog.svg: ../grammar.yml scripts/generate_primitive_catalog.py
	python3 scripts/generate_primitive_catalog.py

catalog: figures/primitive_catalog.svg

# ── Convert SVG figures to PDF ───────────────────────────────
# Include the generated primitive catalog SVG explicitly so it's part of SVG_SOURCES
# even before the wildcard sees it.
SVG_SOURCES = $(sort $(wildcard figures/*.svg) figures/primitive_catalog.svg)
SVG_PDFS    = $(SVG_SOURCES:.svg=.pdf)
TEX_SOURCES = paper.tex predict_paper.tex appendix-primitives.tex appendix-proofs.tex nomenclature.tex

svgs: $(SVG_PDFS)

figures/%.pdf: figures/%.svg
	rsvg-convert -f pdf $< -o $@

# ── Compile paper ────────────────────────────────────────────
paper: paper.pdf

paper.pdf: $(TEX_SOURCES) references.bib $(SVG_PDFS)
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
