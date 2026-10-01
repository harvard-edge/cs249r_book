.PHONY: all paper figures svgs tikz values check-values validate-benchmarks validate-anchors layout-review clean distclean FORCE

# ── Default: convert figures → compile paper ─────────────────
# The default build compiles TikZ diagrams and converts remaining SVGs to PDF,
# then compiles the paper. It reads the committed generated/*.tex files and figure
# sources; it never runs Python. Regenerate computed values explicitly with
# `make values` after any mlsysim change, then commit.
all: paper

# ── Computed values and data figures (explicit, needs uv) ────
# Runs against the in-repo mlsysim package (PYTHONPATH), never a pip install.
# pytest, marimo, scipy, and ortools are only for the test-count step, which
# collects tests/ (optional backends change the collected count).
MLSYSIM_ROOT = ..
UV_RUN = uv run --no-project --python 3.11 \
	--with "pint>=0.24.4" --with "pydantic>=2.10.5" --with "numpy>=2.0" \
	--with typer --with rich --with pyyaml --with matplotlib \
	--with pytest --with marimo --with scipy --with ortools --with plotly
GENERATOR = paper/scripts/generate_paper_values.py

values:
	cd $(MLSYSIM_ROOT) && PYTHONPATH=. $(UV_RUN) python $(GENERATOR)

# Recompute every value and fail if it differs from generated/values.json.
check-values:
	cd $(MLSYSIM_ROOT) && PYTHONPATH=. $(UV_RUN) python $(GENERATOR) --check

# Compare the seven published benchmarks with their reported values.
validate-benchmarks:
	cd $(MLSYSIM_ROOT) && PYTHONPATH=. $(UV_RUN) python paper/scripts/validate_benchmarks.py

validate-anchors: validate-benchmarks

# ── Figure generation (TikZ diagrams + SVG data charts) ───────
TIKZ_STEMS   = architecture-stack mlsysim-overview registry-composition solver-chaining system-anatomy
TIKZ_SOURCES = $(patsubst %,figures/tikz/%.tex,$(TIKZ_STEMS))
TIKZ_PDFS    = $(patsubst %,figures/%.pdf,$(TIKZ_STEMS))

SVG_ALL      = $(wildcard figures/*.svg)
SVG_TIKZ     = $(patsubst %,figures/%.svg,$(TIKZ_STEMS))
SVG_SOURCES  = $(filter-out $(SVG_TIKZ),$(SVG_ALL))
SVG_PDFS     = $(SVG_SOURCES:.svg=.pdf)

ALL_FIGURES  = $(TIKZ_PDFS) $(SVG_PDFS)

figures: $(ALL_FIGURES)
tikz: $(TIKZ_PDFS)
svgs: $(SVG_PDFS)

# Compile TikZ diagram to PDF
figures/%.pdf: figures/tikz/%.tex figures/tikz/glyphs.tex FORCE
	pdflatex -interaction=nonstopmode -output-directory figures/tikz $<
	cp figures/tikz/$*.pdf $@

# Convert non-TikZ SVG data figures to PDF
figures/%.pdf: figures/%.svg FORCE
	rm -f $@
	rsvg-convert -f pdf $< -o $@

FORCE:

# ── Compile paper ────────────────────────────────────────────
GENERATED_TEX = $(wildcard generated/*.tex)

paper: paper.pdf

paper.pdf: paper.tex references.bib $(GENERATED_TEX) $(ALL_FIGURES)
	pdflatex -interaction=nonstopmode paper.tex || true
	bibtex paper || true
	pdflatex -interaction=nonstopmode paper.tex || true
	pdflatex -interaction=nonstopmode paper.tex || true

# ── Rasterize each PDF page to PNG (layout / float debugging) ─
layout-review: paper.pdf
	mkdir -p layout-review
	pdftoppm -png -r 110 paper.pdf layout-review/p

# ── Clean auxiliary files ────────────────────────────────────
clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.toc
	rm -f paper.fls paper.fdb_latexmk paper.synctex.gz paper.lof paper.lot
	rm -f figures/tikz/*.aux figures/tikz/*.log figures/tikz/*.pdf
	rm -rf layout-review

# ── Clean everything including PDF and generated figures ─────
distclean: clean
	rm -f paper.pdf
	rm -f $(ALL_FIGURES)
