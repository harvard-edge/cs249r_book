.PHONY: all paper svgs values check-values validate-anchors layout-review clean distclean FORCE

# ── Default: convert SVGs → compile paper ────────────────────
# The default build needs only LaTeX and rsvg-convert. It reads the committed
# generated/*.tex files and figure SVGs; it never runs Python. Regenerate them
# explicitly with `make values` after any mlsysim change, then commit.
all: paper

# ── Computed values and data figures (explicit, needs uv) ────
# Runs against the in-repo mlsysim package (PYTHONPATH), never a pip install.
# pytest, marimo, scipy, and ortools are only for the test-count step, which
# collects tests/ (optional backends change the collected count).
MLSYSIM_ROOT = ..
UV_RUN = uv run --no-project --python 3.11 \
	--with "pint>=0.24.4" --with "pydantic>=2.10.5" --with "numpy>=2.0" \
	--with typer --with rich --with pyyaml --with matplotlib \
	--with pytest --with marimo --with scipy --with ortools
GENERATOR = paper/scripts/generate_paper_values.py

values:
	cd $(MLSYSIM_ROOT) && PYTHONPATH=. $(UV_RUN) python $(GENERATOR)

# Recompute every value and fail if it differs from generated/values.json.
check-values:
	cd $(MLSYSIM_ROOT) && PYTHONPATH=. $(UV_RUN) python $(GENERATOR) --check

# Compare the seven validation anchors with their published values.
validate-anchors:
	cd $(MLSYSIM_ROOT) && PYTHONPATH=. $(UV_RUN) python paper/scripts/validate_anchors.py

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
GENERATED_TEX = $(wildcard generated/*.tex)

paper: paper.pdf

paper.pdf: paper.tex references.bib $(GENERATED_TEX) $(SVG_PDFS)
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
	rm -rf layout-review

# ── Clean everything including PDF and generated figures ─────
distclean: clean
	rm -f paper.pdf
	rm -f $(SVG_PDFS)
