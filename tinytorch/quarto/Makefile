# =============================================================================
# TinyTorch quarto build
# =============================================================================
# Two Quarto projects share this directory:
#   • the website (this directory's _quarto.yml, project type "website")
#   • the lab guide PDF (pdf/_quarto.yml, project type "book")
#
# Both consume the same chapter qmd files in modules/. Hand-authored SVG
# diagrams in assets/images/diagrams/ are the source-of-truth artifacts —
# edit them directly in Inkscape, Affinity, or a text editor (see
# tools/diagrams/STYLE.md for the visual convention). For the LaTeX
# build we convert each SVG to a sibling PDF via rsvg-convert; a small
# Lua filter (tools/diagrams/svg-to-pdf.lua) rewrites the qmd image
# refs from `.svg` → `.pdf` only for the PDF format, so the website
# keeps serving SVGs while the PDF embeds the pre-converted vector PDFs.
#
# Common entry points:
#   make            — alias for `make pdf`
#   make pdf        — convert SVGs to PDFs + render the lab guide
#   make site       — render the website (HTML)
#   make figures    — only convert SVGs to PDFs (no render)
#   make clean      — remove caches + generated diagram PDFs
#   make distclean  — clean + remove rendered guide & website outputs
#   make help       — list all targets
#   make check-tools — verify rsvg-convert + quarto are on PATH
#
# Dependencies:
#   • quarto         (https://quarto.org)
#   • rsvg-convert   (`brew install librsvg` on macOS,
#                     `apt-get install librsvg2-bin` on Debian/Ubuntu)
# =============================================================================

# Use bash + fail-fast pipelines so a broken sub-step actually fails the
# Make target rather than silently emitting a stale PDF.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

DIAGRAM_DIR := assets/images/diagrams

# Discover diagrams from the filesystem so adding a new SVG (drop a new
# file into $(DIAGRAM_DIR)) automatically gets picked up the next time
# `make figures` runs. We compute the .pdf list from the .svg list:
# SVGs are the source-of-truth, PDFs are the derived LaTeX-embed format.
SVGS := $(wildcard $(DIAGRAM_DIR)/*.svg)
PDFS := $(SVGS:.svg=.pdf)

# rsvg-convert flags:
#   -f pdf                emit PDF (otherwise it picks based on extension)
#   --keep-aspect-ratio   ensure no squashing if a viewBox is unusual
# We deliberately do NOT pass --width/--height — the SVG's own viewBox
# is the canvas size, and forcing a pixel size would defeat vector
# rendering downstream in LaTeX.
RSVG := rsvg-convert
RSVG_FLAGS := -f pdf --keep-aspect-ratio

QUARTO := quarto

# Treat every target below as phony except the actual file recipes.
.PHONY: all pdf site figures clean distclean help check-tools

all: pdf

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} \
	      /^[a-zA-Z_-]+:.*?##/ {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' \
	      $(MAKEFILE_LIST)

# ── SVG → PDF conversion ──────────────────────────────────────────────
# Pattern rule: Make's automatic dependency on $< means a touched SVG
# re-converts only the affected PDF — the build is incremental over
# diagram edits.
$(DIAGRAM_DIR)/%.pdf: $(DIAGRAM_DIR)/%.svg | check-tools
	@$(RSVG) $(RSVG_FLAGS) -o $@ $<
	@printf "  rsvg → %s\n" "$@"

figures: $(PDFS) ## Convert all SVG diagrams to PDF (for LaTeX embedding).

# ── Render targets ────────────────────────────────────────────────────
# `make pdf` is the canonical "rebuild the lab guide" command. It
# converts every SVG to PDF, then runs Quarto. The Lua filter in
# pdf/_quarto.yml rewrites .svg → .pdf for the PDF format only.
#
# Quarto's post-render cleanup step exits non-zero because the book
# project reads chapters at `../modules/...` — outside its own project
# root — so Quarto refuses to remove the auxiliary `_files/` directories
# left behind there. XeLaTeX has already written the PDF by that point,
# so we mirror the CI pattern (.github/workflows/tinytorch-build-pdfs.yml):
# run Quarto, capture its exit code, then *verify the PDF exists* and
# treat the cleanup error as benign only if it does. If the PDF is
# missing, fail loudly — that's a real build failure.
# Quarto writes the rendered book into pdf/_build/ (see `output-dir: _build`
# in pdf/_quarto.yml); we then copy it up to pdf/ so the publish workflow
# (.github/workflows/tinytorch-publish-live.yml) finds it at the stable
# path it expects.
PDF_BUILD := pdf/_build/TinyTorch-Guide.pdf
PDF_OUT   := pdf/TinyTorch-Guide.pdf

pdf: figures ## Build the TinyTorch Lab Guide PDF.
	@set +e; \
	  cd pdf && $(QUARTO) render; \
	  quarto_exit=$$?; \
	  cd ..; \
	  if [ ! -f "$(PDF_BUILD)" ]; then \
	    echo "ERROR: $(PDF_BUILD) was not produced (quarto exit=$$quarto_exit)"; \
	    exit 1; \
	  fi; \
	  cp "$(PDF_BUILD)" "$(PDF_OUT)"; \
	  if [ $$quarto_exit -ne 0 ]; then \
	    echo "  note: quarto exited $$quarto_exit on post-render cleanup (benign — see Makefile)"; \
	  fi; \
	  size=$$(du -h "$(PDF_OUT)" | cut -f1); \
	  echo "  built $(PDF_OUT) ($$size)"

site: ## Build the website (HTML).
	@$(QUARTO) render

# ── Maintenance ───────────────────────────────────────────────────────
clean: ## Remove Quarto caches + generated diagram PDFs.
	@rm -rf _tmp pdf/_build pdf/index_files modules/*_files
	@rm -f $(DIAGRAM_DIR)/*.pdf
	@echo "  cleaned caches and generated diagram PDFs"

distclean: clean ## Also remove rendered guide PDF and website output.
	@rm -f pdf/TinyTorch-Guide.pdf
	@rm -rf _site
	@echo "  removed rendered outputs"

check-tools: ## Verify rsvg-convert + quarto are on PATH.
	@command -v $(RSVG) >/dev/null 2>&1 || { \
	  echo "ERROR: $(RSVG) not found on PATH."; \
	  echo "  macOS:  brew install librsvg"; \
	  echo "  Debian: apt-get install librsvg2-bin"; \
	  exit 1; }
	@command -v $(QUARTO) >/dev/null 2>&1 || { \
	  echo "ERROR: $(QUARTO) not found on PATH."; \
	  echo "  Install from https://quarto.org"; \
	  exit 1; }
