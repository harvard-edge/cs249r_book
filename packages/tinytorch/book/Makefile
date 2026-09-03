# =============================================================================
# TinyTorch Monograph Build Makefile
# =============================================================================

.PHONY: all pdf clean preview

all: pdf

pdf:
	@echo "Rendering TinyTorch Monograph PDF with Quarto & LuaLaTeX..."
	quarto render --to pdf

clean:
	@echo "Cleaning build artifacts..."
	rm -rf _build .quarto index_files index.tex index.log

preview: pdf
	@if command -v open > /dev/null; then \
		open _build/TinyTorch-Book.pdf; \
	fi
