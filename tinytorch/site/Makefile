# TinyTorch Book Build Makefile
# Convenient shortcuts for building HTML and PDF versions

.PHONY: help html pdf clean install test restore-emoji

help:
	@echo "Tiny🔥Torch Build Commands"
	@echo "=========================="
	@echo ""
	@echo "  make html          - Build HTML version (website)"
	@echo "  make pdf           - Build PDF via XeLaTeX"
	@echo "  make clean         - Remove all build artifacts"
	@echo "  make restore-emoji - Restore original markdown files (if build interrupted)"
	@echo "  make install       - Install Python dependencies"
	@echo "  make test          - Test build configuration"
	@echo ""

html:
	@echo "🌐 Building HTML version..."
	@jupyter-book build .

pdf:
	@echo "🔥 Building Tiny🔥Torch PDF via XeLaTeX..."
	@echo ""
	@# Check dependencies
	@command -v jupyter-book >/dev/null 2>&1 || { echo "❌ Error: jupyter-book not installed. Run: pip install jupyter-book"; exit 1; }
	@command -v xelatex >/dev/null 2>&1 || { echo "❌ Error: xelatex not found. Install TeX Live from https://www.tug.org/texlive/"; exit 1; }
	@echo "✅ Dependencies OK"
	@echo ""
	@# Clean previous builds
	@echo "🧹 Cleaning previous builds..."
	@jupyter-book clean . --all 2>/dev/null || true
	@echo ""
	@# Build LaTeX files
	@echo "📚 Generating LaTeX files..."
	@jupyter-book build . --builder latex --config _config_pdf.yml
	@echo ""
	@# Remove emojis for clean professional PDF
	@echo "🧹 Removing emojis for clean PDF..."
	@python3 scripts/latex_postprocessor.py
	@cp _static/logos/fire-emoji.png _build/latex/
	@cp _static/logos/logo-mlsysbook.png _build/latex/
	@echo ""
	@# Compile PDF with XeLaTeX
	@echo "🔥 Compiling PDF with XeLaTeX..."
	@cd _build/latex && latexmk -xelatex -pdf -dvi- -ps- tinytorch-course.tex
	@echo ""
	@# Check if build succeeded
	@if [ -f "_build/latex/tinytorch-course.pdf" ]; then \
		PDF_SIZE=$$(du -h "_build/latex/tinytorch-course.pdf" | cut -f1); \
		echo "✅ PDF build complete!"; \
		echo "📄 Output: _build/latex/tinytorch-course.pdf"; \
		echo "📊 Size: $${PDF_SIZE}"; \
		echo ""; \
		echo "To view: open _build/latex/tinytorch-course.pdf"; \
	else \
		echo "❌ PDF compilation failed - check errors above"; \
		echo "📝 Check _build/latex/tinytorch-course.log for details"; \
		exit 1; \
	fi

clean:
	@echo "🧹 Cleaning build artifacts..."
	jupyter-book clean . --all
	rm -rf _build/

install:
	@echo "📦 Installing dependencies..."
	pip install -U pip
	pip install "jupyter-book<1.0"
	pip install -r requirements.txt

test:
	@echo "🧪 Testing build configuration..."
	jupyter-book config sphinx .
	@echo "✅ Configuration valid"

restore-emoji:
	@echo "🔄 Restoring original markdown files..."
	@python3 scripts/emoji_preprocessor.py --restore

# Default target
.DEFAULT_GOAL := help

