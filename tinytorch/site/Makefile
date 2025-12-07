# TinyTorch Book Build Makefile
# Convenient shortcuts for building HTML and PDF versions

.PHONY: help html pdf clean install test

help:
	@echo "Tiny🔥Torch Build Commands"
	@echo "=========================="
	@echo ""
	@echo "  make html        - Build HTML version (website)"
	@echo "  make pdf         - Build PDF via LuaLaTeX"
	@echo "  make clean       - Remove all build artifacts"
	@echo "  make install     - Install Python dependencies"
	@echo "  make test        - Test build configuration"
	@echo ""

html:
	@echo "🌐 Building HTML version..."
	@jupyter-book build .

pdf:
	@echo "🔥 Building Tiny🔥Torch PDF via LuaLaTeX..."
	@echo ""
	@# Check dependencies
	@command -v jupyter-book >/dev/null 2>&1 || { echo "❌ Error: jupyter-book not installed. Run: pip install jupyter-book"; exit 1; }
	@command -v lualatex >/dev/null 2>&1 || { echo "❌ Error: lualatex not found. Install TeX Live from https://www.tug.org/texlive/"; exit 1; }
	@echo "✅ Dependencies OK"
	@echo ""
	@# Clean previous builds
	@echo "🧹 Cleaning previous builds..."
	@jupyter-book clean . --all 2>/dev/null || true
	@echo ""
	@# Build PDF
	@echo "📚 Building LaTeX/PDF (this may take a few minutes)..."
	@echo "   Using LuaLaTeX engine with Tiny🔥Torch branding"
	@jupyter-book build . --builder pdflatex --config _config_pdf.yml
	@# Check if build succeeded
	@if [ -f "_build/latex/tinytorch-course.pdf" ]; then \
		PDF_SIZE=$$(du -h "_build/latex/tinytorch-course.pdf" | cut -f1); \
		echo ""; \
		echo "✅ PDF build complete!"; \
		echo "📄 Output: _build/latex/tinytorch-course.pdf"; \
		echo "📊 Size: $${PDF_SIZE}"; \
		echo ""; \
		echo "To view: open _build/latex/tinytorch-course.pdf"; \
	else \
		echo ""; \
		echo "❌ PDF build failed - check errors above"; \
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

# Default target
.DEFAULT_GOAL := help

