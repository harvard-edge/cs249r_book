# TinyTorch Book Build Makefile
# Convenient shortcuts for building HTML and PDF versions

.PHONY: help html pdf pdf-simple clean install test

help:
	@echo "TinyTorch Book Build Commands"
	@echo "=============================="
	@echo ""
	@echo "  make html        - Build HTML version (default website)"
	@echo "  make pdf         - Build PDF via LaTeX (requires LaTeX installation)"
	@echo "  make pdf-simple  - Build PDF via HTML (no LaTeX required)"
	@echo "  make clean       - Remove all build artifacts"
	@echo "  make install     - Install Python dependencies"
	@echo "  make install-pdf - Install dependencies for PDF building"
	@echo "  make test        - Test build configuration"
	@echo ""
	@echo "Quick start for PDF:"
	@echo "  make install-pdf && make pdf-simple"
	@echo ""

html:
	@echo "🌐 Building HTML version..."
	@echo "📓 Preparing notebooks for launch buttons..."
	@./prepare_notebooks.sh || echo "⚠️  Notebook preparation skipped (tito not available)"
	@echo ""
	jupyter-book build .

pdf:
	@echo "📚 Building PDF via LaTeX..."
	@./build_pdf.sh

pdf-simple:
	@echo "📚 Building PDF via HTML..."
	@./build_pdf_simple.sh

clean:
	@echo "🧹 Cleaning build artifacts..."
	jupyter-book clean . --all
	rm -rf _build/

install:
	@echo "📦 Installing base dependencies..."
	pip install -U pip
	pip install "jupyter-book<1.0"
	pip install -r requirements.txt

install-pdf:
	@echo "📦 Installing PDF dependencies..."
	pip install -U pip
	pip install "jupyter-book<1.0" pyppeteer
	pip install -r requirements.txt

test:
	@echo "🧪 Testing build configuration..."
	jupyter-book config sphinx .
	@echo "✅ Configuration valid"

# Default target
.DEFAULT_GOAL := help

