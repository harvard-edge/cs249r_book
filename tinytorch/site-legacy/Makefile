# TinyTorch Book Build Makefile
# Convenient shortcuts for building HTML and PDF versions

.PHONY: help html serve pdf paper downloads clean install test restore-emoji

help:
	@echo "Tiny🔥Torch Build Commands"
	@echo "=========================="
	@echo ""
	@echo "  make html          - Build HTML version (website)"
	@echo "  make serve         - Build and serve locally at http://localhost:8000"
	@echo "  make pdf           - Build PDF course guide via XeLaTeX"
	@echo "  make paper         - Build research paper via LuaLaTeX"
	@echo "  make clean         - Remove all build artifacts"
	@echo "  make test          - Test build configuration"
	@echo ""

html:
	@echo "🌐 Building HTML version..."
	@jupyter-book build .

serve: clean html
	@echo ""
	@echo "🚀 Starting local server at http://localhost:8000"
	@echo "   Press Ctrl+C to stop"
	@echo ""
	@cd _build/html && python3 -m http.server 8000

pdf:
	@echo "🔥 Building Tiny🔥Torch PDF via XeLaTeX..."
	@echo ""
	@# PDF build needs Mermaid CLI (for diagrams). Config uses npx; Node.js must be installed.
	@command -v npx >/dev/null 2>&1 || { echo "❌ Error: npx not found. PDF build requires Node.js (for Mermaid diagrams). Install from https://nodejs.org/ or run: brew install node"; exit 1; }
	@echo "✅ Mermaid CLI (via npx) available"
	@# Step 1: Generate LaTeX (matches CI workflow)
	@echo "📚 Generating LaTeX files..."
	jupyter-book build . --builder latex --config _config_pdf.yml --toc _toc_pdf.yml
	python3 scripts/latex_postprocessor.py
	cp _static/logos/fire-emoji.png _build/latex/
	cp _static/logos/logo-mlsysbook.png _build/latex/
	cp _static/images/tito.png _build/latex/
	@echo ""
	@# Step 2: Compile PDF with XeLaTeX (CI uses xu-cheng/latex-action with xelatex)
	@echo "🔥 Compiling PDF with XeLaTeX..."
	cd _build/latex && latexmk -xelatex -pdf -dvi- -ps- tinytorch-course.tex
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
	pip install "jupyter-book>=1.0.0,<2.0.0"
	pip install -r requirements.txt

test:
	@echo "🧪 Testing build configuration..."
	jupyter-book config sphinx .
	@echo "✅ Configuration valid"

restore-emoji:
	@echo "🔄 Restoring original markdown files..."
	@python3 scripts/emoji_preprocessor.py --restore

paper:
	@echo "📑 Building TinyTorch Research Paper via LuaLaTeX..."
	@echo ""
	@# Check dependencies
	@command -v lualatex >/dev/null 2>&1 || { echo "❌ Error: lualatex not found. Install TeX Live from https://www.tug.org/texlive/"; exit 1; }
	@echo "✅ Dependencies OK"
	@echo ""
	@# Build paper
	@cd ../paper && lualatex -interaction=nonstopmode paper.tex
	@cd ../paper && bibtex paper 2>/dev/null || true
	@cd ../paper && lualatex -interaction=nonstopmode paper.tex
	@cd ../paper && lualatex -interaction=nonstopmode paper.tex
	@echo ""
	@# Check if build succeeded
	@if [ -f "../paper/paper.pdf" ]; then \
		PDF_SIZE=$$(du -h "../paper/paper.pdf" | cut -f1); \
		echo "✅ Paper build complete!"; \
		echo "📄 Output: ../paper/paper.pdf"; \
		echo "📊 Size: $${PDF_SIZE}"; \
	else \
		echo "❌ Paper compilation failed - check errors above"; \
		exit 1; \
	fi

downloads: pdf paper
	@echo ""
	@echo "📦 Copying PDFs to assets/downloads/..."
	@mkdir -p _build/html/assets/downloads
	@cp _build/latex/tinytorch-course.pdf _build/html/assets/downloads/TinyTorch-Guide.pdf
	@cp ../paper/paper.pdf _build/html/assets/downloads/TinyTorch-Paper.pdf
	@echo ""
	@echo "✅ Downloads ready:"
	@ls -lh _build/html/assets/downloads/

# Default target
.DEFAULT_GOAL := help
