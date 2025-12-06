#!/bin/bash
# Build PDF version of TinyTorch book
# This script builds the LaTeX/PDF version using jupyter-book

set -e  # Exit on error

echo "🔥 Building TinyTorch PDF..."
echo ""

# Check if we're in the site directory
if [ ! -f "_config.yml" ]; then
    echo "❌ Error: Must run from site/ directory"
    echo "Usage: cd site && ./build_pdf.sh"
    exit 1
fi

# Check dependencies
echo "📋 Checking dependencies..."
if ! command -v jupyter-book &> /dev/null; then
    echo "❌ Error: jupyter-book not installed"
    echo "Install with: pip install jupyter-book"
    exit 1
fi

if ! command -v pdflatex &> /dev/null; then
    echo "⚠️  Warning: pdflatex not found"
    echo "PDF build requires LaTeX installation:"
    echo "  - macOS: brew install --cask mactex-no-gui"
    echo "  - Ubuntu: sudo apt-get install texlive-latex-extra texlive-fonts-recommended"
    echo "  - Windows: Install MiKTeX from miktex.org"
    echo ""
    echo "Alternatively, use HTML-to-PDF build (doesn't require LaTeX):"
    echo "  jupyter-book build . --builder pdfhtml"
    exit 1
fi

echo "✅ Dependencies OK"
echo ""

# Clean previous builds
echo "🧹 Cleaning previous builds..."
jupyter-book clean . --all || true
echo ""

# Prepare notebooks (for consistency, though PDF doesn't need launch buttons)
echo "📓 Preparing notebooks..."
./prepare_notebooks.sh || echo "⚠️  Notebook preparation skipped"

# Build PDF via LaTeX
echo "📚 Building LaTeX/PDF (this may take a few minutes)..."
jupyter-book build . --builder pdflatex

# Check if build succeeded
if [ -f "_build/latex/tinytorch-course.pdf" ]; then
    PDF_SIZE=$(du -h "_build/latex/tinytorch-course.pdf" | cut -f1)
    echo ""
    echo "✅ PDF build complete!"
    echo "📄 Output: docs/_build/latex/tinytorch-course.pdf"
    echo "📊 Size: ${PDF_SIZE}"
    echo ""
    echo "To view the PDF:"
    echo "  open _build/latex/tinytorch-course.pdf    # macOS"
    echo "  xdg-open _build/latex/tinytorch-course.pdf  # Linux"
    echo "  start _build/latex/tinytorch-course.pdf     # Windows"
else
    echo ""
    echo "❌ PDF build failed - check errors above"
    echo ""
    echo "📝 Build artifacts in: _build/latex/"
    echo "Check _build/latex/tinytorch-course.log for detailed errors"
    exit 1
fi

