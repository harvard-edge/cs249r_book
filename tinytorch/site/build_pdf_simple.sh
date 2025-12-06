#!/bin/bash
# Build PDF version of TinyTorch book (Simple HTML-to-PDF method)
# This script builds PDF via HTML conversion - no LaTeX installation required

set -e  # Exit on error

echo "🔥 Building TinyTorch PDF (Simple Method - No LaTeX Required)..."
echo ""

# Check if we're in the site directory
if [ ! -f "_config.yml" ]; then
    echo "❌ Error: Must run from site/ directory"
    echo "Usage: cd site && ./build_pdf_simple.sh"
    exit 1
fi

# Check dependencies
echo "📋 Checking dependencies..."
if ! command -v jupyter-book &> /dev/null; then
    echo "❌ Error: jupyter-book not installed"
    echo "Install with: pip install jupyter-book pyppeteer"
    exit 1
fi

# Check if pyppeteer is installed
python3 -c "import pyppeteer" 2>/dev/null || {
    echo "❌ Error: pyppeteer not installed"
    echo "Install with: pip install pyppeteer"
    echo ""
    echo "Note: First run will download Chromium (~170MB)"
    exit 1
}

echo "✅ Dependencies OK"
echo ""

# Clean previous builds
echo "🧹 Cleaning previous builds..."
jupyter-book clean . --all || true
echo ""

# Prepare notebooks (for consistency, though PDF doesn't need launch buttons)
echo "📓 Preparing notebooks..."
./prepare_notebooks.sh || echo "⚠️  Notebook preparation skipped"

# Build PDF via HTML
echo "📚 Building PDF from HTML (this may take a few minutes)..."
echo "ℹ️  First run will download Chromium browser (~170MB)"
jupyter-book build . --builder pdfhtml

# Check if build succeeded
if [ -f "_build/pdf/book.pdf" ]; then
    # Copy to standard location with better name
    cp "_build/pdf/book.pdf" "_build/tinytorch-course.pdf"
    PDF_SIZE=$(du -h "_build/tinytorch-course.pdf" | cut -f1)
    echo ""
    echo "✅ PDF build complete!"
    echo "📄 Output: docs/_build/tinytorch-course.pdf"
    echo "📊 Size: ${PDF_SIZE}"
    echo ""
    echo "To view the PDF:"
    echo "  open _build/tinytorch-course.pdf           # macOS"
    echo "  xdg-open _build/tinytorch-course.pdf       # Linux"
    echo "  start _build/tinytorch-course.pdf          # Windows"
else
    echo ""
    echo "❌ PDF build failed - check errors above"
    exit 1
fi

