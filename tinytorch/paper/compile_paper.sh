#!/bin/bash
# Compile TinyTorch LaTeX paper to PDF
# Uses LuaLaTeX for emoji support

cd "$(dirname "$0")"

# Check if lualatex is available
if ! command -v lualatex &> /dev/null; then
    echo "Error: lualatex not found"
    echo "Please install MacTeX: brew install --cask mactex"
    echo "Or install BasicTeX: brew install --cask basictex"
    exit 1
fi

echo "Compiling paper.tex with LuaLaTeX (for emoji support)..."

# First pass
lualatex -interaction=nonstopmode paper.tex

# Biber pass (for biblatex)
if command -v biber &> /dev/null; then
    biber paper
elif command -v bibtex &> /dev/null; then
    echo "Warning: biber not found, falling back to bibtex (may not work with biblatex)"
    bibtex paper
fi

# Second pass (resolve references)
lualatex -interaction=nonstopmode paper.tex

# Third pass (final cleanup)
lualatex -interaction=nonstopmode paper.tex

# Check if PDF was created
if [ -f paper.pdf ]; then
    echo "✓ PDF created successfully: paper.pdf"
    echo "✓ Opening PDF..."
    open paper.pdf
else
    echo "✗ PDF compilation failed"
    echo "Check paper.log for errors"
    exit 1
fi

# Clean up auxiliary files (optional)
# rm -f paper.aux paper.log paper.bbl paper.blg paper.out

echo "✓ Compilation complete!"
