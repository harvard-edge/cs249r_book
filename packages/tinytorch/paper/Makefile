# Makefile for TinyTorch LaTeX paper
# Uses LuaLaTeX for emoji support

# Main document name (without extension)
DOC = paper

# LaTeX compiler
LATEX = lualatex
LATEXFLAGS = -interaction=nonstopmode

# Biber compiler (for biblatex)
BIBER = biber

# Output files
PDF = $(DOC).pdf
AUX = $(DOC).aux
BBL = $(DOC).bbl
BLG = $(DOC).blg
LOG = $(DOC).log
OUT = $(DOC).out

# Source files
TEX = $(DOC).tex
BIB = references.bib

# Default target
all: $(PDF)

# Build PDF with full compilation cycle (3 passes for references)
$(PDF): $(TEX) $(BIB)
	@echo "Compiling $(TEX) with LuaLaTeX (for emoji support)..."
	$(LATEX) $(LATEXFLAGS) $(TEX)
	$(BIBER) $(DOC)
	$(LATEX) $(LATEXFLAGS) $(TEX)
	$(LATEX) $(LATEXFLAGS) $(TEX)
	@if [ -f $(PDF) ]; then \
		echo "✓ PDF created successfully: $(PDF)"; \
	else \
		echo "✗ PDF compilation failed"; \
		echo "Check $(LOG) for errors"; \
		exit 1; \
	fi

# Quick compile (single pass, no bibtex)
quick:
	@echo "Quick compile (single pass)..."
	$(LATEX) $(LATEXFLAGS) $(TEX)

# Open the PDF
view: $(PDF)
	@echo "Opening $(PDF)..."
	open $(PDF)

# Build and view
paper: $(PDF) view

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f $(AUX) $(LOG) $(BBL) $(BLG) $(OUT)
	rm -f $(DOC).fls $(DOC).fdb_latexmk $(DOC).synctex.gz
	rm -f $(DOC).bcf $(DOC).run.xml
	@echo "✓ Auxiliary files cleaned"

# Clean everything including PDF
distclean: clean
	@echo "Removing PDF..."
	rm -f $(PDF)
	@echo "✓ All generated files removed"

# Check for required tools
check:
	@echo "Checking for required tools..."
	@command -v $(LATEX) >/dev/null 2>&1 || { \
		echo "Error: $(LATEX) not found"; \
		echo "Please install MacTeX: brew install --cask mactex"; \
		echo "Or install BasicTeX: brew install --cask basictex"; \
		exit 1; \
	}
	@command -v $(BIBER) >/dev/null 2>&1 || { \
		echo "Warning: $(BIBER) not found - bibliography may not compile"; \
	}
	@echo "✓ All required tools found"

# Help target
help:
	@echo "TinyTorch Paper Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all        Build the PDF (default)"
	@echo "  quick      Single-pass compile (fast, may have missing refs)"
	@echo "  view       Open the PDF"
	@echo "  paper      Build and view the PDF"
	@echo "  clean      Remove auxiliary files"
	@echo "  distclean  Remove all generated files including PDF"
	@echo "  check      Verify required tools are installed"
	@echo "  help       Show this help message"

.PHONY: all quick view paper clean distclean check help
