# =============================================================================
# Hardware Kits Build System
# =============================================================================
# Part of the MLSysBook ecosystem
# Uses symlink switching between HTML and PDF configs (like the book)
# =============================================================================

.PHONY: all build html pdf preview clean help

# Default target
all: html

# Build HTML site (default)
html: build
build:
	@echo "🔧 Building Hardware Kits HTML site..."
	ln -sf config/_quarto-html.yml _quarto.yml
	quarto render
	@echo "✅ Build complete: _build/"

# Build PDF with cover page
pdf:
	@echo "📄 Building Hardware Kits PDF..."
	ln -sf config/_quarto-pdf.yml _quarto.yml
	quarto render --to titlepage-pdf
	ln -sf config/_quarto-html.yml _quarto.yml
	@echo "✅ PDF build complete: _build/pdf/"

# Preview the site locally
preview:
	@echo "👀 Starting preview server..."
	ln -sf config/_quarto-html.yml _quarto.yml
	quarto preview

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf _build
	rm -rf .quarto
	@echo "✅ Clean complete"

# Help
help:
	@echo "Hardware Kits Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  make build    - Build HTML site to _build/"
	@echo "  make html     - Build HTML site to _build/ (same as build)"
	@echo "  make pdf      - Build PDF to _build/pdf/"
	@echo "  make preview  - Start local preview server"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make help     - Show this help message"
