# =============================================================================
# Hardware Kits Build System
# =============================================================================
# Part of the MLSysBook ecosystem
# =============================================================================

.PHONY: all build html pdf preview clean help

# Default target
all: html

# Build HTML site (default)
html: build
build:
	@echo "🔧 Building Hardware Kits HTML site..."
	quarto render
	@echo "✅ Build complete: _build/"

# Build PDF
pdf:
	@echo "📄 Building Hardware Kits PDF..."
	quarto render --profile pdf
	@echo "✅ PDF build complete: _build/pdf/"

# Preview the site locally
preview:
	@echo "👀 Starting preview server..."
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
