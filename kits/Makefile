# =============================================================================
# Hardware Kits Build System
# =============================================================================
# Part of the MLSysBook ecosystem
# =============================================================================

.PHONY: all build preview clean help

# Default target
all: build

# Build the kits site
build:
	@echo "🔧 Building Hardware Kits site..."
	quarto render
	@echo "✅ Build complete: _build/"

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
	@echo "  make build    - Build the site to _build/"
	@echo "  make preview  - Start local preview server"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make help     - Show this help message"
