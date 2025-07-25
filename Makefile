# =============================================================================
# MLSysBook Makefile
# =============================================================================
# Common development tasks for the Machine Learning Systems book project
#
# Usage:
#   make clean       - Clean all build artifacts
#   make clean-deep  - Deep clean including caches and environments
#   make build       - Build the book (HTML)
#   make build-pdf   - Build PDF version
#   make preview     - Start development preview server
#   make test        - Run tests and validation
#   make install     - Install dependencies
#   make check       - Check project health
# =============================================================================

.PHONY: help clean clean-deep clean-dry build build-pdf preview test install check setup-hooks lint

# Default target
help:
	@echo "📚 MLSysBook Development Commands"
	@echo "=================================="
	@echo ""
	@echo "🧹 Cleaning:"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make clean-deep  - Deep clean (includes caches, venv)"
	@echo "  make clean-dry   - Show what would be cleaned (dry run)"
	@echo ""
	@echo "🔨 Building:"
	@echo "  make build       - Build HTML version"
	@echo "  make build-pdf   - Build PDF version"
	@echo "  make build-all   - Build all formats"
	@echo ""
	@echo "🔍 Development:"
	@echo "  make preview     - Start development server"
	@echo "  make test        - Run tests and validation"
	@echo "  make check       - Check project health"
	@echo "  make lint        - Run linting checks"
	@echo ""
	@echo "⚙️ Setup:"
	@echo "  make install     - Install dependencies"
	@echo "  make setup-hooks - Setup git hooks"
	@echo ""
	@echo "💡 Examples:"
	@echo "  make clean build preview    # Clean, build, and start preview"
	@echo "  make clean-dry              # See what would be cleaned"

# =============================================================================
# Cleaning Tasks
# =============================================================================

clean:
	@echo "🧹 Cleaning build artifacts..."
	@./tools/scripts/build/clean.sh

clean-deep:
	@echo "🔥 Deep cleaning (including caches and environments)..."
	@./tools/scripts/build/clean.sh --deep

clean-dry:
	@echo "🔍 Dry run - showing what would be cleaned..."
	@./tools/scripts/build/clean.sh --dry-run

# =============================================================================
# Building Tasks
# =============================================================================

build:
	@echo "🔨 Building HTML version..."
	@cd book && quarto render --to html

build-pdf:
	@echo "📄 Building PDF version..."
	@cd book && quarto render --to titlepage-pdf

build-all:
	@echo "📚 Building all formats..."
	@cd book && quarto render

# =============================================================================
# Development Tasks
# =============================================================================

preview:
	@echo "🌐 Starting development preview server..."
	@echo "  -> Open your browser to the URL shown below"
	@cd book && quarto preview

test:
	@echo "🧪 Running tests and validation..."
	@echo "  📋 Checking Quarto configuration..."
	@cd book && quarto check
	@echo "  🔍 Validating project structure..."
	@./tools/scripts/build/clean.sh --dry-run > /dev/null
	@echo "  ✅ Basic validation passed"

check:
	@echo "🔍 Checking project health..."
	@echo ""
	@echo "📊 Project Structure:"
	@find book/contents -name "*.qmd" | wc -l | xargs echo "  QMD files:"
	@find book/contents -name "*.bib" | wc -l | xargs echo "  Bibliography files:"
	@find book/contents -name "*_quizzes.json" | wc -l | xargs echo "  Quiz files:"
	@echo ""
	@echo "🗂️ Git Status:"
	@if git status --porcelain | head -5; then \
		echo "  (Showing first 5 changed files)"; \
	else \
		echo "  Repository is clean"; \
	fi
	@echo ""
	@echo "📦 Dependencies:"
	@if command -v quarto >/dev/null 2>&1; then \
		echo "  ✅ Quarto: $$(quarto --version)"; \
	else \
		echo "  ❌ Quarto: Not installed"; \
	fi
	@if command -v python3 >/dev/null 2>&1; then \
		echo "  ✅ Python: $$(python3 --version)"; \
	else \
		echo "  ❌ Python: Not installed"; \
	fi

lint:
	@echo "🔍 Running linting checks..."
	@echo "  📝 Checking for common issues..."
	@# Check for TODO comments
	@if grep -r "TODO\|FIXME\|XXX" book/contents --include="*.qmd" 2>/dev/null; then \
		echo "  ⚠️  TODO items found (review above)"; \
	else \
		echo "  ✅ No TODO items found"; \
	fi
	@# Check for broken internal links (basic check)
	@echo "  🔗 Checking for potential broken references..."
	@if grep -r "fig-\|tbl-\|sec-" book/contents --include="*.qmd" | grep -v "^[^:]*:\s*#" | head -5; then \
		echo "  📊 Cross-references found (manual review recommended)"; \
	fi

# =============================================================================
# Setup Tasks
# =============================================================================

install:
	@echo "📦 Installing dependencies..."
	@echo "  🐍 Python dependencies..."
	@if [ -f tools/dependencies/requirements.txt ]; then \
		pip install -r tools/dependencies/requirements.txt; \
	else \
		echo "  ⚠️  requirements.txt not found"; \
	fi
	@echo "  📊 R dependencies..."
	@if [ -f tools/dependencies/install_packages.R ] && command -v Rscript >/dev/null 2>&1; then \
		Rscript tools/dependencies/install_packages.R; \
	else \
		echo "  ⚠️  R or install_packages.R not found"; \
	fi
	@echo "  ✅ Dependencies installation completed"

setup-hooks:
	@echo "🔧 Setting up git hooks..."
	@if [ ! -f .git/hooks/pre-commit ]; then \
		echo "  ❌ Pre-commit hook not found. Please run this from project root."; \
		exit 1; \
	fi
	@chmod +x .git/hooks/pre-commit
	@chmod +x tools/scripts/build/clean.sh
	@echo "  ✅ Git hooks are now active"
	@echo "  📋 The pre-commit hook will automatically:"
	@echo "     - Clean build artifacts before commits"
	@echo "     - Check for large files and potential secrets"
	@echo "     - Ensure repository cleanliness"

# =============================================================================
# Compound Tasks
# =============================================================================

dev: clean build preview
	@echo "🚀 Development environment ready!"

full-clean-build: clean-deep install build
	@echo "🎯 Full clean build completed!"

release-check: clean lint test build
	@echo "📋 Release checks completed!"

# =============================================================================
# Utility Tasks
# =============================================================================

status:
	@echo "📊 MLSysBook Project Status"
	@echo "==========================="
	@make check

# Help with specific tasks
help-clean:
	@echo "🧹 Cleaning Help"
	@echo "==============="
	@echo ""
	@echo "clean      - Removes build artifacts (HTML, PDF, LaTeX files, caches)"
	@echo "clean-deep - Also removes virtual environments and all caches"
	@echo "clean-dry  - Shows what would be removed without actually deleting"
	@echo ""
	@echo "The cleanup script removes:"
	@echo "  • Build outputs: *.html, *.pdf, *.tex, *.aux, *.log"
	@echo "  • Cache directories: .quarto/, site_libs/, index_files/"
	@echo "  • Python artifacts: __pycache__/, *.pyc"
	@echo "  • System files: .DS_Store, Thumbs.db"
	@echo "  • Editor files: *.swp, *~"

help-build:
	@echo "🔨 Building Help"
	@echo "==============="
	@echo ""
	@echo "build     - Creates HTML version in _book/ directory"
	@echo "build-pdf - Creates PDF version (requires LaTeX)"
	@echo "build-all - Creates all configured formats"
	@echo ""
	@echo "Build outputs go to:"
	@echo "  • HTML: _book/index.html"
	@echo "  • PDF: book/index.pdf"
	@echo ""
	@echo "Before building, ensure:"
	@echo "  • Quarto is installed and updated"
	@echo "  • All dependencies are installed (make install)"
	@echo "  • Project is clean (make clean)" 