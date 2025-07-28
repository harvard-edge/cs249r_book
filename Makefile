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
#   make fast-<name> - Build single chapter (e.g., make fast-introduction)
# =============================================================================

.PHONY: help clean clean-deep clean-dry build build-html build-pdf build-all preview preview-pdf test install check setup-hooks lint dev full-clean-build release-check status show-build help-clean help-build _fast_build _fast_build_and_preview

# =============================================================================
# Fast Single-Chapter Build System
# =============================================================================
# Clean delegation to shell script for better argument handling
#
# Usage examples:
#   make fast CHAPTER=introduction          # HTML build
#   make fast CHAPTER=introduction FORMAT=pdf  # PDF build
#   make fast-preview CHAPTER=introduction  # HTML + preview
# =============================================================================

# Fast build with arguments
fast:
	@if [ -z "$(CHAPTER)" ]; then \
		echo "❌ Usage: make fast CHAPTER=<name> [FORMAT=html|pdf]"; \
		echo "💡 Examples:"; \
		echo "    make fast CHAPTER=introduction"; \
		echo "    make fast CHAPTER=introduction FORMAT=pdf"; \
		exit 1; \
	fi
	@echo "🚀 Fast $(or $(FORMAT),html) build for chapter: $(CHAPTER)"
	@echo "  🔍 Searching for chapter matching: $(CHAPTER)"
	@TARGET_FILE=$$(find book/contents -name "*$(CHAPTER)*.qmd" | head -1); \
	if [ -z "$$TARGET_FILE" ]; then \
		echo "❌ No .qmd file found matching '$(CHAPTER)'"; \
		echo "💡 Available chapters:"; \
		find book/contents -name "*.qmd" | grep -v "/images/" | sed 's|book/contents/||' | sed 's|\.qmd||' | sort | sed 's|^|     |'; \
		exit 1; \
	fi; \
	echo "  ✅ Found: $$TARGET_FILE"; \
	TARGET_PATH=$$(echo $$TARGET_FILE | sed 's|^book/||'); \
	echo "  📄 Format: $(or $(FORMAT),html)"; \
	if [ "$(FORMAT)" = "pdf" ]; then \
		CONFIG_FILE="_quarto-pdf.yml"; \
		RENDER_CMD="quarto render --to titlepage-pdf"; \
		BUILD_DIR="build/pdf"; \
	else \
		CONFIG_FILE="_quarto-html.yml"; \
		RENDER_CMD="quarto render --to html"; \
		BUILD_DIR="build/html"; \
	fi; \
	CONFIG_PATH="book/$$CONFIG_FILE"; \
	BACKUP_PATH="$$CONFIG_PATH.fast-build-backup"; \
	echo "  📝 Temporarily commenting out non-target .qmd files in $$CONFIG_FILE"; \
	cp "$$CONFIG_PATH" "$$BACKUP_PATH"; \
	sed -i.tmp -E '/\.qmd($|[^a-zA-Z0-9_-])/s/^(.*)$$/# FAST_BUILD_COMMENTED: \1/' "$$CONFIG_PATH"; \
	TARGET_ESCAPED=$$(echo $$TARGET_PATH | sed 's/[\/&]/\\&/g'); \
	sed -i.tmp -E "/(index\.qmd|$$TARGET_ESCAPED)/s/^# FAST_BUILD_COMMENTED: (.*)$$/\1/" "$$CONFIG_PATH"; \
	rm -f "$$CONFIG_PATH.tmp"; \
	COMMENTED_COUNT=$$(grep -c "# FAST_BUILD_COMMENTED:" "$$CONFIG_PATH" || echo 0); \
	ACTIVE_COUNT=$$(grep -c "\.qmd" "$$CONFIG_PATH" || echo 0); \
	echo "  📊 Build scope: $$COMMENTED_COUNT commented, $$ACTIVE_COUNT active"; \
	mkdir -p "$$(dirname book)/$$BUILD_DIR"; \
	cd book && if [ -f "_quarto.yml" ] && [ ! -L "_quarto.yml" ]; then rm -f "_quarto.yml"; fi; \
	cd book && ln -sf "$$CONFIG_FILE" "_quarto.yml"; \
	echo "  🔗 _quarto.yml → $$CONFIG_FILE"; \
	echo "  🔨 Building with reduced config..."; \
	cd book && $$RENDER_CMD; \
	echo "  🔄 Restoring original config..."; \
	mv "$$BACKUP_PATH" "$$CONFIG_PATH"; \
	echo "  ✅ Fast build complete: $$BUILD_DIR/"

# Fast build + preview  
fast-preview:
	@if [ -z "$(CHAPTER)" ]; then \
		echo "❌ Usage: make fast-preview CHAPTER=<name>"; \
		echo "💡 Example: make fast-preview CHAPTER=introduction"; \
		exit 1; \
	fi
	@echo "🌐 Fast preview for chapter: $(CHAPTER)"
	@echo "  🔍 Searching for chapter matching: $(CHAPTER)"
	@TARGET_FILE=$$(find book/contents -name "*$(CHAPTER)*.qmd" | head -1); \
	if [ -z "$$TARGET_FILE" ]; then \
		echo "❌ No .qmd file found matching '$(CHAPTER)'"; \
		echo "💡 Available chapters:"; \
		find book/contents -name "*.qmd" | grep -v "/images/" | sed 's|book/contents/||' | sed 's|\.qmd||' | sort | sed 's|^|     |'; \
		exit 1; \
	fi; \
	echo "  ✅ Found: $$TARGET_FILE"; \
	TARGET_PATH=$$(echo $$TARGET_FILE | sed 's|^book/||'); \
	CONFIG_FILE="_quarto-html.yml"; \
	CONFIG_PATH="book/$$CONFIG_FILE"; \
	BACKUP_PATH="$$CONFIG_PATH.fast-build-backup"; \
	echo "  📝 Temporarily commenting out non-target .qmd files in $$CONFIG_FILE"; \
	cp "$$CONFIG_PATH" "$$BACKUP_PATH"; \
	sed -i.tmp -E '/\.qmd($|[^a-zA-Z0-9_-])/s/^(.*)$$/# FAST_BUILD_COMMENTED: \1/' "$$CONFIG_PATH"; \
	TARGET_ESCAPED=$$(echo $$TARGET_PATH | sed 's/[\/&]/\\&/g'); \
	sed -i.tmp -E "/(index\.qmd|$$TARGET_ESCAPED)/s/^# FAST_BUILD_COMMENTED: (.*)$$/\1/" "$$CONFIG_PATH"; \
	rm -f "$$CONFIG_PATH.tmp"; \
	cd book && if [ -f "_quarto.yml" ] && [ ! -L "_quarto.yml" ]; then rm -f "_quarto.yml"; fi; \
	cd book && ln -sf "$$CONFIG_FILE" "_quarto.yml"; \
	echo "  🔗 _quarto.yml → $$CONFIG_FILE"; \
	echo "  🌐 Starting preview server with reduced config..."; \
	echo "  💡 TIP: You can inspect $$CONFIG_FILE to see what's commented out"; \
	echo "  🛑 Press Ctrl+C to stop the server and restore config"; \
	cd book && trap 'mv "$$BACKUP_PATH" "$$CONFIG_PATH" 2>/dev/null || true' EXIT INT TERM; quarto preview

# Cleanup fast build state
fast-cleanup:
	@echo "🧹 Fast Build Cleanup"
	@echo "💡 Restoring master configs (_quarto-html.yml, _quarto-pdf.yml) only"
	@for config in "_quarto-html.yml" "_quarto-pdf.yml"; do \
		CONFIG_PATH="book/$$config"; \
		BACKUP_PATH="$$CONFIG_PATH.fast-build-backup"; \
		if [ -f "$$BACKUP_PATH" ]; then \
			echo "  🔄 Restoring $$config from backup..."; \
			mv "$$BACKUP_PATH" "$$CONFIG_PATH"; \
			echo "  ✅ $$config restored"; \
		elif grep -q "FAST_BUILD_COMMENTED" "$$CONFIG_PATH" 2>/dev/null; then \
			echo "  🔄 Uncommenting $$config..."; \
			sed -i.tmp 's/^# FAST_BUILD_COMMENTED: //' "$$CONFIG_PATH"; \
			rm -f "$$CONFIG_PATH.tmp"; \
			echo "  ✅ $$config uncommented"; \
		else \
			echo "  ✅ $$config already clean"; \
		fi; \
	done
	@if [ -L "book/_quarto.yml" ]; then \
		CURRENT_TARGET=$$(readlink book/_quarto.yml); \
		echo "  🔗 Current symlink: _quarto.yml → $$CURRENT_TARGET"; \
	fi
	@echo "  ✅ All configs restored to clean state"

# Switch config symlink
switch-html:
	@echo "🔗 Switching to HTML config..."
	@$(MAKE) fast-cleanup >/dev/null 2>&1
	@cd book && rm -f "_quarto.yml" && ln -sf "_quarto-html.yml" "_quarto.yml"
	@echo "  ✅ _quarto.yml → _quarto-html.yml"

switch-pdf:  
	@echo "🔗 Switching to PDF config..."
	@$(MAKE) fast-cleanup >/dev/null 2>&1
	@cd book && rm -f "_quarto.yml" && ln -sf "_quarto-pdf.yml" "_quarto.yml"
	@echo "  ✅ _quarto.yml → _quarto-pdf.yml"

# Legacy pattern targets (for backward compatibility)
fast-%-pdf:
	@$(MAKE) fast CHAPTER="$*" FORMAT=pdf

fast-%-html:
	@$(MAKE) fast CHAPTER="$*" FORMAT=html

preview-%:
	@$(MAKE) fast-preview CHAPTER="$*"

fast-%: 
	@$(MAKE) fast CHAPTER="$*"

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
	@echo "  make build       - Interactive build (choose format)"
	@echo "  make build-html  - Build HTML website"
	@echo "  make build-pdf   - Build PDF book"
	@echo "  make build-all   - Build all formats"
	@echo ""
	@echo "⚡ Fast Single-Chapter Builds:"
	@echo "  make fast CHAPTER=<name> [FORMAT=pdf] - Build single chapter"
	@echo "  make fast-preview CHAPTER=<name>     - Build and preview chapter"
	@echo "  make fast-cleanup                    - Restore configs to clean state"
	@echo "  make switch-html / switch-pdf        - Switch _quarto.yml symlink"
	@echo ""
	@echo "  Legacy patterns (still work):"
	@echo "    make fast-<name>     make fast-<name>-pdf     make preview-<name>"
	@echo ""
	@echo "  Examples:"
	@echo "    make fast CHAPTER=introduction               # HTML build"
	@echo "    make fast CHAPTER=introduction FORMAT=pdf   # PDF build"
	@echo "    make fast-preview CHAPTER=ml_systems        # HTML + preview"
	@echo "    make fast-introduction                       # Legacy pattern"
	@echo "    make fast-cleanup                            # Clean up configs"
	@echo ""
	@echo "🔍 Development:"
	@echo "  make preview     - Start HTML development server"
	@echo "  make preview-pdf - Start PDF development server"
	@echo "  make test        - Run tests and validation"
	@echo "  make check       - Check project health"
	@echo "  make lint        - Run linting checks"
	@echo "  make show-build  - Show build directory structure"
	@echo ""
	@echo "⚙️ Setup:"
	@echo "  make install     - Install dependencies"
	@echo "  make setup-hooks - Setup git hooks"
	@echo ""
	@echo "💡 Examples:"
	@echo "  make clean build-html preview  # Clean, build HTML, and start preview"
	@echo "  make build                     # Interactive build (choose format)"
	@echo "  make clean-dry                 # See what would be cleaned"
	@echo "  make fast-introduction         # Quick build of just introduction"

# =============================================================================
# Cleaning Tasks
# =============================================================================

clean:
	@echo "🧹 Cleaning build artifacts..."
	@./binder clean

clean-deep:
	@echo "🔥 Deep cleaning (including caches and environments)..."
	@./binder clean deep

clean-dry:
	@echo "🔍 Dry run - showing what would be cleaned..."
	@./binder clean dry

# =============================================================================
# Building Tasks
# =============================================================================

build:
	@echo "📚 What would you like to build?"
	@echo ""
	@echo "  1️⃣  HTML website (interactive, fast)"
	@echo "  2️⃣  PDF book (academic, slower)"
	@echo "  3️⃣  Both formats"
	@echo ""
	@read -p "Choose [1/2/3]: " choice; \
	case $$choice in \
		1|html|HTML) echo "🔨 Building HTML..."; make build-html ;; \
		2|pdf|PDF) echo "📄 Building PDF..."; make build-pdf ;; \
		3|both|all) echo "📚 Building both..."; make build-all ;; \
		*) echo "❌ Invalid choice. Use: make build-html, make build-pdf, or make build-all" ;; \
	esac

build-html:
	@echo "🔨 Building HTML version..."
	@echo "  📝 Using HTML configuration..."
	@mkdir -p build/html
	@cd book && ln -sf _quarto-html.yml _quarto.yml
	@cd book && quarto render --to html
	@cd book && rm _quarto.yml
	@echo "  ✅ HTML build complete: build/html/"

build-pdf:
	@echo "📄 Building PDF version..."
	@echo "  📝 Using PDF configuration..."
	@mkdir -p build/pdf
	@cd book && ln -sf _quarto-pdf.yml _quarto.yml
	@cd book && quarto render --to titlepage-pdf
	@cd book && rm _quarto.yml
	@echo "  ✅ PDF build complete: build/pdf/"

build-all:
	@echo "📚 Building all formats..."
	@echo "  🔄 Building HTML..."
	@make build-html
	@echo "  🔄 Building PDF..."
	@make build-pdf
	@echo "  ✅ All formats complete"

# =============================================================================
# Development Tasks
# =============================================================================

preview:
	@echo "🌐 Starting development preview server (HTML)..."
	@echo "  📝 Using HTML configuration for preview..."
	@echo "  -> Open your browser to the URL shown below"
	@echo "  🛑 Press Ctrl+C to stop the server"
	@cd book && ln -sf _quarto-html.yml _quarto.yml
	@cd book && trap 'rm -f _quarto.yml' EXIT INT TERM; quarto preview

preview-pdf:
	@echo "📄 Starting PDF development preview..."
	@echo "  📝 Using PDF configuration for preview..."
	@echo "  -> Open your browser to the URL shown below"
	@echo "  🛑 Press Ctrl+C to stop the server"
	@cd book && ln -sf _quarto-pdf.yml _quarto.yml
	@cd book && trap 'rm -f _quarto.yml' EXIT INT TERM; quarto preview

test:
	@echo "🧪 Running tests and validation..."
	@echo "  📋 Checking Quarto configuration..."
	@cd book && quarto check
	@echo "  🔍 Validating project structure..."
	@./binder check > /dev/null
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
	@chmod +x binder
	@echo "  ✅ Git hooks are now active"
	@echo "  📋 The pre-commit hook will automatically:"
	@echo "     - Clean build artifacts before commits"
	@echo "     - Check for large files and potential secrets"
	@echo "     - Ensure repository cleanliness"

# =============================================================================
# Compound Tasks
# =============================================================================

dev: clean build-html preview
	@echo "🚀 Development environment ready!"

full-clean-build: clean-deep install build-html
	@echo "🎯 Full clean build completed!"

release-check: clean lint test build-html
	@echo "📋 Release checks completed!"

# =============================================================================
# Utility Tasks
# =============================================================================

status:
	@echo "📊 MLSysBook Project Status"
	@echo "==========================="
	@make check

show-build:
	@echo "📁 Build Directory Structure"
	@echo "============================"
	@if [ -d build ]; then \
		echo "✅ Build directory exists"; \
		echo ""; \
		echo "Contents:"; \
		find build -type d | sort | sed 's|^|  |'; \
		echo ""; \
		echo "Files:"; \
		find build -type f | sort | sed 's|^|  |'; \
	else \
		echo "❌ Build directory does not exist"; \
		echo "Run 'make build-html' or 'make build-pdf' to create it"; \
	fi

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
	@echo "build     - Interactive build (choose format)"
	@echo "build-html - Creates HTML version in build/html/"
	@echo "build-pdf - Creates PDF version in build/pdf/"
	@echo "build-all - Creates all configured formats"
	@echo ""
	@echo "Build outputs go to:"
	@echo "  • HTML: build/html/"
	@echo "  • PDF: build/pdf/"
	@echo ""
	@echo "Before building, ensure:"
	@echo "  • Quarto is installed and updated"
	@echo "  • All dependencies are installed (make install)"
	@echo "  • Project is clean (make clean)" 