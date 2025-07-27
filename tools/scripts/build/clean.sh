#!/bin/bash

# =============================================================================
# MLSysBook Cleanup Script
# =============================================================================
# This script removes all build artifacts, cache files, and temporary files
# from the MLSysBook project to ensure a clean repository state.
#
# Usage: ./tools/scripts/clean.sh [options]
# Options:
#   --deep    : Perform deep clean including all caches and virtual environments
#   --dry-run : Show what would be deleted without actually deleting
#   --quiet   : Suppress output except errors
# =============================================================================

set -e  # Exit on any error

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default options
DEEP_CLEAN=false
DRY_RUN=false
QUIET=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --deep)
            DEEP_CLEAN=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--deep] [--dry-run] [--quiet]"
            echo "  --deep    : Perform deep clean including caches and virtual environments"
            echo "  --dry-run : Show what would be deleted without actually deleting"
            echo "  --quiet   : Suppress output except errors"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging function
log() {
    if [[ "$QUIET" == "false" ]]; then
        echo -e "${BLUE}[CLEAN]${NC} $1"
    fi
}

log_success() {
    if [[ "$QUIET" == "false" ]]; then
        echo -e "${GREEN}[SUCCESS]${NC} $1"
    fi
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to remove files/directories
remove_item() {
    local item="$1"
    local description="$2"
    
    if [[ -e "$item" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log "Would remove: $item ($description)"
        else
            log "Removing: $item ($description)"
            rm -rf "$item"
        fi
    fi
}

# Function to find and remove pattern
remove_pattern() {
    local pattern="$1"
    local description="$2"
    local base_dir="${3:-$PROJECT_ROOT}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "Would remove files matching: $pattern ($description)"
        find "$base_dir" -name "$pattern" -type f 2>/dev/null | head -10 | while read -r file; do
            echo "  - $file"
        done
    else
        local count
        count=$(find "$base_dir" -name "$pattern" -type f 2>/dev/null | wc -l)
        if [[ $count -gt 0 ]]; then
            log "Removing $count files matching: $pattern ($description)"
            find "$base_dir" -name "$pattern" -type f -delete 2>/dev/null || true
        fi
    fi
}

log "🧹 Starting MLSysBook cleanup..."
log "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# =============================================================================
# Core Build Artifacts
# =============================================================================
log "📄 Cleaning core build artifacts..."

# Quarto build outputs in book directory
remove_item "book/index.html" "Main HTML output"
remove_item "book/index.pdf" "Main PDF output"
remove_item "book/index.tex" "LaTeX source"
remove_item "book/index.aux" "LaTeX auxiliary"
remove_item "book/index.log" "LaTeX log"
remove_item "book/index.toc" "Table of contents"
remove_item "book/index_files" "Quarto support files"
remove_item "book/site_libs" "Site libraries"

# Quarto build outputs in root
remove_item "_book" "Generated book output (legacy)"
remove_item "index_files" "Root index files (legacy)"

# New build directory structure
remove_item "build" "Build directory (all formats)"

# =============================================================================
# Quarto Cache and Temporary Files
# =============================================================================
log "🗂️ Cleaning Quarto cache and temporary files..."

remove_item "book/.quarto" "Quarto cache (book)"
remove_item ".quarto" "Quarto cache (root)"
remove_pattern "_quarto_test.yml" "Test configuration files"
remove_pattern "*.qmd~" "Quarto backup files"

# =============================================================================
# LaTeX and PDF Build Artifacts
# =============================================================================
log "📝 Cleaning LaTeX and PDF artifacts..."

remove_pattern "*.aux" "LaTeX auxiliary files"
remove_pattern "*.log" "LaTeX log files"
remove_pattern "*.toc" "Table of contents files"
remove_pattern "*.out" "LaTeX outline files"
remove_pattern "*.fdb_latexmk" "Latexmk database files"
remove_pattern "*.fls" "LaTeX file list"
remove_pattern "*.synctex.gz" "SyncTeX files"
remove_pattern "*.bbl" "Bibliography files"
remove_pattern "*.blg" "Bibliography log files"

# =============================================================================
# HTML and Web Artifacts
# =============================================================================
log "🌐 Cleaning HTML and web artifacts..."

# HTML files in content directories (should be build artifacts)
remove_pattern "*.html" "HTML build artifacts" "book/contents"
remove_pattern "*.html" "HTML build artifacts" "contents"

# =============================================================================
# Python and Development Artifacts
# =============================================================================
log "🐍 Cleaning Python artifacts..."

remove_pattern "__pycache__" "Python cache directories"
remove_pattern "*.pyc" "Python compiled files"
remove_pattern "*.pyo" "Python optimized files"
remove_pattern "*.pyd" "Python extension modules"
remove_pattern ".pytest_cache" "Pytest cache"
remove_item "*.egg-info" "Python package info"

# =============================================================================
# System and Editor Files
# =============================================================================
log "💻 Cleaning system and editor files..."

remove_pattern ".DS_Store" "macOS metadata files"
remove_pattern "Thumbs.db" "Windows thumbnail cache"
remove_pattern "*.swp" "Vim swap files"
remove_pattern "*.swo" "Vim swap files"
remove_pattern "*~" "Editor backup files"
remove_pattern ".#*" "Emacs lock files"

# =============================================================================
# Log and Debug Files
# =============================================================================
log "📋 Cleaning log and debug files..."

remove_pattern "debug.log" "Debug log files"
remove_pattern "error.log" "Error log files"
remove_pattern "*.log" "General log files" "tools"
remove_item "source_analysis_report.json" "Analysis reports"
remove_item "content_map.json" "Content mapping files"

# =============================================================================
# Deep Clean (Optional)
# =============================================================================
if [[ "$DEEP_CLEAN" == "true" ]]; then
    log "🔥 Performing deep clean..."
    
    # Virtual environments
    remove_item ".venv" "Python virtual environment"
    remove_item "venv" "Python virtual environment"
    remove_item "env" "Python virtual environment"
    
    # Node modules (if any)
    remove_item "node_modules" "Node.js modules"
    remove_item "package-lock.json" "Node.js lock file"
    
    # R artifacts
    remove_item ".Rproj.user" "R project user files"
    remove_pattern "*.Rhistory" "R history files"
    remove_pattern "*.RData" "R data files"
    
    # All figure directories (regeneratable)
    remove_pattern "figure-*" "Generated figure directories"
    remove_pattern "mediabag" "Pandoc media bags"
    
    log_warning "Deep clean completed - you may need to reinstall dependencies"
fi

# =============================================================================
# Summary and Verification
# =============================================================================
if [[ "$DRY_RUN" == "false" ]]; then
    log "🔍 Verifying cleanup..."
    
    # Check for remaining build artifacts
    remaining_artifacts=$(find . -name "*.html" -o -name "*.pdf" -o -name "*.aux" -o -name "*.log" | grep -v ".venv" | grep -v "node_modules" | wc -l)
    
    if [[ $remaining_artifacts -eq 0 ]]; then
        log_success "✅ Project is clean - no build artifacts found"
    else
        log_warning "⚠️  Some artifacts may remain ($remaining_artifacts files)"
        find . -name "*.html" -o -name "*.pdf" -o -name "*.aux" -o -name "*.log" | grep -v ".venv" | grep -v "node_modules" | head -5
    fi
    
    # Git status check
    if command -v git >/dev/null 2>&1 && [[ -d .git ]]; then
        untracked_count=$(git status --porcelain | grep "^??" | wc -l)
        if [[ $untracked_count -eq 0 ]]; then
            log_success "✅ No untracked files in git"
        else
            log_warning "⚠️  $untracked_count untracked files in git (may be new content)"
        fi
    fi
else
    log "🔍 Dry run completed - no files were actually removed"
fi

log_success "🎉 Cleanup completed successfully!"

# Return to original directory
cd - >/dev/null 2>&1 || true 