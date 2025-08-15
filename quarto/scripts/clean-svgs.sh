#!/usr/bin/env bash

set -euo pipefail

# ==============================================================================
# SVG Cleanup Script
# ==============================================================================
# Removes control characters from SVG files that can cause rendering issues
# in browsers and other tools. Control characters are often introduced by
# LaTeX -> SVG conversion tools.
# ==============================================================================

# Configuration
BUILD_DIR="${1:-_build}"
SCRIPT_NAME="$(basename "$0")"

# Colors for output (if terminal supports them)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    PURPLE='\033[0;35m'
    CYAN='\033[0;36m'
    NC='\033[0m' # No Color
else
    RED='' GREEN='' YELLOW='' BLUE='' PURPLE='' CYAN='' NC=''
fi

# Logging functions
log_info() {
    echo -e "${CYAN}[INFO]${NC} ${SCRIPT_NAME}: $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} ${SCRIPT_NAME}: $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} ${SCRIPT_NAME}: $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} ${SCRIPT_NAME}: $*"
}

log_file() {
    echo -e "${PURPLE}  → ${NC}$*"
}

# Main execution
main() {
    log_info "Starting SVG cleanup in directory: ${BUILD_DIR}"
    
    # Check if build directory exists
    if [[ ! -d "$BUILD_DIR" ]]; then
        log_warning "Build directory '${BUILD_DIR}' does not exist"
        log_info "Script completed (nothing to clean)"
        exit 0
    fi
    
    # Find all SVG files
    local svg_files=()
    while IFS= read -r -d '' file; do
        svg_files+=("$file")
    done < <(find "$BUILD_DIR" -type f -name '*.svg' -print0 2>/dev/null)
    
    if [[ ${#svg_files[@]} -eq 0 ]]; then
        log_info "No SVG files found in ${BUILD_DIR}"
        log_info "Script completed (nothing to clean)"
        exit 0
    fi
    
    # Process each SVG file
    local cleaned_count=0
    
    for svg_file in "${svg_files[@]}"; do
        # Check if file contains control characters before cleaning
        if LC_ALL=C grep -q '[[:cntrl:]]' "$svg_file" 2>/dev/null; then
            log_file "Cleaning: ${svg_file}"
            
            # Clean control characters
            if LC_ALL=C perl -i -pe 's/[\x00-\x08\x0B\x0C\x0E-\x1F]//g' "$svg_file"; then
                ((cleaned_count++))
            else
                log_error "Failed to clean: ${svg_file}"
            fi
        fi
    done
    
    # Only show summary if files were actually cleaned
    if [[ $cleaned_count -gt 0 ]]; then
        log_success "Cleaned ${cleaned_count} SVG file(s)"
    fi
}

# Run main function
main "$@"
