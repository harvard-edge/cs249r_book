#!/usr/bin/env bash

# ==============================================================================
# SVG Cleanup Script
# ==============================================================================
# Removes control characters from SVG files that can cause rendering issues
# in browsers and other tools. Control characters are often introduced by
# LaTeX -> SVG conversion tools.
# ==============================================================================

# Configuration
# Use Quarto's output directory if available, otherwise fall back to command line arg or default
if [[ -n "${QUARTO_PROJECT_OUTPUT_DIR:-}" ]]; then
    BUILD_DIR="$QUARTO_PROJECT_OUTPUT_DIR"
    DEBUG_MODE="${1:-false}"  # First arg becomes debug mode when run by Quarto
else
    BUILD_DIR="${1:-_build}"
    DEBUG_MODE="${2:-false}"
fi
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

log_debug() {
    if [[ "$DEBUG_MODE" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} ${SCRIPT_NAME}: $*"
    fi
}

# Usage function
usage() {
    echo "Usage: $0 [BUILD_DIR] [DEBUG_MODE]"
    echo ""
    echo "This script can be run in two modes:"
    echo ""
    echo "1. Quarto Post-Render Script (automatic):"
    echo "   When run by Quarto, uses QUARTO_PROJECT_OUTPUT_DIR environment variable"
    echo "   Usage: $0 [DEBUG_MODE]"
    echo "   DEBUG_MODE   Enable debug output (true/false, default: false)"
    echo ""
    echo "2. Standalone Script:"
    echo "   Usage: $0 [BUILD_DIR] [DEBUG_MODE]"
    echo "   BUILD_DIR    Directory to clean SVG files in (default: _build)"
    echo "   DEBUG_MODE   Enable debug output (true/false, default: false)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Clean SVGs in _build directory (or Quarto output dir)"
    echo "  $0 true               # Enable debug mode (when run by Quarto)"
    echo "  $0 output             # Clean SVGs in output directory (standalone)"
    echo "  $0 _build true        # Clean with debug output enabled (standalone)"
}

# Main execution
main() {
    # Handle help option
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi
    
    log_info "Starting SVG cleanup in directory: ${BUILD_DIR}"
    if [[ "$DEBUG_MODE" == "true" ]]; then
        log_debug "Debug mode enabled"
        if [[ -n "${QUARTO_PROJECT_OUTPUT_DIR:-}" ]]; then
            log_debug "Running as Quarto post-render script"
            log_debug "QUARTO_PROJECT_OUTPUT_DIR=${QUARTO_PROJECT_OUTPUT_DIR}"
            if [[ -n "${QUARTO_PROJECT_RENDER_ALL:-}" ]]; then
                log_debug "Full project render detected (QUARTO_PROJECT_RENDER_ALL=${QUARTO_PROJECT_RENDER_ALL})"
            else
                log_debug "Incremental or preview render"
            fi
        else
            log_debug "Running as standalone script"
        fi
    fi
    
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
    local error_count=0
    
    for svg_file in "${svg_files[@]}"; do
        log_debug "Processing: ${svg_file}"
        
        # Check if file contains control characters before cleaning
        if LC_ALL=C grep -q '[[:cntrl:]]' "$svg_file" 2>/dev/null; then
            log_file "Cleaning: ${svg_file}"
            log_debug "Control characters detected in: ${svg_file}"
            
            # Check file permissions before attempting to modify
            if [[ ! -w "$svg_file" ]]; then
                log_error "No write permission for: ${svg_file}"
                continue
            fi
            
            # Create a backup and clean control characters
            local backup_file="${svg_file}.bak.$$"
            log_debug "Creating backup: ${backup_file}"
            
            if cp "$svg_file" "$backup_file" 2>/dev/null; then
                # Clean control characters using tr (no external dependencies needed)
                log_debug "Running tr cleaning on: ${svg_file}"
                
                # Try to clean the file - capture any errors for debugging
                local tr_error_msg
                if tr_error_msg=$(LC_ALL=C tr -d '\000-\010\013\014\016-\037' < "$svg_file" > "${svg_file}.tmp" 2>&1); then
                    # tr succeeded, now try to replace the original file
                    if mv "${svg_file}.tmp" "$svg_file" 2>/dev/null; then
                        ((cleaned_count++))
                        log_debug "Successfully cleaned: ${svg_file}"
                        # Remove backup on success
                        rm -f "$backup_file" 2>/dev/null
                        log_debug "Removed backup: ${backup_file}"
                    else
                        log_error "Failed to replace cleaned file: ${svg_file}"
                        rm -f "${svg_file}.tmp" 2>/dev/null
                        # Restore from backup
                        if mv "$backup_file" "$svg_file" 2>/dev/null; then
                            log_info "Restored original file from backup"
                        else
                            log_error "Failed to restore backup for: ${svg_file}"
                        fi
                    fi
                else
                    log_error "tr cleaning failed for: ${svg_file}"
                    if [[ -n "$tr_error_msg" ]]; then
                        log_error "tr error details: $tr_error_msg"
                    fi
                    ((error_count++))
                    rm -f "${svg_file}.tmp" 2>/dev/null
                    # Restore from backup
                    if mv "$backup_file" "$svg_file" 2>/dev/null; then
                        log_info "Restored original file from backup"
                    else
                        log_error "Failed to restore backup for: ${svg_file}"
                    fi
                fi
            else
                log_error "Cannot create backup for: ${svg_file}"
                ((error_count++))
            fi
        else
            log_debug "No control characters found in: ${svg_file}"
        fi
    done
    
    # Show summary of processing results
    if [[ $cleaned_count -gt 0 ]]; then
        log_success "Cleaned ${cleaned_count} SVG file(s)"
    fi
    
    if [[ $error_count -gt 0 ]]; then
        log_warning "Failed to clean ${error_count} SVG file(s) - see error messages above"
        log_info "Script completed with some errors but continuing"
    else
        log_info "Script completed successfully"
    fi
}

# Run main function
main "$@"
