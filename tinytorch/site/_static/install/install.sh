#!/bin/bash
# ============================================================================
# TinyTorch Installer
# ============================================================================
#
# Usage:
#   curl -sSL tinytorch.ai/install | bash
#
# What this script does:
#   1. Checks you're in a sensible location
#   2. Checks prerequisites (git, Python 3.8+)
#   3. Shows you what it's about to do and asks for confirmation
#   4. Downloads the TinyTorch folder using git sparse checkout
#   5. Creates a Python virtual environment
#   6. Installs dependencies and the tito CLI
#
# The installer creates a 'tinytorch' directory in your current location.
#
# For more information: https://mlsysbook.ai/tinytorch/
# ============================================================================

set -e  # Exit on any error

# ============================================================================
# Configuration
# ============================================================================
REPO_URL="https://github.com/harvard-edge/cs249r_book.git"
INSTALL_DIR="tinytorch"
SPARSE_PATH="tinytorch"
MIN_PYTHON_VERSION="3.8"
TINYTORCH_VERSION="0.1.0"

# ============================================================================
# ANSI Color Codes
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'  # No Color / Reset

# ============================================================================
# Cleanup on exit
# ============================================================================
TEMP_DIR=""
cleanup() {
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}
trap cleanup EXIT
trap 'echo ""; echo -e "${RED}Installation cancelled.${NC}"; exit 1' INT TERM

# ============================================================================
# Output Helpers
# ============================================================================
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}!${NC} $1"; }
print_info() { echo -e "${BLUE}→${NC} $1"; }

print_banner() {
    echo ""
    echo -e "  ${BOLD}Tiny${NC}${YELLOW}🔥Torch${NC} ${DIM}v${TINYTORCH_VERSION}${NC}"
    echo -e "  ${DIM}Don't import it. Build it.${NC}"
    echo ""
}

# ============================================================================
# Utility Functions
# ============================================================================
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

get_python_cmd() {
    if command_exists python3; then
        echo "python3"
    elif command_exists python && python --version 2>&1 | grep -q "Python 3"; then
        echo "python"
    else
        echo ""
    fi
}

check_python_version() {
    local python_cmd="$1"
    local version major minor
    version=$($python_cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    major=$($python_cmd -c "import sys; print(sys.version_info.major)" 2>/dev/null)
    minor=$($python_cmd -c "import sys; print(sys.version_info.minor)" 2>/dev/null)

    if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
        echo "$version"
        return 0
    else
        echo "$version"
        return 1
    fi
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

check_location() {
    local current_dir="$PWD"

    # Block dangerous locations
    case "$current_dir" in
        /)
            print_error "You're in the root directory (/)"
            echo "  Please cd to a project folder first."
            exit 1
            ;;
        /tmp|/var/tmp|/private/tmp)
            print_error "You're in a temporary directory ($current_dir)"
            echo "  Files here will be deleted. Please cd somewhere permanent."
            exit 1
            ;;
        /System*|/usr*|/bin*|/sbin*|/etc*)
            print_error "You're in a system directory ($current_dir)"
            echo "  Please cd to a user directory."
            exit 1
            ;;
        "$HOME")
            print_warning "You're in your home directory ($current_dir)"
            echo "  This will create ~/tinytorch which may clutter your home folder."
            echo ""
            read -p "  Continue anyway? [y/N] " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "  Suggestion: mkdir -p ~/projects && cd ~/projects"
                exit 1
            fi
            ;;
        */Downloads|*/Downloads/*)
            print_warning "You're in your Downloads folder"
            echo "  This is probably not where you want to install TinyTorch."
            echo ""
            read -p "  Continue anyway? [y/N] " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "  Suggestion: mkdir -p ~/projects && cd ~/projects"
                exit 1
            fi
            ;;
    esac

    # Check if we're inside another git repo
    if git rev-parse --git-dir >/dev/null 2>&1; then
        local repo_root
        repo_root=$(git rev-parse --show-toplevel 2>/dev/null)
        print_warning "You're inside a git repository: $repo_root"
        echo "  Installing here will create a nested repo."
        echo ""
        read -p "  Continue anyway? [y/N] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "  Suggestion: cd to a folder outside this repo"
            exit 1
        fi
    fi

    # Check write permission
    if ! touch ".tinytorch_write_test" 2>/dev/null; then
        print_error "Cannot write to this directory"
        echo "  Check your permissions or cd somewhere else."
        exit 1
    fi
    rm -f ".tinytorch_write_test"
}

check_prerequisites() {
    local errors=0

    # Check for git
    if command_exists git; then
        GIT_VERSION=$(git --version | cut -d' ' -f3)
        print_success "Git $GIT_VERSION"
    else
        print_error "Git not found"
        echo "  Install: https://git-scm.com/downloads"
        errors=$((errors + 1))
    fi

    # Check for Python 3.8+
    PYTHON_CMD=$(get_python_cmd)
    if [ -n "$PYTHON_CMD" ]; then
        if PY_VERSION=$(check_python_version "$PYTHON_CMD"); then
            print_success "Python $PY_VERSION"
        else
            print_error "Python $PY_VERSION found, but 3.8+ required"
            echo "  Upgrade: https://python.org/downloads"
            errors=$((errors + 1))
        fi
    else
        print_error "Python 3 not found"
        echo "  Install: https://python.org/downloads"
        errors=$((errors + 1))
    fi

    # Check for venv module
    if [ -n "$PYTHON_CMD" ]; then
        if $PYTHON_CMD -c "import venv" 2>/dev/null; then
            print_success "Python venv module"
        else
            print_error "Python venv module not found"
            echo "  Install: sudo apt install python3-venv (Debian/Ubuntu)"
            errors=$((errors + 1))
        fi
    fi

    if [ $errors -gt 0 ]; then
        echo ""
        print_error "Missing prerequisites. Please fix the issues above."
        exit 1
    fi
}

check_existing_directory() {
    # Check for existing TinyTorch directory (case-insensitive check)
    if [ -d "$INSTALL_DIR" ]; then
        echo ""
        print_warning "Directory '$INSTALL_DIR' already exists"
        echo ""
        echo "  [1] Overwrite (delete existing and reinstall)"
        echo "  [2] Cancel"
        echo ""
        read -p "  Choice [2]: " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[1]$ ]]; then
            rm -rf "$INSTALL_DIR"
            print_success "Removed existing directory"
        else
            print_info "Installation cancelled"
            exit 0
        fi
    fi

}

# ============================================================================
# Main Installation
# ============================================================================

show_plan_and_confirm() {
    echo ""
    echo -e "This will create a ${CYAN}tinytorch${NC} folder here:"
    echo -e "  ${BOLD}$PWD/tinytorch${NC}"
    echo ""
    echo "What will be installed:"
    echo -e "  • Tiny${YELLOW}🔥Torch${NC} learning modules"
    echo "  • Python virtual environment"
    echo "  • tito CLI tool"
    echo ""
}

do_install() {
    echo ""

    # Step 1: Download
    echo -e "${BLUE}[1/4]${NC} Downloading from GitHub..."

    TEMP_DIR=$(mktemp -d)

    # Clone with progress (always from main branch for stability)
    if ! git clone --depth 1 --filter=blob:none --sparse --progress --branch main \
        "$REPO_URL" "$TEMP_DIR/repo" 2>&1 | while read -r line; do
            # Show git progress lines (Receiving/Resolving with percentages)
            if [[ "$line" =~ ^Receiving|^Resolving|^remote:|^Cloning ]]; then
                printf "\r      %s" "$line"
            fi
        done; then
        echo ""
        print_error "Failed to download from GitHub"
        echo "  Check your internet connection and try again."
        exit 1
    fi
    printf "\r      %-60s\n" "Done"

    local original_dir="$PWD"
    cd "$TEMP_DIR/repo"
    git sparse-checkout set "$SPARSE_PATH" 2>/dev/null
    cd "$original_dir"

    # Move to final location
    mv "$TEMP_DIR/repo/$SPARSE_PATH" "$INSTALL_DIR"
    rm -rf "$TEMP_DIR"
    TEMP_DIR=""
    print_success "Downloaded TinyTorch"

    # Step 2: Create virtual environment
    echo -e "${BLUE}[2/4]${NC} Creating Python environment..."
    cd "$INSTALL_DIR"
    $PYTHON_CMD -m venv .venv
    source .venv/bin/activate
    print_success "Created virtual environment"

    # Step 3: Install dependencies
    echo -e "${BLUE}[3/4]${NC} Installing dependencies..."

    # Upgrade pip (show that something is happening)
    echo -n "      Upgrading pip..."
    pip install --upgrade pip -q 2>/dev/null
    echo " done"

    # Install requirements
    if [ -f "requirements.txt" ]; then
        # Count packages for progress
        total_pkgs=$(grep -c -E "^[^#]" requirements.txt 2>/dev/null || echo "?")
        echo -n "      Installing $total_pkgs packages..."
        pip install -r requirements.txt -q 2>/dev/null
        echo " done"
    fi

    # Install tinytorch
    echo -n "      Installing TinyTorch..."
    pip install -e . -q 2>/dev/null
    echo " done"

    print_success "Installed dependencies"

    # Step 4: Verify
    echo -e "${BLUE}[4/4]${NC} Verifying installation..."
    if python -c "import tinytorch" 2>/dev/null; then
        print_success "Verified installation"
    else
        print_warning "Installation completed but verification failed"
        echo "  Try: cd tinytorch && source .venv/bin/activate && pip install -e ."
    fi
}

print_success_message() {
    local install_path="$PWD"

    echo ""
    echo -e "${GREEN}✓${NC} Tiny${YELLOW}🔥Torch${NC} installed successfully!"
    echo ""
    echo -e "${BOLD}Get started:${NC}"
    echo ""
    echo -e "  ${CYAN}cd $install_path${NC}"
    echo -e "  ${CYAN}source .venv/bin/activate${NC}"
    echo -e "  ${CYAN}tito setup${NC}                 # First-time setup & verification"
    echo ""
    echo -e "${BOLD}Common commands:${NC}"
    echo ""
    echo -e "  ${CYAN}tito module start 01${NC}       # Start Module 01 (Tensors)"
    echo -e "  ${CYAN}tito module complete 01${NC}    # Test & submit your work"
    echo -e "  ${CYAN}tito milestones${NC}            # View your progress"
    echo -e "  ${CYAN}tito system doctor${NC}         # Check environment health"
    echo ""
    echo -e "${BOLD}Stay updated:${NC}"
    echo ""
    echo -e "  ${CYAN}tito update${NC}                # Check for updates"
    echo ""
    echo -e "${BOLD}Resources:${NC}"
    echo ""
    echo -e "  Documentation   ${DIM}https://tinytorch.ai/docs${NC}"
    echo -e "  Community       ${DIM}https://discord.gg/tinyml${NC}"
    echo -e "  Issues          ${DIM}https://github.com/harvard-edge/cs249r_book/issues${NC}"
    echo ""
    echo -e "${DIM}To uninstall: rm -rf $install_path${NC}"
    echo ""
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Show banner first
    print_banner

    # Phase 1: Pre-flight checks
    check_location

    echo "Checking prerequisites..."
    check_prerequisites

    check_existing_directory

    # Phase 2: Show plan and confirm
    show_plan_and_confirm

    read -p "Continue? [Y/n] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_info "Installation cancelled"
        exit 0
    fi

    # Phase 3: Do the install
    do_install

    # Phase 4: Success
    print_success_message
}

main
