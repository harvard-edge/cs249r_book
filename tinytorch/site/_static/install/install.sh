#!/bin/bash
# ============================================================================
# TinyTorch Installer
# ============================================================================
#
# Usage:
#   curl -sSL tinytorch.ai/install | bash
#
# What this script does:
#   1. Checks prerequisites (git, Python 3.8+)
#   2. Shows you what it's about to do and asks for confirmation
#   3. Downloads the TinyTorch folder using git sparse checkout
#   4. Creates a Python virtual environment
#   5. Installs dependencies and the tito CLI
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
REPO_SHORT="harvard-edge/cs249r_book"
BRANCH="main"
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

# Spinner for long-running tasks
spin() {
    local pid=$1
    local msg=$2
    local spinchars='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0
    while kill -0 "$pid" 2>/dev/null; do
        printf "\r      ${DIM}%s${NC} %s" "${spinchars:i++%10:1}" "$msg"
        sleep 0.1
    done
    printf "\r      %-50s\r" ""
}

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

check_write_permission() {
    # Check write permission
    if ! touch ".tinytorch_write_test" 2>/dev/null; then
        print_error "Cannot write to this directory"
        echo "  Check your permissions or cd somewhere else."
        exit 1
    fi
    rm -f ".tinytorch_write_test"
}

check_not_in_venv() {
    # Warn if already in a virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        print_warning "You're inside a virtual environment: $VIRTUAL_ENV"
        echo "  Consider deactivating first: deactivate"
        echo ""
    fi
}

check_internet() {
    # Quick connectivity check
    if ! git ls-remote --exit-code "$REPO_URL" >/dev/null 2>&1; then
        print_error "Cannot reach GitHub"
        echo "  Check your internet connection and try again."
        exit 1
    fi
    print_success "GitHub reachable"
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
    if [ -d "$INSTALL_DIR" ]; then
        print_error "Directory '$INSTALL_DIR' already exists"
        echo "  Remove it first or cd to a different location."
        exit 1
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
    echo -e "${DIM}Source: ${REPO_SHORT} (${BRANCH} branch)${NC}"
    echo ""
}

do_install() {
    echo ""

    # Step 1: Download
    echo -e "${BLUE}[1/4]${NC} Downloading from GitHub..."

    TEMP_DIR=$(mktemp -d)

    # Clone in background with spinner
    git clone --depth 1 --filter=blob:none --sparse --branch "$BRANCH" \
        "$REPO_URL" "$TEMP_DIR/repo" >/dev/null 2>&1 &
    local clone_pid=$!
    spin $clone_pid "Cloning repository..."
    wait $clone_pid
    local clone_status=$?

    if [ $clone_status -ne 0 ]; then
        print_error "Failed to download from GitHub"
        echo "  Check your internet connection and try again."
        exit 1
    fi

    local original_dir="$PWD"
    cd "$TEMP_DIR/repo"
    git sparse-checkout set "$SPARSE_PATH" 2>/dev/null

    # Capture commit hash for provenance
    COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    cd "$original_dir"

    # Move to final location
    mv "$TEMP_DIR/repo/$SPARSE_PATH" "$INSTALL_DIR"
    rm -rf "$TEMP_DIR"
    TEMP_DIR=""

    # Remove dev-only files that students don't need
    rm -rf "$INSTALL_DIR/paper" \
           "$INSTALL_DIR/instructor" \
           "$INSTALL_DIR/site" \
           "$INSTALL_DIR/scripts" \
           "$INSTALL_DIR/tools" \
           "$INSTALL_DIR/binder" \
           "$INSTALL_DIR/etc" \
           "$INSTALL_DIR/assignments" \
           "$INSTALL_DIR/milestones" \
           "$INSTALL_DIR/benchmark_results" \
           "$INSTALL_DIR/.git-hooks" \
           "$INSTALL_DIR/.claude" \
           "$INSTALL_DIR/.cursor" \
           "$INSTALL_DIR/.vscode" \
           "$INSTALL_DIR/Makefile" \
           "$INSTALL_DIR/activate.sh" \
           "$INSTALL_DIR/setup-dev.sh" \
           "$INSTALL_DIR/setup-environment.sh" \
           "$INSTALL_DIR/CONTRIBUTING.md" \
           "$INSTALL_DIR/INSTRUCTOR.md" \
           "$INSTALL_DIR/settings.ini" \
           "$INSTALL_DIR/MANIFEST.in" \
           "$INSTALL_DIR/.pre-commit-config.yaml" \
           "$INSTALL_DIR/.shared-ai-rules.md" \
           "$INSTALL_DIR/.tinyrc" \
           "$INSTALL_DIR/.editorconfig" \
           "$INSTALL_DIR/.gitattributes" \
           2>/dev/null || true

    # Clear modules/ folder - students populate via CLI exports from src/
    if [ -d "$INSTALL_DIR/modules" ]; then
        find "$INSTALL_DIR/modules" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
    fi

    print_success "Downloaded TinyTorch ${DIM}(${COMMIT_HASH})${NC}"

    # Step 2: Create virtual environment
    echo -e "${BLUE}[2/4]${NC} Creating Python environment..."
    cd "$INSTALL_DIR"
    $PYTHON_CMD -m venv .venv
    source .venv/bin/activate
    print_success "Created virtual environment"

    # Step 3: Install dependencies
    echo -e "${BLUE}[3/4]${NC} Installing dependencies..."

    # Upgrade pip
    pip install --upgrade pip -q 2>/dev/null &
    local pip_pid=$!
    spin $pip_pid "Upgrading pip..."
    wait $pip_pid

    # Install requirements
    if [ -f "requirements.txt" ]; then
        total_pkgs=$(grep -c -E "^[^#]" requirements.txt 2>/dev/null || echo "?")
        pip install -r requirements.txt -q 2>/dev/null &
        local req_pid=$!
        spin $req_pid "Installing $total_pkgs packages..."
        wait $req_pid
    fi

    # Install tinytorch
    pip install -e . -q 2>/dev/null &
    local tt_pid=$!
    spin $tt_pid "Installing TinyTorch..."
    wait $tt_pid

    print_success "Installed dependencies"

    # Step 4: Verify
    echo -e "${BLUE}[4/4]${NC} Verifying installation..."
    if command -v tito >/dev/null 2>&1; then
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
    echo -e "  ${CYAN}tito module start 01${NC}       # Start building!"
    echo ""
    echo -e "${BOLD}Useful commands:${NC}"
    echo ""
    echo -e "  ${CYAN}tito module status${NC}         # View your progress"
    echo -e "  ${CYAN}tito system health${NC}         # Check environment health"
    echo -e "  ${CYAN}tito update${NC}                # Check for updates"
    echo ""
    echo -e "${DIM}Documentation: https://tinytorch.ai${NC}"
    echo ""
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Show banner first
    print_banner

    # Phase 1: Pre-flight checks
    check_write_permission
    check_existing_directory
    check_not_in_venv

    echo "Checking prerequisites..."
    check_prerequisites
    check_internet

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
