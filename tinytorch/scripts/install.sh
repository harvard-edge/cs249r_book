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
#   2. Downloads only the tinytorch/ folder using git sparse checkout
#      (This avoids downloading the entire book repository)
#   3. Creates a Python virtual environment
#   4. Installs dependencies and the tito CLI
#   5. Verifies the installation
#
# Requirements:
#   - git (any recent version)
#   - Python 3.8 or later
#   - Internet connection
#
# The installer creates a 'tinytorch' directory in your current location.
# All work happens inside this directory with an isolated virtual environment.
#
# For more information: https://mlsysbook.ai/tinytorch/
# ============================================================================

set -e  # Exit on any error

# ============================================================================
# ANSI Color Codes
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'  # No Color / Reset

# ============================================================================
# Configuration
# ============================================================================
REPO_URL="https://github.com/harvard-edge/cs249r_book.git"
INSTALL_DIR="tinytorch"
SPARSE_PATH="tinytorch"
MIN_PYTHON_VERSION="3.8"
TINYTORCH_VERSION="0.1.0"

# ============================================================================
# Output Helpers
# ============================================================================

# Print the TinyTorch ASCII banner (matches tito CLI logo style)
print_banner() {
    echo ""
    echo -e "${YELLOW}    🔥                                    🔥${NC}"
    echo -e "${BOLD}    ████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗${NC}"
    echo -e "${BOLD}    ╚${YELLOW}T${NC}${BOLD}═██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║${NC}"
    echo -e "${BOLD}     ${YELLOW}I${NC}${BOLD} ██║   ██║   ██║██████╔╝██║     ███████║${NC}"
    echo -e "${BOLD}     ${YELLOW}N${NC}${BOLD} ██║   ██║   ██║██╔══██╗██║     ██╔══██║${NC}"
    echo -e "${BOLD}     ${YELLOW}Y${NC}${BOLD} ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║${NC}"
    echo -e "${BOLD}       ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝${NC}"
    echo ""
    echo -e "        ${YELLOW}🔥${NC} ${BOLD}Don't import it. Build it.${NC}"
    echo -e "        ${DIM}Version ${TINYTORCH_VERSION}${NC}"
    echo ""
}

# Print a section header
print_header() {
    echo ""
    echo -e "${BOLD}━━━ $1 ━━━${NC}"
}

# Print a step indicator (for actions in progress)
print_step() {
    echo -e "${BLUE}▶${NC} $1"
}

# Print a sub-step (indented detail)
print_substep() {
    echo -e "  ${DIM}$1${NC}"
}

# Print success message
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Print error message
print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Print warning message
print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Print info message
print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

# ============================================================================
# Utility Functions
# ============================================================================

# Check if a command exists in PATH
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get the appropriate Python command (python3 preferred over python)
get_python_cmd() {
    if command_exists python3; then
        echo "python3"
    elif command_exists python; then
        # Verify it's actually Python 3
        if python --version 2>&1 | grep -q "Python 3"; then
            echo "python"
        else
            echo ""
        fi
    else
        echo ""
    fi
}

# Check Python version meets minimum requirement
# Returns: version string, exit code 0 if meets requirement
check_python_version() {
    local python_cmd="$1"
    local version
    local major
    local minor

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
# Installation Steps
# ============================================================================

# Step 1: Check all prerequisites before starting
check_prerequisites() {
    print_header "Checking Prerequisites"

    local errors=0

    # Check for git
    if command_exists git; then
        local git_version
        git_version=$(git --version | cut -d' ' -f3)
        print_success "Git ${git_version}"
    else
        print_error "Git not found"
        print_substep "Install git: https://git-scm.com/downloads"
        errors=$((errors + 1))
    fi

    # Check for Python 3.8+
    local python_cmd
    python_cmd=$(get_python_cmd)
    if [ -n "$python_cmd" ]; then
        local py_version
        py_version=$(check_python_version "$python_cmd")
        if [ $? -eq 0 ]; then
            print_success "Python ${py_version} (${python_cmd})"
            PYTHON_CMD="$python_cmd"
        else
            print_error "Python ${py_version} found, but ${MIN_PYTHON_VERSION}+ required"
            print_substep "Upgrade Python: https://python.org/downloads"
            errors=$((errors + 1))
        fi
    else
        print_error "Python 3 not found"
        print_substep "Install Python ${MIN_PYTHON_VERSION}+: https://python.org/downloads"
        errors=$((errors + 1))
    fi

    # Check for venv module
    if [ -n "$PYTHON_CMD" ]; then
        if $PYTHON_CMD -c "import venv" 2>/dev/null; then
            print_success "Python venv module"
        else
            print_error "Python venv module not found"
            print_substep "Install: sudo apt install python3-venv (Debian/Ubuntu)"
            errors=$((errors + 1))
        fi
    fi

    # Check if install directory already exists
    if [ -d "$INSTALL_DIR" ]; then
        echo ""
        print_warning "Directory '${INSTALL_DIR}' already exists"
        echo -e "  ${DIM}This will remove the existing directory and reinstall.${NC}"
        echo ""
        read -p "  Continue? [y/N] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$INSTALL_DIR"
            print_success "Removed existing directory"
        else
            echo ""
            print_error "Installation cancelled by user"
            exit 1
        fi
    fi

    # Fail if any prerequisites are missing
    if [ $errors -gt 0 ]; then
        echo ""
        print_error "Prerequisites check failed (${errors} issue(s))"
        print_substep "Please fix the issues above and try again."
        exit 1
    fi

    print_success "All prerequisites met"
}

# Step 2: Download TinyTorch using git sparse checkout
# This only downloads the tinytorch/ folder, not the entire book repo
clone_sparse() {
    print_header "Downloading TinyTorch"

    print_step "Using sparse checkout (faster, smaller download)"
    print_substep "Only downloading tinytorch/ folder from repository"

    # Create a temporary directory for the clone operation
    local temp_dir
    temp_dir=$(mktemp -d)

    # Clone with minimal data transfer:
    # --depth 1        : Only latest commit (no history)
    # --filter=blob:none : Don't download file contents yet
    # --sparse         : Enable sparse checkout
    print_substep "Cloning repository skeleton..."
    if ! git clone --depth 1 --filter=blob:none --sparse \
        "$REPO_URL" "$temp_dir" 2>&1 | grep -v "^remote:" | head -5; then
        print_error "Failed to clone repository"
        rm -rf "$temp_dir"
        exit 1
    fi

    # Configure sparse checkout to only get tinytorch folder
    print_substep "Fetching tinytorch/ folder..."
    cd "$temp_dir"
    git sparse-checkout set "$SPARSE_PATH" 2>/dev/null

    # Move the tinytorch folder to the final location
    cd - > /dev/null
    mv "$temp_dir/$SPARSE_PATH" "$INSTALL_DIR"

    # Clean up temporary directory
    rm -rf "$temp_dir"

    # Show what was downloaded
    local file_count
    file_count=$(find "$INSTALL_DIR" -type f | wc -l | tr -d ' ')
    print_success "Downloaded TinyTorch (${file_count} files)"
}

# Step 3: Create an isolated Python virtual environment
setup_venv() {
    print_header "Setting Up Environment"

    print_step "Creating Python virtual environment"

    cd "$INSTALL_DIR"

    # Create the virtual environment
    $PYTHON_CMD -m venv .venv

    # Activate it for the rest of the installation
    # shellcheck disable=SC1091
    source .venv/bin/activate

    print_success "Created .venv/"
    print_substep "Python: $(which python)"
}

# Step 4: Install Python dependencies
install_deps() {
    print_step "Installing dependencies"

    # Upgrade pip first (quietly)
    print_substep "Upgrading pip..."
    pip install --upgrade pip -q 2>/dev/null

    # Install from requirements.txt if it exists
    if [ -f "requirements.txt" ]; then
        print_substep "Installing requirements.txt..."
        pip install -r requirements.txt -q 2>/dev/null
        print_success "Installed dependencies"
    fi

    # Install tinytorch package in editable mode
    # This makes the 'tito' command available
    print_substep "Installing TinyTorch package..."
    pip install -e . -q 2>/dev/null
    print_success "Installed tito CLI"
}

# Step 5: Verify the installation works
verify_install() {
    print_header "Verifying Installation"

    local all_good=true

    # Check tito CLI is available
    if command_exists tito; then
        print_success "tito command available"
    else
        # Try via python module as fallback
        if python -m tito --version >/dev/null 2>&1; then
            print_success "tito available (via python -m tito)"
        else
            print_warning "tito CLI not in PATH"
            print_substep "Will work after activating the virtual environment"
            all_good=false
        fi
    fi

    # Check tinytorch module imports
    if python -c "import tinytorch" 2>/dev/null; then
        print_success "tinytorch module importable"
    else
        print_warning "tinytorch module not importable"
        all_good=false
    fi

    # Quick sanity check on module count
    local module_count
    module_count=$(find modules -name "*.ipynb" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$module_count" -gt 0 ]; then
        print_success "Found ${module_count} learning modules"
    fi

    if [ "$all_good" = true ]; then
        print_success "All checks passed"
    fi
}

# Final: Print success message and next steps
print_final_message() {
    local abs_path
    abs_path=$(pwd)

    echo ""
    echo -e "${GREEN}${BOLD}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${NC}"
    echo -e "${GREEN}${BOLD}┃      🎉 TinyTorch installed successfully! 🎉         ┃${NC}"
    echo -e "${GREEN}${BOLD}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${NC}"
    echo ""
    echo -e "${BOLD}Get started:${NC}"
    echo ""
    echo -e "  ${CYAN}cd ${INSTALL_DIR}${NC}"
    echo -e "  ${CYAN}source .venv/bin/activate${NC}"
    echo -e "  ${CYAN}tito${NC}"
    echo ""
    echo -e "${DIM}Or copy this one-liner:${NC}"
    echo ""
    echo -e "  ${CYAN}cd ${INSTALL_DIR} && source .venv/bin/activate && tito${NC}"
    echo ""
    echo -e "${BOLD}Useful commands:${NC}"
    echo ""
    echo -e "  ${YELLOW}tito setup${NC}           First-time setup & verification"
    echo -e "  ${YELLOW}tito module start 01${NC} Start Module 01 (Tensors)"
    echo -e "  ${YELLOW}tito system doctor${NC}   Check environment health"
    echo -e "  ${YELLOW}tito update${NC}          Check for updates"
    echo ""
    echo -e "${BOLD}Resources:${NC}"
    echo ""
    echo -e "  📚 Documentation  ${CYAN}https://mlsysbook.ai/tinytorch/${NC}"
    echo -e "  💬 Community      ${CYAN}https://discord.gg/tinyml${NC}"
    echo -e "  🐛 Issues         ${CYAN}https://github.com/harvard-edge/cs249r_book/issues${NC}"
    echo ""
    echo -e "${DIM}Installation path: ${abs_path}${NC}"
    echo ""
}

# ============================================================================
# Main Entry Point
# ============================================================================

main() {
    # Show banner
    print_banner

    # Run installation steps
    check_prerequisites
    clone_sparse
    setup_venv
    install_deps
    verify_install
    print_final_message
}

# Run the installer
main
