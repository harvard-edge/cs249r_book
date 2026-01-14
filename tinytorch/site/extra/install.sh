#!/bin/bash
# ============================================================================
# TinyTorch Installer
# ============================================================================
#
# USAGE
# -----
#   curl -sSL mlsysbook.ai/tinytorch/install.sh | bash
#
# WHAT THIS SCRIPT DOES
# ---------------------
#   1. Checks prerequisites (git, Python 3.10+, venv module)
#   2. Asks where to install (default: ./tinytorch)
#   3. Shows installation plan and asks for confirmation
#   4. Downloads TinyTorch via git sparse checkout (minimal download)
#   5. Creates an isolated Python virtual environment (.venv/)
#   6. Installs all dependencies and the tito CLI
#
# AFTER INSTALLATION
# ------------------
#   cd tinytorch
#   source .venv/bin/activate
#   tito setup                    # First-time profile setup
#
# WHAT GETS CREATED
# -----------------
#   tinytorch/                    # Created in your current directory
#   ├── .venv/                    # Python virtual environment
#   ├── src/                      # Module source notebooks (20 modules)
#   ├── tinytorch/                # Package - your implementations go here
#   ├── tito/                     # CLI tool source
#   ├── milestones/               # Historical ML recreations
#   ├── tests/                    # Test suites for your code
#   ├── datasets/                 # Sample datasets (tinydigits, tinytalks)
#   ├── modules/                  # Working directory (populated by tito)
#   ├── bin/                      # CLI entry point
#   ├── requirements.txt          # Python dependencies
#   ├── pyproject.toml            # Package configuration
#   ├── settings.ini              # nbdev configuration
#   └── README.md, LICENSE        # Documentation
#
# REQUIREMENTS
# ------------
#   - git (any recent version)
#   - Python 3.10 or higher
#   - Python venv module (usually included; on Debian/Ubuntu: apt install python3-venv)
#   - Internet connection to GitHub
#
# DOCUMENTATION
# -------------
#   https://tinytorch.ai
#   https://mlsysbook.ai/tinytorch/
#
# SOURCE
# ------
#   https://github.com/harvard-edge/cs249r_book (tinytorch/ subdirectory)
#
# ============================================================================

set -e  # Exit on any error

# ============================================================================
# Configuration
# ============================================================================
REPO_URL="https://github.com/harvard-edge/cs249r_book.git"
REPO_SHORT="harvard-edge/cs249r_book"
BRANCH="dev"
INSTALL_DIR="tinytorch"
SPARSE_PATH="tinytorch"
TINYTORCH_VERSION="0.1.2"

# ============================================================================
# ANSI Color Codes (for terminal output)
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
# Cleanup Handler
# Removes temporary files if script exits unexpectedly
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

# Spinner animation for long-running background tasks
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

# Check if Python version is 3.8+
check_python_version() {
    local python_cmd="$1"
    local version major minor
    version=$($python_cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    major=$($python_cmd -c "import sys; print(sys.version_info.major)" 2>/dev/null)
    minor=$($python_cmd -c "import sys; print(sys.version_info.minor)" 2>/dev/null)

    # Check for Python 3.8+ (Required for TinyTorch)
    if [ "$major" -eq 3 ] && [ "$minor" -ge 8 ]; then
        echo "$version"
        return 0
    elif [ "$major" -gt 3 ]; then
        echo "$version"
        return 0
    else
        echo "$version"
        return 1
    fi
}

# Find the best Python command (prioritize newer versions)
get_python_cmd() {
    # Check specific versions first, prioritizing newer versions
    local candidates=("python3.13" "python3.12" "python3.11" "python3.10" "python3.9" "python3.8" "python3" "python")
    
    for cmd in "${candidates[@]}"; do
        if command_exists "$cmd"; then
            # Verify this specific candidate actually meets the version requirement
            if check_python_version "$cmd" >/dev/null 2>&1; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    echo ""
}

# ============================================================================
# Pre-flight Checks
# These run before any installation to catch problems early
# ============================================================================

check_write_permission() {
    if ! touch ".tinytorch_write_test" 2>/dev/null; then
        print_error "Cannot write to this directory"
        echo "  Check your permissions or cd to a writable directory."
        exit 1
    fi
    rm -f ".tinytorch_write_test"
}

check_not_in_venv() {
    if [ -n "$VIRTUAL_ENV" ]; then
        print_warning "You're inside a virtual environment: $VIRTUAL_ENV"
        echo "  Consider deactivating first: deactivate"
        echo ""
    fi
}

check_internet() {
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

    # Check for Python 3.10+
    PYTHON_CMD=$(get_python_cmd)
    if [ -n "$PYTHON_CMD" ]; then
        # We know it's good because get_python_cmd validates it, but we run check again to get the version string
        PY_VERSION=$(check_python_version "$PYTHON_CMD")
        print_success "Python $PY_VERSION ($PYTHON_CMD)"
    else
        # Diagnostic: Check if they have ANY python, just too old
        if command_exists python3; then
             CURRENT_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
             print_error "Found Python $CURRENT_VER, but 3.8+ is required"
        else
             print_error "Python 3.8+ not found"
        fi
        echo "  Install: https://python.org/downloads or 'brew install python'"
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
        echo "  Remove it first: rm -rf $INSTALL_DIR"
        echo "  Or cd to a different location."
        exit 1
    fi
}

# ============================================================================
# Installation Steps
# ============================================================================

prompt_install_directory() {
    echo ""
    echo -e "Where would you like to install Tiny${YELLOW}🔥Torch${NC}?"
    echo -e "  ${DIM}Press Enter for default: ${BOLD}$PWD/tinytorch${NC}"
    echo ""
    printf "Install directory [tinytorch]: "
    read -r user_dir </dev/tty

    if [ -n "$user_dir" ]; then
        INSTALL_DIR="$user_dir"
    fi
}

show_plan_and_confirm() {
    echo ""
    echo -e "This will create a ${CYAN}${INSTALL_DIR}${NC} folder here:"
    echo -e "  ${BOLD}$PWD/${INSTALL_DIR}${NC}"
    echo ""
    echo "What will be installed:"
    echo -e "  - Tiny${YELLOW}🔥Torch${NC} learning modules"
    echo "  - Python virtual environment (.venv/)"
    echo "  - tito CLI tool"
    echo ""
    echo -e "${DIM}Source: ${REPO_SHORT} (${BRANCH} branch)${NC}"
    echo ""
}

do_install() {
    echo ""

    # -------------------------------------------------------------------------
    # Step 1: Download from GitHub using sparse checkout
    # This downloads only the tinytorch/ subdirectory, not the entire repo
    # -------------------------------------------------------------------------
    echo -e "${BLUE}[1/4]${NC} Downloading from GitHub..."

    TEMP_DIR=$(mktemp -d)

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

    # Capture commit hash for provenance tracking
    COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    cd "$original_dir"

    # Move to final location
    mv "$TEMP_DIR/repo/$SPARSE_PATH" "$INSTALL_DIR"
    rm -rf "$TEMP_DIR"
    TEMP_DIR=""

    # -------------------------------------------------------------------------
    # Clean up dev-only files that students don't need
    #
    # KEEP (students need these):
    #   src/           - Module source notebooks
    #   tinytorch/     - Package where student code goes
    #   tito/          - CLI tool source
    #   milestones/    - Historical ML recreations
    #   modules/       - Working directory (cleared, populated by tito)
    #   tests/         - Test suites for student code
    #   datasets/      - Sample datasets (tinydigits, tinytalks)
    #   bin/           - CLI entry point script
    #   requirements.txt, pyproject.toml - Package dependencies
    #   settings.ini   - nbdev config (needed for exports)
    #   README.md, LICENSE - Documentation
    #
    # REMOVE (dev-only):
    # -------------------------------------------------------------------------
    rm -rf "$INSTALL_DIR/paper" \
           "$INSTALL_DIR/instructor" \
           "$INSTALL_DIR/site" \
           "$INSTALL_DIR/scripts" \
           "$INSTALL_DIR/tools" \
           "$INSTALL_DIR/binder" \
           "$INSTALL_DIR/etc" \
           "$INSTALL_DIR/assignments" \
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
           "$INSTALL_DIR/MANIFEST.in" \
           "$INSTALL_DIR/.pre-commit-config.yaml" \
           "$INSTALL_DIR/.shared-ai-rules.md" \
           "$INSTALL_DIR/.tinyrc" \
           "$INSTALL_DIR/.editorconfig" \
           "$INSTALL_DIR/.gitattributes" \
           "$INSTALL_DIR/settings.json" \
           "$INSTALL_DIR/.tinytorch" \
           2>/dev/null || true

    # Clear modules/ folder - students populate this via tito CLI exports
    if [ -d "$INSTALL_DIR/modules" ]; then
        find "$INSTALL_DIR/modules" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
    fi

    # Reset progress tracking - students start fresh
    rm -f "$INSTALL_DIR/progress.json" 2>/dev/null || true
    rm -rf "$INSTALL_DIR/.tito" 2>/dev/null || true

    # Clear tinytorch/core/ implementation files - students build these
    # Keep __init__.py files (package structure)
    if [ -d "$INSTALL_DIR/tinytorch/core" ]; then
        find "$INSTALL_DIR/tinytorch/core" -name "*.py" ! -name "__init__.py" -type f -delete 2>/dev/null || true
    fi

    print_success "Downloaded TinyTorch ${DIM}(${COMMIT_HASH})${NC}"

    # -------------------------------------------------------------------------
    # Step 2: Create Python virtual environment
    # -------------------------------------------------------------------------
    echo -e "${BLUE}[2/4]${NC} Creating Python environment..."
    cd "$INSTALL_DIR"
    
    # Use the detected 3.10+ command explicitly
    $PYTHON_CMD -m venv .venv
    source .venv/bin/activate
    print_success "Created virtual environment using $PYTHON_CMD"

    # -------------------------------------------------------------------------
    # Step 3: Install dependencies
    # -------------------------------------------------------------------------
    echo -e "${BLUE}[3/4]${NC} Installing dependencies..."

    # Upgrade pip first
    pip install --upgrade pip -q 2>/dev/null &
    local pip_pid=$!
    spin $pip_pid "Upgrading pip..."
    wait $pip_pid

    # Install from requirements.txt
    if [ -f "requirements.txt" ]; then
        total_pkgs=$(grep -c -E "^[^#]" requirements.txt 2>/dev/null || echo "?")
        pip install -r requirements.txt -q 2>/dev/null &
        local req_pid=$!
        spin $req_pid "Installing $total_pkgs packages..."
        wait $req_pid
    fi

    # Install TinyTorch package in editable mode (includes tito CLI)
    pip install -e . -q 2>/dev/null &
    local tt_pid=$!
    spin $tt_pid "Installing TinyTorch..."
    wait $tt_pid

    print_success "Installed dependencies"

    # -------------------------------------------------------------------------
    # Step 4: Verify installation
    # -------------------------------------------------------------------------
    echo -e "${BLUE}[4/4]${NC} Verifying installation..."
    if command -v tito >/dev/null 2>&1; then
        print_success "Verified tito CLI"
    else
        print_warning "Installation completed but tito not found in PATH"
        echo "  This is normal - activate the venv first."
    fi
}

print_success_message() {
    local install_path="$PWD"

    echo ""
    echo -e "${GREEN}✓${NC} Tiny${YELLOW}🔥Torch${NC} installed successfully!"
    echo ""
    echo -e "${BOLD}Next steps:${NC}"
    echo ""
    echo -e "  ${CYAN}cd $install_path${NC}"
    echo -e "  ${CYAN}source .venv/bin/activate${NC}"
    echo -e "  ${CYAN}tito setup${NC}"
    echo ""
    echo -e "${BOLD}Then start building:${NC}"
    echo ""
    echo -e "  ${CYAN}tito module start 01${NC}"
    echo ""
    echo -e "${DIM}Documentation: https://tinytorch.ai${NC}"
    echo ""
}

# ============================================================================
# Main Entry Point
# ============================================================================

main() {
    print_banner

    # Pre-flight checks
    check_write_permission
    check_not_in_venv

    echo "Checking prerequisites..."
    check_prerequisites
    check_internet

    # Ask where to install
    prompt_install_directory

    # Check directory doesn't exist (after user chooses)
    check_existing_directory

    # Show plan and confirm
    show_plan_and_confirm

    printf "Continue? [Y/n] "
    read -r REPLY </dev/tty
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_info "Installation cancelled"
        exit 0
    fi

    # Run installation
    do_install

    # Success message with next steps
    print_success_message
}

main