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
# These can be overridden via environment variables for testing:
#   TINYTORCH_BRANCH=dev curl -sSL mlsysbook.ai/tinytorch/install.sh | bash
#   TINYTORCH_VERSION=0.1.5 TINYTORCH_BRANCH=feature/foo ./install.sh
#   TINYTORCH_NON_INTERACTIVE=1 ./install.sh  # Skip all prompts (for CI)
REPO_URL="https://github.com/harvard-edge/cs249r_book.git"
REPO_SHORT="harvard-edge/cs249r_book"
TAGS_API="https://api.github.com/repos/harvard-edge/cs249r_book/tags"
TAG_PREFIX="tinytorch-v"
BRANCH="${TINYTORCH_BRANCH:-main}"
INSTALL_DIR="${TINYTORCH_INSTALL_DIR:-tinytorch}"
SPARSE_PATH="tinytorch"
# Non-interactive mode: skip prompts, use defaults (for CI/testing)
NON_INTERACTIVE="${TINYTORCH_NON_INTERACTIVE:-}"
# Version is fetched from GitHub tags (single source of truth)
# Can be overridden for testing: TINYTORCH_VERSION=0.1.5 ./install.sh
TINYTORCH_VERSION="${TINYTORCH_VERSION:-}"

# ============================================================================
# Timeouts
# ============================================================================
# Every external command below (Python detection, network calls, downloads,
# package installs, interactive prompts) is capped by one of these. None of
# them should ever be reached in a healthy environment; they exist purely so
# that a broken environment produces a clear, timed-out error message instead
# of an installer that just sits there forever with no output and no way to
# tell what's wrong. See https://github.com/harvard-edge/cs249r_book/issues/1960
PYTHON_CHECK_TIMEOUT=5      # Detecting/validating a python command
NETWORK_CHECK_TIMEOUT=15    # Checking GitHub is reachable
CLONE_TIMEOUT=120           # Downloading TinyTorch (shallow sparse clone)
VENV_CREATE_TIMEOUT=60      # Creating the virtual environment
PIP_INSTALL_TIMEOUT=600     # Installing Python packages (10 min ceiling)
TTY_READ_TIMEOUT=30         # Waiting on an interactive prompt

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

# Run a command with a hard time limit, so a hung or misbehaving external
# program (a broken Python launcher, a stalled network call, a package
# manager waiting on a credential prompt nobody can answer) fails loudly
# after a bounded wait instead of freezing this installer forever with no
# output. That silent-forever failure is exactly what was reported in
# https://github.com/harvard-edge/cs249r_book/issues/1960: the script
# printed "Git OK" and then simply never continued, with no error at all.
#
# Usage: run_with_timeout <seconds> <command> [args...]
run_with_timeout() {
    local seconds="$1"
    shift
    if command_exists timeout; then
        timeout "$seconds" "$@"
    elif command_exists gtimeout; then
        # macOS without GNU coreutils names it gtimeout to avoid clobbering
        # the BSD tools of the same short names.
        gtimeout "$seconds" "$@"
    else
        # No timeout utility available at all. Running unguarded is better
        # than refusing to run, but a hang here can no longer be caught.
        "$@"
    fi
}

# Fetch latest version from GitHub tags API (single source of truth)
# Sets TINYTORCH_VERSION global variable
fetch_latest_version() {
    # Skip if already set via environment variable
    if [ -n "$TINYTORCH_VERSION" ]; then
        return 0
    fi

    # Try curl first (more reliable across platforms)
    if command_exists curl; then
        local response
        response=$(curl -fsSL --max-time 10 "$TAGS_API" 2>/dev/null) || true
        if [ -n "$response" ]; then
            # Parse JSON to find first tinytorch-v* tag
            # Uses grep/sed for portability (no jq dependency)
            local tag_name
            tag_name=$(echo "$response" | grep -o "\"name\": *\"${TAG_PREFIX}[^\"]*\"" | head -1 | sed 's/.*"name": *"\([^"]*\)".*/\1/')
            if [ -n "$tag_name" ]; then
                TINYTORCH_VERSION="${tag_name#$TAG_PREFIX}"
                return 0
            fi
        fi
    fi

    # Fallback: fetch pyproject.toml directly from raw.githubusercontent.com
    if command_exists curl; then
        local pyproject_url="https://raw.githubusercontent.com/${REPO_SHORT}/${BRANCH}/tinytorch/pyproject.toml"
        local pyproject
        pyproject=$(curl -fsSL --max-time 10 "$pyproject_url" 2>/dev/null) || true
        if [ -n "$pyproject" ]; then
            local version
            version=$(echo "$pyproject" | grep -E "^version" | head -1 | sed 's/.*= *"\([^"]*\)".*/\1/')
            if [ -n "$version" ]; then
                TINYTORCH_VERSION="$version"
                return 0
            fi
        fi
    fi

    # Final fallback: unknown version (will still work, just won't show version)
    TINYTORCH_VERSION="latest"
}

# Check if Python version is 3.10+
#
# Every candidate is run through run_with_timeout: on Windows, "python" can
# resolve to a Store "App Execution Alias" placeholder
# (%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe) that Windows registers
# by default even when no real Python is installed. Running that
# placeholder can silently try to pop open the Microsoft Store and never
# return, which otherwise hangs this whole installer with zero output
# (issue #1960) instead of just failing over to the next candidate.
check_python_version() {
    local python_cmd="$1"
    local output version major minor
    # One combined call instead of three separate ones, so a hung/broken
    # launcher only costs one timeout wait, not three.
    output=$(run_with_timeout "$PYTHON_CHECK_TIMEOUT" "$python_cmd" -c \
        "import sys; print(sys.version_info.major, sys.version_info.minor, f'{sys.version_info.major}.{sys.version_info.minor}')" \
        2>/dev/null)
    read -r major minor version <<< "$output"

    # Check for Python 3.10+ (Required for TinyTorch)
    if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
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

# Look for a working Python install in the places Windows installers
# commonly put one, for when nothing usable is on PATH at all. This is
# deliberately a last resort after get_python_cmd's normal PATH-based
# search fails: it exists so a user whose real Python just isn't on PATH
# (a very common state after installing via Anaconda, the Microsoft
# Store, or python.org without checking "Add to PATH") gets TinyTorch
# working immediately, rather than being sent off to fix PATH themselves
# first. Every candidate still goes through check_python_version, so
# this can't pick a broken or too-old Python, and it can't hang: each
# attempt is bounded by the same PYTHON_CHECK_TIMEOUT as everything else.
search_for_python_windows() {
    local candidates=()

    # The Python Launcher (py.exe) keeps its own registry of installs
    # independent of PATH, so it can find a working Python even when the
    # bare "python" command resolves to Windows' Store placeholder.
    if command_exists py; then
        candidates+=("py")
    fi

    # Common install locations, newest-looking first. Using glob patterns
    # directly (with nullglob) rather than `find`, since these are all
    # shallow, well-known paths and this needs to stay fast.
    local search_globs=(
        "$LOCALAPPDATA/Programs/Python/Python3*/python.exe"
        "$PROGRAMFILES/Python3*/python.exe"
        "/c/Program Files (x86)/Python3*/python.exe"
        "$USERPROFILE/anaconda3/python.exe"
        "$USERPROFILE/miniconda3/python.exe"
        "$USERPROFILE/AppData/Local/anaconda3/python.exe"
        "/c/ProgramData/Anaconda3/python.exe"
        "/c/ProgramData/miniconda3/python.exe"
        "$USERPROFILE/.pyenv/pyenv-win/versions/*/python.exe"
        "$LOCALAPPDATA/uv/python/*/python.exe"
    )

    local pattern path
    shopt -s nullglob
    for pattern in "${search_globs[@]}"; do
        for path in $pattern; do
            candidates+=("$path")
        done
    done
    shopt -u nullglob

    local candidate
    for candidate in "${candidates[@]}"; do
        if check_python_version "$candidate" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# Find the best Python command (prioritize newer versions)
get_python_cmd() {
    local platform
    platform=$(get_platform)

    # Check specific versions first, prioritizing newer versions
    # On Windows, prefer 'python' to avoid Microsoft Store alias that
    # resolves 'python3' to a stub and creates Unix-style venv paths.
    # Contributed by @adil-mubashir-ch (PR #1169)
    #
    # Windows Python installs are not all the same shape. python.org's
    # installer and Anaconda create a real python.exe. But Python installed
    # through a version manager (`uv python install`, pyenv-win, etc.) is
    # commonly exposed only as a python.cmd or python.bat shim. Git Bash
    # (MSYS) does not apply Windows' own PATHEXT extension search to a
    # bare, extension-less name the way cmd.exe/PowerShell do, so
    # `command -v python` can fail to find a perfectly good Python that
    # only exists as python.cmd -- reporting "Python not found" even
    # though it is installed and working. Try every extension Windows
    # itself would search, in the same order, and use whichever actually
    # runs.
    if [ "$platform" = "windows" ]; then
        local suffix candidate
        for suffix in "" .exe .cmd .bat; do
            candidate="python${suffix}"
            if command_exists "$candidate" && check_python_version "$candidate" >/dev/null 2>&1; then
                echo "$candidate"
                return 0
            fi
        done

        # Nothing usable on PATH. Rather than stop here and make the user
        # go fix PATH or Windows settings themselves, look in the handful
        # of places Windows Python installers actually put python.exe and
        # use the first working one we find. This is echoed as an
        # absolute path (unlike the bare command names above), which is
        # how the caller tells apart "found on PATH" from "found by
        # searching" to explain itself to the user.
        candidate=$(search_for_python_windows)
        if [ -n "$candidate" ]; then
            echo "$candidate"
            return 0
        fi

        echo ""
        return 0
    fi

    local candidates=("python3.13" "python3.12" "python3.11" "python3.10" "python3.9" "python3" "python")

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

# Find the system platform (linux, macos, windows)
# Contributed by @rnjema (PR #1105)
get_platform() {
    local uname_out
    uname_out=$(uname -s)
    case "${uname_out}" in
        Linux*)     echo "linux";;
        Darwin*)    echo "macos";;
        CYGWIN*|MINGW*|MSYS*) echo "windows";;
        *)          echo "unknown";;
    esac
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
    # git ls-remote has no built-in timeout: a firewall or proxy that
    # silently drops packets (rather than rejecting the connection) can
    # leave this hanging for minutes with no feedback. Bound it so a dead
    # network produces a clear message quickly instead of an installer
    # that just appears to be stuck.
    if ! run_with_timeout "$NETWORK_CHECK_TIMEOUT" git ls-remote --exit-code "$REPO_URL" >/dev/null 2>&1; then
        print_error "Cannot reach GitHub"
        echo "  This usually means one of:"
        echo "    - Your internet connection is down or very slow"
        echo "    - A firewall, VPN, or antivirus is blocking git/GitHub"
        echo "    - You're on a work/school network that blocks GitHub directly"
        echo "  Try opening https://github.com in a browser first to confirm you can reach it,"
        echo "  then run this installer again."
        exit 1
    fi
    print_success "GitHub reachable"
}

# Check if git version > 2.25.1
check_git_version() {
    local major minor rev
    major="$(echo "$1" | cut -d '.' -f 1)" 2>/dev/null
    minor="$(echo "$1" | cut -d '.' -f 2)" 2>/dev/null
    rev="$(echo "$1" | cut -d '.' -f 3)" 2>/dev/null

    # Check for git >= 2.25.2 (sparse url bug fixed)
    if [ "$major" -eq 2 ] && [ "$minor" -ge 26 ]; then
        return 0
    elif [ "$major" -gt 3 ]; then
        return 0
    elif [ "$major" -eq 2 ] && [ "$minor" -eq 25 ] &&  [ "$rev" -ge 2 ] ; then
        return 0
    else
        return 1
    fi
}

check_prerequisites() {
    local errors=0

    # Check for git
    if command_exists git; then
        GIT_VERSION="$(git --version | cut -d ' ' -f3)"
        if check_git_version "$GIT_VERSION"; then
            print_success "Git $GIT_VERSION"
        else
            print_error "Git version $GIT_VERSION is too old (2.25.2 or newer is required)"
            echo "  Download the latest version: https://git-scm.com/downloads"
            echo "  (On Ubuntu 20.04, upgrading to 22.04+ also fixes this.)"
            errors=$((errors + 1))
        fi
    else
        print_error "Git not found"
        echo "  TinyTorch uses git to download itself. Install it from:"
        echo "    https://git-scm.com/downloads"
        echo "  then restart your terminal and try again."
        errors=$((errors + 1))
    fi

    # Check for Python 3.10+
    PYTHON_CMD=$(get_python_cmd)
    PLATFORM=$(get_platform)
    if [ -n "$PYTHON_CMD" ]; then
        # We know it's good because get_python_cmd validates it, but we run check again to get the version string
        PY_VERSION=$(check_python_version "$PYTHON_CMD")
        print_success "Python $PY_VERSION ($PYTHON_CMD)"
        # An absolute path (as opposed to a bare command name like "python"
        # or "python.cmd") means this came from search_for_python_windows,
        # not from PATH -- tell the user, since silently using a Python
        # they don't know about is more confusing than helpful.
        case "$PYTHON_CMD" in
            */*)
                print_info "This Python isn't on your PATH -- using it just for this install"
                echo "  To use it directly later too, add it to your PATH: $PYTHON_CMD"
                ;;
        esac
    else
        # Diagnostic: Check if they have ANY python, just too old
        if command_exists python3; then
             CURRENT_VER=$(run_with_timeout "$PYTHON_CHECK_TIMEOUT" python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
             print_error "Found Python $CURRENT_VER, but 3.10+ is required"
             echo "  Please install a newer Python: https://python.org/downloads"
        else
             print_error "Python 3.10+ not found"
             echo "  Install it from: https://python.org/downloads"
             echo "  (macOS: 'brew install python'. Debian/Ubuntu: 'apt install python3.12'.)"
        fi
        if [ "$(get_platform)" = "windows" ]; then
            echo ""
            echo "  This installer already checked the common install locations"
            echo "  (python.org, Anaconda/Miniconda, the Python Launcher, pyenv-win,"
            echo "  uv) and none of them had a working Python 3.10+, so it really"
            echo "  does look like Python isn't installed, or is somewhere unusual."
            echo ""
            echo "  If you're sure Python IS installed somewhere, Windows may be"
            echo "  silently redirecting the 'python' command to its own placeholder"
            echo "  instead of your real install. To fix that:"
            echo "    Settings > Apps > Advanced app settings > App execution aliases"
            echo "    then turn OFF the 'python.exe' and 'python3.exe' entries."
            echo "  After that, close and reopen your terminal and run this again."
        fi
        errors=$((errors + 1))
    fi

    # Check for venv module
    if [ -n "$PYTHON_CMD" ]; then
        if run_with_timeout "$PYTHON_CHECK_TIMEOUT" "$PYTHON_CMD" -c "import venv" 2>/dev/null; then
            print_success "Python venv module"
        else
            print_error "Python's built-in 'venv' module is missing or unresponsive"
            echo "  Debian/Ubuntu: sudo apt install python3-venv"
            echo "  Other platforms: reinstall Python from https://python.org/downloads"
            echo "  (make sure not to select a 'minimal'/custom install that skips it)"
            errors=$((errors + 1))
        fi
    fi

    # Show Windows-specific guidance (contributed by @rnjema)
    if [ "$PLATFORM" = "windows" ]; then
        print_info "Windows detected - using Git Bash/WSL compatible mode"
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
    # Non-interactive mode: use INSTALL_DIR as-is (from env var or default)
    if [ -n "$NON_INTERACTIVE" ]; then
        return
    fi

    # No TTY available: use defaults silently
    if ! [ -t 0 ] && ! [ -e /dev/tty ]; then
        return
    fi

    echo ""
    echo -e "Where would you like to install Tiny${YELLOW}🔥Torch${NC}?"
    echo -e "  ${DIM}Press Enter for default: ${BOLD}$PWD/tinytorch${NC}"
    echo ""
    printf "Install directory [tinytorch]: "

    # /dev/tty can exist as a path without actually delivering input (some
    # IDE-embedded terminals, containers, restricted shells). Without a
    # timeout, `read` would wait for that input forever with no way for the
    # user to tell the difference between "still thinking" and "stuck".
    local user_dir
    if ! read -r -t "$TTY_READ_TIMEOUT" user_dir </dev/tty; then
        echo ""
        print_warning "No response after ${TTY_READ_TIMEOUT}s, using the default location."
        return
    fi

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

    # A shallow sparse clone of one small folder should take a few seconds.
    # If it's still running after CLONE_TIMEOUT, something is wrong
    # (dead network, proxy silently dropping the connection, etc.) --
    # better to fail with a clear message than leave the spinner running
    # forever with no way to tell it apart from a slow-but-working download.
    run_with_timeout "$CLONE_TIMEOUT" git clone --depth 1 --filter=blob:none --sparse --branch "$BRANCH" \
        "$REPO_URL" "$TEMP_DIR/repo" >/dev/null 2>&1 &
    local clone_pid=$!
    spin $clone_pid "Cloning repository..."
    wait $clone_pid
    local clone_status=$?

    if [ $clone_status -eq 124 ]; then
        print_error "Download timed out after ${CLONE_TIMEOUT}s"
        echo "  This usually means a firewall, VPN, or proxy is silently blocking"
        echo "  the connection to GitHub rather than rejecting it outright."
        echo "  Try a different network, or disable VPN/proxy, then try again."
        exit 1
    elif [ $clone_status -ne 0 ]; then
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

    # Use the detected 3.10+ command explicitly. A generous timeout guards
    # against a corrupted Python install hanging during its internal
    # ensurepip bootstrap; venv creation itself is a local, fast operation.
    if ! run_with_timeout "$VENV_CREATE_TIMEOUT" "$PYTHON_CMD" -m venv .venv; then
        print_error "Could not create the virtual environment"
        echo "  This usually means the detected Python ($PYTHON_CMD) is broken or"
        echo "  incomplete. Try reinstalling Python from https://python.org/downloads"
        echo "  and running this installer again."
        exit 1
    fi

    # Activate venv (handle Windows Git Bash vs Unix)
    if [ -f ".venv/Scripts/activate" ]; then
        # Windows (Git Bash)
        source .venv/Scripts/activate
    else
        # macOS/Linux
        source .venv/bin/activate
    fi
    print_success "Created virtual environment using $PYTHON_CMD"

    # -------------------------------------------------------------------------
    # Step 3: Install dependencies
    # Uses $PYTHON_CMD -m pip for reliability (contributed by @rnjema)
    # -------------------------------------------------------------------------
    echo -e "${BLUE}[3/4]${NC} Installing dependencies..."

    # A generous timeout (PIP_INSTALL_TIMEOUT) rather than a short one:
    # a slow-but-working install on a bad connection can legitimately take
    # a few minutes and should be allowed to finish. What this actually
    # guards against is a true hang -- e.g. a global pip config pointing at
    # a private package index that silently waits for credentials nobody
    # can type in a piped `curl | bash` session.
    pip_install_step() {
        local description="$1"
        shift
        run_with_timeout "$PIP_INSTALL_TIMEOUT" "$PYTHON_CMD" -m pip "$@" -q 2>/dev/null &
        local step_pid=$!
        spin $step_pid "$description"
        wait $step_pid
        local status=$?
        if [ $status -eq 124 ]; then
            print_error "Timed out after ${PIP_INSTALL_TIMEOUT}s: $description"
            echo "  This can happen with a very slow connection, or if pip is configured"
            echo "  to use a private/internal package index that needs credentials."
            echo "  Check for a pip.ini/pip.conf pointing at a custom index, or try again"
            echo "  on a different network."
            exit 1
        elif [ $status -ne 0 ]; then
            print_error "Failed: $description"
            echo "  Check your internet connection and try again. If it keeps failing,"
            echo "  re-run with pip's quiet flag removed to see the underlying error:"
            echo "    $PYTHON_CMD -m pip $*"
            exit 1
        fi
    }

    pip_install_step "Upgrading pip..." install --upgrade pip

    if [ -f "requirements.txt" ]; then
        total_pkgs=$(grep -c -E "^[^#]" requirements.txt 2>/dev/null || echo "?")
        pip_install_step "Installing $total_pkgs packages..." install -r requirements.txt
    fi

    pip_install_step "Installing TinyTorch..." install -e .

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

    # Determine correct activation command for the platform
    local activate_cmd="source .venv/bin/activate"
    if [ -f ".venv/Scripts/activate" ]; then
        # Windows (Git Bash)
        activate_cmd="source .venv/Scripts/activate"
    fi

    echo ""
    echo -e "${GREEN}✓${NC} Tiny${YELLOW}🔥Torch${NC} installed successfully!"
    echo ""
    echo -e "${BOLD}Next steps:${NC}"
    echo ""
    echo -e "  ${CYAN}cd $install_path${NC}"
    echo -e "  ${CYAN}$activate_cmd${NC}"
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
    # Fetch version from GitHub (single source of truth: pyproject.toml via tags)
    fetch_latest_version

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

    # Show plan and confirm (skip in non-interactive mode)
    if [ -z "$NON_INTERACTIVE" ] && { [ -t 0 ] || [ -e /dev/tty ]; }; then
        show_plan_and_confirm

        printf "Continue? [Y/n] "
        # See prompt_install_directory for why this needs a timeout: a tty
        # that exists but never delivers input must not hang forever.
        if ! read -r -t "$TTY_READ_TIMEOUT" REPLY </dev/tty; then
            echo ""
            print_warning "No response after ${TTY_READ_TIMEOUT}s, continuing automatically."
            REPLY="Y"
        fi
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            print_info "Installation cancelled"
            exit 0
        fi
    fi

    # Run installation
    do_install

    # Success message with next steps
    print_success_message
}

main
