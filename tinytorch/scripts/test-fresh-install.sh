#!/bin/bash
# =============================================================================
# TinyTorch Fresh Install Test
# =============================================================================
# Simulates exactly what a student experiences: fresh machine, curl install,
# run through modules and milestones.
#
# Usage:
#   ./scripts/test-fresh-install.sh                    # Test against main
#   ./scripts/test-fresh-install.sh --branch dev       # Test against dev
#   ./scripts/test-fresh-install.sh --branch feature/foo --ci  # CI mode
#
# This catches issues like:
#   - Git LFS files not being pulled correctly
#   - Missing dependencies in requirements.txt
#   - Interactive prompts blocking non-interactive use
#   - Broken install script
# =============================================================================

set -e

# Defaults
BRANCH="main"
CI_MODE=false
INSTALL_SCRIPT_URL="https://raw.githubusercontent.com/harvard-edge/cs249r_book/main/tinytorch/site/extra/install.sh"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        --ci)
            CI_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--branch BRANCH] [--ci]"
            exit 1
            ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

print_step() { echo -e "\n${CYAN}▶ $1${NC}"; }
print_pass() { echo -e "${GREEN}✓ $1${NC}"; }
print_fail() { echo -e "${RED}✗ $1${NC}"; }

# Build the test script that runs inside container (or CI)
# Using a function to allow variable interpolation
build_test_script() {
    cat << 'OUTER_EOF'
set -e

BRANCH="__BRANCH__"

echo "══════════════════════════════════════════════════════════════"
echo "  TinyTorch Fresh Install Test"
echo "  Branch: $BRANCH"
echo "══════════════════════════════════════════════════════════════"

# Step 1: Install from specified branch
echo ""
echo "▶ Step 1: Running install script (branch: $BRANCH)..."
export TINYTORCH_BRANCH="$BRANCH"
export TINYTORCH_NON_INTERACTIVE=1
curl -fsSL "https://raw.githubusercontent.com/harvard-edge/cs249r_book/${BRANCH}/tinytorch/site/extra/install.sh" -o /tmp/install.sh || {
    echo "✗ Failed to download install script for branch: $BRANCH"
    echo "  URL: https://raw.githubusercontent.com/harvard-edge/cs249r_book/${BRANCH}/tinytorch/site/extra/install.sh"
    echo "  Hint: Does the branch '${BRANCH}' exist and contain tinytorch/site/extra/install.sh?"
    exit 1
}
bash /tmp/install.sh

cd tinytorch
source .venv/bin/activate

# Step 2: Verify tito works
echo ""
echo "▶ Step 2: Verifying tito CLI..."
tito --version

# Step 3: Verify datasets are real files (not LFS pointers)
echo ""
echo "▶ Step 3: Checking dataset files..."
TRAIN_PKL="datasets/tinydigits/train.pkl"
if [ -f "$TRAIN_PKL" ]; then
    # Check first bytes - pickle files start with 0x80, LFS pointers start with "version"
    FIRST_CHAR=$(head -c 1 "$TRAIN_PKL" | xxd -p)
    if [ "$FIRST_CHAR" = "80" ]; then
        echo "✓ train.pkl is valid pickle data"
    else
        echo "✗ train.pkl appears to be an LFS pointer, not actual data"
        head -c 100 "$TRAIN_PKL"
        exit 1
    fi
else
    echo "✗ train.pkl not found"
    exit 1
fi

# Step 4: Test loading the dataset directly
echo ""
echo "▶ Step 4: Testing dataset loading..."
python3 -c "
import pickle
with open('datasets/tinydigits/train.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'✓ Loaded {len(data[\"images\"])} training images')
"

# Step 5: Run milestone 01 (Perceptron - simplest)
echo ""
echo "▶ Step 5: Running Milestone 01 (Perceptron)..."
timeout 120 tito milestone run 01 --non-interactive || {
    echo "⚠ Milestone 01 did not complete (may need module implementations)"
}

# Step 6: Run milestone 03 (MLP with TinyDigits - the one that caught the LFS bug)
echo ""
echo "▶ Step 6: Running Milestone 03 (MLP/TinyDigits)..."
timeout 180 tito milestone run 03 --non-interactive || {
    echo "⚠ Milestone 03 did not complete (may need module implementations)"
}

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ✓ Fresh install test completed!"
echo "══════════════════════════════════════════════════════════════"
OUTER_EOF
}

# =============================================================================
# Main
# =============================================================================

echo "Testing TinyTorch installation from branch: $BRANCH"

# Build test script with branch substituted
TEST_SCRIPT=$(build_test_script | sed "s|__BRANCH__|$BRANCH|g")

if [ "$CI_MODE" = true ]; then
    print_step "Running in CI mode (no Docker)"

    # Install git and curl if needed (for CI environments)
    if ! command -v git &> /dev/null; then
        apt-get update && apt-get install -y git curl xxd
    fi

    eval "$TEST_SCRIPT"
else
    print_step "Running via Docker (simulates clean student machine)"

    # Check Docker is available
    if ! command -v docker &> /dev/null; then
        print_fail "Docker not found. Install Docker or run with --ci in a clean environment."
        exit 1
    fi

    # Run in Docker - note: no git-lfs installed, just like a typical student machine
    docker run --rm \
        -e DEBIAN_FRONTEND=noninteractive \
        python:3.11-slim \
        bash -c "
            apt-get update && apt-get install -y git curl xxd > /dev/null 2>&1
            $TEST_SCRIPT
        "

    print_pass "Fresh install test completed successfully"
fi
