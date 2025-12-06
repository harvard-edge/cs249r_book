#!/bin/bash
# Validate that all demo workflows actually work
# Run this before generating GIFs to catch any broken commands

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🔥 TinyTorch Demo Validation${NC}"
echo "========================================"
echo ""

# Track results
PASSED=0
FAILED=0
WARNINGS=0

# Test helper
test_command() {
    local test_name="$1"
    local command="$2"
    local check_output="$3"  # Optional: grep pattern to verify output

    echo -e "${BLUE}Testing: ${test_name}${NC}"

    if eval "$command" > /tmp/test_output.txt 2>&1; then
        if [ -n "$check_output" ]; then
            if grep -q "$check_output" /tmp/test_output.txt; then
                echo -e "  ${GREEN}✓ PASS${NC}"
                ((PASSED++))
            else
                echo -e "  ${YELLOW}⚠ WARNING: Command ran but output doesn't match expected pattern${NC}"
                echo -e "  Expected pattern: $check_output"
                ((WARNINGS++))
            fi
        else
            echo -e "  ${GREEN}✓ PASS${NC}"
            ((PASSED++))
        fi
    else
        echo -e "  ${RED}✗ FAIL${NC}"
        echo -e "  ${RED}Error output:${NC}"
        tail -5 /tmp/test_output.txt | sed 's/^/    /'
        ((FAILED++))
    fi
    echo ""
}

echo -e "${CYAN}📋 Demo 01: Zero to Ready${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Clean slate
rm -rf /tmp/TinyTorch_validate
cd /tmp

test_command "git clone" \
    "git clone https://github.com/mlsysbook/TinyTorch.git TinyTorch_validate" \
    "Cloning into"

cd TinyTorch_validate

test_command "setup-environment.sh" \
    "./setup-environment.sh" \
    "TinyTorch environment setup complete"

test_command "activate.sh exists" \
    "test -f activate.sh"

test_command "source activate.sh" \
    "source activate.sh && echo 'Environment activated'"

test_command "tito module status" \
    "source activate.sh && tito module status" \
    "Module"

test_command "tito logo" \
    "source activate.sh && tito logo" \
    "TinyTorch"

echo ""
echo -e "${CYAN}📋 Demo 02: Build, Test, Ship (Module 01)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd /tmp/TinyTorch_validate

test_command "tito module reset 01" \
    "source activate.sh && tito module reset 01 --force --no-backup"

test_command "Check src/01_tensor exists" \
    "test -d src/01_tensor"

test_command "jupyter lab --help" \
    "source activate.sh && jupyter lab --help" \
    "jupyter-lab"

test_command "tito module complete 01 (dry run check)" \
    "source activate.sh && tito module test 01" \
    ""

test_command "Check import tinytorch works" \
    "source activate.sh && python -c 'import sys; sys.path.insert(0, \"src\"); import tinytorch; print(tinytorch.__version__ if hasattr(tinytorch, \"__version__\") else \"imported successfully\")'"

echo ""
echo -e "${CYAN}📋 Demo 03: Milestone Unlocked${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

test_command "tito milestones progress" \
    "source activate.sh && tito milestones progress" \
    ""

test_command "tito milestones list" \
    "source activate.sh && tito milestones list" \
    ""

# Check which milestones exist
echo -e "${BLUE}Checking available milestones...${NC}"
source activate.sh && tito milestones list 2>&1 | head -10

echo ""
echo -e "${CYAN}📋 Demo 04: Share Journey${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

test_command "tito module status (for demo 04)" \
    "source activate.sh && tito module status" \
    "Module"

test_command "tito community join --help" \
    "source activate.sh && tito community join --help || tito community --help" \
    ""

test_command "tito community update --help" \
    "source activate.sh && tito community update --help || tito community --help" \
    ""

echo ""
echo "========================================"
echo -e "${CYAN}📊 Validation Summary${NC}"
echo "========================================"
echo ""
echo -e "  ${GREEN}Passed:   $PASSED${NC}"
echo -e "  ${YELLOW}Warnings: $WARNINGS${NC}"
echo -e "  ${RED}Failed:   $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All critical tests passed!${NC}"
    echo ""
    echo -e "${CYAN}Next Steps:${NC}"
    echo ""
    echo "  1. [Optional] Calibrate timing for your machine:"
    echo "     ./docs/_static/demos/scripts/demo.sh --calibrate"
    echo ""
    echo "  2. Generate a demo:"
    echo "     ./docs/_static/demos/scripts/demo.sh --tape 01"
    echo ""
    echo "  3. Preview:"
    echo "     open docs/_static/demos/01-zero-to-ready.gif"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please fix issues before generating demos.${NC}"
    echo ""
    echo "Debug by running the failing command manually:"
    echo "  cd /tmp/TinyTorch_validate"
    echo "  source activate.sh"
    echo "  # Run the failing command"
    exit 1
fi
