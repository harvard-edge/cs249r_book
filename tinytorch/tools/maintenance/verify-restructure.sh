#!/bin/bash
# Verify TinyTorch structure after reorganization
# Tests that all critical functionality still works

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔍 TinyTorch Structure Verification"
echo "===================================="
echo ""

FAILED=0

# Test 1: Check directory structure
echo "📁 Test 1: Verifying directory structure..."
REQUIRED_DIRS=(
    "tools/dev"
    "tools/build"
    "tools/maintenance"
    "docs/_static/demos/scripts"
    "docs/development"
    "tito"
    "tinytorch"
    "src"
    "tests"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir"
    else
        echo "  ❌ $dir - MISSING"
        FAILED=$((FAILED + 1))
    fi
done
echo ""

# Test 2: Check critical files
echo "📄 Test 2: Verifying critical files..."
CRITICAL_FILES=(
    "README.md"
    "requirements.txt"
    "setup-environment.sh"
    "activate.sh"
    "tools/dev/setup.sh"
    "docs/_static/demos/scripts/generate.sh"
    "docs/_static/demos/scripts/optimize.sh"
    "docs/_static/demos/scripts/validate.sh"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file - MISSING"
        FAILED=$((FAILED + 1))
    fi
done
echo ""

# Test 3: Check TITO CLI
echo "🚀 Test 3: Testing TITO CLI..."
if command -v tito &> /dev/null; then
    echo "  ✅ tito command available"

    # Test basic commands
    if tito --help &> /dev/null; then
        echo "  ✅ tito --help works"
    else
        echo "  ❌ tito --help failed"
        FAILED=$((FAILED + 1))
    fi

    if tito --version &> /dev/null; then
        echo "  ✅ tito --version works"
    else
        echo "  ⚠️  tito --version failed (may be expected)"
    fi
else
    echo "  ❌ tito command not found"
    echo "     Try: source activate.sh"
    FAILED=$((FAILED + 1))
fi
echo ""

# Test 4: Check Python imports
echo "🐍 Test 4: Testing Python imports..."
if python3 -c "import tinytorch" 2>/dev/null; then
    echo "  ✅ import tinytorch works"
else
    echo "  ❌ import tinytorch failed"
    FAILED=$((FAILED + 1))
fi

if python3 -c "import tito" 2>/dev/null; then
    echo "  ✅ import tito works"
else
    echo "  ❌ import tito failed"
    FAILED=$((FAILED + 1))
fi
echo ""

# Test 5: Check GIF generation setup
echo "🎬 Test 5: Checking GIF generation..."
if [ -d "docs/_static/demos/tapes" ]; then
    echo "  ✅ VHS tapes directory exists"
    tape_count=$(ls docs/_static/demos/tapes/*.tape 2>/dev/null | wc -l)
    echo "  ✅ Found $tape_count VHS tape files"
else
    echo "  ❌ VHS tapes directory missing"
    FAILED=$((FAILED + 1))
fi

if command -v vhs &> /dev/null; then
    echo "  ✅ VHS installed"
else
    echo "  ⚠️  VHS not installed (optional for maintainers)"
fi
echo ""

# Test 6: Check documentation structure
echo "📚 Test 6: Checking documentation..."
DOC_DIRS=(
    "docs/development"
    "docs/instructor"
    "docs/_static"
)

for dir in "${DOC_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir"
    else
        echo "  ❌ $dir - MISSING"
        FAILED=$((FAILED + 1))
    fi
done
echo ""

# Summary
echo "================================"
if [ $FAILED -eq 0 ]; then
    echo "✅ All verification tests passed!"
    echo ""
    echo "Next steps:"
    echo "  1. Test website build: cd docs && ./build.sh"
    echo "  2. Test module workflow: tito module status"
    echo "  3. Run test suite: pytest tests/"
    exit 0
else
    echo "❌ $FAILED test(s) failed"
    echo ""
    echo "Some issues detected. Please review the output above."
    exit 1
fi
