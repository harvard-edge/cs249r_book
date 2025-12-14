#!/bin/bash
# Prepare notebooks for site build
# This script ensures notebooks exist in site/ for launch buttons to work
# Called automatically during site build
#
# Workflow:
# 1. Uses existing assignment notebooks if available (from tito nbgrader generate)
# 2. Falls back to generating notebooks from modules if needed
# 3. Copies notebooks to docs/chapters/modules/ for Jupyter Book launch buttons

set -e

# Get the site directory (where this script lives)
SITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SITE_DIR/.." && pwd)"

echo "📓 Preparing notebooks for site build..."

# Create notebooks directory in site if it doesn't exist
NOTEBOOKS_DIR="$SITE_DIR/chapters/modules"
mkdir -p "$NOTEBOOKS_DIR"

cd "$REPO_ROOT"

# Strategy: Use existing assignment notebooks if available, otherwise generate
# This is faster and uses already-processed notebooks
echo "🔄 Looking for existing assignment notebooks..."

MODULES=$(ls -1 modules/ 2>/dev/null | grep -E "^[0-9]" | sort -V || echo "")

if [ -z "$MODULES" ]; then
    echo "⚠️  No modules found. Skipping notebook preparation."
    exit 0
fi

NOTEBOOKS_COPIED=0
NOTEBOOKS_GENERATED=0

for module in $MODULES; do
    TARGET_NB="$NOTEBOOKS_DIR/${module}.ipynb"

    # Check if assignment notebook already exists
    ASSIGNMENT_NB="$REPO_ROOT/assignments/source/$module/${module}.ipynb"

    if [ -f "$ASSIGNMENT_NB" ]; then
        # Use existing assignment notebook
        cp "$ASSIGNMENT_NB" "$TARGET_NB"
        echo "  ✅ Copied existing notebook: $module"
        NOTEBOOKS_COPIED=$((NOTEBOOKS_COPIED + 1))
    elif command -v tito &> /dev/null; then
        # Try to generate notebook if tito is available
        echo "  🔄 Generating notebook for $module..."
        if tito nbgrader generate "$module" >/dev/null 2>&1; then
            if [ -f "$ASSIGNMENT_NB" ]; then
                cp "$ASSIGNMENT_NB" "$TARGET_NB"
                echo "    ✅ Generated and copied: $module"
                NOTEBOOKS_GENERATED=$((NOTEBOOKS_GENERATED + 1))
            fi
        else
            echo "    ⚠️  Could not generate notebook for $module (module may not be ready)"
        fi
    else
        echo "  ⚠️  No notebook found for $module (install tito CLI to generate)"
    fi
done

echo ""
if [ $NOTEBOOKS_COPIED -gt 0 ] || [ $NOTEBOOKS_GENERATED -gt 0 ]; then
    echo "✅ Notebook preparation complete!"
    echo "   Copied: $NOTEBOOKS_COPIED | Generated: $NOTEBOOKS_GENERATED"
    echo "   Notebooks available in: $NOTEBOOKS_DIR"
    echo "   Launch buttons will now work on notebook pages!"
else
    echo "⚠️  No notebooks prepared. Launch buttons may not appear."
    echo "   Run 'tito nbgrader generate --all' first to create assignment notebooks."
fi
