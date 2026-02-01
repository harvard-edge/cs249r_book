#!/bin/bash
# =============================================================================
# MIT Press Release Builder
# =============================================================================
# Builds PDF for MIT Press submission. Produces either a regular or copy-edit
# PDF for Vol I or Vol II.
#
# Usage:
#   ./mit-press-release.sh --vol1                 # Regular Vol I PDF
#   ./mit-press-release.sh --vol2                 # Regular Vol II PDF
#   ./mit-press-release.sh --vol1 --copyedit      # Copy-edit Vol I PDF
#   ./mit-press-release.sh --vol2 --copyedit      # Copy-edit Vol II PDF
# =============================================================================

set -e

# =============================================================================
# Configurable settings — edit these as needed
# =============================================================================
REGULAR_FONTSIZE="9pt"        # Font size for regular PDF
COPYEDIT_FONTSIZE="12pt"      # Font size for copy-edit PDF
COPYEDIT_LINESTRETCH="2"      # Line spacing for copy-edit PDF (2 = double)

# =============================================================================
# Paths (derived automatically — no need to edit)
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
QUARTO_DIR="$REPO_ROOT/book/quarto"
CONFIG_DIR="$QUARTO_DIR/config"

VOLUME=""
COPYEDIT=false

# =============================================================================
# Parse arguments
# =============================================================================
usage() {
    cat <<EOF
Usage: $(basename "$0") --vol1|--vol2 [--copyedit]

Options:
  --vol1        Build Volume I PDF
  --vol2        Build Volume II PDF
  --copyedit    Produce ${COPYEDIT_FONTSIZE}, double-spaced PDF for copy editing
  -h, --help    Show this help message

Examples:
  $(basename "$0") --vol1                  Regular ${REGULAR_FONTSIZE} single-spaced PDF
  $(basename "$0") --vol1 --copyedit       ${COPYEDIT_FONTSIZE} double-spaced PDF for editors
  $(basename "$0") --vol2                  Regular ${REGULAR_FONTSIZE} single-spaced PDF
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vol1) VOLUME="vol1"; shift ;;
        --vol2) VOLUME="vol2"; shift ;;
        --copyedit) COPYEDIT=true; shift ;;
        -h|--help) usage ;;
        *) echo "Error: Unknown option '$1'"; usage ;;
    esac
done

if [[ -z "$VOLUME" ]]; then
    echo "Error: Must specify --vol1 or --vol2"
    usage
fi

# =============================================================================
# Resolve paths
# =============================================================================
SOURCE_CONFIG="$CONFIG_DIR/_quarto-pdf-${VOLUME}.yml"
TARGET_CONFIG="$QUARTO_DIR/_quarto.yml"

if [[ "$COPYEDIT" == true ]]; then
    OUTPUT_DIR="_build/pdf-${VOLUME}-copyedit"
else
    OUTPUT_DIR="_build/pdf-${VOLUME}"
fi

if [[ ! -f "$SOURCE_CONFIG" ]]; then
    echo "Error: Config not found: $SOURCE_CONFIG"
    exit 1
fi

# =============================================================================
# Build
# =============================================================================
echo "============================================="
echo "MIT Press PDF Build"
echo "  Volume:    $VOLUME"
if $COPYEDIT; then
    echo "  Mode:      copy-edit (${COPYEDIT_FONTSIZE}, linestretch ${COPYEDIT_LINESTRETCH})"
else
    echo "  Mode:      regular (${REGULAR_FONTSIZE}, single-spaced)"
fi
echo "  Config:    $SOURCE_CONFIG"
echo "  Output:    $QUARTO_DIR/$OUTPUT_DIR"
echo "============================================="

# Copy base config
cp "$SOURCE_CONFIG" "$TARGET_CONFIG"

# For copyedit mode, patch fontsize and linestretch in-place
if [[ "$COPYEDIT" == true ]]; then
    echo "Applying copy-edit overrides..."

    # Update output directory so it doesn't clobber the regular build
    sed -i.bak "s|output-dir:.*|output-dir: $OUTPUT_DIR|" "$TARGET_CONFIG"

    # Replace fontsize
    sed -i.bak "s/fontsize: ${REGULAR_FONTSIZE}/fontsize: ${COPYEDIT_FONTSIZE}/" "$TARGET_CONFIG"

    # Add linestretch after fontsize line
    sed -i.bak "/fontsize: ${COPYEDIT_FONTSIZE}/a\\
    linestretch: ${COPYEDIT_LINESTRETCH}" "$TARGET_CONFIG"

    # Clean up sed backup files
    rm -f "$TARGET_CONFIG.bak"
fi

echo "Building PDF..."
cd "$QUARTO_DIR"
quarto render --to titlepage-pdf

echo ""
echo "============================================="
echo "Build complete: $QUARTO_DIR/$OUTPUT_DIR"
echo "============================================="
