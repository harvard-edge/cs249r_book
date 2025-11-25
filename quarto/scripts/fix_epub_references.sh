#!/usr/bin/env bash
# Post-process EPUB to fix cross-references
# This script extracts the EPUB, fixes references, and re-packages it

set -e

EPUB_FILE="$1"

if [ -z "$EPUB_FILE" ]; then
    # Running as post-render hook - find the EPUB
    EPUB_FILE="_build/epub/Machine-Learning-Systems.epub"
fi

if [ ! -f "$EPUB_FILE" ]; then
    echo "⚠️ EPUB file not found: $EPUB_FILE"
    exit 0
fi

echo "📚 Post-processing EPUB: $EPUB_FILE"

# Get absolute path to EPUB file
EPUB_ABS=$(cd "$(dirname "$EPUB_FILE")" && pwd)/$(basename "$EPUB_FILE")

# Create temporary directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Extract EPUB
echo "   Extracting EPUB..."
unzip -q "$EPUB_ABS" -d "$TEMP_DIR"

# Fix cross-references using Python script
echo "   Fixing cross-references..."
cd "$TEMP_DIR"
python3 "$(dirname "$0")/fix_cross_references.py" >/dev/null 2>&1 || true

# Re-package EPUB
echo "   Re-packaging EPUB..."
cd "$TEMP_DIR"
# EPUB requires mimetype to be first and uncompressed
zip -0 -X fixed.epub mimetype
# Add all other files recursively
zip -r -X fixed.epub META-INF EPUB

# Replace original with fixed version
mv "$TEMP_DIR/fixed.epub" "$EPUB_ABS"

echo "✅ EPUB post-processing complete"
