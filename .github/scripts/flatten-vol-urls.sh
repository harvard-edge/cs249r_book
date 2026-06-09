#!/usr/bin/env bash
# flatten-vol-urls.sh — Flatten Quarto's nested vol URL structure at deploy time.
#
# Quarto renders chapters at:
#   html-vol{N}/contents/vol{N}/{chapter}/{chapter}.html   (depth 3)
# which deploys to:
#   mlsysbook.ai/vol{N}/contents/vol{N}/{chapter}/{chapter}.html
#
# This script moves all content under contents/vol{N}/ up to the site root so
# deployed URLs become the clean form:
#   mlsysbook.ai/vol{N}/{chapter}/{chapter}.html
#
# Why relative paths need fixing:
#   Quarto generates ../../../site_libs/ (3 levels up from contents/vol{N}/chapter/).
#   After the move to {chapter}/ (depth 1), those same assets are at ../site_libs/.
#   We fix ONLY the files being moved — files staying in place (e.g. contents/frontmatter/)
#   are left untouched.
#
# Usage:
#   flatten-vol-urls.sh <site-dir> <vol>
#   flatten-vol-urls.sh ./vol1-site vol1
#   flatten-vol-urls.sh ./vol2-site vol2

set -euo pipefail

SITE_DIR="${1:?Usage: flatten-vol-urls.sh <site-dir> <vol>}"
VOL="${2:?Usage: flatten-vol-urls.sh <site-dir> <vol>}"

NESTED_DIR="$SITE_DIR/contents/$VOL"

if [ ! -d "$NESTED_DIR" ]; then
  echo "ℹ️  $NESTED_DIR not found — nothing to flatten for $VOL"
  exit 0
fi

echo "🔧 Flattening $VOL: contents/$VOL/* → site root..."

# Step 1: Fix 3-level relative asset paths → 1-level in files being moved.
# Chapters are at depth 3 (contents/vol{N}/chapter/); after the move they're
# at depth 1 ({chapter}/). All Quarto-generated relative refs use exactly 3
# levels of ../../../  since every rendered file is at the same depth.
echo "  Fixing relative paths (../../../ → ../) in moved files..."
find "$NESTED_DIR" -name "*.html" -type f \
  -exec sed -i 's|\.\./\.\./\.\./|../|g' {} \;

# Step 2: Fix absolute https://mlsysbook.ai/{vol}/contents/{vol}/ refs.
# These appear in og:image, twitter:image, and any full-URL inline refs.
echo "  Fixing absolute URL refs in moved files..."
find "$NESTED_DIR" -name "*.html" -type f \
  -exec sed -i "s|https://mlsysbook\.ai/${VOL}/contents/${VOL}/|https://mlsysbook.ai/${VOL}/|g" {} \;

# Step 3: Move all content from contents/vol{N}/ up to the site root.
# --ignore-existing: the real homepage is at root index.html; any
# contents/vol{N}/index.html (the sidebar "Homepage" entry) is skipped.
echo "  Moving contents/$VOL/* → site root..."
rsync -a --ignore-existing "$NESTED_DIR/" "$SITE_DIR/"

# Step 4: Fix search.json hrefs (root-relative, no leading slash).
# Before: "contents/vol{N}/chapter/chapter.html"
# After:  "chapter/chapter.html"
if [ -f "$SITE_DIR/search.json" ]; then
  echo "  Fixing search.json hrefs..."
  sed -i "s|\"contents/${VOL}/|\"|g" "$SITE_DIR/search.json"
fi

# Step 5: Fix sitemap.xml (full absolute URLs).
# Before: https://mlsysbook.ai/vol{N}/contents/vol{N}/...
# After:  https://mlsysbook.ai/vol{N}/...
if [ -f "$SITE_DIR/sitemap.xml" ]; then
  echo "  Fixing sitemap.xml..."
  sed -i "s|/${VOL}/contents/${VOL}/|/${VOL}/|g" "$SITE_DIR/sitemap.xml"
fi

# Step 6: Remove the now-copied nested directory.
rm -rf "$NESTED_DIR"

echo "✅ $VOL flattened — chapter URLs now at mlsysbook.ai/$VOL/{chapter}/{chapter}.html"
