#!/usr/bin/env bash
# Rewrite absolute mlsysbook.ai URLs to relative paths for the dev preview site.
#
# Usage: rewrite-dev-urls.sh <subsite> <build-dir> [--shallow]
#
#   subsite   — the caller's subsite name on the dev site (e.g. "slides", "instructors").
#               Use "root" for content deployed at the dev site root (the landing page).
#   build-dir — path to the directory containing rendered HTML files to rewrite.
#   --shallow — only process HTML files directly in build-dir (no recursion into subdirs).
#               Useful for unified site builds where subsites are rewritten separately.
#
# The script knows the full site map and computes relative paths automatically.
# When a new subsite is added, update SUBSITES below — one place, all workflows.

set -euo pipefail

SUBSITE="${1:?Usage: rewrite-dev-urls.sh <subsite> <build-dir> [--shallow]}"
BUILD_DIR="${2:?Usage: rewrite-dev-urls.sh <subsite> <build-dir> [--shallow]}"
SHALLOW="${3:-}"

# ── Site map ──────────────────────────────────────────────────────────────────
# Maps mlsysbook.ai/<key>/ to the dev site path <value>/.
# "vol1" and "vol2" live under "book/" on the dev site; everything else is 1:1.
declare -A SUBSITES=(
  [vol1]="book/vol1"
  [vol2]="book/vol2"
  [tinytorch]="tinytorch"
  [kits]="kits"
  [labs]="labs"
  [mlsysim]="mlsysim"
  [slides]="slides"
  [instructors]="instructors"
  [interviews]="interviews"
  [staffml]="staffml"
  [newsletter]="newsletter"
  [community]="community"
  [about]="about"
)

# ── Compute prefix ────────────────────────────────────────────────────────────
# Content at the dev root needs no "../" prefix; subsites need "../" to reach siblings.
if [[ "$SUBSITE" == "root" ]]; then
  PREFIX=""
  SELF_PREFIX="./"
else
  PREFIX="../"
  SELF_PREFIX="./"
fi

# ── Rewrite ───────────────────────────────────────────────────────────────────
echo "🔗 Rewriting mlsysbook.ai URLs for dev site (subsite=$SUBSITE)..."

FIND_DEPTH=""
if [[ "$SHALLOW" == "--shallow" ]]; then
  FIND_DEPTH="-maxdepth 1"
fi

find "$BUILD_DIR" $FIND_DEPTH -name "*.html" -type f | while read -r f; do
  for key in "${!SUBSITES[@]}"; do
    dev_path="${SUBSITES[$key]}"
    if [[ "$SUBSITE" != "root" && "$dev_path" == "$SUBSITE" ]]; then
      # Self-links: mlsysbook.ai/slides/ → ./ (when building slides)
      sed -i "s|https://mlsysbook.ai/${key}/|${SELF_PREFIX}|g" "$f"
    else
      sed -i "s|https://mlsysbook.ai/${key}/|${PREFIX}${dev_path}/|g" "$f"
    fi
  done
  # Catch-all: any remaining mlsysbook.ai/ base URL → dev root
  if [[ "$SUBSITE" == "root" ]]; then
    sed -i 's|https://mlsysbook.ai/|./|g' "$f"
  else
    sed -i "s|https://mlsysbook.ai/|${PREFIX}|g" "$f"
  fi
done

echo "✅ URL rewriting complete ($(find "$BUILD_DIR" $FIND_DEPTH -name '*.html' -type f | wc -l) files processed)"
