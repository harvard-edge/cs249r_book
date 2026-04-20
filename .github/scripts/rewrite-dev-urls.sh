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
# PREFIX is the number of "../" hops needed to climb from the *current build
# location* on dev back to the dev site root. It depends on how deep the
# subsite lives on dev, NOT on the mlsysbook.ai key. e.g. vol1 lives at
# /book/vol1/ on dev → 2 hops → PREFIX="../../". This was previously hard-
# coded to "../" which broke nested subsites (vol1, vol2) — every cross-site
# link, including the navbar title-href, landed one level too shallow.
#
# SELF_PREFIX is "./" — a self-link rewrites to the current dir.
SELF_PREFIX="./"
if [[ "$SUBSITE" == "root" ]]; then
  PREFIX=""
else
  # Note: under `set -u`, indexing an associative array with a missing key
  # triggers an unbound-variable error in some bash builds. Probe with the
  # `key in array` test instead of parameter-expansion default.
  found=0
  for k in "${!SUBSITES[@]}"; do
    [[ "$k" == "$SUBSITE" ]] && { found=1; break; }
  done
  if [[ "$found" -eq 0 ]]; then
    echo "❌ Unknown subsite '$SUBSITE'. Add it to SUBSITES in rewrite-dev-urls.sh." >&2
    exit 1
  fi
  SELF_DEV_PATH="${SUBSITES[$SUBSITE]}"
  # Depth = number of path segments in the dev-side path (book/vol1 → 2).
  DEPTH=$(awk -F/ '{print NF}' <<< "$SELF_DEV_PATH")
  PREFIX=""
  for ((i = 0; i < DEPTH; i++)); do PREFIX+="../"; done
fi

# ── Rewrite ───────────────────────────────────────────────────────────────────
echo "🔗 Rewriting mlsysbook.ai URLs for dev site (subsite=$SUBSITE, prefix='$PREFIX')..."

FIND_DEPTH=""
if [[ "$SHALLOW" == "--shallow" ]]; then
  FIND_DEPTH="-maxdepth 1"
fi

find "$BUILD_DIR" $FIND_DEPTH -name "*.html" -type f | while read -r f; do
  for key in "${!SUBSITES[@]}"; do
    dev_path="${SUBSITES[$key]}"
    # Self-link match is by mlsysbook.ai key, not dev_path. Previously this
    # compared dev_path to SUBSITE which silently failed for nested subsites.
    if [[ "$SUBSITE" != "root" && "$key" == "$SUBSITE" ]]; then
      # Self-links: mlsysbook.ai/<key>/ → ./ (when building <key>)
      sed -i "s|https://mlsysbook.ai/${key}/|${SELF_PREFIX}|g" "$f"
    else
      sed -i "s|https://mlsysbook.ai/${key}/|${PREFIX}${dev_path}/|g" "$f"
    fi
  done
  # Catch-all: any remaining mlsysbook.ai/ base URL → dev root.
  # (Used by the navbar title-href and footer site links.)
  if [[ "$SUBSITE" == "root" ]]; then
    sed -i 's|https://mlsysbook.ai/|./|g' "$f"
  else
    sed -i "s|https://mlsysbook.ai/|${PREFIX}|g" "$f"
  fi
done

echo "✅ URL rewriting complete ($(find "$BUILD_DIR" $FIND_DEPTH -name '*.html' -type f | wc -l) files processed)"
