#!/usr/bin/env bash
# =============================================================================
# inject-build-stamp.sh — replace the <!-- MLSB_BUILD_STAMP --> token in HTML
# =============================================================================
#
# Why this exists
# ---------------
# Each subsite under mlsysbook.ai is rebuilt on its own cadence. Readers want
# a quick "is this fresh?" signal without us having to maintain manual
# version dates. Quarto's per-page `date-modified` covers individual chapter
# files, but it doesn't capture site-level rebuilds (theme tweaks, navbar
# changes, deploy reruns). This script injects a build-time stamp into the
# page footer at the end of the build, just before deploy.
#
# Usage
# -----
#   bash shared/scripts/inject-build-stamp.sh <build-dir> <site-label> [commit]
#
#   <build-dir>   directory of rendered HTML files (recursively)
#   <site-label>  short site label shown in the stamp (e.g. "Vol I", "Kits")
#   [commit]      optional short git SHA; defaults to GITHUB_SHA[:7] or HEAD
#
# Footer wiring
# -------------
# Place the literal token `<!-- MLSB_BUILD_STAMP -->` somewhere in the
# footer (footer-common.yml / footer-site.yml / per-site equivalents).
# The script swaps that token for an inline span like:
#
#   <span class="mlsb-build-stamp">Last updated 2026-04-19 · Vol I · 9ab1234</span>
#
# Sites that haven't adopted the token yet are unaffected — the script is
# a pure search-and-replace and silently skips files without the token.
# =============================================================================

set -euo pipefail

BUILD_DIR="${1:?Usage: inject-build-stamp.sh <build-dir> <site-label> [commit]}"
SITE_LABEL="${2:?Usage: inject-build-stamp.sh <build-dir> <site-label> [commit]}"
COMMIT="${3:-}"

if [[ -z "$COMMIT" ]]; then
  if [[ -n "${GITHUB_SHA:-}" ]]; then
    COMMIT="${GITHUB_SHA:0:7}"
  elif command -v git >/dev/null 2>&1; then
    COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  else
    COMMIT="unknown"
  fi
fi

# UTC date keeps stamps comparable across builds run from different timezones
# (CI is UTC, maintainer laptops aren't). YYYY-MM-DD is intentional — the
# clock-of-day jitter from sequential CI runs adds noise without information.
DATE="$(date -u +"%Y-%m-%d")"

# Use a unique sed delimiter so the SITE_LABEL or COMMIT can contain '/'
# without escaping (paranoid but cheap).
STAMP="<span class=\"mlsb-build-stamp\">Last updated ${DATE} · ${SITE_LABEL} · ${COMMIT}</span>"

# Search-and-replace across HTML files. Stay quiet on files without the
# token so adopting sites can roll the change in incrementally.
count=0
while IFS= read -r f; do
  if grep -q "<!-- MLSB_BUILD_STAMP -->" "$f"; then
    # GNU sed (CI / Linux) syntax. macOS sed users should run via Docker/Linux.
    sed -i "s|<!-- MLSB_BUILD_STAMP -->|${STAMP}|g" "$f"
    count=$((count + 1))
  fi
done < <(find "$BUILD_DIR" -name "*.html" -type f)

echo "🕐 Stamped ${count} page(s) in ${BUILD_DIR} (${DATE} · ${SITE_LABEL} · ${COMMIT})"
