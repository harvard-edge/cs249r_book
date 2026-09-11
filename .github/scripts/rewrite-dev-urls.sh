#!/usr/bin/env bash
# Rewrite absolute mlsysbook.ai URLs to relative paths for the dev preview site.
#
# Usage: rewrite-dev-urls.sh <subsite> <build-dir> [--shallow] [--live-external]
#
#   subsite   — the caller's subsite name on the dev site (e.g. "slides", "instructors").
#               Use "root" for content deployed at the dev site root (the landing page).
#   build-dir — path to the directory containing rendered HTML files to rewrite.
#   --shallow       — only process HTML files directly in build-dir (no recursion into subdirs).
#                     Useful for unified site builds where subsites are rewritten separately.
#   --live-external — rewrite the current volume pair for dev preview, but leave
#                     other ecosystem subsites on mlsysbook.ai when they are not
#                     deployed in that preview repository.
#
# The script knows the full site map and computes relative paths automatically.
# When a new subsite is added, update SUBSITES below — one place, all workflows.

set -euo pipefail

SUBSITE="${1:?Usage: rewrite-dev-urls.sh <subsite> <build-dir> [--shallow]}"
BUILD_DIR="${2:?Usage: rewrite-dev-urls.sh <subsite> <build-dir> [--shallow]}"
SHALLOW=""
LIVE_EXTERNALS=0
for arg in "${@:3}"; do
  case "$arg" in
    --shallow)
      SHALLOW="--shallow"
      ;;
    --live-external)
      LIVE_EXTERNALS=1
      ;;
    *)
      echo "❌ Unknown option '$arg'." >&2
      exit 1
      ;;
  esac
done

# ── Site map ──────────────────────────────────────────────────────────────────
# Maps mlsysbook.ai/<key>/ to the dev site path <value>/.
# Every subsite (including the volumes) is 1:1 — top-level on both targets.
declare -A SUBSITES=(
  [vol1]="vol1"
  [vol2]="vol2"
  [tinytorch]="tinytorch"
  [physical]="physical"
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
# We still compute from the subsite path, but the actual "../" prefix now has
# to be derived per rendered HTML file, not once per subsite. That lets nested
# pages like kits/contents/... rewrite navbar URLs correctly.
if [[ "$SUBSITE" == "root" ]]; then
  SUBSITE_DEPTH=0
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
  SUBSITE_DEPTH=$(awk -F/ '{print NF}' <<< "$SELF_DEV_PATH")
fi

repeat_parent_prefix() {
  local depth="${1:-0}"
  local prefix=""
  for ((i = 0; i < depth; i++)); do
    prefix+="../"
  done
  printf '%s' "$prefix"
}

replace_url() {
  local search="$1"
  local replacement="$2"
  local file="$3"
  SEARCH="$search" REPLACEMENT="$replacement" perl -0pi -e 's|\Q$ENV{SEARCH}\E|$ENV{REPLACEMENT}|g' "$file"
}

# ── Rewrite ───────────────────────────────────────────────────────────────────
echo "🔗 Rewriting mlsysbook.ai URLs for dev site (subsite=$SUBSITE, subsite_depth=$SUBSITE_DEPTH)..."

FIND_ARGS=("$BUILD_DIR")
if [[ "$SHALLOW" == "--shallow" ]]; then
  FIND_ARGS+=("-maxdepth" "1")
fi
FIND_ARGS+=("-name" "*.html" "-type" "f")

find "${FIND_ARGS[@]}" | while IFS= read -r f; do
  file_dir="$(dirname "$f")"
  rel_dir="${file_dir#$BUILD_DIR}"
  rel_dir="${rel_dir#/}"

  if [[ -z "$rel_dir" || "$rel_dir" == "." ]]; then
    FILE_DEPTH=0
  else
    FILE_DEPTH=$(awk -F/ '{print NF}' <<< "$rel_dir")
  fi

  ROOT_PREFIX="$(repeat_parent_prefix $((SUBSITE_DEPTH + FILE_DEPTH)))"
  if [[ "$FILE_DEPTH" -eq 0 ]]; then
    SELF_PREFIX="./"
  else
    SELF_PREFIX="$(repeat_parent_prefix "$FILE_DEPTH")"
  fi

  for key in "${!SUBSITES[@]}"; do
    dev_path="${SUBSITES[$key]}"

    # Self-link match is by mlsysbook.ai key, not dev_path. Previously this
    # compared dev_path to SUBSITE which silently failed for nested subsites.
    if [[ "$SUBSITE" != "root" && "$key" == "$SUBSITE" ]]; then
      # Self-links: mlsysbook.ai/<key>/ → current subsite root from this file.
      replace_url "https://mlsysbook.ai/${key}/" "$SELF_PREFIX" "$f"
    else
      if [[ "$LIVE_EXTERNALS" -eq 1 && "$key" != "vol1" && "$key" != "vol2" ]]; then
        continue
      fi
      replace_url "https://mlsysbook.ai/${key}/" "${ROOT_PREFIX}${dev_path}/" "$f"
    fi
  done

  # The book's release-pill metadata uses a production-root absolute path.
  # GitHub Pages serves the dev site below /cs249r_book_dev/, so make that
  # manifest reference relative to the current volume page before deployment.
  if [[ "$SUBSITE" == "vol1" || "$SUBSITE" == "vol2" ]]; then
    replace_url \
      "content=\"/${SUBSITE}/release-manifest.json\"" \
      "content=\"${SELF_PREFIX}release-manifest.json\"" \
      "$f"
  fi

  # Catch-all: any remaining mlsysbook.ai/ base URL → dev root.
  # (Used by the navbar title-href and footer site links.)
  if [[ "$LIVE_EXTERNALS" -eq 0 ]]; then
    if [[ "$SUBSITE" == "root" ]]; then
      if [[ "$FILE_DEPTH" -eq 0 ]]; then
        replace_url "https://mlsysbook.ai/" "./" "$f"
      else
        replace_url "https://mlsysbook.ai/" "$SELF_PREFIX" "$f"
      fi
    else
      replace_url "https://mlsysbook.ai/" "$ROOT_PREFIX" "$f"
    fi
  fi
done

echo "✅ URL rewriting complete ($(find "${FIND_ARGS[@]}" | wc -l) files processed)"
