#!/usr/bin/env bash
# =============================================================================
# Sync shared assets to their per-subsite mirrors.
# =============================================================================
# Quarto's resource-copy step preserves symlinks rather than dereferencing
# them, which breaks builds and deploys when a shared asset is symlinked into
# a subsite's resource path. So we keep real file copies in each consumer
# subsite and use this script to sync them from the canonical sources.
#
# Run after editing any canonical source listed below. The script is also
# safe to run idempotently (no-op when everything is already in sync).
#
# Usage:
#   bash shared/scripts/sync-mirrors.sh           # sync (overwrite mirrors)
#   bash shared/scripts/sync-mirrors.sh --check   # exit non-zero if any mirror is stale
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
fi

# canonical_path|mirror_path_1,mirror_path_2,...
SYNC_MAP=(
  "shared/scripts/subscribe-modal.js|site/assets/scripts/subscribe-modal.js,book/quarto/assets/scripts/subscribe-modal.js,labs/assets/scripts/subscribe-modal.js,kits/assets/scripts/subscribe-modal.js,mlsysim/docs/scripts/subscribe-modal.js"
  "shared/assets/img/logo-seas-shield.png|book/quarto/assets/images/icons/logo-seas-shield.png"

  # SCSS partials consumed by the book Quarto build. Sass resolves @import
  # relative to the importing file's physical location (not the symlink
  # target), so each consumer needs a real file with the same content.
  "shared/styles/_brand.scss|book/quarto/assets/styles/_brand.scss"
  "shared/styles/themes/_theme-harvard.scss|book/quarto/assets/styles/themes/_theme-harvard.scss"
  "shared/styles/themes/_theme-eth.scss|book/quarto/assets/styles/themes/_theme-eth.scss"
)

# Mirrors that are intentionally NOT synced (different content per subsite):
#   tinytorch/site-quarto/assets/scripts/subscribe-modal.js  (TinyTorch-branded)
#   tinytorch/site/_static/subscribe-modal.js                (legacy Sphinx site)

stale=0
synced=0

for entry in "${SYNC_MAP[@]}"; do
  src="${entry%%|*}"
  mirrors="${entry#*|}"

  if [[ ! -f "$src" ]]; then
    echo "ERROR: canonical missing: $src" >&2
    exit 2
  fi

  IFS=',' read -r -a mirror_arr <<< "$mirrors"
  for dst in "${mirror_arr[@]}"; do
    if [[ ! -f "$dst" ]] || ! cmp -s "$src" "$dst"; then
      if (( CHECK_ONLY )); then
        echo "stale: $dst (out of sync with $src)"
        stale=$((stale + 1))
      else
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
        echo "synced: $src -> $dst"
        synced=$((synced + 1))
      fi
    fi
  done
done

if (( CHECK_ONLY )); then
  if (( stale > 0 )); then
    echo ""
    echo "$stale mirror(s) out of sync. Run: bash shared/scripts/sync-mirrors.sh"
    exit 1
  fi
  echo "All mirrors up to date."
  exit 0
fi

if (( synced == 0 )); then
  echo "All mirrors already up to date."
fi
