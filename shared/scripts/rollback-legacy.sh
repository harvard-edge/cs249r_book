#!/usr/bin/env bash
# =============================================================================
# rollback-legacy.sh — snapshot + restore the legacy site at gh-pages root
# =============================================================================
#
# Why this script exists
# ----------------------
# The mlsysbook.ai launch is a phased rollout. New properties (Vol I, Vol II,
# TinyTorch, labs, kits, slides, instructors, staffml, mlsysim, unified
# landing) are deployed into subdirectories under gh-pages, while the legacy
# single-volume site continues to live at the *root* of mlsysbook.ai. Once
# everything is verified, the unified landing page replaces the legacy root.
#
# This script is the panic button for that root cutover. It does NOT touch
# subsite directories — the safety story for those is "redeploy from main".
#
# Modes
# -----
#   snapshot        Take a timestamped backup of the legacy root files
#                   (everything at the gh-pages root that is NOT a known
#                   subsite directory) and push it to a `legacy-backup/<TS>/`
#                   path on the gh-pages branch.
#
#   restore <ID>    Copy a previously-snapshotted set back to the gh-pages
#                   root, OVERWRITING the current root files but leaving
#                   subsite directories alone.
#
#   list            List available snapshots on the gh-pages branch.
#
# Subsite preservation
# --------------------
# A "subsite" is any top-level directory we knowingly publish into. The list
# is hard-coded below (and should be kept in sync with book-publish-live.yml
# and other *-publish-live workflows). Snapshots intentionally exclude these
# so a rollback never wipes out actively-deployed properties.
#
# Safety
# ------
#   - Default is --dry-run. Pass --apply to actually push.
#   - Always operates against a fresh `gh-pages` clone in a tempdir; never
#     mutates the working tree of the calling repo.
#   - Refuses to restore a snapshot that doesn't exist.
#   - On restore, snapshot path is ALSO retained (we copy, not move) so a
#     rollback is itself reversible.
#
# Typical sequence on launch day
# ------------------------------
#   1. Run `./rollback-legacy.sh snapshot --apply` BEFORE deploying the
#      unified landing. Note the printed snapshot ID.
#   2. Deploy the unified landing the usual way (book-publish-live with
#      target=all).
#   3. Verify mlsysbook.ai. If anything looks wrong:
#        ./rollback-legacy.sh restore <SNAPSHOT_ID> --apply
#      Wait for gh-pages CDN to invalidate (~5 min on GitHub Pages).
#
# Requires: bash 4+, git, awk. Run from any clone of the repo.
# =============================================================================

set -euo pipefail

# Subsite directories at the gh-pages root that should NEVER be included in
# legacy-snapshots and NEVER be touched on restore. Keep this list in sync
# with the deploy workflows.
SUBSITES=(
  book
  tinytorch
  kits
  labs
  mlsysim
  slides
  instructors
  interviews
  staffml
  about
  community
  newsletter
  legacy-backup        # don't snapshot snapshots
  .git                 # never touch git internals
)

REPO_URL="${REPO_URL:-https://github.com/harvard-edge/cs249r_book.git}"
GH_PAGES_BRANCH="gh-pages"
DRY_RUN=1
MODE=""
SNAPSHOT_ID=""

usage() {
  cat <<'EOF'
Usage:
  rollback-legacy.sh snapshot [--apply]
      Take a snapshot of the current legacy root on gh-pages.

  rollback-legacy.sh restore <SNAPSHOT_ID> [--apply]
      Restore a previously-taken snapshot to the legacy root.

  rollback-legacy.sh list
      List available snapshots.

Flags:
  --apply       Actually push to gh-pages (default is dry-run).
  --repo URL    Override clone URL (default: $REPO_URL).
EOF
  exit 2
}

# --- Arg parse -----------------------------------------------------------------
[[ $# -eq 0 ]] && usage
MODE="$1"; shift || true
case "$MODE" in
  snapshot|list) ;;
  restore)
    [[ $# -ge 1 ]] || { echo "❌ restore requires a snapshot ID" >&2; usage; }
    SNAPSHOT_ID="$1"; shift
    ;;
  -h|--help) usage ;;
  *) echo "❌ Unknown mode: $MODE" >&2; usage ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)   DRY_RUN=0 ;;
    --repo)    REPO_URL="$2"; shift ;;
    -h|--help) usage ;;
    *)         echo "❌ Unknown flag: $1" >&2; usage ;;
  esac
  shift
done

# --- Workspace ----------------------------------------------------------------
WORKDIR="$(mktemp -d -t mlsb-rollback.XXXXXX)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "📁 Workspace: $WORKDIR"
echo "🌐 Cloning $REPO_URL @ $GH_PAGES_BRANCH ..."
git clone --quiet --depth=1 --branch="$GH_PAGES_BRANCH" "$REPO_URL" "$WORKDIR/gh-pages"
cd "$WORKDIR/gh-pages"

# Build a regex of subsite top-level paths to exclude.
exclude_args=()
for s in "${SUBSITES[@]}"; do
  exclude_args+=( "--exclude=$s" )
done

# --- Mode dispatch ------------------------------------------------------------
case "$MODE" in
  list)
    echo "📜 Snapshots on $GH_PAGES_BRANCH:"
    if [[ -d legacy-backup ]]; then
      (cd legacy-backup && ls -1 | sort)
    else
      echo "  (none — legacy-backup/ does not exist yet)"
    fi
    ;;

  snapshot)
    TS="$(date -u +%Y%m%dT%H%M%SZ)"
    DEST="legacy-backup/$TS"
    echo "📸 Creating snapshot $TS ..."
    mkdir -p "$DEST"

    shopt -s extglob nullglob
    # Subsite-aware copy: every top-level entry except known subsites + dotfiles
    # we never want in a snapshot (.git, .nojekyll is fine to capture).
    for entry in *; do
      skip=0
      for s in "${SUBSITES[@]}"; do
        [[ "$entry" == "$s" ]] && { skip=1; break; }
      done
      [[ $skip -eq 1 ]] && continue
      cp -R "$entry" "$DEST/"
    done
    shopt -u extglob nullglob

    file_count=$(find "$DEST" -type f | wc -l | tr -d ' ')
    echo "✅ Snapshot prepared: $DEST ($file_count files)"

    if [[ $DRY_RUN -eq 1 ]]; then
      echo "🚧 DRY RUN — not pushing. Pass --apply to push."
      echo "   Snapshot ID would be: $TS"
      exit 0
    fi

    git config user.name  "github-actions[bot]"
    git config user.email "github-actions[bot]@users.noreply.github.com"
    git add "$DEST"
    git commit -m "chore(rollback): snapshot legacy root → $DEST"
    git push origin "$GH_PAGES_BRANCH"
    echo "🎯 Snapshot ID: $TS"
    echo "   To restore: rollback-legacy.sh restore $TS --apply"
    ;;

  restore)
    SRC="legacy-backup/$SNAPSHOT_ID"
    if [[ ! -d "$SRC" ]]; then
      echo "❌ Snapshot $SNAPSHOT_ID not found at $SRC" >&2
      exit 1
    fi

    echo "♻️  Restoring snapshot $SNAPSHOT_ID to gh-pages root ..."

    # Remove root-level files that AREN'T subsites (so the restore is clean
    # and stale legacy files don't survive). DON'T touch subsite directories.
    shopt -s extglob nullglob
    for entry in *; do
      skip=0
      for s in "${SUBSITES[@]}"; do
        [[ "$entry" == "$s" ]] && { skip=1; break; }
      done
      [[ $skip -eq 1 ]] && continue
      rm -rf "$entry"
    done
    shopt -u extglob nullglob

    cp -R "$SRC"/* .

    file_count=$(git status --porcelain | wc -l | tr -d ' ')
    echo "✅ Restore prepared. $file_count file changes vs current gh-pages."

    if [[ $DRY_RUN -eq 1 ]]; then
      echo "🚧 DRY RUN — not pushing. Pass --apply to push."
      echo "   Diff preview (first 40 lines):"
      git status --porcelain | head -40
      exit 0
    fi

    git config user.name  "github-actions[bot]"
    git config user.email "github-actions[bot]@users.noreply.github.com"
    git add -A
    git commit -m "revert(launch): restore legacy root from snapshot $SNAPSHOT_ID"
    git push origin "$GH_PAGES_BRANCH"
    echo "🎯 Restore pushed. CDN invalidation typically completes within ~5 minutes."
    ;;
esac
