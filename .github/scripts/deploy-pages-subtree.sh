#!/usr/bin/env bash
# Replace one subsite's folder on gh-pages with a fresh build.
#
# Usage: deploy-pages-subtree.sh <subdir> <build-dir> <commit-message>
#
#   subdir         — the subsite's folder at the gh-pages root (e.g. "labs").
#   build-dir      — the rendered site to publish there.
#   commit-message — message for the gh-pages commit.
#
# Env: GH_TOKEN (a token that can push gh-pages), GITHUB_REPOSITORY.
#
# Why not peaceiris/actions-gh-pages with keep_files: true? That overlays the
# build onto the old folder, so pages that a build no longer produces (retired
# pages, renamed labs, unreleased volumes) stay live forever. This script
# deletes the folder first, the way the TinyTorch deploy does, so the live
# folder always matches the build exactly. Other subsites are untouched.

set -euo pipefail

SUBDIR="${1:?Usage: deploy-pages-subtree.sh <subdir> <build-dir> <commit-message>}"
BUILD_DIR="$(cd "${2:?missing build-dir}" && pwd)"
MESSAGE="${3:?missing commit-message}"
: "${GH_TOKEN:?GH_TOKEN must be set}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"

case "$SUBDIR" in
  ""|.|..|/*|*..*) echo "❌ Refusing unsafe subdir '$SUBDIR'"; exit 1 ;;
esac
[ -f "$BUILD_DIR/index.html" ] || { echo "❌ $BUILD_DIR/index.html missing; refusing to deploy an empty build"; exit 1; }

WORK="$(mktemp -d)"
git clone --depth=1 --branch=gh-pages \
  "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" "$WORK/gh-pages"
cd "$WORK/gh-pages"
git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"

before=$(find "$SUBDIR" -type f 2>/dev/null | wc -l | tr -d ' ')
rm -rf "$SUBDIR"
mkdir -p "$SUBDIR"
cp -a "$BUILD_DIR"/. "$SUBDIR"/
touch .nojekyll
after=$(find "$SUBDIR" -type f | wc -l | tr -d ' ')
echo "📦 $SUBDIR: $before file(s) before, $after after"

git add -A "$SUBDIR" .nojekyll
if git diff --cached --quiet; then
  echo "🟡 Nothing changed in $SUBDIR"
  exit 0
fi
git commit -q -m "$MESSAGE"

for attempt in 1 2 3; do
  if git push origin gh-pages; then
    echo "✅ Deployed $SUBDIR (attempt $attempt)"
    exit 0
  fi
  echo "⚠️ Push failed (attempt $attempt/3); rebasing onto the latest gh-pages"
  git fetch --depth=50 origin gh-pages
  git rebase origin/gh-pages || { echo "❌ Rebase failed"; exit 1; }
done
echo "❌ Push failed after 3 attempts"
exit 1
