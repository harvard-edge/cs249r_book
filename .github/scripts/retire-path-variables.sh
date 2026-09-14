#!/usr/bin/env bash
# retire-path-variables.sh — Delete the retired directory-path repository variables.
#
# Workflows write repository paths literally (docs/CI-VARIABLES.md explains why).
# These variables remain only while a branch on GitHub still has workflows that
# read them. Run this after the next dev -> main publish carries dev's workflows
# to main.
#
# Usage: .github/scripts/retire-path-variables.sh [--apply]
#
#   (no flag) — check origin/main and origin/dev, list the variables still set.
#   --apply   — delete them, but only when neither branch has a workflow that
#               still reads one of them.

set -euo pipefail

REPO="harvard-edge/cs249r_book"
VARS=(
  BOOK_ROOT BOOK_QUARTO BOOK_TOOLS BOOK_DOCKER BOOK_DEPS
  TINYTORCH_ROOT TINYTORCH_SITE TINYTORCH_SRC TINYTORCH_TESTS
  MLSYSIM_ROOT MLSYSIM_DOCS SLIDES_ROOT INSTRUCTORS_ROOT
  STAFFML_ROOT VAULT_DIR VAULT_CLI_DIR
  KITS_ROOT KITS_DOCS LABS_ROOT LABS_DOCS
)

APPLY=0
case "${1:-}" in
  "") ;;
  --apply) APPLY=1 ;;
  *) echo "Usage: $0 [--apply]" >&2; exit 2 ;;
esac

cd "$(git rev-parse --show-toplevel)"
git fetch --quiet origin main dev

names=$(IFS='|'; echo "${VARS[*]}")
pattern="vars\\.(${names})([^A-Za-z0-9_]|\$)"

blocked=0
for ref in origin/main origin/dev; do
  hits=$(git grep -nE "$pattern" "$ref" -- .github/workflows .github/actions || true)
  if [ -n "$hits" ]; then
    count=$(printf '%s\n' "$hits" | wc -l | tr -d ' ')
    echo "✗ $ref: $count workflow lines still read retired path variables, for example:"
    printf '%s\n' "$hits" | head -5 | sed 's/^/    /'
    blocked=1
  else
    echo "✓ $ref: no workflow reads a retired path variable"
  fi
done

existing=$(gh variable list -R "$REPO" --json name --jq '.[].name')
to_delete=()
for v in "${VARS[@]}"; do
  if printf '%s\n' "$existing" | grep -qx "$v"; then
    to_delete+=("$v")
  fi
done

if [ "${#to_delete[@]}" -eq 0 ]; then
  echo "Nothing to delete: none of the retired variables are set."
  exit 0
fi
echo "Still set on $REPO (${#to_delete[@]}): ${to_delete[*]}"

if [ "$APPLY" -eq 0 ]; then
  echo "Dry run. Re-run with --apply to delete them."
  exit 0
fi
if [ "$blocked" -ne 0 ]; then
  echo "Refusing to delete while a branch still reads them. Publish dev to main first." >&2
  exit 1
fi
for v in "${to_delete[@]}"; do
  gh variable delete "$v" -R "$REPO"
  echo "deleted $v"
done
