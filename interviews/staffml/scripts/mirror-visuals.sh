#!/usr/bin/env bash
# mirror-visuals.sh — Copy SVG visuals from the vault to public/question-visuals/
#
# Standalone fallback for the `vault build --local` SVG mirroring step.
# Runs without the vault CLI (pip install -e interviews/vault-cli).
#
# Usage:
#   interviews/staffml/scripts/mirror-visuals.sh
#
# The script auto-detects its own location and resolves paths relative to the
# repo root, so it works from any working directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAFFML_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VAULT_VISUALS="$(cd "$STAFFML_DIR/../../interviews/vault/visuals" 2>/dev/null && pwd)" || {
  echo "[mirror-visuals] ERROR: Cannot find interviews/vault/visuals/ relative to this script." >&2
  exit 1
}

TARGET_DIR="$STAFFML_DIR/public/question-visuals"
TRACKS=(cloud edge mobile tinyml)
total=0

for track in "${TRACKS[@]}"; do
  src="$VAULT_VISUALS/$track"
  dst="$TARGET_DIR/$track"

  if [ ! -d "$src" ]; then
    echo "[mirror-visuals] WARN: source directory missing: $src" >&2
    continue
  fi

  mkdir -p "$dst"

  count=0
  for svg in "$src"/*.svg; do
    [ -f "$svg" ] || continue
    cp "$svg" "$dst/"
    count=$((count + 1))
  done

  total=$((total + count))
  echo "[mirror-visuals] $track: $count SVGs"
done

echo "[mirror-visuals] done — $total SVGs mirrored to public/question-visuals/"
