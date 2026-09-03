#!/usr/bin/env bash
# Re-run verify_lego_chapter.sh for every chapter listed in lego_chapter_failures.txt.
#
# Usage (repo root):
#   ./book/tools/audit/verify_lego_failed.sh
#   ./book/tools/audit/verify_lego_failed.sh book/tools/audit/artifacts/lego_chapter_failures.txt
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"

FAILS="${1:-$REPO/book/tools/audit/artifacts/lego_chapter_failures.txt}"
if [[ ! -s "$FAILS" ]]; then
  echo "No failures in $FAILS"
  exit 0
fi

: > "${FAILS}.retry"
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  vol="${line%%/*}"
  ch="${line#*/}"
  echo "=== retry $vol/$ch ==="
  if ./book/tools/audit/verify_lego_chapter.sh "$vol" "$ch"; then
    echo "OK $line"
  else
    echo "$line" >> "${FAILS}.retry"
    echo "FAIL $line"
  fi
done < "$FAILS"

mv "${FAILS}.retry" "$FAILS"
echo "Remaining failures: $(wc -l < "$FAILS" | tr -d ' ') — see $FAILS"
