#!/usr/bin/env bash
# Pre-flight: "the PDF looks clean" gate. Run from this worktree root.
# 1) content gate  2) fresh build  3) nothing-steps-out  4) no-ugly-gaps
set -uo pipefail
cd "$(dirname "$0")"
echo "== 1. CONTENT GATE =="; pre-commit run --all-files
for V in 1 2; do
  P="book/quarto/_build/pdf-vol$V/Machine-Learning-Systems-Vol$V.pdf"
  echo "== 2. BUILD vol$V =="; ./book/binder build pdf --vol$V --skip-validate -v > /tmp/pf_build_vol$V.log 2>&1
  echo "== 3a. overfull/underfull (must be 0) vol$V =="; grep -c 'Overfull\|Underfull' /tmp/pf_build_vol$V.log
  echo "== 3b. collisions vol$V =="; ./book/binder layout collisions "$P" 2>&1 | grep -E 'collisions:'
  echo "== 3c. margins vol$V =="; ./book/binder layout margins "$P" --include-overlaps 2>&1 | grep -E 'overlaps [0-9]'
  echo "== 4. gaps vol$V =="; ./book/binder layout check "$P" --skip-frontmatter 2>&1 | grep -E 'flagged'
  echo "== 5. purpose-overflow vol$V (must be 0) =="; python3 book/tools/audit/check_purpose_overflow.py "$P" --vol vol$V 2>&1 | grep -E '✗|PASS|FAIL'
done
