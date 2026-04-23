#!/bin/bash
# Autonomous periodic table improvement loop.
# Each iteration completes, then the next starts immediately.
# Usage: ./iterate.sh [num_iterations]
#   Default: 6 iterations

set -e

ITERATIONS=${1:-6}
LOG="$(dirname "$0")/iteration-log.md"
TABLE="$(dirname "$0")/index.html"

echo "╔═══════════════════════════════════════════════╗"
echo "║  Periodic Table Autonomous Improvement Loop   ║"
echo "║  Iterations: $ITERATIONS                              ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

for i in $(seq 1 "$ITERATIONS"); do
  echo "━━━ Iteration $i/$ITERATIONS ━━━"
  echo ""

  claude -p "$(cat <<'PROMPT'
You are running iteration of an autonomous improvement loop for the Periodic Table of ML Systems.

## Instructions

1. Read /Users/VJ/GitHub/MLSysBook/periodic-table/iteration-log.md to see what previous iterations found and changed.
2. Read /Users/VJ/GitHub/MLSysBook/periodic-table/index.html to see the current table state.
3. Pick ONE specific improvement from this priority list:
   a. Fix a misplaced element (wrong column/row based on the axes)
   b. Add a critically missing element
   c. Remove or merge a duplicate/weak element
   d. Fix a broken bond reference (element references a symbol that doesn't exist)
   e. Test a gap: does the empty cell predict a real concept?
   f. Improve a weak "whyHere" explanation
4. Make the change by editing index.html.
5. Append a brief entry to iteration-log.md documenting what you changed and why.

## Rules
- ONE change per iteration. Small and focused.
- Don't repeat a change already logged in iteration-log.md.
- Don't add elements just to fill gaps — only add if the concept is genuinely important.
- Always verify the same-column test passes after your change.
- If you find nothing to improve, say so in the log and stop.
PROMPT
  )" --allowedTools Read,Edit,Write,Grep,Glob

  echo ""
  echo "✓ Iteration $i complete."
  echo ""
done

echo "═══ All $ITERATIONS iterations complete. ═══"
echo "Review changes: cat $LOG"
