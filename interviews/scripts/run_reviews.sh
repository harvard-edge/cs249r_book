#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Launch parallel Gemini math reviews on all 63 chunks
# Usage: ./run_reviews.sh [model] [parallel]
# Default: gemini-2.5-flash, 8 parallel
# For Pro: GEMINI_MODEL=gemini-3.1-pro-preview ./run_reviews.sh
# ═══════════════════════════════════════════════════════════════
set -uo pipefail

MODEL="${GEMINI_MODEL:-gemini-2.5-flash}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
CHUNKS_DIR="/tmp/staffml-review-chunks"
REVIEWS_DIR="$(cd "$(dirname "$0")" && pwd)/_reviews"
mkdir -p "$REVIEWS_DIR"

PROMPT='Review these ML interview questions for math errors. For each question output EXACTLY one line:
ERROR | <title> | <error-type> | <description> | <correct-value>
WARN | <title> | <issue> | <description>
OK | <title>

Check: (1) napkin math arithmetic (2) hardware specs realism (H100=80GB/3.35TB/s/989TFLOPS, A100=80GB/2TB/s/312TFLOPS, Orin=32GB/275TOPS) (3) unit consistency (4) answer supports conclusion. Flag only genuine errors (>2x off).'

CHUNKS=($(ls "$CHUNKS_DIR"/*.md 2>/dev/null | sort))
TOTAL=${#CHUNKS[@]}
echo "═══ Math Review: ${TOTAL} chunks, ${MAX_PARALLEL} parallel, model=${MODEL} ═══"
echo "Start: $(date)"
echo ""

RUNNING=0
DONE=0
ERRORS_TOTAL=0

for i in "${!CHUNKS[@]}"; do
    CHUNK="${CHUNKS[$i]}"
    NAME="$(basename "$CHUNK" .md)"
    REPORT="${REVIEWS_DIR}/${NAME}-review.txt"

    # Skip if already reviewed
    if [[ -f "$REPORT" ]] && [[ $(wc -l < "$REPORT") -gt 2 ]]; then
        echo "[SKIP] ${NAME} (already reviewed)"
        DONE=$((DONE + 1))
        continue
    fi

    # Wait if at max parallel
    while [[ $RUNNING -ge $MAX_PARALLEL ]]; do
        wait -n 2>/dev/null || true
        RUNNING=$((RUNNING - 1))
    done

    # Launch review
    (
        gemini -m "$MODEL" -p "${PROMPT}

$(cat "$CHUNK")" -o text > "$REPORT" 2>&1

        ERRS=$(grep -c "^ERROR" "$REPORT" 2>/dev/null || echo 0)
        OKS=$(grep -c "^OK" "$REPORT" 2>/dev/null || echo 0)
        echo "[DONE] ${NAME}: ${ERRS} errors, ${OKS} OK"
    ) &

    RUNNING=$((RUNNING + 1))
    echo "[LAUNCH ${i}/${TOTAL}] ${NAME}"
done

# Wait for remaining
wait
echo ""
echo "═══ All reviews complete! ═══"
echo "End: $(date)"

# Summary
echo ""
echo "═══ Summary ═══"
TOTAL_ERRORS=$(grep -rch "^ERROR" "$REVIEWS_DIR"/*-review.txt 2>/dev/null | paste -sd+ - | bc 2>/dev/null || echo 0)
TOTAL_WARNS=$(grep -rch "^WARN" "$REVIEWS_DIR"/*-review.txt 2>/dev/null | paste -sd+ - | bc 2>/dev/null || echo 0)
TOTAL_OK=$(grep -rch "^OK" "$REVIEWS_DIR"/*-review.txt 2>/dev/null | paste -sd+ - | bc 2>/dev/null || echo 0)
echo "  Errors:   ${TOTAL_ERRORS}"
echo "  Warnings: ${TOTAL_WARNS}"
echo "  OK:       ${TOTAL_OK}"

# Aggregate errors
echo "" > "${REVIEWS_DIR}/ALL_ERRORS.txt"
echo "═══ All Math Errors Found ═══" >> "${REVIEWS_DIR}/ALL_ERRORS.txt"
echo "Model: ${MODEL}" >> "${REVIEWS_DIR}/ALL_ERRORS.txt"
echo "Date: $(date)" >> "${REVIEWS_DIR}/ALL_ERRORS.txt"
echo "" >> "${REVIEWS_DIR}/ALL_ERRORS.txt"
grep -rh "^ERROR" "$REVIEWS_DIR"/*-review.txt >> "${REVIEWS_DIR}/ALL_ERRORS.txt" 2>/dev/null
echo ""
echo "Error report: ${REVIEWS_DIR}/ALL_ERRORS.txt"
