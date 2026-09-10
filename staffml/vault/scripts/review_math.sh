#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Math Review: Launch parallel Gemini reviews of all questions
# Uses gemini-3.1-pro-preview to verify napkin math in every question
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

MODEL="${GEMINI_MODEL:-gemini-3.1-pro-preview}"
INTERVIEWS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORTS_DIR="${INTERVIEWS_DIR}/_reviews"
CHUNKS_DIR="/tmp/staffml-review-chunks"
MAX_PARALLEL="${MAX_PARALLEL:-8}"

mkdir -p "$REPORTS_DIR" "$CHUNKS_DIR"

# Split large markdown files into chunks of ~50 questions each
echo "═══ Splitting files into review chunks ═══"

split_file() {
    local file="$1"
    local basename="$(basename "$file" .md)"
    local dirname="$(basename "$(dirname "$file")")"
    local prefix="${dirname}-${basename}"

    # Split on <details> blocks — each is one question
    python3 -c "
import sys
content = open('$file').read()
blocks = content.split('<details>')
header = blocks[0]
questions = ['<details>' + b for b in blocks[1:]]
chunk_size = 50
chunk_num = 0
for i in range(0, len(questions), chunk_size):
    chunk = questions[i:i+chunk_size]
    chunk_num += 1
    outfile = f'$CHUNKS_DIR/${prefix}-chunk{chunk_num:02d}.md'
    with open(outfile, 'w') as f:
        if chunk_num == 1:
            f.write(header)
        f.write('\n'.join(chunk))
    print(f'  {outfile}: {len(chunk)} questions')
"
}

# Process all question files
for track in cloud edge mobile tinyml; do
    for file in "${INTERVIEWS_DIR}/${track}"/*.md; do
        [[ "$(basename "$file")" == "README.md" ]] && continue
        split_file "$file"
    done
done

CHUNK_COUNT=$(ls "$CHUNKS_DIR"/*.md 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "═══ ${CHUNK_COUNT} chunks ready for review ═══"
echo "═══ Model: ${MODEL} | Max parallel: ${MAX_PARALLEL} ═══"
echo ""

# The review prompt
REVIEW_PROMPT='You are a math verification expert reviewing ML Systems interview questions.

For EACH question in the file below, check:
1. **Napkin Math Accuracy**: Are the calculations correct? Check arithmetic, unit conversions, order of magnitude.
2. **Hardware Specs**: Are GPU/TPU/MCU specs realistic? (e.g., H100 = 80GB HBM3, 3.35 TB/s bandwidth, 989 TFLOPS FP16)
3. **Answer Consistency**: Does the napkin math in the answer actually support the conclusion?
4. **Unit Errors**: GB vs GiB, MB/s vs Mb/s, FLOPS vs FLOPs, etc.
5. **Logical Errors**: Does the "common mistake" actually describe a real mistake? Does the "realistic solution" make sense?

OUTPUT FORMAT — For each error found, output EXACTLY:
```
ERROR | <question-title> | <error-type> | <what-is-wrong> | <correct-value>
```

For warnings (not wrong but could be better):
```
WARN | <question-title> | <issue-type> | <description>
```

If a question is correct, output:
```
OK | <question-title>
```

Review ALL questions. Do not skip any. Be rigorous but fair — napkin math is approximate by nature, so flag only genuine errors (>2x off for cloud, >50% off for tinyml).'

# Launch reviews in parallel
review_chunk() {
    local chunk="$1"
    local chunk_name="$(basename "$chunk" .md)"
    local report="${REPORTS_DIR}/${chunk_name}-review.txt"

    echo "[START] ${chunk_name}"

    if gemini -m "$MODEL" "${REVIEW_PROMPT}" < "$chunk" > "$report" 2>/dev/null; then
        local errors=$(grep -c "^ERROR" "$report" 2>/dev/null || echo 0)
        local warns=$(grep -c "^WARN" "$report" 2>/dev/null || echo 0)
        local oks=$(grep -c "^OK" "$report" 2>/dev/null || echo 0)
        echo "[DONE] ${chunk_name}: ${errors} errors, ${warns} warnings, ${oks} OK"
    else
        echo "[FAIL] ${chunk_name}: Gemini call failed"
        echo "GEMINI_CALL_FAILED" > "$report"
    fi
}

export -f review_chunk
export MODEL REPORTS_DIR REVIEW_PROMPT

# Run with GNU parallel or xargs fallback
if command -v parallel &>/dev/null; then
    ls "$CHUNKS_DIR"/*.md | parallel -j "$MAX_PARALLEL" review_chunk
else
    ls "$CHUNKS_DIR"/*.md | xargs -P "$MAX_PARALLEL" -I {} bash -c 'review_chunk "$@"' _ {}
fi

echo ""
echo "═══ Review complete! ═══"
echo "Reports in: ${REPORTS_DIR}/"
echo ""

# Summary
echo "═══ Summary ═══"
total_errors=$(grep -rch "^ERROR" "$REPORTS_DIR"/*-review.txt 2>/dev/null | paste -sd+ - | bc 2>/dev/null || echo 0)
total_warns=$(grep -rch "^WARN" "$REPORTS_DIR"/*-review.txt 2>/dev/null | paste -sd+ - | bc 2>/dev/null || echo 0)
total_ok=$(grep -rch "^OK" "$REPORTS_DIR"/*-review.txt 2>/dev/null | paste -sd+ - | bc 2>/dev/null || echo 0)
echo "  Errors:   ${total_errors}"
echo "  Warnings: ${total_warns}"
echo "  OK:       ${total_ok}"
echo ""

# Aggregate all errors into one file
echo "═══ All Errors ═══" > "${REPORTS_DIR}/ALL_ERRORS.txt"
grep -rh "^ERROR" "$REPORTS_DIR"/*-review.txt >> "${REPORTS_DIR}/ALL_ERRORS.txt" 2>/dev/null || echo "(none)"
echo ""
echo "Full error list: ${REPORTS_DIR}/ALL_ERRORS.txt"
