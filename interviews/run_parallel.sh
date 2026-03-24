#!/bin/bash
# run_parallel.sh — Orchestrate parallel question generation across (level, track) combos.
#
# Architecture:
#   1. vault.py plan-all → creates _plans/plan-{level}-{track}.json files
#   2. This script runs generate-batch on each plan in waves (to respect rate limits)
#   3. vault.py merge → combines all _generated/ batches into corpus
#   4. vault.py release → rebuild chains, stats, export
#
# Usage:
#   ./run_parallel.sh                    # Full run (all gaps)
#   ./run_parallel.sh --wave-size 2      # Conservative (2 concurrent batches)
#   ./run_parallel.sh --workers 2        # Fewer Gemini calls per batch
#   ./run_parallel.sh --dry-run          # Plan only, don't generate

set -euo pipefail
cd "$(dirname "$0")"

WAVE_SIZE=${1:-3}       # Concurrent batches per wave
WORKERS=${2:-4}         # Gemini workers per batch
DRY_RUN=${3:-""}

LOG_DIR="_logs"
mkdir -p "$LOG_DIR" "_generated"

TIMESTAMP=$(date +%Y%m%d-%H%M)
MASTER_LOG="$LOG_DIR/parallel-$TIMESTAMP.log"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$MASTER_LOG"; }

log "═══ StaffML Parallel Generation ═══"
log "  Wave size:  $WAVE_SIZE concurrent batches"
log "  Workers:    $WORKERS per batch"
log "  Timestamp:  $TIMESTAMP"

# Step 1: Generate all plans
log ""
log "── Step 1: Generating plans ──"
python3 vault.py plan-all 2>&1 | tee -a "$MASTER_LOG"

# Step 2: Run generation in waves
log ""
log "── Step 2: Generating questions in waves ──"

if [ -n "$DRY_RUN" ]; then
    log "DRY RUN — skipping generation"
    exit 0
fi

PLAN_DIR="_plans"
PLAN_FILES=($(ls "$PLAN_DIR"/plan-*.json 2>/dev/null | sort))
TOTAL_PLANS=${#PLAN_FILES[@]}

if [ "$TOTAL_PLANS" -eq 0 ]; then
    log "No plan files found"
    exit 0
fi

log "  Total plan files: $TOTAL_PLANS"

# Process in waves
WAVE=1
IDX=0
TOTAL_GENERATED=0
TOTAL_FAILED=0

while [ $IDX -lt $TOTAL_PLANS ]; do
    WAVE_END=$((IDX + WAVE_SIZE))
    if [ $WAVE_END -gt $TOTAL_PLANS ]; then
        WAVE_END=$TOTAL_PLANS
    fi

    WAVE_FILES=("${PLAN_FILES[@]:$IDX:$WAVE_SIZE}")
    log ""
    log "── Wave $WAVE: batches $((IDX+1))-$WAVE_END of $TOTAL_PLANS ──"

    PIDS=()
    WAVE_LOGS=()

    for PLAN in "${WAVE_FILES[@]}"; do
        PLAN_NAME=$(basename "$PLAN" .json)
        BATCH_LOG="$LOG_DIR/$PLAN_NAME-$TIMESTAMP.log"

        log "  Starting: $PLAN_NAME (log: $BATCH_LOG)"
        python3 -u vault.py generate-batch "$PLAN" --workers "$WORKERS" > "$BATCH_LOG" 2>&1 &
        PIDS+=($!)
        WAVE_LOGS+=("$BATCH_LOG")
    done

    # Wait for all in this wave
    WAVE_OK=0
    WAVE_FAIL=0
    for i in "${!PIDS[@]}"; do
        PID=${PIDS[$i]}
        BLOG=${WAVE_LOGS[$i]}
        if wait "$PID"; then
            WAVE_OK=$((WAVE_OK + 1))
            # Extract counts from log
            GEN=$(grep -oP 'Generated: \K\d+' "$BLOG" 2>/dev/null | tail -1 || echo "0")
            TOTAL_GENERATED=$((TOTAL_GENERATED + GEN))
            log "  ✓ PID $PID done (+$GEN)"
        else
            WAVE_FAIL=$((WAVE_FAIL + 1))
            TOTAL_FAILED=$((TOTAL_FAILED + 1))
            log "  ✗ PID $PID failed (see $BLOG)"
        fi
    done

    log "  Wave $WAVE complete: $WAVE_OK ok, $WAVE_FAIL failed"

    IDX=$WAVE_END
    WAVE=$((WAVE + 1))

    # Brief pause between waves to let rate limits reset
    if [ $IDX -lt $TOTAL_PLANS ]; then
        log "  Cooling down 10s..."
        sleep 10
    fi
done

log ""
log "═══ Generation Complete ═══"
log "  Total generated: ~$TOTAL_GENERATED"
log "  Failed batches:  $TOTAL_FAILED"

# Step 3: Merge
log ""
log "── Step 3: Merging into corpus ──"
python3 vault.py merge 2>&1 | tee -a "$MASTER_LOG"

# Step 4: Release
log ""
log "── Step 4: Release ──"
python3 vault.py release --skip figures 2>&1 | tee -a "$MASTER_LOG"

log ""
log "═══ All Done ═══"
log "  Log: $MASTER_LOG"
