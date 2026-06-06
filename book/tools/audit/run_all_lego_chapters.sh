#!/usr/bin/env bash
# Verify every chapter one-at-a-time (binder build html per chapter).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"
LOG="$REPO/book/tools/audit/artifacts/lego_chapter_batch.log"
FAILS="$REPO/book/tools/audit/artifacts/lego_chapter_failures.txt"
LOCKDIR="$REPO/book/tools/audit/artifacts/lego_chapter_batch.lock.d"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "Another LEGO chapter batch is running (lock: $LOCKDIR)" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

: > "$LOG"
: > "$FAILS"

vol1_chapters=(
  introduction ml_systems ml_workflow data_engineering nn_computation
  nn_architectures frameworks training data_selection model_compression
  hw_acceleration benchmarking model_serving ml_ops responsible_engr conclusion
  appendix_algorithm appendix_assumptions appendix_dam appendix_data appendix_machine
)

vol2_chapters=(
  introduction compute_infrastructure network_fabrics data_storage
  distributed_training collective_communication fault_tolerance fleet_orchestration
  performance_engineering inference edge_intelligence ops_scale security_privacy
  robust_ai sustainable_ai responsible_ai conclusion
  appendix_dam appendix_fleet appendix_communication appendix_reliability
  appendix_inference appendix_c3 appendix_assumptions
)

run_vol() {
  local vol=$1
  shift
  local chapters=("$@")
  for ch in "${chapters[@]}"; do
    echo "========== BATCH $vol/$ch $(date -Iseconds) ==========" | tee -a "$LOG"
    if ./book/tools/audit/verify_lego_chapter.sh "$vol" "$ch" >> "$LOG" 2>&1; then
      echo "OK $vol/$ch" | tee -a "$LOG"
    else
      echo "FAIL $vol/$ch" | tee -a "$LOG"
      echo "$vol/$ch" >> "$FAILS"
    fi
  done
}

run_vol vol1 "${vol1_chapters[@]}"
run_vol vol2 "${vol2_chapters[@]}"

echo "DONE $(date -Iseconds)" | tee -a "$LOG"
echo "Failures: $(wc -l < "$FAILS" | tr -d ' ') — see $FAILS"
