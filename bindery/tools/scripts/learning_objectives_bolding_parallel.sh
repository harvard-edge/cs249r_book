#!/usr/bin/env bash
# Learning Objectives Bolding: Run Gemini CLI in parallel across all chapters
#
# Usage: From repo root, run:
#   ./bindery/tools/scripts/learning_objectives_bolding_parallel.sh
#
# Requires: Gemini CLI installed (npm install -g @google/gemini-cli or brew install gemini-cli)
# Rate limits: Free tier is 60 req/min. Default MAX_PARALLEL=8 to stay under limit.
#              Adjust with: MAX_PARALLEL=4 ./bindery/tools/scripts/learning_objectives_bolding_parallel.sh

set -e

# Repo root (script lives at bindery/tools/scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# Limit concurrent Gemini invocations to avoid rate limits (60/min free tier)
MAX_PARALLEL="${MAX_PARALLEL:-2}"

# All chapters with Learning Objectives (exclude compute_infrastructure_expanded—draft)
CHAPTERS=(
  "books/vol1/introduction/introduction.qmd"
  "books/vol1/ml_systems/ml_systems.qmd"
  "books/vol1/data_engineering/data_engineering.qmd"
  "books/vol1/data_selection/data_selection.qmd"
  "books/vol1/nn_architectures/nn_architectures.qmd"
  "books/vol1/nn_computation/nn_computation.qmd"
  "books/vol1/training/training.qmd"
  "books/vol1/hw_acceleration/hw_acceleration.qmd"
  "books/vol1/model_compression/model_compression.qmd"
  "books/vol1/frameworks/frameworks.qmd"
  "books/vol1/benchmarking/benchmarking.qmd"
  "books/vol1/ml_ops/ml_ops.qmd"
  "books/vol1/model_serving/model_serving.qmd"
  "books/vol1/responsible_engr/responsible_engr.qmd"
  "books/vol1/ml_workflow/ml_workflow.qmd"
  "books/vol1/conclusion/conclusion.qmd"
  "books/vol1/backmatter/appendix_algorithm.qmd"
  "books/vol1/backmatter/appendix_data.qmd"
  "books/vol1/backmatter/appendix_machine.qmd"
  "books/vol1/backmatter/appendix_dam.qmd"
  "books/vol1/backmatter/appendix_assumptions.qmd"
  "books/vol2/introduction/introduction.qmd"
  "books/vol2/compute_infrastructure/compute_infrastructure.qmd"
  "books/vol2/network_fabrics/network_fabrics.qmd"
  "books/vol2/data_storage/data_storage.qmd"
  "books/vol2/distributed_training/distributed_training.qmd"
  "books/vol2/collective_communication/collective_communication.qmd"
  "books/vol2/fault_tolerance/fault_tolerance.qmd"
  "books/vol2/fleet_orchestration/fleet_orchestration.qmd"
  "books/vol2/inference/inference.qmd"
  "books/vol2/edge_intelligence/edge_intelligence.qmd"
  "books/vol2/ops_scale/ops_scale.qmd"
  "books/vol2/performance_engineering/performance_engineering.qmd"
  "books/vol2/sustainable_ai/sustainable_ai.qmd"
  "books/vol2/responsible_ai/responsible_ai.qmd"
  "books/vol2/robust_ai/robust_ai.qmd"
  "books/vol2/security_privacy/security_privacy.qmd"
  "books/vol2/conclusion/conclusion.qmd"
)

run_chapter() {
  local file="$1"
  local prompt="Read the instructions at .claude/docs/learning-objectives-gemini-instructions.md, then edit this chapter file to add bolding to Learning Objectives: $file"
  echo "[$(date +%H:%M:%S)] Starting: $file"
  if gemini -p "$prompt" --include-directories "books,.claude/docs"; then
    echo "[$(date +%H:%M:%S)] Done: $file"
  else
    echo "[$(date +%H:%M:%S)] FAILED: $file" >&2
    return 1
  fi
}

export -f run_chapter
export MAX_PARALLEL

echo "Learning Objectives Bolding: Running Gemini CLI on ${#CHAPTERS[@]} chapters (max $MAX_PARALLEL parallel)"
echo "Instructions: .claude/docs/learning-objectives-gemini-instructions.md"
echo ""

# Run in parallel with semaphore
for ch in "${CHAPTERS[@]}"; do
  while [ "$(jobs -r 2>/dev/null | wc -l | tr -d ' ')" -ge "$MAX_PARALLEL" ]; do
    sleep 2
  done
  run_chapter "$ch" &
done

wait
echo ""
echo "All chapters processed. Review changes with: git diff books/"
