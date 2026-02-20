#!/usr/bin/env bash
# Learning Objectives Bolding: Run Gemini CLI in parallel across all chapters
#
# Usage: From repo root, run:
#   ./book/tools/scripts/learning_objectives_bolding_parallel.sh
#
# Requires: Gemini CLI installed (npm install -g @google/gemini-cli or brew install gemini-cli)
# Rate limits: Free tier is 60 req/min. Default MAX_PARALLEL=8 to stay under limit.
#              Adjust with: MAX_PARALLEL=4 ./book/tools/scripts/learning_objectives_bolding_parallel.sh

set -e

# Repo root (script lives at book/tools/scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# Limit concurrent Gemini invocations to avoid rate limits (60/min free tier)
MAX_PARALLEL="${MAX_PARALLEL:-2}"

# All chapters with Learning Objectives (exclude compute_infrastructure_expanded—draft)
CHAPTERS=(
  "book/quarto/contents/vol1/introduction/introduction.qmd"
  "book/quarto/contents/vol1/ml_systems/ml_systems.qmd"
  "book/quarto/contents/vol1/data_engineering/data_engineering.qmd"
  "book/quarto/contents/vol1/data_selection/data_selection.qmd"
  "book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd"
  "book/quarto/contents/vol1/nn_computation/nn_computation.qmd"
  "book/quarto/contents/vol1/training/training.qmd"
  "book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd"
  "book/quarto/contents/vol1/optimizations/model_compression.qmd"
  "book/quarto/contents/vol1/frameworks/frameworks.qmd"
  "book/quarto/contents/vol1/benchmarking/benchmarking.qmd"
  "book/quarto/contents/vol1/ml_ops/ml_ops.qmd"
  "book/quarto/contents/vol1/model_serving/model_serving.qmd"
  "book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd"
  "book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd"
  "book/quarto/contents/vol1/conclusion/conclusion.qmd"
  "book/quarto/contents/vol1/backmatter/appendix_algorithm.qmd"
  "book/quarto/contents/vol1/backmatter/appendix_data.qmd"
  "book/quarto/contents/vol1/backmatter/appendix_machine.qmd"
  "book/quarto/contents/vol1/backmatter/appendix_dam.qmd"
  "book/quarto/contents/vol1/backmatter/appendix_assumptions.qmd"
  "book/quarto/contents/vol2/introduction/introduction.qmd"
  "book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd"
  "book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd"
  "book/quarto/contents/vol2/data_storage/data_storage.qmd"
  "book/quarto/contents/vol2/distributed_training/distributed_training.qmd"
  "book/quarto/contents/vol2/collective_communication/collective_communication.qmd"
  "book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd"
  "book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd"
  "book/quarto/contents/vol2/inference/inference.qmd"
  "book/quarto/contents/vol2/edge_intelligence/edge_intelligence.qmd"
  "book/quarto/contents/vol2/ops_scale/ops_scale.qmd"
  "book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd"
  "book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd"
  "book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd"
  "book/quarto/contents/vol2/robust_ai/robust_ai.qmd"
  "book/quarto/contents/vol2/security_privacy/security_privacy.qmd"
  "book/quarto/contents/vol2/conclusion/conclusion.qmd"
)

run_chapter() {
  local file="$1"
  local prompt="Read the instructions at .claude/docs/learning-objectives-gemini-instructions.md, then edit this chapter file to add bolding to Learning Objectives: $file"
  echo "[$(date +%H:%M:%S)] Starting: $file"
  if gemini -p "$prompt" --include-directories "book/quarto/contents,.claude/docs"; then
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
echo "All chapters processed. Review changes with: git diff book/quarto/contents/"
