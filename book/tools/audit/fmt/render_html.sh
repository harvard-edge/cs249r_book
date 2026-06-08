#!/usr/bin/env bash
# Render inline-python chapters sequentially and scan archived HTML for spurious .0
#
# Usage (from repo root):
#   ./book/tools/audit/fmt/render_html.sh vol1
#   ./book/tools/audit/fmt/render_html.sh vol2
#
# Each chapter HTML is copied to book/quarto/_build/html-audit/<vol>/<chapter>.html
# before the next build overwrites the fast-build output directory.
set -euo pipefail

VOL="${1:-vol1}"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"

AUDIT_PY="book/tools/audit/fmt/audit_html.py"
ARCHIVE_DIR="book/quarto/_build/html-audit/${VOL}"
mkdir -p "$ARCHIVE_DIR"

case "$VOL" in
  vol1)
    BINDER_VOL=(--vol1)
    BUILD_DIR="book/quarto/_build/html-vol1/contents/vol1"
    CHAPTERS=(
      introduction/introduction
      ml_systems/ml_systems
      ml_workflow/ml_workflow
      data_engineering/data_engineering
      nn_computation/nn_computation
      nn_architectures/nn_architectures
      frameworks/frameworks
      training/training
      data_selection/data_selection
      model_compression/model_compression
      hw_acceleration/hw_acceleration
      benchmarking/benchmarking
      model_serving/model_serving
      ml_ops/ml_ops
      responsible_engr/responsible_engr
      conclusion/conclusion
      backmatter/appendix_algorithm
      backmatter/appendix_assumptions
      backmatter/appendix_dam
      backmatter/appendix_data
      backmatter/appendix_machine
    )
    ;;
  vol2)
    BINDER_VOL=(--vol2)
    BUILD_DIR="book/quarto/_build/html-vol2/contents/vol2"
    CHAPTERS=(
      introduction/introduction
      compute_infrastructure/compute_infrastructure
      network_fabrics/network_fabrics
      data_storage/data_storage
      distributed_training/distributed_training
      collective_communication/collective_communication
      fault_tolerance/fault_tolerance
      fleet_orchestration/fleet_orchestration
      performance_engineering/performance_engineering
      inference/inference
      edge_intelligence/edge_intelligence
      ops_scale/ops_scale
      security_privacy/security_privacy
      robust_ai/robust_ai
      sustainable_ai/sustainable_ai
      responsible_ai/responsible_ai
      conclusion/conclusion
      backmatter/appendix_dam
      backmatter/appendix_fleet
      backmatter/appendix_communication
      backmatter/appendix_reliability
      backmatter/appendix_inference
      backmatter/appendix_c3
      backmatter/appendix_assumptions
    )
    ;;
  *)
    echo "Unknown volume: $VOL (expected vol1 or vol2)" >&2
    exit 2
    ;;
esac

echo "$(echo "$VOL" | tr '[:lower:]' '[:upper:]') HTML RENDER + AUDIT"
echo "========================"
FAIL=0
for ch in "${CHAPTERS[@]}"; do
  name="${ch##*/}"
  binder_ch="${VOL}/${name}"
  printf "%-28s " "$name"
  if ! ./book/binder build html "${BINDER_VOL[@]}" "$binder_ch" --skip-hygiene --skip-validate >/tmp/render_${VOL}_${name}.log 2>&1; then
    echo "BUILD FAIL (see /tmp/render_${VOL}_${name}.log)"
    FAIL=$((FAIL + 1))
    continue
  fi
  html="${BUILD_DIR}/${ch}.html"
  archive="${ARCHIVE_DIR}/${name}.html"
  if [[ ! -f "$html" ]]; then
    echo "HTML MISSING"
    FAIL=$((FAIL + 1))
    continue
  fi
  cp "$html" "$archive"
  result=$(python3 "$AUDIT_PY" "$archive" 2>&1 || true)
  if [[ "$result" == "CLEAN" ]]; then
    echo "OK + CLEAN"
  else
    echo "OK + HTML FLAGS"
    echo "$result" | head -5
    FAIL=$((FAIL + 1))
  fi
done
echo "========================"
echo "Failed: $FAIL / ${#CHAPTERS[@]}"
echo "Archived HTML: ${ARCHIVE_DIR}/"
exit $(( FAIL > 0 ? 1 : 0 ))
