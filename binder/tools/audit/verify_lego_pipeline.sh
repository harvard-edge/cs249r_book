#!/usr/bin/env bash
# Full LEGO verification pipeline: gates → render → cell audit → LLM prose → PDF
#
# Usage (repo root):
#   ./binder/tools/audit/verify_lego_pipeline.sh
#   ./binder/tools/audit/verify_lego_pipeline.sh --skip-render --skip-pdf
#   ./binder/tools/audit/verify_lego_pipeline.sh --skip-llm
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

SKIP_RENDER=0
SKIP_LLM=0
SKIP_PDF=0
for arg in "$@"; do
  case "$arg" in
    --skip-render) SKIP_RENDER=1 ;;
    --skip-llm) SKIP_LLM=1 ;;
    --skip-pdf) SKIP_PDF=1 ;;
  esac
done

export PYTHONPATH=mlsysim

echo "══════════════════════════════════════════════════════════════════════"
echo "Phase 0 — binder gates"
echo "══════════════════════════════════════════════════════════════════════"
./binder/binder check registry
./binder/binder check code --scope lego-dead-code
./binder/binder check code --scope lego-prose-literals
./binder/binder check math --scope canonical
$PYTHON binder/tools/audit/lego_focal_verify.py books/vol1 books/vol2

if [[ "$SKIP_RENDER" -eq 0 ]]; then
  echo ""
  echo "══════════════════════════════════════════════════════════════════════"
  echo "Phase 1 — HTML render + .0 audit (vol1 + vol2)"
  echo "══════════════════════════════════════════════════════════════════════"
  ./binder/tools/audit/fmt/render_html.sh vol1
  ./binder/tools/audit/fmt/render_html.sh vol2
else
  echo "(skipping Phase 1 render — using archived HTML)"
fi

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "Phase 2 — per-LEGO-cell exec + export + HTML ref verification"
echo "══════════════════════════════════════════════════════════════════════"
$PYTHON binder/tools/audit/fmt/audit_lego_cells.py \
  --report binder/tools/audit/artifacts/lego_cells_verify_report.json
$PYTHON binder/tools/audit/fmt/audit_lego_html.py \
  --report binder/tools/audit/artifacts/lego_html_verify_report.json

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "Phase 3 — per-chapter rendered prose (optional; run verify_lego_chapter.sh for full PASS)"
echo "══════════════════════════════════════════════════════════════════════"
echo "  ./binder/tools/audit/run_all_lego_chapters.sh"
echo "  or: ./binder/tools/audit/verify_lego_chapter.sh vol1 introduction"

if [[ "$SKIP_LLM" -eq 0 ]]; then
  echo ""
  echo "══════════════════════════════════════════════════════════════════════"
  echo "Phase 4 — LLM prose coherence review"
  echo "══════════════════════════════════════════════════════════════════════"
  $PYTHON binder/tools/audit/lego_prose_coherence.py --all --workers 4 \
    --report binder/tools/audit/artifacts/lego_prose_coherence_report.json
else
  echo "(skipping Phase 4 LLM prose review)"
fi

if [[ "$SKIP_PDF" -eq 0 ]]; then
  echo ""
  echo "══════════════════════════════════════════════════════════════════════"
  echo "Phase 5 — PDF build (vol1 + vol2)"
  echo "══════════════════════════════════════════════════════════════════════"
  ./binder/binder build pdf --vol1
  ./binder/binder build pdf --vol2
else
  echo "(skipping Phase 5 PDF build)"
fi

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "Pipeline complete"
echo "  lego_cells:  binder/tools/audit/artifacts/lego_cells_verify_report.json"
echo "  lego_html:   binder/tools/audit/artifacts/lego_html_verify_report.json"
echo "  coherence:   binder/tools/audit/artifacts/lego_prose_coherence_report.json"
echo "══════════════════════════════════════════════════════════════════════"
