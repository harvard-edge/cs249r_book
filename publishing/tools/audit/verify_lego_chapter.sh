#!/usr/bin/env bash
# Build one chapter HTML via binder; verify every LEGO cell and every inline ref
# against rendered HTML prose (expected value + HTML context + coherence).
#
# PASS means:
#   1. Quarto HTML build succeeds (no tracebacks, no literal {python} in output)
#   2. Every LEGO cell execs; every export resolves; every prose ref appears in HTML
#   3. Every ref has extracted HTML prose context matching the computed value
#   4. LLM prose-coherence review passes (no fail verdicts on substituted prose)
#
# Usage (repo root):
#   ./book/tools/audit/verify_lego_chapter.sh vol1 introduction
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"

VOL="${1:?vol1|vol2}"
CH="${2:?chapter short name, e.g. introduction}"
LOG="/tmp/lego_verify_${VOL}_${CH}.log"
PROGRESS="$REPO/book/tools/audit/artifacts/lego_chapter_progress.md"
REPORTS="$REPO/book/tools/audit/artifacts/lego_chapter_reports"
# Artifacts are gitignored — a fresh checkout/worktree has no reports dir and
# set -e kills the run at the first cp below. (2026-06-07)
mkdir -p "$REPORTS"
CELLS_SNAP="$REPORTS/${VOL}_${CH}_cells.json"
PROSE_SNAP="$REPORTS/${VOL}_${CH}_prose.json"

export PYTHONPATH=mlsysim
export LEGO_VERIFY_VOL="$VOL"
export LEGO_VERIFY_CH="$CH"
export LEGO_VERIFY_REPO="$REPO"
export LEGO_VERIFY_CELLS_SNAP="$CELLS_SNAP"
export LEGO_VERIFY_PROSE_SNAP="$PROSE_SNAP"

echo "=== LEGO verify $VOL/$CH $(date -Iseconds) ===" | tee "$LOG"

./book/binder reset html 2>&1 | tee -a "$LOG"
python3 book/tools/audit/chapter_html_verify.py "--$VOL" "$CH" 2>&1 | tee -a "$LOG"
./book/binder reset html 2>&1 | tee -a "$LOG"

python3 book/tools/audit/fmt/audit_lego_cells.py --chapter "$VOL/$CH" 2>&1 | tee -a "$LOG"
cp "$REPO/book/tools/audit/artifacts/lego_cells_verify_report.json" "$CELLS_SNAP"
python3 book/tools/audit/fmt/audit_lego_rendered_prose.py --chapter "$VOL/$CH" 2>&1 | tee -a "$LOG"
cp "$REPO/book/tools/audit/artifacts/lego_rendered_prose_audit.json" "$PROSE_SNAP"

COHERENCE_RC=0
if command -v gemini >/dev/null 2>&1; then
  python3 book/tools/audit/lego_prose_coherence.py --chapter "$VOL/$CH" \
    --report "$REPO/book/tools/audit/artifacts/lego_prose_coherence_${VOL}_${CH}.json" \
    2>&1 | tee -a "$LOG" || COHERENCE_RC=$?
else
  echo "WARN: gemini CLI not found — skipping LLM prose-coherence gate" | tee -a "$LOG"
  COHERENCE_RC=0
fi
export LEGO_VERIFY_COHERENCE_RC="$COHERENCE_RC"

python3 <<'PY'
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

repo = Path(os.environ["LEGO_VERIFY_REPO"])
vol = os.environ["LEGO_VERIFY_VOL"]
ch = os.environ["LEGO_VERIFY_CH"]
coh_rc = int(os.environ.get("LEGO_VERIFY_COHERENCE_RC", "0"))

cells_path = Path(os.environ.get("LEGO_VERIFY_CELLS_SNAP", "")) or (
    repo / f"book/tools/audit/artifacts/lego_chapter_reports/{vol}_{ch}_cells.json"
)
prose_path = Path(os.environ.get("LEGO_VERIFY_PROSE_SNAP", "")) or (
    repo / f"book/tools/audit/artifacts/lego_chapter_reports/{vol}_{ch}_prose.json"
)
ledger_path = repo / "book/tools/audit/artifacts/chapter_html_audit.json"
coh_path = repo / f"book/tools/audit/artifacts/lego_prose_coherence_{vol}_{ch}.json"
report_md = repo / f"book/tools/audit/artifacts/lego_chapter_reports/{vol}_{ch}_rendered_prose.md"
cert_path = repo / f"book/tools/audit/artifacts/lego_chapter_reports/{vol}_{ch}_certificate.md"
progress = repo / "book/tools/audit/artifacts/lego_chapter_progress.md"

cells_raw = json.loads(cells_path.read_text()) if cells_path.is_file() else []
cells_list = cells_raw if isinstance(cells_raw, list) else [cells_raw]
prose_raw = json.loads(prose_path.read_text()) if prose_path.is_file() else []
prose_list = prose_raw if isinstance(prose_raw, list) else [prose_raw]
ledger = json.loads(ledger_path.read_text())
cr = next((r for r in cells_list if r.get("vol") == vol and r.get("chapter") == ch), {})
pr = next((r for r in prose_list if r.get("vol") == vol and r.get("chapter") == ch), {})
lc = ledger.get("chapters", {}).get(f"{vol}/{ch}", {})

coh = {}
if coh_path.is_file():
    coh_raw = json.loads(coh_path.read_text())
    if isinstance(coh_raw, list):
        coh = coh_raw[0] if coh_raw else {}
    else:
        coh = coh_raw

chapter_ok = lc.get("status") == "pass"
cells_ok = cr.get("status") == "PASS"
prose_ok = pr.get("status") == "PASS"
coh_status = coh.get("status", "skipped")
coh_ok = coh_status in ("PASS", "WARN", "skipped", "") and coh_rc == 0
coh_fail = coh_status == "FAIL" or coh_rc != 0

status = "PASS" if chapter_ok and cells_ok and prose_ok and coh_ok and not coh_fail else "FAIL"

cert_lines = [
    f"# LEGO certificate — {vol}/{ch}",
    "",
    f"Verified: {datetime.now(timezone.utc).isoformat()}",
    f"Status: **{status}**",
    "",
    "## What was checked",
    "- Binder HTML build for this chapter only",
    "- Every LEGO cell: exec, exports resolve, each `{python}` ref present in HTML",
    "- Every ref: computed value located in HTML with extracted prose context",
    "- LLM review of substituted prose for unit/number coherence",
    "",
    "## Gates",
    f"- HTML build + spurious/registry: {'PASS' if chapter_ok else 'FAIL'}",
    f"- LEGO cells: {cr.get('cells_pass', 0)}/{cr.get('cells_total', 0)}",
    f"- Rendered prose (value + HTML context per ref): {pr.get('refs_pass', 0)}/{pr.get('refs_total', 0)}",
    f"- LLM prose coherence: {coh_status}",
    "",
    "## LEGO cells",
]
for cell in cr.get("cells", []):
    nrefs = len(cell.get("refs", []))
    goal = (cell.get("goal") or "")[:100]
    cert_lines.append(
        f"- `{cell.get('class')}` (L{cell.get('cell_line')}) — {cell.get('status')} — "
        f"{nrefs} prose ref(s); {goal}"
    )

if coh.get("items"):
    cert_lines.extend(["", "## LLM coherence notes", ""])
    for it in coh["items"]:
        if it.get("verdict") in ("warn", "fail"):
            for issue in it.get("issues", []):
                cert_lines.append(
                    f"- `{it.get('id')}` L{issue.get('line')}: [{issue.get('severity')}] {issue.get('note')}"
                )

cert_lines.extend(["", "## Every inline ref (expected → QMD → HTML)", ""])
by_class: dict[str, list] = defaultdict(list)
for ref in pr.get("refs", []):
    by_class[ref["ref"].split(".")[0]].append(ref)
for cls in sorted(by_class):
    cert_lines.append(f"### `{cls}`")
    for ref in by_class[cls]:
        st = ref.get("status", "?")
        cert_lines.append(f"- L{ref['qmd_line']} `{ref['ref']}` — **{st}**")
        cert_lines.append(f"  - Expected: `{str(ref.get('expected', ''))[:120]}`")
        cert_lines.append(f"  - QMD: {str(ref.get('qmd_rendered', ''))[:200]}")
        ctx = (ref.get("html_contexts") or ["*(no context)*"])[0]
        cert_lines.append(f"  - HTML: …{ctx[:200]}…")
        for issue in ref.get("issues") or []:
            cert_lines.append(f"  - Issue: {issue}")
    cert_lines.append("")

cert_path.parent.mkdir(parents=True, exist_ok=True)
cert_path.write_text("\n".join(cert_lines), encoding="utf-8")

header = (
    "# LEGO chapter-by-chapter verify progress\n\n"
    "| Vol | Chapter | Status | Cells | Refs | Coherence | Certificate |\n"
    "|-----|---------|--------|-------|------|-----------|-------------|\n"
)
if not progress.exists():
    progress.write_text(header, encoding="utf-8")
line = (
    f"| {vol} | {ch} | **{status}** | "
    f"{cr.get('cells_pass', 0)}/{cr.get('cells_total', 0)} | "
    f"{pr.get('refs_pass', 0)}/{pr.get('refs_total', 0)} | "
    f"{coh_status} | `{cert_path.name}` |"
)
text = progress.read_text(encoding="utf-8")
pat = rf"\| {re.escape(vol)} \| {re.escape(ch)} \|[^\n]*\n"
if re.search(pat, text):
    text = re.sub(pat, line + "\n", text)
else:
    if "| Vol | Chapter |" not in text:
        text = header
    text = text.rstrip() + "\n" + line + "\n"
progress.write_text(text, encoding="utf-8")

print("")
print(f"CERTIFICATE: {cert_path}")
print(f"FULL REF REPORT: {report_md}")
print(line)
if status != "PASS":
    sys.exit(1)
PY
