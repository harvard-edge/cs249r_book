#!/usr/bin/env bash
# Rebuild the student syllabus HTML from README.md (source of truth).
set -euo pipefail
cd "$(dirname "$0")"

python3 - <<'PY'
from pathlib import Path
lines = Path("README.md").read_text().splitlines(True)
i = 0
if lines and lines[0].startswith("# "):
    i = 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1
Path("_syllabus-body.md").write_text("".join(lines[i:]))
PY

quarto render Physical-AI-Systems.qmd --to html
echo "Wrote Physical-AI-Systems.html"
