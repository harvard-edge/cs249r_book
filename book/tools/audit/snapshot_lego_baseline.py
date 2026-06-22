#!/usr/bin/env python3
"""Snapshot LEGO cell inventory and focal-verify status for migration baseline."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "book" / "tools" / "audit" / "artifacts"
OUT_PATH = OUT_DIR / "lego_baseline.json"

CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
CLASS = re.compile(r"^class\s+(\w+)", re.M)
IMPORT_CONSTANTS = re.compile(
    r"from\s+mlsysim\.core\.constants\s+import|from\s+mlsysim\.core\.constants\s+import\s+\*"
)


def qmd_files() -> list[Path]:
    root = REPO_ROOT / "book" / "quarto" / "contents"
    return sorted(root.rglob("*.qmd"))


def scan_qmd(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    cells: list[dict] = []
    i = 0
    while i < len(lines):
        if CELL_START.match(lines[i]):
            start_line = i + 1
            j = i + 1
            while j < len(lines) and not CELL_END.match(lines[j]):
                j += 1
            block = "\n".join(lines[i : j + 1])
            cls_m = CLASS.search(block)
            cells.append(
                {
                    "start_line": start_line,
                    "class": cls_m.group(1) if cls_m else None,
                    "uses_constants_import": bool(IMPORT_CONSTANTS.search(block)),
                    "lines": len(block.splitlines()),
                }
            )
            i = j + 1
        else:
            i += 1
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "cell_count": len(cells),
        "cells": cells,
        "constants_import_cells": sum(1 for c in cells if c["uses_constants_import"]),
    }


def run_focal_verify() -> dict:
    script = REPO_ROOT / "book" / "tools" / "audit" / "lego_focal_verify.py"
    if not script.exists():
        return {"skipped": True, "reason": "lego_focal_verify.py missing"}
    vol1 = REPO_ROOT / "book" / "quarto" / "contents" / "vol1"
    vol2 = REPO_ROOT / "book" / "quarto" / "contents" / "vol2"
    proc = subprocess.run(
        [sys.executable, str(script), str(vol1), str(vol2)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-8000:] if proc.stdout else "",
        "stderr": proc.stderr[-4000:] if proc.stderr else "",
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chapters = [scan_qmd(p) for p in qmd_files()]
    payload = {
        "generated_by": "book/tools/audit/snapshot_lego_baseline.py",
        "chapter_count": len(chapters),
        "total_cells": sum(c["cell_count"] for c in chapters),
        "total_constants_import_cells": sum(c["constants_import_cells"] for c in chapters),
        "chapters": chapters,
        "focal_verify": run_focal_verify(),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(
        f"  {payload['total_cells']} LEGO cells, "
        f"{payload['total_constants_import_cells']} with constants import"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
