#!/usr/bin/env python3
"""struct_qa_pdf.py

Structural QA for a built PDF in the MLSysBook camera-ready sweep.
Shells out to ``pdftotext``, ``pdfimages``, and ``mutool`` to inspect a
single PDF and emit a one-line JSON summary.

Usage:
    python3 struct_qa_pdf.py --vol vol1 \
        --pdf /abs/path/to/build.pdf \
        --qmd-fig-count 412 \
        --out /abs/path/to/run-dir/qa/structural-pdf-vol1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


RAW_REF_RE = re.compile(r"\?\?|@(?:fig|sec|tbl|eq)-[A-Za-z0-9_:-]+|\[@[^\]]+\]")
RAW_CITE_RE = re.compile(r"\\cite[a-z]*\{|\(\?\?\)|\[\?\]")
BROKEN_XREF_RE = re.compile(r"\?\?+|\b(?:Figure|Table|Section|Equation)\s+\?\?+", re.I)


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        **kw,
    )


def pdf_page_count(pdf: Path) -> int:
    if have("mutool"):
        cp = run(["mutool", "info", str(pdf)])
        m = re.search(r"Pages:\s*(\d+)", cp.stdout)
        if m:
            return int(m.group(1))
    if have("pdfinfo"):
        cp = run(["pdfinfo", str(pdf)])
        m = re.search(r"Pages:\s*(\d+)", cp.stdout)
        if m:
            return int(m.group(1))
    return -1


def pdf_text(pdf: Path) -> str:
    if have("pdftotext"):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            out_txt = Path(tmp.name)
        try:
            run(["pdftotext", "-layout", str(pdf), str(out_txt)])
            return out_txt.read_text(encoding="utf-8", errors="replace")
        finally:
            out_txt.unlink(missing_ok=True)
    return ""


def pdf_image_count(pdf: Path) -> int:
    if not have("pdfimages"):
        return -1
    cp = run(["pdfimages", "-list", str(pdf)])
    if cp.returncode != 0:
        return -1
    lines = [l for l in cp.stdout.splitlines() if l.strip()]
    # First two lines are header + separator; subsequent lines are images.
    return max(0, len(lines) - 2)


def pdf_has_toc(pdf: Path, text: str) -> bool:
    if have("mutool"):
        cp = run(["mutool", "show", str(pdf), "outline"])
        if cp.returncode == 0 and cp.stdout.strip():
            return True
    return bool(re.search(r"\bContents\b|\bTable of Contents\b", text[:8000], re.I))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vol", required=True, choices=["vol1", "vol2"])
    p.add_argument("--pdf", required=True)
    p.add_argument("--qmd-fig-count", required=True, type=int)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    pdf = Path(args.pdf).resolve()
    if not pdf.is_file():
        print(f"struct_qa_pdf: missing pdf {pdf}", file=sys.stderr)
        return 2

    size_bytes = pdf.stat().st_size
    page_count = pdf_page_count(pdf)
    text = pdf_text(pdf)
    broken_xrefs = len(BROKEN_XREF_RE.findall(text))
    raw_ref_leaks = len(RAW_REF_RE.findall(text))
    raw_cite_leaks = len(RAW_CITE_RE.findall(text))
    image_count = pdf_image_count(pdf)
    qmd_fig_count = max(args.qmd_fig_count, 0)
    image_ratio = (image_count / qmd_fig_count) if qmd_fig_count > 0 and image_count >= 0 else None
    toc_present = pdf_has_toc(pdf, text)

    ok = (
        broken_xrefs == 0
        and raw_ref_leaks == 0
        and raw_cite_leaks == 0
        and toc_present
        and (image_ratio is None or image_ratio >= 0.5)
    )

    rec = {
        "vol": args.vol,
        "format": "pdf",
        "pdf_path": str(pdf),
        "size_bytes": size_bytes,
        "page_count": page_count,
        "broken_xrefs": broken_xrefs,
        "raw_ref_leaks": raw_ref_leaks,
        "raw_cite_leaks": raw_cite_leaks,
        "image_count": image_count,
        "qmd_figure_count": qmd_fig_count,
        "image_ratio": image_ratio,
        "toc_present": toc_present,
        "ok": ok,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    print(json.dumps(rec, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
