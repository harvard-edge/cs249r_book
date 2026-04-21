#!/usr/bin/env python3
"""
PDF math-rendering audit for MLSysBook chapters.

For each chapter:
  1. Builds a PDF via `./book/binder build pdf <vol>/<chap>`.
     A build failure is itself a strong signal of a math-rendering bug
     (raw \\command outside math mode causes LaTeX to error out).
  2. Extracts text via `pdftotext` and applies the same leak-pattern
     detector used by the HTML audit. Catches LaTeX leakage that compiled
     successfully but typeset literally.
  3. Renders pages to PNG via `pdftoppm -r 150` so a human can spot-check.
     Optionally limited to the first N pages for sampling.

NOTE: This script must NOT run concurrently with `audit_math_rendering.py`
or any other `binder build` invocation -- they all mutate the shared
`book/quarto/_quarto.yml`.

Usage (run from repo root):
    python3 tools/audit/audit_math_pdf.py vol1/introduction vol2/inference
    python3 tools/audit/audit_math_pdf.py --fixed     # only chapters we touched
    python3 tools/audit/audit_math_pdf.py --all       # every chapter
    python3 tools/audit/audit_math_pdf.py --max-pages 0  # render every page
"""

from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BINDER = REPO / "book" / "binder"
BUILD_DIRS = {
    "vol1": REPO / "book" / "quarto" / "_build" / "pdf-vol1",
    # binder writes vol2 PDFs into the per-volume directory if it's
    # configured (config/_quarto-pdf-vol2.yml). If not present at runtime,
    # the fallback in find_pdf() searches both candidate dirs.
    "vol2": REPO / "book" / "quarto" / "_build" / "pdf-vol2",
}
OUT_DIR = REPO / "audit-pdf-output"

# Same leak patterns as HTML audit (kept in sync intentionally).
LEAK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("backslash-cmd", re.compile(r"\\(?:times|frac|approx|alpha|beta|gamma|delta|sigma|mu|eta|lambda|theta|sqrt|sum|int|cdot|leq|geq|neq|partial|nabla|infty|log|ln|exp|min|max|pm|mp|forall|exists|in|notin|to|gets|mapsto|rightarrow|Rightarrow|prod|equiv|sim|propto|land|lor|neg|text|mathbf|mathit|mathcal|mathbb)\b")),  # codespell:ignore notin
    ("caret-brace", re.compile(r"(?<![A-Za-z0-9_])\^\{[^}\n]{1,40}\}")),
    ("under-brace", re.compile(r"(?<![A-Za-z0-9_])_\{[^}\n]{1,40}\}")),
    ("bare-pow10", re.compile(r"\b10\^\{[+-]?\d{1,4}\}")),
    ("dollar-math-text", re.compile(r"\$\\[a-zA-Z]+|\\[a-zA-Z]+\$")),
]

# Chapters that received our LaTeX rendering fixes (touched in commits
# 08bf690d0, 3ba9e8b06, adcb0b5e1).
FIXED_CHAPTERS = [
    ("introduction", "vol1"),
    ("benchmarking", "vol1"),
    ("appendix_dam", "vol1"),
    ("appendix_algorithm", "vol1"),
    ("model_compression", "vol1"),
    ("conclusion", "vol2"),
    ("inference", "vol2"),
    ("introduction", "vol2"),
    ("appendix_dam", "vol2"),
    ("appendix_c3", "vol2"),
    ("robust_ai", "vol2"),
]


@dataclass
class Leak:
    pattern: str
    match: str
    context: str


@dataclass
class PdfReport:
    name: str
    volume: str
    qmd_path: str
    build_ok: bool
    build_seconds: float
    pdf_path: str | None = None
    pages: int = 0
    image_dir: str | None = None
    image_count: int = 0
    leaks: list[Leak] = field(default_factory=list)
    error: str | None = None


def list_all_chapters() -> list[tuple[str, str, Path]]:
    chapters = []
    for vol in ("vol1", "vol2"):
        contents = REPO / "book" / "quarto" / "contents" / vol
        for qmd in sorted(contents.glob("*/*.qmd")):
            if qmd.name.startswith("_") and qmd.name not in (
                "_notation_body.qmd",
                "_notation_distributed.qmd",
            ):
                continue
            chapters.append((qmd.stem, vol, qmd))
    return chapters


def build_pdf(name: str, volume: str) -> tuple[bool, float, str]:
    # Per-volume index symlink: PDF builds need `index.qmd` to point
    # to either `index-vol1.qmd` or `index-vol2.qmd`. CI does this
    # per-job; we replicate it here.
    index_link = REPO / "book" / "quarto" / "index.qmd"
    target = f"index-{volume}.qmd"
    try:
        if index_link.is_symlink() or index_link.exists():
            index_link.unlink()
        index_link.symlink_to(target)
    except OSError:
        pass
    vol_flag = f"--{volume}"
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["python3", str(BINDER), "build", "pdf", name, vol_flag],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return False, time.time() - t0, "build timed out after 900s"
    secs = time.time() - t0
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    ok = proc.returncode == 0 and (
        "build completed" in out.lower()
        or "output created" in out.lower()
        or "render complete" in out.lower()
    )
    return ok, secs, out


def find_pdf(name: str, volume: str, qmd_path: Path) -> Path | None:
    candidate_dirs = [BUILD_DIRS[volume],
                      REPO / "book" / "quarto" / "_build" / "pdf-vol1",
                      REPO / "book" / "quarto" / "_build" / "pdf-vol2"]
    for base in candidate_dirs:
        if not base.exists():
            continue
        candidates = [
            base / f"{name}.pdf",
            base / f"Machine-Learning-Systems-{volume.title()}.pdf",
            base / "Machine-Learning-Systems.pdf",
            base / "MLSysBook.pdf",
        ]
        for c in candidates:
            if c.exists():
                return c
        pdfs = sorted(base.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pdfs:
            return pdfs[0]
    return None


def pdf_pages(pdf: Path) -> int:
    try:
        out = subprocess.run(
            ["pdfinfo", str(pdf)], capture_output=True, text=True, timeout=30
        ).stdout
        for line in out.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return 0


def extract_text(pdf: Path) -> str:
    try:
        return subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            capture_output=True, text=True, timeout=120,
        ).stdout
    except Exception:
        return ""


def scan_text(text: str) -> list[Leak]:
    leaks: list[Leak] = []
    for label, pat in LEAK_PATTERNS:
        for m in pat.finditer(text):
            start, end = max(0, m.start() - 40), min(len(text), m.end() + 40)
            ctx = text[start:end].replace("\n", " ").strip()
            leaks.append(Leak(label, m.group(0), ctx))
            if len(leaks) > 200:
                return leaks
    return leaks


def render_images(pdf: Path, dest: Path, max_pages: int) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    args = ["pdftoppm", "-r", "150", "-png", str(pdf), str(dest / "page")]
    if max_pages > 0:
        args.extend(["-l", str(max_pages)])
    try:
        subprocess.run(args, check=True, capture_output=True, timeout=600)
    except Exception as e:
        print(f"    pdftoppm failed: {e}")
        return 0
    return len(list(dest.glob("page-*.png")))


def audit_chapter(name: str, volume: str, qmd_path: Path,
                  max_pages: int, build: bool) -> PdfReport:
    rep = PdfReport(
        name=name, volume=volume,
        qmd_path=str(qmd_path.relative_to(REPO)),
        build_ok=True, build_seconds=0.0,
    )
    if build:
        ok, secs, output = build_pdf(name, volume)
        rep.build_ok = ok
        rep.build_seconds = secs
        if not ok:
            rep.error = output[-2000:]
            return rep

    pdf = find_pdf(name, volume, qmd_path)
    if pdf is None:
        rep.error = "pdf output not found"
        rep.build_ok = False
        return rep

    # Persist the PDF to a chapter-specific path; the next build will
    # otherwise overwrite the per-volume output.
    chapter_dir = OUT_DIR / volume / name
    if chapter_dir.exists():
        shutil.rmtree(chapter_dir)
    chapter_dir.mkdir(parents=True, exist_ok=True)
    saved_pdf = chapter_dir / f"{name}.pdf"
    shutil.copy2(pdf, saved_pdf)

    rep.pdf_path = str(saved_pdf.relative_to(REPO))
    rep.pages = pdf_pages(saved_pdf)

    text = extract_text(saved_pdf)
    rep.leaks = scan_text(text)

    rep.image_count = render_images(saved_pdf, chapter_dir / "pages", max_pages)
    rep.image_dir = str((chapter_dir / "pages").relative_to(REPO))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chapters", nargs="*", help="vol/<chap> tokens")
    ap.add_argument("--fixed", action="store_true", help="audit only chapters we patched")
    ap.add_argument("--all", action="store_true", help="audit every chapter")
    ap.add_argument("--max-pages", type=int, default=10,
                    help="limit pages rendered to PNG; 0 = all")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--report", default="audit-pdf-report.json")
    ap.add_argument("--text-report", default="audit-pdf-report.md")
    args = ap.parse_args()

    all_chap = list_all_chapters()
    by_key = {f"{v}/{n}": (n, v, q) for (n, v, q) in all_chap}

    if args.fixed:
        targets = [by_key[f"{v}/{n}"] for (n, v) in FIXED_CHAPTERS if f"{v}/{n}" in by_key]
    elif args.all:
        targets = all_chap
    elif args.chapters:
        targets = [by_key[c] for c in args.chapters if c in by_key]
    else:
        ap.error("must pass chapters, --fixed, or --all")

    OUT_DIR.mkdir(exist_ok=True)
    print(f"PDF auditing {len(targets)} chapters; max-pages={args.max_pages}", flush=True)

    reports: list[PdfReport] = []
    for i, (name, vol, qmd) in enumerate(targets, 1):
        print(f"[{i:2d}/{len(targets)}] {vol}/{name} ... ", end="", flush=True)
        rep = audit_chapter(name, vol, qmd, args.max_pages, build=not args.skip_build)
        if not rep.build_ok:
            tag = "BUILD FAILED"
        elif rep.error:
            tag = f"ERROR: {rep.error[:60]}"
        else:
            counts: dict[str, int] = {}
            for L in rep.leaks:
                counts[L.pattern] = counts.get(L.pattern, 0) + 1
            tag = (f"{len(rep.leaks)} leak(s) {dict(counts)} | "
                   f"pages={rep.pages} imgs={rep.image_count}")
        print(f"{tag} ({rep.build_seconds:.0f}s)", flush=True)
        reports.append(rep)

    Path(args.report).write_text(json.dumps([asdict(r) for r in reports], indent=2))

    lines = ["# PDF Math Rendering Audit", "",
             f"Total chapters: **{len(reports)}**", ""]
    failed = [r for r in reports if not r.build_ok]
    leaky = [r for r in reports if r.build_ok and r.leaks]
    clean = [r for r in reports if r.build_ok and not r.leaks and not r.error]
    lines += [f"- Clean: **{len(clean)}**",
              f"- Leaky: **{len(leaky)}**",
              f"- Build failures: **{len(failed)}**", ""]

    if failed:
        lines += ["## Build failures (likely raw LaTeX outside math mode)", ""]
        for r in failed:
            lines += [f"### `{r.volume}/{r.name}`",
                      f"- qmd: `{r.qmd_path}`",
                      "```", (r.error or "")[-1500:], "```", ""]

    if leaky:
        lines += ["## Chapters with LaTeX leakage in PDF text", ""]
        for r in leaky:
            counts: dict[str, int] = {}
            for L in r.leaks:
                counts[L.pattern] = counts.get(L.pattern, 0) + 1
            lines += [f"### `{r.volume}/{r.name}` ({len(r.leaks)} leaks: {dict(counts)})",
                      f"- pdf: `{r.pdf_path}` ({r.pages} pages)",
                      f"- images: `{r.image_dir}/` ({r.image_count} png)",
                      ""]
            shown: dict[str, int] = {}
            for L in r.leaks[:30]:
                if shown.get(L.pattern, 0) >= 5:
                    continue
                shown[L.pattern] = shown.get(L.pattern, 0) + 1
                lines += [f"  - **{L.pattern}** `{L.match}`",
                          f"    > ...{L.context}..."]
            lines.append("")

    if clean:
        lines += ["## Clean chapters (review images for visual confirmation)", ""]
        for r in clean:
            lines += [f"- `{r.volume}/{r.name}` — pdf `{r.pdf_path}` "
                      f"({r.pages} pages) — images `{r.image_dir}/` "
                      f"({r.image_count} png)"]
        lines.append("")

    Path(args.text_report).write_text("\n".join(lines))
    print(f"\nReports: {args.report}, {args.text_report}", flush=True)
    print(f"Images:  {OUT_DIR}/", flush=True)
    print(f"Summary: {len(clean)} clean | {len(leaky)} leaky | "
          f"{len(failed)} build-failed", flush=True)
    sys.exit(0 if not leaky and not failed else 1)


if __name__ == "__main__":
    main()
