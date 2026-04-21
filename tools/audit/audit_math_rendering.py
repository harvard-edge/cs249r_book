#!/usr/bin/env python3
"""
Math-rendering audit for MLSysBook HTML output.

For each chapter, builds the HTML (via `./book/binder build html <chap>`),
then scans the rendered HTML for raw LaTeX leakage that escaped MathJax.

A "leak" is a LaTeX command or pattern that appears in user-visible prose
(outside <script>, <style>, <code>, <pre>, or <span class="math ...">
elements). These render as literal text in browsers and indicate a bug.

Usage (run from repo root):
    python3 tools/audit/audit_math_rendering.py               # all chapters
    python3 tools/audit/audit_math_rendering.py vol1/intro    # one chapter
    python3 tools/audit/audit_math_rendering.py --skip-build  # just scan
"""

from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
BUILD_DIRS = {
    "vol1": REPO / "book" / "quarto" / "_build" / "html-vol1",
    "vol2": REPO / "book" / "quarto" / "_build" / "html-vol1",
}
BINDER = REPO / "book" / "binder"

# Patterns that, if found in user-visible HTML text, indicate a rendering bug.
# Order matters for reporting; most diagnostic first.
LEAK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("backslash-cmd", re.compile(r"\\(?:times|frac|approx|alpha|beta|gamma|delta|sigma|mu|eta|lambda|theta|sqrt|sum|int|cdot|leq|geq|neq|partial|nabla|infty|log|ln|exp|min|max|pm|mp|forall|exists|in|notin|to|gets|mapsto|rightarrow|Rightarrow|prod|approx|equiv|sim|propto|land|lor|neg|text|mathbf|mathit|mathcal|mathbb)\b")),  # codespell:ignore notin
    ("caret-brace", re.compile(r"(?<![A-Za-z0-9_])\^\{[^}\n]{1,40}\}")),
    ("under-brace", re.compile(r"(?<![A-Za-z0-9_])_\{[^}\n]{1,40}\}")),
    ("bare-pow10", re.compile(r"\b10\^\{[+-]?\d{1,4}\}")),
    ("dollar-math-text", re.compile(r"\$\\[a-zA-Z]+|\\[a-zA-Z]+\$")),
]

# Patterns to strip BEFORE scanning (legitimate LaTeX zones).
STRIP_PATTERNS: list[re.Pattern] = [
    re.compile(r"<script\b[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<style\b[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<pre\b[^>]*>.*?</pre>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<code\b[^>]*>.*?</code>", re.DOTALL | re.IGNORECASE),
    re.compile(r'<span class="math[^"]*">.*?</span>', re.DOTALL),
    # MathJax/Katex containers and accessible labels
    re.compile(r'<mjx-container\b.*?</mjx-container>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<math\b.*?</math>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<annotation\b.*?</annotation>', re.DOTALL | re.IGNORECASE),
    # HTML comments and doctype
    re.compile(r"<!--.*?-->", re.DOTALL),
    re.compile(r"<!DOCTYPE[^>]*>", re.IGNORECASE),
    # Head section often holds raw LaTeX in citation metadata
    re.compile(r"<head\b.*?</head>", re.DOTALL | re.IGNORECASE),
]

TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class Leak:
    pattern: str
    match: str
    context: str
    char_offset: int


@dataclass
class ChapterReport:
    name: str
    volume: str
    qmd_path: str
    build_ok: bool
    build_seconds: float
    html_path: str | None = None
    leaks: list[Leak] = field(default_factory=list)
    error: str | None = None


def list_chapters() -> list[tuple[str, str, Path]]:
    """Return list of (name, volume, qmd_path) for all chapters in vol1+vol2."""
    chapters = []
    for vol in ("vol1", "vol2"):
        contents = REPO / "book" / "quarto" / "contents" / vol
        for qmd in sorted(contents.glob("*/*.qmd")):
            # Skip non-chapter files (parts, frontmatter sub-files starting with _)
            if qmd.name.startswith("_") and qmd.name not in (
                "_notation_body.qmd",
                "_notation_distributed.qmd",
            ):
                continue
            # Use the chapter slug (filename stem) -- binder accepts that
            chapters.append((qmd.stem, vol, qmd))
    return chapters


def build_chapter(name: str, volume: str) -> tuple[bool, float, str]:
    """Run `./book/binder build html <vol>/<name>` and return (ok, secs, output)."""
    chap = f"{volume}/{name}"
    t0 = time.time()
    proc = subprocess.run(
        [str(BINDER), "build", "html", chap],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    secs = time.time() - t0
    ok = proc.returncode == 0 and "build completed" in (proc.stdout or "").lower()
    return ok, secs, (proc.stdout or "") + (proc.stderr or "")


def find_html(name: str, volume: str, qmd_path: Path) -> Path | None:
    """Locate the rendered HTML for this chapter."""
    base = BUILD_DIRS[volume]
    rel = qmd_path.relative_to(REPO / "book" / "quarto").with_suffix(".html")
    candidate = base / rel
    if candidate.exists():
        return candidate
    # Fallback: search by filename stem
    for hit in base.rglob(f"{name}.html"):
        return hit
    return None


def scan_html(html: str) -> list[Leak]:
    cleaned = html
    for pat in STRIP_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    # After stripping the safe zones, also strip tags so we look at text only.
    text = TAG_RE.sub(" ", cleaned)

    leaks: list[Leak] = []
    for label, pat in LEAK_PATTERNS:
        for m in pat.finditer(text):
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            ctx = text[start:end].replace("\n", " ").strip()
            leaks.append(Leak(label, m.group(0), ctx, m.start()))
            if len(leaks) > 200:  # cap per-chapter
                return leaks
    return leaks


def audit_chapter(name: str, volume: str, qmd_path: Path, skip_build: bool) -> ChapterReport:
    rep = ChapterReport(
        name=name,
        volume=volume,
        qmd_path=str(qmd_path.relative_to(REPO)),
        build_ok=True,
        build_seconds=0.0,
    )
    if not skip_build:
        try:
            ok, secs, output = build_chapter(name, volume)
        except subprocess.TimeoutExpired:
            rep.build_ok = False
            rep.error = "build timed out"
            return rep
        rep.build_ok = ok
        rep.build_seconds = secs
        if not ok:
            rep.error = (output[-2000:] if output else "no output")
            return rep

    html_path = find_html(name, volume, qmd_path)
    if html_path is None:
        rep.error = "html output not found"
        return rep
    rep.html_path = str(html_path.relative_to(REPO))
    try:
        html = html_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        rep.error = f"read error: {e}"
        return rep
    rep.leaks = scan_html(html)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chapters", nargs="*", help="vol1/<chap> tokens to audit; default = all")
    ap.add_argument("--skip-build", action="store_true", help="reuse existing _build HTML")
    ap.add_argument("--report", default="audit-math-report.json")
    ap.add_argument("--text-report", default="audit-math-report.md")
    args = ap.parse_args()

    all_chapters = list_chapters()
    if args.chapters:
        wanted = set(args.chapters)
        chapters = [c for c in all_chapters if f"{c[1]}/{c[0]}" in wanted or c[0] in wanted]
    else:
        chapters = all_chapters

    print(f"Auditing {len(chapters)} chapters", flush=True)
    reports: list[ChapterReport] = []
    for i, (name, vol, qmd) in enumerate(chapters, 1):
        print(f"[{i:3d}/{len(chapters)}] {vol}/{name} ... ", end="", flush=True)
        try:
            rep = audit_chapter(name, vol, qmd, args.skip_build)
        except Exception as e:
            print(f"EXCEPTION: {e}", flush=True)
            rep = ChapterReport(
                name=name, volume=vol, qmd_path=str(qmd.relative_to(REPO)),
                build_ok=False, build_seconds=0.0, error=f"exception: {e}",
            )
        if not rep.build_ok:
            print(f"BUILD FAILED ({rep.error[:80] if rep.error else '?'})", flush=True)
        elif rep.error:
            print(f"ERROR: {rep.error[:80]}", flush=True)
        elif rep.leaks:
            counts: dict[str, int] = {}
            for L in rep.leaks:
                counts[L.pattern] = counts.get(L.pattern, 0) + 1
            print(f"{len(rep.leaks)} leak(s): {dict(counts)} ({rep.build_seconds:.0f}s)", flush=True)
        else:
            print(f"clean ({rep.build_seconds:.0f}s)", flush=True)
        reports.append(rep)

    # JSON dump
    Path(args.report).write_text(json.dumps([asdict(r) for r in reports], indent=2))

    # Human-readable report
    lines = ["# Math Rendering Audit Report", "",
             f"Total chapters audited: **{len(reports)}**", ""]
    failed = [r for r in reports if not r.build_ok]
    leaky = [r for r in reports if r.build_ok and r.leaks]
    clean = [r for r in reports if r.build_ok and not r.leaks and not r.error]
    errored = [r for r in reports if r.build_ok and r.error]

    lines += [f"- Clean: **{len(clean)}**",
              f"- Leaky: **{len(leaky)}**",
              f"- Build failures: **{len(failed)}**",
              f"- Other errors: **{len(errored)}**", ""]

    if leaky:
        lines += ["## Chapters with LaTeX leakage", ""]
        for r in leaky:
            counts: dict[str, int] = {}
            for L in r.leaks:
                counts[L.pattern] = counts.get(L.pattern, 0) + 1
            lines += [f"### `{r.volume}/{r.name}` ({len(r.leaks)} leaks: {dict(counts)})",
                      f"- qmd: `{r.qmd_path}`",
                      f"- html: `{r.html_path}`",
                      ""]
            shown: dict[str, int] = {}
            for L in r.leaks[:30]:
                if shown.get(L.pattern, 0) >= 5:
                    continue
                shown[L.pattern] = shown.get(L.pattern, 0) + 1
                lines += [f"  - **{L.pattern}** match `{L.match}`",
                          f"    > ...{L.context}..."]
            lines.append("")

    if failed:
        lines += ["## Build failures", ""]
        for r in failed:
            lines += [f"- `{r.volume}/{r.name}`: {(r.error or '?')[:200]}"]
        lines.append("")

    if errored:
        lines += ["## Other errors", ""]
        for r in errored:
            lines += [f"- `{r.volume}/{r.name}`: {(r.error or '?')[:200]}"]
        lines.append("")

    if clean:
        lines += ["## Clean chapters", ""]
        for r in clean:
            lines += [f"- `{r.volume}/{r.name}` ({r.build_seconds:.0f}s)"]
        lines.append("")

    Path(args.text_report).write_text("\n".join(lines))
    print(f"\nReports written: {args.report}, {args.text_report}", flush=True)
    print(f"Summary: {len(clean)} clean | {len(leaky)} leaky | "
          f"{len(failed)} build-failed | {len(errored)} errored", flush=True)
    sys.exit(0 if not leaky and not failed else 1)


if __name__ == "__main__":
    main()
