#!/usr/bin/env python3
"""
audit_prose_semantics.py — read the book the way a reader does and flag prose
that is wrong once the LEGO values are substituted in.

`fmt_prose_contract.py` proves the *formatter* and prose agree about who owns a
glyph (static, AST-level). This tool is complementary and operates on the
**rendered composite**: it executes each chapter, substitutes every
``{python} ref`` with its real value, normalizes LaTeX to visible Unicode
(`$\\times$`→`×`, `\\%`→`%`, ``\\$``→`$`), and then scans the resulting sentence
for things that only show up once the numbers are in place:

  * duplicated unit/glyph after a value — "26 percent percentage points",
    "50% percent", "6× times", "5 GB GB", "$$5";
  * an unresolved / empty / sentinel ref left in the text;
  * leaked markup (`{python}`, raw ``\\times``, stray ``_str``).

It is intentionally HIGH-PRECISION (few, confident patterns) so a non-empty
report means "go look here", not "stylistic nit". Read-only. Exit 1 if findings.

Usage::

    PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_prose_semantics.py --root book/quarto/contents
    PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_prose_semantics.py <chapter.qmd> [-v]
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from assess_equiv import snapshot_file  # noqa: E402

# units whose accidental repetition ("5 GB GB") is a real defect
_UNIT = (r"GB/s|TB/s|MB/s|Gb/s|GB|MB|TB|KB|PB|TFLOP/s|GFLOP/s|PFLOP/s|FLOP/s|"
         r"GHz|MHz|kHz|ms|ns|µs|us|QPS|req/s|tokens?/s|img/s|kW|MW|GW|"
         r"hours?|minutes?|seconds?|days?|years?|months?|weeks?|GPUs?|W")

# A "value" we can anchor a duplication test on: a digit run, optionally with the
# unit/glyph already attached.
_NUM = r"\d[\d,\.]*"

CHECKS: list[tuple[str, re.Pattern]] = [
    # "26 percent percentage points", "50 percent percent", "50% percent"
    ("percent_then_percent",
     re.compile(rf"{_NUM}\s*(?:percent|%)\s+(?:percent|percentage points?|%)\b", re.I)),
    # "5 percentage points percentage points"
    ("points_dup",
     re.compile(r"percentage points?\s+percentage points?", re.I)),
    # multiplier glyph then a redundant word: "6× times", "6× fold", "6× x"
    ("times_then_word",
     re.compile(r"×\s*(?:times|fold|x)\b", re.I)),
    # "6 times ×" / "6 x ×"
    ("word_then_times",
     re.compile(r"\b(?:times|fold)\s*×")),
    # doubled glyphs
    ("double_glyph", re.compile(r"%\s*%|×\s*×|‰\s*‰")),
    # doubled currency around a number: "$$5" or "$ $5"
    ("double_dollar", re.compile(rf"\$\s*\$\s*{_NUM}")),
    # same unit twice in a row after a number: "5 GB GB", "10 ms ms"
    ("double_unit",
     re.compile(rf"{_NUM}\s*({_UNIT})\s+\1\b")),
    # unit-bearing value followed by a redundant "+ unit": "600 GB/s+ GB/s"
    ("double_unit_after_plus",
     re.compile(rf"{_NUM}\s*({_UNIT})\s*\+\s*\1\b")),
    # abbreviation immediately followed by its spelled-out word (or vice-versa):
    # "7.6 PB petabytes", "350 GB gigabytes", "10 ms milliseconds"
    ("unit_abbr_plus_word",
     re.compile(
         rf"{_NUM}\s*"
         r"(?:"
         r"PB\s+petabytes?|TB\s+terabytes?|GB\s+gigabytes?|MB\s+megabytes?|"
         r"KB\s+kilobytes?|ms\s+milliseconds?|ns\s+nanoseconds?|"
         r"[µu]s\s+microseconds?|GHz\s+gigahertz|MHz\s+megahertz|"
         r"kW\s+kilowatts?|MW\s+megawatts?|GW\s+gigawatts?|W\s+watts?|"
         r"petabytes?\s+PB|terabytes?\s+TB|gigabytes?\s+GB|megabytes?\s+MB"
         r")\b", re.I)),
    # an unresolved or sentinel ref leaked into rendered prose
    ("unresolved_ref", re.compile(r"\{python\}|<MISSING:[^>]*>|\bNameError\b")),
    # a LaTeX glyph-command that belongs to PROSE (not math) leaking literally —
    # \times / \percent should have been a glyph; we exclude \frac etc. which are
    # legitimate inside the equation lines this composite also contains.
    ("leaked_glyph_cmd", re.compile(r"\\times\b|\\percent\b")),
]


# words that, after an "N×"/"N times" multiplier, assert the quantity went UP.
# "0.5× faster" / "0.3× larger" is self-contradictory: a sub-unit multiple means
# the thing got *smaller/slower*, so the comparative word is wrong.
_UP_WORDS = (r"faster|larger|bigger|greater|higher|more|longer|heavier|"
             r"wider|deeper|denser|hotter|stronger")
_MULT_DIR = re.compile(rf"(\d[\d,\.]*)\s*(?:×|x\b|times)\s+(?:as\s+\w+\s+|)({_UP_WORDS})\b", re.I)
# a currency value described as a percentage: "$5 percent", "$1,200 %"
_CCY_PCT = re.compile(r"\$\s*\d[\d,\.]*\s*(?:percent\b|%)")


def _numeric_semantic_findings(text: str):
    """Numeric-aware checks that need the substituted value, not just a pattern."""
    for m in _MULT_DIR.finditer(text):
        try:
            n = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if n < 1.0:
            yield "mult_direction", m
    m = _CCY_PCT.search(text)
    if m:
        yield "currency_as_percent", m


@dataclass
class Finding:
    code: str
    chapter: str
    snippet: str


def scan_chapter(qmd: Path) -> tuple[list[Finding], str | None]:
    values, prose, fail = snapshot_file(qmd)
    if fail:
        return [], fail[0][:120]
    rel = str(qmd).split("contents/")[-1]
    out: list[Finding] = []
    seen: set[tuple[str, str]] = set()
    def _emit(code: str, m):
        lo = max(0, m.start() - 30)
        hi = min(len(text), m.end() + 30)
        snip = ("…" if lo else "") + text[lo:hi].strip() + ("…" if hi < len(text) else "")
        sig = (code, snip)
        if sig in seen:
            return
        seen.add(sig)
        out.append(Finding(code, rel, snip))

    for _key, text in prose.items():
        for code, pat in CHECKS:
            m = pat.search(text)
            if m:
                _emit(code, m)
        for code, m in _numeric_semantic_findings(text):
            _emit(code, m)
    return out, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("qmd", nargs="*")
    ap.add_argument("--root")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    files = [Path(p) for p in args.qmd]
    if args.root:
        files += sorted(Path(args.root).rglob("*.qmd"))
    if not files:
        ap.error("pass qmd file(s) or --root")

    all_findings: list[Finding] = []
    exec_fail: list[tuple[str, str]] = []
    for f in files:
        if not f.is_file():
            continue
        finds, fail = scan_chapter(f)
        if fail:
            exec_fail.append((str(f).split("contents/")[-1], fail))
        all_findings.extend(finds)

    by_code: dict[str, int] = {}
    for fi in all_findings:
        by_code[fi.code] = by_code.get(fi.code, 0) + 1

    if args.verbose or all_findings:
        for fi in all_findings:
            print(f"[{fi.code}] {fi.chapter}\n    {fi.snippet}")
    if exec_fail:
        print("\n-- chapters that did not execute (skipped) --")
        for nm, why in exec_fail:
            print(f"   {nm}: {why}")
    print(f"\n=== {len(all_findings)} finding(s) across {len(files)} file(s): {by_code or 'CLEAN'} ===")
    return 1 if all_findings else 0


if __name__ == "__main__":
    sys.exit(main())
