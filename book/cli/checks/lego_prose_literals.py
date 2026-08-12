#!/usr/bin/env python3
"""Flag hardcoded numeric literals in LEGO walkthrough prose.

Policy (computation-sensitive, not number-free prose)
-----------------------------------------------------
Flag literals when they appear in a *computational walkthrough*: a callout
notebook or worked example whose nearby LEGO cell exports ``{python} *_str``
values derived from those inputs.

Three tiers:

1. **Calc lines** — any prose line that already mixes ``{python}`` refs with
   arithmetic (`=`, `×`, `/`, `÷`, `≈`) must not also contain hand-typed
   operands or results (``70B × 2``, ``/365``, ``10×``, ``1,287,000 kWh``).

2. **Setup lines** — inside the same ``{python}`` callout, **Problem** /
   **Setup** / **Consider** / **Strategy** / **Scenario** lines, numbered
   list items, and bullet points must not hardcode scenario inputs that the
   cell uses (``100 GPUs``, ``1,000 QPS``, ``$2/GPU-hour``, ``140 GB``).

3. **Callout quantity lines** — inside any callout that has a ``{python}``
   cell, *any* prose line containing a unit-bearing quantity (``N GB``,
   ``N GPUs``, ``N MW``, ``N percent``, ``N seconds``, ``$N/hour``, etc.)
   is flagged.  This is the broadest tier: if a number participates in or
   feeds a computation anywhere in the callout, it must come from the cell.

Do *not* flag:
- Narrative teaching ranges with no nearby LEGO export (``100--1,000×``)
- Significance / Distinction / Common pitfall bullets (conventions, not math)
- Inline math dimensions (``$224×224$``), footnotes, or figure/table captions
- Table rows (pipe-delimited lines)
- Tier-3-only regions between ``<!-- lego-ok-block: ... -->`` and ``<!-- end lego-ok-block -->``
- Lines with no ``{python}`` callout context (Tiers 2–3 require a python callout)

Always flag inline LEGO suppression comments, including comments inside fenced
code cells. They should not appear in authored chapter source; convert the value
to a LEGO export, cite an mlsysIM source, or use a queued ``lego-ok-block``
exception for a Tier-3 narrative region.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTENTS = REPO_ROOT / "book" / "quarto" / "contents"

CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
FENCE_START = re.compile(r"^```")
CALLOUT_START = re.compile(r"^:::+\s+\{#")
CALLOUT_END = re.compile(r"^:::\s*$")
PY_REF = re.compile(r"\{python\}\s+[A-Za-z_][A-Za-z0-9_.]*")

WALKTHROUGH_MARK = re.compile(
    r"^\*\*(?:Step|Math|Operational|Embodied|Comparison|Problem|Solution|Result|Parameters)\b",
    re.I,
)
WALKTHROUGH_CALC = re.compile(
    r"(?:=|≈|\\approx).*\{python\}|\{python\}.*(?:=|≈|\\approx|\s/\s|\s×\s|\stimes\s|\$\\times\$|\$times\$)"
)
SCENARIO_SKIP = re.compile(
    r"^\s*\d+\.\s+\*\*(?:Significance|Distinction|Common pitfall|Resolution|Engineering lesson)\b",
    re.I,
)
SCENARIO_MARK = re.compile(
    r"^\*\*(?:Problem|Setup|Scenario|Variables|Initial Configuration|Strategy|Case|Cloud|Break-Even|Final decision)\b",
    re.I,
)
SCENARIO_OPEN = re.compile(
    r"^(?:Consider|Imagine|Suppose|Your (?:team|organization))\b",
    re.I,
)

SCI_NOTATION = re.compile(r"10\s*\^|10\^\{")
TENSOR_DIM = re.compile(r"\d+\s*[×x]\s*\d+")
CALC_OPERAND_SHORTHAND = re.compile(
    r"(?:×|\\times|\$\\times\$|÷|=|≈|\\approx|\s/\s).*\b\d+(?:\.\d+)?[KMBT]\b"
    r"|\b\d+(?:\.\d+)?[KMBT]\b.*(?:×|\\times|\$\\times\$|÷|=|≈|\\approx|\s/\s)"
)
TIKZ_LINE = re.compile(
    r"^\\(?:draw|node|pic|fill|colorlet|addplot|begin|end|tikz|scope)\b|^\s*fill=|node distance="
)
PEDAGOGICAL_RANGE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:percent|%)\s*[–-]\s*\d+(?:\.\d+)?\s*(?:percent|%)"
    r"|\d+\s+to\s+\d+\s+percent"
    r"|\d+[–-]\d+\s+percent"
)
GPU_SKU_MEMORY = re.compile(r"\b[A-Za-z]\d+(?:-\d+)?-\d+(?:\.\d+)?\s+GB\b")
LEGO_OK_BLOCK_START = re.compile(r"<!--\s*lego-ok-block:")
LEGO_OK_BLOCK_END = re.compile(r"<!--\s*end lego-ok-block\s*-->")
INLINE_LEGO_OK = re.compile(
    r"<!--\s*lego-ok\b(?!-block)|(?<![-\w])(?:#|//|/\*|%)\s*lego-ok\b|(?<!<!)--\s*lego-ok\b"
)

LITERAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\d{1,3}(?:,\d{3})+"), "comma-formatted number"),
    (
        re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:kWh|kg(?:\s*CO[₂2~]?)?|MW|MWh|Wh|GB|MB|KB|"
            r"g/kWh|metric tons|hours|seconds|GPUs?|months|years|flights|TB/s|QPS)\b",
            re.I,
        ),
        "unit-suffixed quantity literal",
    ),
    (CALC_OPERAND_SHORTHAND, "calc operand K/M/B/T shorthand"),
    (re.compile(r"\b\d+\$\times\$"), "hardcoded multiplier ($\\times$ form)"),
    (
        re.compile(r"(?:times|×|\\times)\s*\d{2,}(?:\.\d+)?"),
        "numeric multiplier (2+ digits)",
    ),
    (re.compile(r"÷\s*\d+(?:\.\d+)?[KMBT]\b"), "division by K/M/B/T rate literal"),
    (re.compile(r"/(?:365|8760)\b"), "division by calendar constant"),
    (
        re.compile(r"/\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?|\s/\s*\d{4,}(?:\.\d+)?"),
        "division by numeric literal",
    ),
)

SETUP_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b\d+\s+distinct\b"), "hardcoded distinct count"),
    (re.compile(r"\(\d+\s+GB\s+footprint\s+each\)"), "hardcoded footprint literal"),
    (re.compile(r"\$\d+(?:\.\d+)?(?:/|\s+per\b)"), "hardcoded price rate"),
    (re.compile(r"\\\$\d+(?:\.\d+)?/"), "hardcoded price rate (escaped $)"),
    (
        re.compile(
            r"\d{1,3}(?:,\d{3})+\s+(?:queries|images|transactions|GPUs?|tokens/s|samples/s)\b",
            re.I,
        ),
        "hardcoded scenario throughput/count",
    ),
    (
        re.compile(r"\b(?:Consider|Imagine|Suppose)\s+(?:a\s+)?(?:fleet|serving|cluster)\s+(?:of\s+)?\d+\b", re.I),
        "hardcoded fleet/serving size in setup",
    ),
)

CALLOUT_QUANTITY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b\d{1,3}(?:,\d{3})+\s+(?:GPUs?|nodes?|servers?|models?|workers?|machines?)\b",
            re.I,
        ),
        "comma-count + hardware/entity unit",
    ),
    (
        re.compile(
            r"\b\d{2,}\s+(?:GPUs?|nodes?|servers?)\b",
        ),
        "hardcoded hardware count",
    ),
    (
        re.compile(
            r"\b\d+(?:\.\d+)?\s+(?:GB|TB|MB|GiB|TiB)\b",
        ),
        "hardcoded storage/memory size",
    ),
    (
        re.compile(
            r"\b\d+(?:\.\d+)?\s+(?:MW|GW|kW|kWh|MWh|Wh)\b",
            re.I,
        ),
        "hardcoded power/energy quantity",
    ),
    (
        re.compile(r"\b\d{2,}\s+percent\b"),
        "hardcoded percentage",
    ),
    (
        re.compile(
            r"\b\d+(?:\.\d+)?\s+(?:seconds?|minutes?|hours?|weeks?|months?)\b",
        ),
        "hardcoded time duration",
    ),
    (
        re.compile(r"\\\$\d+(?:\.\d+)?/|\$\d+(?:\.\d+)?(?:/|\s+per\b)"),
        "hardcoded dollar rate",
    ),
    (
        re.compile(
            r"\b\d+(?:\.\d+)?\s+(?:tokens?/s|QPS|FLOPS|TFLOP/?s|PFLOP/?s|samples?/s|images?/s)\b",
            re.I,
        ),
        "hardcoded throughput rate",
    ),
    (
        re.compile(r"\d{1,3}(?:,\d{3})+"),
        "comma-formatted number in callout",
    ),
)


def _strip_allowed_fragments(line: str) -> str:
    out = line
    while TENSOR_DIM.search(out):
        out = TENSOR_DIM.sub("", out)
    out = PEDAGOGICAL_RANGE.sub("", out)
    out = GPU_SKU_MEMORY.sub("", out)
    out = SCI_NOTATION.sub("", out)
    out = PY_REF.sub("", out)
    out = re.sub(r"#(?:sec|tbl|fig|eq|lst)-[^\s`]+", "", out)
    out = re.sub(r"@[A-Za-z0-9_-]+", "", out)
    out = re.sub(r"<!--.*?-->", "", out)
    # Drop inline math blocks — often carry fixed scenario dimensions.
    out = re.sub(r"\$[^$]+\$", "", out)
    return out


def _is_excluded_prose(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped or stripped.startswith("|") or "fig-cap=" in line:
        return True
    if stripped.startswith("#") or stripped.startswith("[^fn-"):
        return True
    if stripped.startswith(": **"):
        return True
    if "fig-alt=" in line or "tbl-cap=" in line or "lst-cap=" in line:
        return True
    if "**Systems insight**" in line or "**Significance (quantitative)**" in line:
        return True
    if SCENARIO_SKIP.search(stripped):
        return True
    if TIKZ_LINE.search(stripped):
        return True
    return False


def _is_walkthrough_line(line: str, in_callout: bool) -> bool:
    stripped = line.lstrip()
    if _is_excluded_prose(line):
        return False
    if WALKTHROUGH_MARK.search(stripped):
        return True
    if not WALKTHROUGH_CALC.search(line):
        return False
    if in_callout and re.match(r"^\s*(?:[-*]|\d+\.)", line):
        return True
    return bool(re.search(r"(?:=|≈|\\approx|\s/\s)", line))


def _is_setup_line(line: str) -> bool:
    if _is_excluded_prose(line):
        return False
    stripped = line.lstrip()
    if SCENARIO_MARK.search(stripped):
        return True
    if SCENARIO_OPEN.search(stripped):
        return True
    if re.match(r"^\d+\.\s+\*\*", stripped):
        return True
    if re.match(r"^[-*]\s+\*\*", stripped):
        return True
    return False


def _literal_hits(line: str, patterns: tuple[tuple[re.Pattern[str], str], ...]) -> list[str]:
    stripped = _strip_allowed_fragments(line)
    hits: list[str] = []
    for pattern, label in patterns:
        if pattern.search(stripped):
            hits.append(label)
    return hits


def _line_violations(
    line: str,
    in_callout: bool,
    callout_has_python: bool,
    *,
    strict: bool = False,
    suppress_strict: bool = False,
) -> list[str]:
    if _is_excluded_prose(line):
        return []
    if INLINE_LEGO_OK.search(line):
        return ["inline LEGO suppression"]
    if "<!-- lego-ok" in line:
        return []

    hits: list[str] = []
    # Tier 1: calc lines mixing {python} refs with arithmetic operators.
    if "{python}" in line and _is_walkthrough_line(line, in_callout):
        hits.extend(_literal_hits(line, LITERAL_PATTERNS))

    # Tier 2: setup/problem/bullet lines in a python-backed callout.
    if (
        in_callout
        and callout_has_python
        and "{python}" not in line
        and _is_setup_line(line)
    ):
        hits.extend(_literal_hits(line, SETUP_PATTERNS))

    # Tier 3 (--strict): any prose line in a python-backed callout with
    # unit-bearing quantities.  If a number has units and lives in a
    # computational callout, it should come from the cell.
    if (
        strict
        and in_callout
        and callout_has_python
        and not suppress_strict
        and not hits
    ):
        hits.extend(_literal_hits(line, CALLOUT_QUANTITY_PATTERNS))

    return hits


def _iter_callout_blocks(lines: list[str]):
    """Yield (start_lineno, end_lineno, block_lines).

    Only prose lines are included — content inside any fenced code block
    (including ``{python}`` cells) is excluded from the block.
    """
    in_callout = False
    block_start = 0
    block: list[tuple[int, str]] = []
    in_fence = False
    in_python = False

    for idx, raw in enumerate(lines):
        lineno = idx + 1
        line = raw.rstrip()

        if in_callout:
            if CALLOUT_END.match(line):
                yield block_start, lineno, block
                in_callout = False
                in_fence = False
                in_python = False
                block = []
                continue

            if CELL_START.match(line):
                in_python = True
                continue
            if in_python:
                if CELL_END.match(line):
                    in_python = False
                continue

            if FENCE_START.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue

            block.append((lineno, line))
            continue

        if CALLOUT_START.match(line):
            in_callout = True
            in_fence = False
            in_python = False
            block_start = lineno
            block = []


def check_file(path: Path, *, strict: bool = False) -> list[tuple[int, str, list[str]]]:
    issues: list[tuple[int, str, list[str]]] = []
    lines = path.read_text(encoding="utf-8").splitlines()

    in_python = False
    in_fence = False
    for lineno, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        if INLINE_LEGO_OK.search(line):
            issues.append((lineno, line.strip()[:120], ["inline LEGO suppression"]))
            continue
        if CELL_START.match(line):
            in_python = True
            continue
        if in_python:
            if CELL_END.match(line):
                in_python = False
            continue
        if FENCE_START.match(line) and not line.startswith("```{python}"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        hits = _line_violations(line, in_callout=False, callout_has_python=False, strict=strict)
        if hits:
            issues.append((lineno, line.strip()[:120], hits))

    for _start, _end, block in _iter_callout_blocks(lines):
        callout_has_python = any("{python}" in text for _, text in block)
        suppress_strict = False
        for lineno, line in block:
            if LEGO_OK_BLOCK_START.search(line):
                suppress_strict = True
            if LEGO_OK_BLOCK_END.search(line):
                suppress_strict = False
            hits = _line_violations(
                line,
                in_callout=True,
                callout_has_python=callout_has_python,
                strict=strict,
                suppress_strict=suppress_strict,
            )
            if hits:
                issues.append((lineno, line.strip()[:120], hits))

    issues.sort(key=lambda item: item[0])
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="QMD files (default: all contents)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable Tier 3: flag all unit-bearing quantities in python-backed callouts",
    )
    args = parser.parse_args()
    if args.paths:
        expanded: list[Path] = []
        for path in args.paths:
            p = path if path.is_absolute() else REPO_ROOT / path
            if p.is_dir():
                expanded.extend(sorted(p.rglob("*.qmd")))
            elif p.suffix == ".qmd":
                expanded.append(p)
        paths = expanded
    else:
        paths = sorted(CONTENTS.rglob("*.qmd"))
    failures = 0
    total = 0
    for path in paths:
        p = path if path.is_absolute() else REPO_ROOT / path
        if not p.exists() or p.suffix != ".qmd":
            continue
        issues = check_file(p, strict=args.strict)
        total += 1
        if not issues:
            continue
        failures += 1
        print(f"\n{p.relative_to(REPO_ROOT)}")
        for lineno, snippet, labels in issues:
            uniq = ", ".join(dict.fromkeys(labels))
            print(f"  L{lineno}: {uniq}")
            print(f"    {snippet}")
    if failures:
        print(f"\n{failures} file(s) with LEGO prose literal violations")
        return 1
    print(f"OK LEGO prose literals ({total} QMD files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
