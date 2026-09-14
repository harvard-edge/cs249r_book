#!/usr/bin/env python3
"""Advisory audit for stronger LEGO quantity-flow/source discipline.

This is intentionally not wired as a blocking gate yet. It finds patterns that
the current unit linter allows but the hardening plan wants to burn down:

* avoidable scalar extraction and later unit reattachment;
* ``ureg.<unit>`` use where ``mlsysim.core.units`` exports a book-facing alias;
* ``fmt_count(..., scale=..., precision=0)`` boilerplate;
* LOAD-stage numeric ``* unit`` literals that may belong in an MLSysIM registry.

The output is a work queue for coordinator/agent passes. Promote a rule to
``lint_lego_units.py`` only after the corpus is clean or honestly baselined.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTENTS = REPO_ROOT  / "books"

CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
LEGO_MARK = re.compile(
    r"#\s*[│┌].*LEGO"
    r"|#\s*│\s*(?:Context|Goal|Scope|Show|How):"
    r"|#.*\b4\.\s*OUTPUT\b"
)

LOAD_MARK = re.compile(r"#.*\bLOAD\b", re.I)
NEXT_STAGE_MARK = re.compile(r"#.*\b(?:EXECUTE|GUARD|OUTPUT)\b", re.I)

UNIT_ALIAS_FOR_UREG = {
    "byte": "byte",
    "day": "day",
    "GB": "GB",
    "gigawatt": "GW",
    "joule": "J",
    "kilojoule": "kJ",
    "kilowatt": "kilowatt",
    "kilowatt_hour": "kWh",
    "MB": "MB",
    "megawatt": "MW",
    "megawatt_hour": "MWh",
    "microjoule": "uJ",
    "microsecond": "microsecond",
    "microwatt": "uW",
    "millijoule": "mJ",
    "millisecond": "millisecond",
    "milliwatt": "milliwatt",
    "minute": "minute",
    "param": "param",
    "second": "second",
    "watt": "watt",
    "watt_hour": "Wh",
}

UNIT_NAME = (
    r"(?:byte|bit|KB|MB|GB|TB|PB|KiB|MiB|GiB|TiB|"
    r"J|kJ|mJ|uJ|pJ|joule|kilojoule|microjoule|"
    r"Wh|kWh|MWh|GWh|"
    r"watt|kilowatt|milliwatt|GW|MW|kW|mW|uW|"
    r"second|ms|microsecond|millisecond|nanosecond|hour|day|"
    r"flop|TFLOP|TFLOPs|GFLOP|GFLOPs|PFLOPs|"
    r"param|Kparam|Mparam|Bparam|Tparam|kg|kilogram|metric_ton|gram)"
)

FMT_QTY_REATTACH = re.compile(
    rf"\b(?P<func>fmt_qty|fmt_qty_int)\s*\(\s*(?P<expr>[A-Za-z_]\w*)\s*\*\s*(?P<unit>{UNIT_NAME})\b"
)
TO_MAG_UNIT_SUFFIX_ASSIGN = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*_(?:gb|tb|mb|pb|gib|tib|ms|us|ns|s|w|kw|mw|gw|j|kj|mj|kwh|mwh|gwh|kg|tons?|tonnes?|tflops?|gflops?|gbs|tbs|bps))\s*=\s*.*\.to\([^)]+\)\.magnitude\b",
    re.I,
)
TO_MAG_REATTACH_INLINE = re.compile(
    rf"\.to\([^)]+\)\.magnitude\s*\*\s*(?P<unit>{UNIT_NAME})\b"
)
UREG_ALIAS = re.compile(r"\bureg\.(?P<unit>[A-Za-z_][A-Za-z0-9_]*)\b")
FMT_COUNT_PRECISION_ZERO = re.compile(
    r"\bfmt_count\s*\([^)]*\bscale\s*=\s*['\"](?P<scale>[KMBT]|thousand|million|billion|trillion)['\"][^)]*\bprecision\s*=\s*0\b",
    re.I,
)
FORMATTER_ASSIGN = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*(?:fmt\w*|MarkdownStr)\s*\(",
)
RATE_WITHOUT_PARENS = re.compile(
    rf"=\s*\d[\d_]*(?:\.\d+)?\s*\*\s*(?P<unit>{UNIT_NAME})\s*/\s*(?:second|hour|day)\b"
)
LOAD_NUMERIC_UNIT = re.compile(
    rf"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<num>\d[\d_]*(?:\.\d+)?(?:e[+-]?\d+)?)\s*\*\s*(?P<unit>{UNIT_NAME})\b",
    re.I,
)

SCENARIO_LOCAL_HINT = re.compile(
    r"pedagogical|toy|example|assumption|illustrative|hypothetical|scenario",
    re.I,
)


@dataclass(frozen=True)
class Issue:
    """One advisory finding.

    ``rule`` is a ``QF00x`` or ``ST001`` code, ``file`` is repo-relative,
    ``line`` is the 1-based QMD line, ``snippet`` is the stripped source line,
    and ``cell`` and ``stage`` give the LEGO class name and stage.
    """

    rule: str
    file: str
    line: int
    message: str
    snippet: str
    cell: str
    stage: str = ""


def _repo_rel(path: Path) -> str:
    """Return ``path`` relative to the repo root, or as given if it lies outside."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _python_cells(path: Path) -> list[tuple[int, str, bool]]:
    """Return ``(fence_line, code, is_lego)`` for each ``{python}`` cell.

    ``fence_line`` is the 1-based QMD line of the opening fence, so code line
    ``n`` (1-based) sits at ``fence_line + n``. ``is_lego`` is True when the
    code carries a LEGO header or OUTPUT marker. An unterminated cell runs to
    end of file.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    cells: list[tuple[int, str, bool]] = []
    i = 0
    while i < len(lines):
        if CELL_START.match(lines[i]):
            start = i + 1
            buf: list[str] = []
            j = i + 1
            while j < len(lines) and not CELL_END.match(lines[j]):
                buf.append(lines[j])
                j += 1
            code = "\n".join(buf)
            cells.append((start, code, bool(LEGO_MARK.search(code))))
            i = j + 1
        else:
            i += 1
    return cells


def _class_name(code: str) -> str:
    """Return the name of the first ``class`` statement in the cell, or ``""``."""
    match = re.search(r"^\s*class\s+([A-Za-z_]\w*)", code, re.M)
    return match.group(1) if match else ""


def _stage_by_line(code: str) -> dict[int, str]:
    """Map each 1-based cell line to the current LEGO stage.

    A comment containing LOAD switches to ``LOAD``; otherwise a comment
    containing EXECUTE, GUARD, or OUTPUT (case-insensitive) switches to that
    stage. Lines before any marker map to an empty string.
    """
    stage = ""
    stages: dict[int, str] = {}
    for idx, line in enumerate(code.splitlines(), start=1):
        if LOAD_MARK.search(line):
            stage = "LOAD"
        elif NEXT_STAGE_MARK.search(line):
            marker = NEXT_STAGE_MARK.search(line)
            stage = marker.group(0).upper() if marker else stage
            if "EXECUTE" in stage:
                stage = "EXECUTE"
            elif "GUARD" in stage:
                stage = "GUARD"
            elif "OUTPUT" in stage:
                stage = "OUTPUT"
        stages[idx] = stage
    return stages


def check_file(path: Path, *, all_cells: bool = False) -> list[Issue]:
    """Scan one QMD file for quantity-flow issues, line by line.

    Only LEGO cells are scanned unless ``all_cells`` is set; blank and comment
    lines are skipped. Rules: QF001 ``fmt_qty`` reattaching a unit to a scalar;
    QF002 a unit-suffixed name assigned from ``.to(...).magnitude``; QF003 an
    inline magnitude immediately multiplied by a unit; QF004 ``ureg.<unit>``
    where an exported alias exists; QF005 ``fmt_count(scale=..., precision=0)``;
    QF006 an unparenthesized compound rate unit; QF007 ``float()`` applied to a
    variable assigned from a formatter; ST001 a LOAD-stage ``number * unit``
    literal on a line without a scenario hint word. Returns issues in source
    order.
    """
    issues: list[Issue] = []
    rel = _repo_rel(path)
    for cell_start, code, is_lego in _python_cells(path):
        if not all_cells and not is_lego:
            continue
        cls = _class_name(code)
        stages = _stage_by_line(code)
        formatted_vars: set[str] = set()
        for line in code.splitlines():
            if match := FORMATTER_ASSIGN.search(line):
                formatted_vars.add(match.group("name"))
        for offset, line in enumerate(code.splitlines(), start=1):
            lineno = cell_start + offset
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            stage = stages.get(offset, "")

            if match := FMT_QTY_REATTACH.search(line):
                issues.append(Issue(
                    "QF001",
                    rel,
                    lineno,
                    (
                        f"{match.group('func')} reattaches {match.group('unit')} "
                        f"to scalar {match.group('expr')!r}; prefer keeping a Quantity."
                    ),
                    stripped,
                    cls,
                    stage,
                ))

            if match := TO_MAG_UNIT_SUFFIX_ASSIGN.search(line):
                issues.append(Issue(
                    "QF002",
                    rel,
                    lineno,
                    (
                        f"{match.group('name')!r} stores a unit-suffixed scalar from "
                        ".to(...).magnitude; review whether the Quantity can stay attached."
                    ),
                    stripped,
                    cls,
                    stage,
                ))

            if match := TO_MAG_REATTACH_INLINE.search(line):
                issues.append(Issue(
                    "QF003",
                    rel,
                    lineno,
                    (
                        f"Inline .to(...).magnitude is immediately reattached to "
                        f"{match.group('unit')}; keep the original Quantity."
                    ),
                    stripped,
                    cls,
                    stage,
                ))

            for match in UREG_ALIAS.finditer(line):
                unit = match.group("unit")
                alias = UNIT_ALIAS_FOR_UREG.get(unit)
                if alias is None:
                    continue
                issues.append(Issue(
                    "QF004",
                    rel,
                    lineno,
                    f"Use exported alias {alias!r} instead of ureg.{unit}.",
                    stripped,
                    cls,
                    stage,
                ))

            if match := FMT_COUNT_PRECISION_ZERO.search(line):
                issues.append(Issue(
                    "QF005",
                    rel,
                    lineno,
                    (
                        f"fmt_count(scale={match.group('scale')!r}, precision=0) "
                        "usually matches the default; remove boilerplate or use a domain helper."
                    ),
                    stripped,
                    cls,
                    stage,
                ))

            for name in formatted_vars:
                if re.search(rf"\bfloat\s*\(\s*{re.escape(name)}\s*\)", line):
                    issues.append(Issue(
                        "QF007",
                        rel,
                        lineno,
                        (
                            f"Formatted string {name!r} is converted back to "
                            "float for arithmetic; keep a numeric/Quantity variable."
                        ),
                        stripped,
                        cls,
                        stage,
                    ))

            if RATE_WITHOUT_PARENS.search(line):
                issues.append(Issue(
                    "QF006",
                    rel,
                    lineno,
                    "Parenthesize compound units: value * (unit / denominator).",
                    stripped,
                    cls,
                    stage,
                ))

            if stage == "LOAD" and (match := LOAD_NUMERIC_UNIT.search(line)):
                if SCENARIO_LOCAL_HINT.search(line):
                    continue
                issues.append(Issue(
                    "ST001",
                    rel,
                    lineno,
                    (
                        f"LOAD literal {match.group('name')!r} = {match.group('num')} * "
                        f"{match.group('unit')} may need an MLSysIM registry/source home."
                    ),
                    stripped,
                    cls,
                    stage,
                ))

    return issues


def _resolve_paths(paths: list[Path]) -> list[Path]:
    """Expand CLI paths into QMD files.

    With no paths, returns every ``.qmd`` under ``books/``. Relative paths are
    resolved against the repo root, directories are searched recursively, and
    non-QMD files are dropped. Argument order is kept and duplicates are not
    removed.
    """
    if not paths:
        return sorted(CONTENTS.rglob("*.qmd"))
    out: list[Path] = []
    for path in paths:
        p = path if path.is_absolute() else REPO_ROOT / path
        if p.is_dir():
            out.extend(sorted(p.rglob("*.qmd")))
        elif p.suffix == ".qmd":
            out.append(p)
    return out


def _print_summary(issues: list[Issue]) -> None:
    """Print issue counts per rule and the 20 files with the most issues."""
    by_rule: dict[str, int] = {}
    by_file: dict[str, int] = {}
    for issue in issues:
        by_rule[issue.rule] = by_rule.get(issue.rule, 0) + 1
        by_file[issue.file] = by_file.get(issue.file, 0) + 1
    print("By rule:")
    for rule, count in sorted(by_rule.items()):
        print(f"  {rule}: {count}")
    print("Top files:")
    for file, count in sorted(by_file.items(), key=lambda item: item[1], reverse=True)[:20]:
        print(f"  {count:4d}  {file}")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Scans the given QMD files or directories (all of ``books/`` by default) and
    prints issues as text, JSON, or a grouped summary; ``--format json`` takes
    precedence over ``--summary``. Returns 1 only when ``--fail-on-findings``
    is set and issues exist, otherwise 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="QMD files or directories")
    parser.add_argument("--all-cells", action="store_true", help="Scan all Python cells, not just LEGO cells")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--summary", action="store_true", help="Print grouped counts instead of full findings")
    parser.add_argument("--fail-on-findings", action="store_true", help="Exit 1 when findings exist")
    args = parser.parse_args(argv)

    paths = _resolve_paths(args.paths)
    issues: list[Issue] = []
    for path in paths:
        if path.exists() and path.suffix == ".qmd":
            issues.extend(check_file(path, all_cells=args.all_cells))

    if args.format == "json":
        print(json.dumps([asdict(issue) for issue in issues], indent=2))
    elif args.summary:
        _print_summary(issues)
    else:
        for issue in issues:
            where = f"{issue.file}:{issue.line}"
            context = f" [{issue.cell} {issue.stage}]".rstrip()
            print(f"{where}: [{issue.rule}]{context} {issue.message}")
            print(f"    {issue.snippet}")

    return 1 if issues and args.fail_on_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
