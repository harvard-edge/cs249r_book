"""Multiplier and ``\\times`` style checks for QMD prose.

This checker is intentionally example-heavy because most failures are visual
typography mistakes that look plausible in source.

It catches:

* ``body_multiplier_suffix``:
  ``speedup_str = fmt(speedup, suffix="×")`` or ``suffix="x"``.
  Fix by using the typed multiplier formatter, whose output owns the glyph::

      speedup_mult_str = fmt_multiple(speedup, precision=1, commas=False)
      `{python} speedup_mult_str` speedup

* ``mult_double_glyph``:
  a computed multiplier such as ``{python} speedup_mult_str`` followed by
  another ``$\\times$`` / ``×`` / ``x``. The formatter already emits the glyph.

* ``unicode_times_in_prose``:
  raw ``×`` in rendered prose or tables, e.g. ``3× faster``.
  Fix with LaTeX in rendered prose: ``3$\\times$ faster``.
  Raw ``×`` is allowed only in plain-text contexts such as ``fig-alt`` /
  ``tbl-alt`` / ``alt``, Matplotlib labels, code fences, comments/docstrings,
  and ASCII diagrams.

* ``times_product_spacing``:
  arithmetic products with tight ``$\\times$``, e.g.
  `` `{python} a_str`$\\times$`{python} b_str` ``.
  Fix products with spaces around the operator::

      `{python} a_str` $\\times$ `{python} b_str`

  Compact literal multiplier prose remains valid: ``10$\\times$ faster``.
  Computed multiplier prose uses the closed formatter:
  `` `{python} speedup_mult_str` speedup ``.

* ``fmt_sci_math_context``:
  ``fmt_sci()`` used inside math output, e.g. ``value_math =
  fmt_math(fmt_sci(x))``. ``fmt_sci()`` emits Unicode scientific notation for
  plain text. Preferred prose math uses ``sci_latex(...)`` wrapped in
  ``fmt_math(...)``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CELL_START = re.compile(r"^```\{[^}]*\}|^```python\b|^```")
CELL_END = re.compile(r"^```\s*$")
PYTHON_CELL_START = re.compile(r"^```\{python\}|^```python\b")
SUFFIX_VALUE = re.compile(r"""suffix\s*=\s*(?:"([^"]*)"|'([^']*)')""")
ASSIGN_NAME = re.compile(r"^\s*([A-Za-z_]\w*)\s*=")
FMT_SCI_CALL = re.compile(r"\bfmt_sci\s*\(")
MATH_SUFFIX_ASSIGN = re.compile(r"^\s*[A-Za-z_]\w*(?:_math|_eq)\s*=")
MARKDOWN_MATH_LITERAL = re.compile(r"\bMarkdownStr\s*\(\s*f?[\"']\$")
INLINE_PY = re.compile(r"`\{python\}\s+([A-Za-z_][\w.]*)`")
MULT_AFTER = re.compile(r"^\s*(\$\\times\$|\\times|×|x\b)")


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    code: str
    message: str
    context: str = ""
    suggestion: str = ""


def _qmd_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.qmd")))
        elif path.suffix == ".qmd" and path.exists():
            files.append(path)
    return sorted(dict.fromkeys(files))


def _attribute_intervals(line: str, names: tuple[str, ...]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for name in names:
        for match in re.finditer(rf"\b{re.escape(name)}\s*=\s*(['\"])", line):
            quote = match.group(1)
            start = match.end()
            end = start
            escaped = False
            while end < len(line):
                ch = line[end]
                if ch == "\\" and not escaped:
                    escaped = True
                    end += 1
                    continue
                if ch == quote and not escaped:
                    break
                escaped = False
                end += 1
            intervals.append((start, end))
    return intervals


def _all_positions_in_intervals(line: str, needle: str, intervals: list[tuple[int, int]]) -> bool:
    positions = [match.start() for match in re.finditer(re.escape(needle), line)]
    if not positions:
        return True
    return all(any(start <= pos < end for start, end in intervals) for pos in positions)


def _next_nonspace(line: str, start: int) -> tuple[int, str] | None:
    for idx in range(start, len(line)):
        if not line[idx].isspace():
            return idx, line[idx]
    return None


def _prev_nonspace(line: str, start: int) -> tuple[int, str] | None:
    for idx in range(start - 1, -1, -1):
        if not line[idx].isspace():
            return idx, line[idx]
    return None


def _is_right_product_operand(ch: str) -> bool:
    return ch in "`$\\(" or ch.isdigit()


def _has_mult_token(ref: str) -> bool:
    bare = ref.split(".")[-1]
    if not bare.endswith("_str"):
        return False
    return "mult" in bare[:-4].split("_")


def _audit_file(path: Path) -> list[Violation]:
    violations: list[Violation] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_fence = False
    in_python = False

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if not in_fence and CELL_START.match(line):
            in_fence = True
            in_python = bool(PYTHON_CELL_START.match(line))
            continue
        if in_fence and CELL_END.match(line):
            in_fence = False
            in_python = False
            continue

        if in_python:
            for match in SUFFIX_VALUE.finditer(line):
                value = match.group(1) if match.group(1) is not None else match.group(2)
                stripped_value = value.strip().lower()
                if stripped_value == "x" or "×" in value:
                    violations.append(
                        Violation(
                            file=str(path),
                            line=lineno,
                            code="body_multiplier_suffix",
                            message=(
                                f"Body-prose multiplier uses suffix={value!r}; "
                                "use fmt_multiple so the typed formatter owns the glyph."
                            ),
                            context=line.strip()[:160],
                            suggestion=(
                                "Replace fmt(..., suffix=...) with fmt_multiple(...), "
                                "rename the export to *_mult_str, and use the computed "
                                "ref by itself in prose."
                            ),
                        )
                    )

            if FMT_SCI_CALL.search(line) and (
                MATH_SUFFIX_ASSIGN.search(line)
                or "fmt_math(" in line
                or MARKDOWN_MATH_LITERAL.search(line)
            ):
                violations.append(
                    Violation(
                        file=str(path),
                        line=lineno,
                        code="fmt_sci_math_context",
                        message=(
                            "fmt_sci() emits Unicode scientific notation for plain-text "
                            "contexts, not prose math."
                        ),
                        context=line.strip()[:160],
                        suggestion=(
                            "For prose math, use sci_latex(...) wrapped with fmt_math(...). "
                            "Keep fmt_sci() for plain-text examples, labels, or logs."
                        ),
                    )
                )
            continue

        if in_fence:
            continue

        # LaTeX is NOT processed in these attributes, so Unicode × is the correct
        # (and only) choice there: fig-alt/tbl-alt/alt (screen-reader text) and
        # title=/page-title (callout titles, HTML <title>, PDF bookmarks). Every
        # context that DOES render (body prose, fig-cap/tbl-cap, table cells) must
        # use $\times$, so × outside these attributes is still flagged.
        noproc_intervals = _attribute_intervals(
            line, ("fig-alt", "tbl-alt", "alt", "title", "page-title")
        )
        if "×" in line and not _all_positions_in_intervals(line, "×", noproc_intervals):
            violations.append(
                Violation(
                    file=str(path),
                    line=lineno,
                    code="unicode_times_in_prose",
                    message="Raw Unicode × appears in rendered prose or table text.",
                    context=line.strip()[:160],
                    suggestion=(
                        "Use `$\\times$` in rendered prose/tables. Raw × is reserved "
                        "for plain-text contexts such as alt text, plot labels, code "
                        "fences, and ASCII diagrams."
                    ),
                )
            )

        for match in INLINE_PY.finditer(line):
            ref = match.group(1)
            if _has_mult_token(ref) and MULT_AFTER.match(line[match.end():]):
                violations.append(
                    Violation(
                        file=str(path),
                        line=lineno,
                        code="mult_double_glyph",
                        message="Computed multiplier ref is followed by another times glyph.",
                        context=line.strip()[:160],
                        suggestion=(
                            "Remove the prose times glyph. fmt_multiple/fmt_multiple_range "
                            "already emits ×; use the `{python} *_mult_str` ref alone."
                        ),
                    )
                )

        marker = "$\\times$"
        search_from = 0
        while True:
            op_start = line.find(marker, search_from)
            if op_start == -1:
                break
            op_end = op_start + len(marker)
            search_from = op_end

            # A genuine arithmetic product glues an operand IMMEDIATELY after the
            # operator (e.g. `a_str`$\times$`b_str`, or $a$$\times$$b$). A compact
            # multiplier always leaves a space after the operator ("N$\times$
            # faster", "N$\times$ (clarification)") and is valid prose, not a
            # product — so a space (or end of line) after $\times$ means skip.
            after = line[op_end] if op_end < len(line) else ""
            if after == "" or after.isspace() or not _is_right_product_operand(after):
                continue

            left = _prev_nonspace(line, op_start)
            if left is None:
                continue

            context = line[max(0, op_start - 30) : min(len(line), op_end + 30)].strip()
            violations.append(
                Violation(
                    file=str(path),
                    line=lineno,
                    code="times_product_spacing",
                    message="Arithmetic product has tight `$\\times$` without spaces.",
                    context=context,
                    suggestion=(
                        "Put spaces around product operators: `a $\\times$ b`. "
                        "Compact multiplier prose remains `N$\\times$ faster`."
                    ),
                )
            )

    return violations


def audit(paths: Iterable[Path]) -> list[Violation]:
    violations: list[Violation] = []
    for path in _qmd_files(paths):
        try:
            violations.extend(_audit_file(path))
        except (OSError, UnicodeDecodeError):
            continue
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("book/quarto/contents")])
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--by-file", action="store_true", help="Summarize by file")
    args = parser.parse_args(argv)

    violations = audit(args.paths or [Path("book/quarto/contents")])
    if args.json:
        print(json.dumps([v.__dict__ for v in violations], indent=2, ensure_ascii=False))
        return 1 if violations else 0

    if args.by_file:
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for violation in violations:
            counts[violation.file][violation.code] += 1
        for file in sorted(counts):
            total = sum(counts[file].values())
            details = ", ".join(f"{code}={count}" for code, count in sorted(counts[file].items()))
            print(f"{file}: {total} ({details})")
        print(f"Total violations: {len(violations)} across {len(counts)} files")
        return 1 if violations else 0

    for violation in violations:
        print(f"{violation.file}:{violation.line} [{violation.code}] {violation.message}")
        if violation.context:
            print(f"  source: {violation.context}")
        if violation.suggestion:
            print(f"  fix: {violation.suggestion}")
    print(f"Total violations: {len(violations)}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
