#!/usr/bin/env python3
"""Verify prose ``A / B = C`` equations backed by ``{python}`` refs.

Finds walkthrough lines with three inline-python refs in division form::

    `{python} Class.a_str` / `{python} Class.b_str` = `{python} Class.c_str`

Execs chapter cells (shared namespace), parses numeric values from resolved
exports, and checks ``A / B ≈ C`` within tolerance.

Legacy inline LEGO suppressions are skipped by this checker,
but authored chapters should not introduce them; use source-of-truth values or a
queued ``lego-ok-block`` exception instead.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "mlsysim"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fmt"))

from cell_exec import exec_cell_code, make_exec_namespace  # noqa: E402

CONTENTS = REPO_ROOT / "book" / "quarto" / "contents"

CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
FENCE_START = re.compile(r"^```")
PY_REF = re.compile(r"`\{python\}\s+([A-Za-z_][\w.]*)`")

REL_TOL = 0.025
ABS_TOL = 0.05


def _exec_python_cells(lines: list[str]) -> dict:
    ns = make_exec_namespace()
    in_cell = False
    buf: list[str] = []
    for line in lines:
        if CELL_START.match(line):
            in_cell = True
            buf = []
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            exec_cell_code("\n".join(buf), ns)
            continue
        if in_cell:
            buf.append(line)
    return ns


def _resolve_ref(ref: str, ns: dict) -> str:
    parts = ref.split(".")
    obj = ns[parts[0]]
    for part in parts[1:]:
        obj = getattr(obj, part)
    return str(obj)


def _resolve_number(ref: str, ns: dict) -> float | None:
    """Prefer ``*_value`` attrs; fall back to parsing formatted ``*_str``."""
    parts = ref.split(".")
    try:
        obj = ns[parts[0]]
        for part in parts[1:-1]:
            obj = getattr(obj, part)
        attr = parts[-1]
        if attr.endswith("_str"):
            value_attr = attr[:-4] + "_value"
            if hasattr(obj, value_attr):
                return float(getattr(obj, value_attr))
    except (TypeError, ValueError, AttributeError, KeyError):
        pass
    return _parse_numeric(_resolve_ref(ref, ns))


def _parse_numeric(text: str) -> float | None:
    s = text.strip()
    s = re.sub(r"^~+", "", s)
    s = re.sub(r"^\$|\\?\$", "", s)
    s = s.replace(",", "")
    s = re.sub(
        r"(?i)\s*(?:ms|mw|gb|mb|kb|gib|tib|tb|kwh|mwh|wh|"
        r"seconds?|secs?|minutes?|mins?|hours?|hrs?|"
        r"percent|gpus?|qps|flights?|tokens?|images?|nodes?|usd)\s*$",
        "",
        s,
    )
    s = re.sub(r"(?i)x$", "", s)
    s = s.rstrip("%").strip()
    if not s or s.startswith("<MISSING"):
        return None
    try:
        return float(s)
    except ValueError:
        m = re.search(r"-?\d[\d,]*\.?\d*", s)
        if m:
            try:
                return float(m.group(0).replace(",", ""))
            except ValueError:
                return None
        return None


def _ratio_scale(a_ref: str, b_ref: str, c_ref: str) -> float:
    """Unit-aware scale for common LEGO patterns (e.g. mJ/ms → mW)."""
    al, bl, cl = a_ref.lower(), b_ref.lower(), c_ref.lower()
    if "mj" in al and "ms" in bl and "mw" in cl:
        return 1000.0
    if "kwh" in al and ("kg" in bl or "co" in bl) and ("kg" in cl or "emission" in cl):
        return 1.0
    return 1.0


def _find_div_equations(line: str) -> list[tuple[str, str, str]]:
    refs = list(PY_REF.finditer(line))
    if len(refs) < 3:
        return []
    out: list[tuple[str, str, str]] = []
    for i in range(len(refs) - 2):
        ma, mb, mc = refs[i], refs[i + 1], refs[i + 2]
        between = [r for r in refs if ma.end() < r.start() < mc.start()]
        if between:
            continue
        if "/" not in line[ma.end() : mb.start()]:
            continue
        if "=" not in line[mb.end() : mc.start()]:
            continue
        out.append((ma.group(1), mb.group(1), mc.group(1)))
    return out


def _approx_equal(a: float, b: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= max(ABS_TOL, REL_TOL * scale)


def check_file(path: Path) -> list[tuple[int, str, list[str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        ns = _exec_python_cells(lines)
    except Exception as exc:
        return [(0, "", [f"cell exec failed: {exc}"])]

    issues: list[tuple[int, str, list[str]]] = []
    in_python = False
    in_fence = False
    for lineno, raw in enumerate(lines, start=1):
        line = raw.rstrip()
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
        if in_fence or "<!-- lego-ok" in line:
            continue

        for a_ref, b_ref, c_ref in _find_div_equations(line):
            try:
                a_val = _resolve_number(a_ref, ns)
                b_val = _resolve_number(b_ref, ns)
                c_val = _resolve_number(c_ref, ns)
            except (KeyError, AttributeError) as exc:
                issues.append(
                    (
                        lineno,
                        line.strip()[:120],
                        [f"unresolved ref in equation: {exc}"],
                    )
                )
                continue
            if a_val is None or b_val is None or c_val is None:
                issues.append(
                    (
                        lineno,
                        line.strip()[:120],
                        [f"non-numeric equation operands ({a_ref}, {b_ref}, {c_ref})"],
                    )
                )
                continue
            if b_val == 0:
                issues.append(
                    (
                        lineno,
                        line.strip()[:120],
                        [f"division by zero in equation ({b_ref})"],
                    )
                )
                continue
            expected = (a_val / b_val) * _ratio_scale(a_ref, b_ref, c_ref)
            if not _approx_equal(expected, c_val):
                issues.append(
                    (
                        lineno,
                        line.strip()[:120],
                        [
                            f"equation mismatch: {a_ref}({a_val})/{b_ref}({b_val})"
                            f"={expected:.4g} ≠ {c_ref}({c_val})"
                        ],
                    )
                )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="QMD files (default: all contents)")
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
        issues = check_file(p)
        total += 1
        if not issues:
            continue
        failures += 1
        print(f"\n{p.relative_to(REPO_ROOT)}")
        for lineno, snippet, labels in issues:
            uniq = ", ".join(dict.fromkeys(labels))
            loc = f"L{lineno}" if lineno else "exec"
            print(f"  {loc}: {uniq}")
            if snippet:
                print(f"    {snippet}")
    if failures:
        print(f"\n{failures} file(s) with LEGO equation violations")
        return 1
    print(f"OK LEGO equations ({total} QMD files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
