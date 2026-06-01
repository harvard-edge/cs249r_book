#!/usr/bin/env python3
"""Flag redundant unit/currency tokens after closed ``{python} *_str`` prose refs.

Policy
------
Only **closed** exports (domain formatters, ``fmt_qty``, ``fmt_percent``, or
``*_unit_str`` names fed by open ``fmt()``) carry their own unit glyph. Prose
must not repeat that unit after the ref.

**Open** exports (``fmt()``, ``fmt_int()`` on bare scalars) intentionally leave
the unit to prose — ``32 GPUs``, ``95 percent``, ``10 ms`` are valid.

Lines with ``<!-- lego-ok: ... -->`` are skipped.
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
CLASS = re.compile(r"^class\s+(\w+)", re.M)

# Closed export assignments in LEGO OUTPUT sections.
CLOSED_DOMAIN_FMT = re.compile(
    r"^\s*(?P<name>\w+_str)\s*=\s*"
    r"(?:fmt_qty|fmt_power|fmt_energy|fmt_bandwidth|fmt_memory|fmt_emissions|"
    r"fmt_latency|fmt_percent|fmt_rate|fmt_usd|fmt_eur|fmt_time|fmt_tokens|fmt_params|"
    r"fmt_flop_rate|fmt_flops|fmt_ops_rate|fmt_arithmetic_intensity|"
    r"fmt_compute_efficiency|fmt_carbon_intensity|fmt_throughput|"
    r"fmt_sci_qty)\s*\(",
    re.M | re.I,
)
CLOSED_NAME_OPEN_FMT = re.compile(
    r"^\s*(?P<name>\w+_(?:w|kw|mw|j|mj|wh|kwh|mwh|gwh|gb|tb|gib|ms|s|kg|gbps|tflop|tonnes?)_str)\s*=\s*fmt\s*\(",
    re.M | re.I,
)
FMT_SUFFIX = re.compile(
    r"(\w+_str)\s*=\s*fmt(?:_int|_percent|_unit|_val)?\([^)]*suffix\s*=\s*['\"]([^'\"]+)['\"]",
    re.M,
)

# Unit/currency tokens immediately after a closing backtick on a _str ref.
PROSE_UNIT_AFTER_REF = re.compile(
    r"`\{python\}\s+([A-Za-z_][\w.]*_str)`\s*"
    r"(ms|mW|MW|GW|kW|Wh|kWh|MWh|GWh|"
    r"GB|MB|KB|GiB|TiB|TB|"
    r"seconds?|secs?|minutes?|mins?|hours?|hrs?|weeks?|months?|years?|"
    r"percent|GPUs?|QPS|FLOPS|TFLOP/?s|PFLOP/?s|"
    r"flights?|tokens?|images?|nodes?|servers?|"
    r"USD|\$|%|×|x\b|tonnes?|metric tons?)",
    re.I,
)

# Name suffix → prose tokens that duplicate the closed export.
_SUFFIX_UNITS: dict[str, frozenset[str]] = {
    "w": frozenset({"w", "watt", "watts", "mw"}),
    "kw": frozenset({"kw", "kilowatt", "kilowatts"}),
    "mw": frozenset({"mw", "megawatt", "megawatts"}),
    "wh": frozenset({"wh", "watt-hour", "watt-hours"}),
    "kwh": frozenset({"kwh", "kilowatt-hour", "kilowatt-hours"}),
    "mwh": frozenset({"mwh", "megawatt-hour", "megawatt-hours"}),
    "gwh": frozenset({"gwh"}),
    "gb": frozenset({"gb", "gib", "gigabyte", "gigabytes"}),
    "tb": frozenset({"tb", "tib", "terabyte", "terabytes"}),
    "gib": frozenset({"gib", "gb"}),
    "ms": frozenset({"ms", "millisecond", "milliseconds"}),
    "s": frozenset({"s", "sec", "secs", "second", "seconds"}),
    "kg": frozenset({"kg", "kilogram", "kilograms"}),
    "tonnes": frozenset({"t", "tonne", "tonnes", "metric ton", "metric tons"}),
    "tonne": frozenset({"t", "tonne", "tonnes", "metric ton", "metric tons"}),
    "gbps": frozenset({"gbps", "gb/s", "gib/s"}),
    "tflop": frozenset({"tflop", "tflops", "tflop/s", "tflops/s"}),
}

_FMT_UNITS: dict[str, frozenset[str]] = {
    "fmt_percent": frozenset({"percent", "%"}),
    "fmt_usd": frozenset({"$", "usd"}),
    "fmt_eur": frozenset({"eur"}),
    "fmt_rate": frozenset({"qps", "tflop/s", "tflops/s", "gb/s", "tb/s"}),
    "fmt_flop_rate": frozenset({"flop/s", "flops/s", "tflop/s", "tflops/s", "pflop/s", "pflops/s"}),
    "fmt_flops": frozenset({"flop", "flops", "kflop", "mflop", "gflop", "tflop", "pflop"}),
    "fmt_ops_rate": frozenset({"ops/s", "tops", "tops/s"}),
    "fmt_arithmetic_intensity": frozenset({"flop/byte", "flops/byte"}),
    "fmt_compute_efficiency": frozenset({"tflop/s/w", "tflops/s/w"}),
    "fmt_carbon_intensity": frozenset({"g/kwh", "kg/kwh"}),
    "fmt_emissions": frozenset({"t", "tonne", "tonnes", "metric ton", "metric tons", "kg", "g"}),
    "fmt_latency": frozenset({"ns", "us", "µs", "ms", "s", "sec", "secs", "second", "seconds",
                               "minute", "minutes", "hour", "hours"}),
    "fmt_time": frozenset({"ns", "us", "µs", "ms", "s", "sec", "secs", "second", "seconds",
                           "minute", "minutes", "hour", "hours"}),
    "fmt_power": frozenset({"w", "kw", "mw", "gw", "watt", "watts", "kilowatt", "megawatt"}),
    "fmt_energy": frozenset({"wh", "kwh", "mwh", "gwh", "j", "kj", "mj"}),
    "fmt_memory": frozenset({"b", "kb", "mb", "gb", "tb", "gib", "tib"}),
    "fmt_qty": frozenset(),  # resolved from name suffix when present
}


def _name_suffix_units(name: str) -> frozenset[str]:
    m = re.search(
        r"_(w|kw|mw|j|mj|wh|kwh|mwh|gwh|gb|tb|gib|ms|s|kg|gbps|tflop|tonnes?|tonne)_str$",
        name,
        re.I,
    )
    if not m:
        return frozenset()
    key = m.group(1).lower()
    if key.endswith("s") and key not in _SUFFIX_UNITS and key[:-1] in _SUFFIX_UNITS:
        key = key[:-1]
    return _SUFFIX_UNITS.get(key, frozenset())


def _closed_units_from_code(code: str, cls: str) -> dict[str, frozenset[str]]:
    """Map ``Class.export_str`` → normalized unit tokens the export already carries."""
    out: dict[str, frozenset[str]] = {}

    for m in CLOSED_DOMAIN_FMT.finditer(code):
        name = m.group("name")
        fmt_name = re.search(
            r"=\s*(fmt_\w+)", m.group(0), re.I
        ).group(1).lower()
        units = set(_FMT_UNITS.get(fmt_name, frozenset()))
        units |= _name_suffix_units(name)
        out[f"{cls}.{name}"] = frozenset(u.lower() for u in units)

    for m in CLOSED_NAME_OPEN_FMT.finditer(code):
        name = m.group("name")
        out[f"{cls}.{name}"] = _name_suffix_units(name)

    for attr, suffix in FMT_SUFFIX.findall(code):
        norm = suffix.strip().lower().lstrip("~")
        tokens = {norm}
        if norm in {"%", "percent"}:
            tokens |= {"percent", "%"}
        out[f"{cls}.{attr}"] = frozenset(tokens)

    return out


def _closed_map_from_cells(lines: list[str]) -> dict[str, frozenset[str]]:
    out: dict[str, frozenset[str]] = {}
    in_cell = False
    buf: list[str] = []
    for line in lines:
        if CELL_START.match(line):
            in_cell = True
            buf = []
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            code = "\n".join(buf)
            cls_m = CLASS.search(code)
            if cls_m:
                out.update(_closed_units_from_code(code, cls_m.group(1)))
            continue
        if in_cell:
            buf.append(line)
    return out


def _normalize_unit(token: str) -> str:
    t = token.strip().lower()
    if t in {"$", "usd"}:
        return "$"
    if t in {"%", "percent"}:
        return "percent"
    if t in {"x", "×"}:
        return "x"
    if t.startswith("metric ton"):
        return "metric tons" if t.endswith("s") else "metric ton"
    return t


def _unit_duplicates_closed(token: str, closed_units: frozenset[str]) -> bool:
    if not closed_units:
        return False
    norm = _normalize_unit(token)
    for cu in closed_units:
        if norm == cu:
            return True
        if norm in cu or cu in norm:
            return True
        # ms ↔ millisecond style loose match for short tokens only
        if len(norm) <= 4 and (norm.endswith(cu) or cu.endswith(norm)):
            return True
    return False


def _line_hits(line: str, closed_map: dict[str, frozenset[str]]) -> list[str]:
    if "<!-- lego-ok" in line:
        return []
    hits: list[str] = []
    for m in PROSE_UNIT_AFTER_REF.finditer(line):
        ref = m.group(1)
        unit_token = m.group(2)
        after = line[m.end() :].lstrip()
        # LaTeX math delimiter ($\times$, $\approx$, …) — not a duplicate USD glyph.
        if unit_token == "$" and after.startswith("\\"):
            continue
        # LaTeX math close delimiter, not currency.
        if unit_token == "$" and (after.startswith(")") or after.startswith("**") or after.startswith("|")):
            continue
        closed_units = closed_map.get(ref, frozenset())
        if not _unit_duplicates_closed(unit_token, closed_units):
            continue
        hits.append(
            f"duplicate unit '{unit_token}' after closed export {ref.split('.')[-1]}"
        )
    return hits


def check_file(path: Path) -> list[tuple[int, str, list[str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    closed_map = _closed_map_from_cells(lines)
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
        if in_fence:
            continue

        hits = _line_hits(line, closed_map)
        if hits:
            issues.append((lineno, line.strip()[:120], hits))

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
            print(f"  L{lineno}: {uniq}")
            print(f"    {snippet}")
    if failures:
        print(f"\n{failures} file(s) with LEGO prose unit violations")
        return 1
    print(f"OK LEGO prose units ({total} QMD files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
