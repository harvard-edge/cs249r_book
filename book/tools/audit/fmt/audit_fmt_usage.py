#!/usr/bin/env python3
"""
audit_fmt_usage.py — Ground-truth inventory of every fmt-family call site.

Parses the ```{python}``` code cells of every chapter, AST-walks each cell, and
classifies every call to the fmt-family helpers by the *semantic kind* of value
it formats and by any
"abuse" pattern that the typed-formatter migration intends to eliminate.

This is read-only. It writes a JSON inventory + prints a Markdown summary.

Usage:
    PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_fmt_usage.py \
        [--root book/quarto/contents] [--json /tmp/fmt_audit.json]
"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

def _discover_fmt_family() -> set[str]:
    """Return every public formatter exported by mlsysim.fmt.

    The formatter surface is intentionally growing as more LEGO domains get
    typed helpers. Keeping this audit dynamic prevents a new helper such as
    ``fmt_energy_per_flop`` from becoming invisible to the inventory.
    """
    try:
        fmt_mod = importlib.import_module("mlsysim.fmt")
    except Exception:
        return set()
    names = {
        name for name, value in vars(fmt_mod).items()
        if (name.startswith("fmt_") or name in {"fmt", "sci_latex"})
        and (inspect.isfunction(value) or inspect.isclass(value))
    }
    if hasattr(fmt_mod, "MarkdownStr"):
        names.add("MarkdownStr")
    return names


FMT_FAMILY = _discover_fmt_family() or {
    "fmt", "fmt_int", "fmt_usd", "fmt_percent", "fmt_qty", "fmt_qty_int",
    "fmt_val", "fmt_unit", "fmt_sci", "fmt_math", "fmt_frac", "sci_latex",
    "fmt_pp", "fmt_multiple", "fmt_count", "fmt_ratio", "fmt_range",
    "fmt_time", "fmt_rate", "fmt_qty_range", "fmt_time_range",
    "fmt_count_range", "fmt_usd_range", "fmt_percent_range",
    "fmt_multiple_range", "fmt_sci_qty", "MarkdownStr",
}
NUMERIC_FMT = FMT_FAMILY - {"MarkdownStr", "fmt_math", "fmt_frac"}

FENCE_RE = re.compile(r"^([ \t]*)```+\s*\{python\}\s*$")
CLOSE_RE = re.compile(r"^([ \t]*)```+\s*$")

PERCENT_SUFFIXES = {"%", " percent", "percent"}
PP_SUFFIXES = {" pp", "pp", " percentage points", " percentage point",
               "percentage points", "percentage point"}
MULTIPLIER_SUFFIXES = {"x", "×", " x", " ×"}
SERVICE_RATE_SUFFIXES = {
    "QPS", "FPS", "tokens/s", "img/s", "images/s", "req/s", "samples/s",
}
SCALE_WORDS = {"thousand", "million", "billion", "trillion", "Million"}
SCALE_GLYPHS = {"K", "M", "B", "T"}
RESOURCE_TIME_TOKENS = {
    "GPU-hours", "GPU-hr", "TPUv4-hours", "PFLOP-days", "person-years",
    "instance-seconds",
}
RATE_OR_PER_MARKERS = {
    "/day", " per day", "/photo", "/patient", "/household-year", "/kWh",
    "per W", "/MB", "/inference", "/byte", "kJ per hour", "/year",
}
TIME_COMPOUND_MARKERS = {
    "ms latency", "ms round-trip", "ms/step", "ms+", "hours/day", "s/hr",
    "μs/op",
}


def extract_python_cells(text: str):
    """Yield (start_line, source) for each ```{python}``` fenced block.

    start_line is the 1-based line number of the fence itself, so a node at
    ast lineno N maps to file line start_line + N.
    """
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        m = FENCE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        fence_line = i + 1  # 1-based
        indent = m.group(1)
        body = []
        j = i + 1
        while j < n:
            cm = CLOSE_RE.match(lines[j])
            if cm and len(cm.group(1)) <= len(indent) + 0:
                break
            # strip the fence indentation so ast sees valid code
            ln = lines[j]
            if indent and ln.startswith(indent):
                ln = ln[len(indent):]
            body.append(ln)
            j += 1
        yield fence_line, "\n".join(body)
        i = j + 1


def literal_str(node):
    """Return the python string value if node is a constant str, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def classify_suffix(suffix: str):
    s = suffix
    stripped = s.strip()
    if s in PERCENT_SUFFIXES or s.strip() in {"%", "percent"}:
        return "percent"
    if s in PP_SUFFIXES or s.strip() in {"pp", "percentage points",
                                         "percentage point"}:
        return "percentage_point"
    if s.strip() in {"x", "×"} or s.startswith("×") or s.startswith("x") and len(s.strip()) <= 6 and "×" in s:
        return "multiplier"
    if "×" in s or s.strip() == "x":
        return "multiplier"
    if s.startswith("/"):
        return "rate_denominator"
    if s.strip() in SERVICE_RATE_SUFFIXES:
        return "service_rate"
    if s.strip() in {"K", "M", "B", "T"}:
        return "scale_glyph"
    if s.strip() in {"million", "billion", "thousand", "trillion"}:
        return "scale_word"
    if (
        re.search(r"(^|[- ])(" + "|".join(SCALE_WORDS | SCALE_GLYPHS) + r")\b", stripped)
        and stripped not in SCALE_GLYPHS
    ):
        return "compound_scale"
    if any(tok in stripped for tok in RESOURCE_TIME_TOKENS) or stripped.endswith(("-hour", "-minute")):
        return "resource_time"
    if "FLOP" in stripped and "/s" not in stripped and "FLOP/byte" not in stripped:
        return "op_count"
    if any(tok in stripped for tok in TIME_COMPOUND_MARKERS):
        return "time_compound"
    if any(tok in stripped for tok in RATE_OR_PER_MARKERS):
        return "unit_rate_or_denominator"
    if s.strip() in {"GPUs", "GPU", "nodes", "tokens", "layers", "queries",
                     "images", "img", "QPS", "ops", "operations"}:
        return "count_label"
    if s.strip() in {"hours", "hour", "minutes", "minute", "seconds", "second",
                     "milliseconds", "millisecond", "microseconds", "microsecond",
                     "days", "day", "weeks", "week", "years", "year", "months",
                     "month", "h", "min", "s", "ms", "μs", "ns"}:
        return "time_unit"
    return "physical_unit"


def get_src(src_segment_fn, node):
    try:
        seg = src_segment_fn(node)
        return seg or ""
    except Exception:
        return ""


def analyze_cell(cell_src: str, fence_line: int, file_rel: str, records: list):
    try:
        tree = ast.parse(cell_src)
    except SyntaxError:
        return 0  # skip un-parseable cells (rare; inline shell etc.)

    def src_of(node):
        return ast.get_source_segment(cell_src, node) or ""

    # Map child fmt-calls that are arguments of an outer numeric fmt call,
    # to detect double-wrap abuse.
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        fname = None
        if isinstance(func, ast.Name):
            fname = func.id
        elif isinstance(func, ast.Attribute):
            fname = func.attr
        if fname not in FMT_FAMILY:
            continue
        count += 1

        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        suffix = literal_str(kwargs["suffix"]) if "suffix" in kwargs else None
        extra_suffix = (
            literal_str(kwargs["extra_suffix"])
            if "extra_suffix" in kwargs else None
        )
        prefix = literal_str(kwargs["prefix"]) if "prefix" in kwargs else None
        first_arg_src = src_of(node.args[0]) if node.args else ""

        # --- abuse detection ---
        abuses = []
        # double-wrap: a numeric fmt receiving a fmt-family call as arg
        if fname in NUMERIC_FMT:
            for a in node.args:
                for sub in ast.walk(a):
                    if isinstance(sub, ast.Call):
                        sf = sub.func
                        sn = sf.id if isinstance(sf, ast.Name) else (
                            sf.attr if isinstance(sf, ast.Attribute) else None)
                        if sn in FMT_FAMILY:
                            abuses.append("double_wrap")
                            break
        # MarkdownStr wrapping a fmt call (manual semantics assembly)
        if fname == "MarkdownStr":
            for sub in ast.walk(node):
                if sub is node:
                    continue
                if isinstance(sub, ast.Call):
                    sf = sub.func
                    sn = sf.id if isinstance(sf, ast.Name) else (
                        sf.attr if isinstance(sf, ast.Attribute) else None)
                    if sn in (FMT_FAMILY - {"MarkdownStr"}):
                        abuses.append("markdownstr_wraps_fmt")
                        break

        # semantic-in-suffix abuses
        kind = None
        if suffix is not None:
            kind = classify_suffix(suffix)
            if kind == "percent" and fname not in {"fmt_percent"}:
                abuses.append("percent_via_suffix")
            if kind == "percentage_point" and fname not in {"fmt_pp"}:
                abuses.append("pp_via_suffix")
            if kind == "multiplier":
                abuses.append("multiplier_via_suffix")
            if kind == "scale_glyph" and fname == "fmt":
                # bare fmt with K/M/B — could be currency-scale OR count;
                # currency is already handled by fmt_usd, so a K/M/B on fmt is
                # a count-scale that should be explicit.
                abuses.append("scale_glyph_on_fmt")
        if prefix is not None and prefix not in {"", "\\$", "~\\$"}:
            abuses.append("nonempty_prefix")

        # percent ratio-vs-scaled ambiguity signal
        percent_input_scaled = None
        if (kind == "percent") or fname == "fmt_percent":
            seg = first_arg_src.replace(" ", "")
            if "*100" in seg:
                percent_input_scaled = "scales_inline(*100)"
            elif "/100" in seg:
                percent_input_scaled = "divides(/100)"
            else:
                percent_input_scaled = "passes_raw_value"

        records.append({
            "file": file_rel,
            "line": fence_line + (node.lineno or 0),
            "func": fname,
            "suffix": suffix,
            "extra_suffix": extra_suffix,
            "prefix": prefix,
            "kind": kind,
            "first_arg": first_arg_src[:80],
            "percent_input": percent_input_scaled,
            "abuses": sorted(set(abuses)),
        })
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="book/quarto/contents")
    ap.add_argument("--json", default="/tmp/fmt_audit.json")
    args = ap.parse_args()

    root = Path(args.root)
    records: list = []
    files = sorted(root.rglob("*.qmd"))
    parsed_calls = 0
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        rel = str(f)
        for fence_line, cell in extract_python_cells(text):
            parsed_calls += analyze_cell(cell, fence_line, rel, records)

    # ---- aggregate ----
    by_func = Counter(r["func"] for r in records)
    by_kind = Counter(r["kind"] for r in records if r["kind"])
    by_extra_kind = Counter(
        classify_suffix(r["extra_suffix"]) for r in records
        if r.get("extra_suffix")
    )
    abuse_counter = Counter()
    for r in records:
        for a in r["abuses"]:
            abuse_counter[a] += 1
    percent_inputs = Counter(r["percent_input"] for r in records
                             if r["percent_input"])
    abuse_by_file = defaultdict(Counter)
    for r in records:
        for a in r["abuses"]:
            abuse_by_file[r["file"]][a] += 1

    Path(args.json).write_text(json.dumps(records, indent=2))

    def table(counter, title):
        print(f"\n### {title}")
        for k, v in counter.most_common():
            print(f"  {v:>6}  {k}")

    print(f"# fmt-family usage audit")
    print(f"\nFiles scanned: {len(files)}   AST-parsed fmt calls: {len(records)}")
    table(by_func, "Calls by function")
    table(by_kind, "suffix= calls by semantic kind")
    table(by_extra_kind, "extra_suffix= calls by semantic kind")
    table(abuse_counter, "Abuse patterns (targets for migration)")
    table(percent_inputs, "Percent input form (ratio vs pre-scaled)")

    print("\n### Top files by total abuse count")
    totals = sorted(abuse_by_file.items(),
                    key=lambda kv: -sum(kv[1].values()))[:15]
    for fpath, c in totals:
        short = fpath.split("contents/")[-1]
        print(f"  {sum(c.values()):>5}  {short}  {dict(c)}")

    print(f"\nFull record JSON: {args.json}")


if __name__ == "__main__":
    main()
