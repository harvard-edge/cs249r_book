#!/usr/bin/env python3
"""
run_percent_lane.py — move ``fmt(x, suffix='%'|' percent')`` into the guarded
``fmt_percent`` formatter with a BYTE-IDENTICAL acceptance gate (WS1 percent).

The migration is value- and prose-preserving by construction:

    cell:  X = fmt(x, …, suffix='%')        → X = fmt_percent((x)/100, …, style='symbol')
    cell:  X = fmt(x, …, suffix=' percent')  → X = fmt_percent((x)/100, …, style='prose')

``x`` is the already-scaled percentage number; dividing by 100 yields the 0-1
ratio ``fmt_percent`` expects, which renders ``x`` back with the SAME glyph
('%') or word (' percent'). The reader sees nothing change — but the value now
flows through ``fmt_percent``'s guard, so a ratio that blows past ``max_ratio``
(the "10,000%" class of bug) can no longer render silently.

Because output must be unchanged, the gate is the strictest possible: every
exported ``*_str`` AND every visible prose preview must be byte-identical
before and after. Anything that differs — a float round-trip drift, a comma
default mismatch (≥1000%), or a value above ``max_ratio`` (>150%, which raises
in fmt_percent) — fails the gate and the file is reverted untouched, then
surfaced for manual adjudication (add ``max_ratio=``/``commas=`` by hand).

Usage::

    python3 book/tools/audit/fmt/run_percent_lane.py --write <qmd> ...
    python3 book/tools/audit/fmt/run_percent_lane.py --write --all
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from assess_equiv import snapshot_file  # noqa: E402
from fmt_prose_contract import check_file  # noqa: E402
from codemod_fmt import scan_percent, _apply_cell_edits  # noqa: E402

PCT_CODES = ("percent_dup", "pp_dup")
_CELL_HEAD = re.compile(r"^```\{python\}")
_CELL_END = re.compile(r"^```\s*$")
_FMT_IMPORT = re.compile(r"^(\s*from mlsysim\.fmt import )([^\n(]+)$")


def _revert(path: Path):
    subprocess.run(["git", "checkout", "--", str(path)], check=False)


def _ensure_percent_import(text: str, edit_lines: set[int]) -> str:
    """Make fmt_percent resolvable in every cell that received a percent edit.

    Cells share one namespace (a top ``from mlsysim import *`` covers all), but
    chapters built purely from per-cell selective imports need fmt_percent added
    to the cells we touched. We append it to the cell's ``from mlsysim.fmt
    import`` list, or insert a fresh import after the cell header if there is
    none. Cells are processed bottom-to-top so inserts never shift the line
    numbers of cells still to be examined.
    """
    lines = text.split("\n")
    cells: list[tuple[int, int]] = []
    start = None
    for i, ln in enumerate(lines):
        if _CELL_HEAD.match(ln):
            start = i
        elif start is not None and _CELL_END.match(ln):
            cells.append((start, i))
            start = None

    for s, e in reversed(cells):
        if not any(s + 1 <= el <= e + 1 for el in edit_lines):
            continue
        body = lines[s:e + 1]
        if any(re.search(r"import \*", b) for b in body):
            continue  # star import already brings fmt_percent into scope
        import_idx = None
        for j in range(s, e + 1):
            m = _FMT_IMPORT.match(lines[j])
            if not m:
                continue
            names = m.group(2)
            if re.search(r"\bfmt_percent\b", names):
                import_idx = -1  # already imported here
                break
            import_idx = j
        if import_idx == -1:
            continue
        if import_idx is not None:
            lines[import_idx] = lines[import_idx].rstrip()
            sep = " fmt_percent" if lines[import_idx].endswith(",") else ", fmt_percent"
            lines[import_idx] += sep
        else:
            ins = s + 1
            while ins <= e and (lines[ins].lstrip().startswith("#|") or lines[ins].strip() == ""):
                ins += 1
            lines.insert(ins, "from mlsysim.fmt import fmt_percent")
    return "\n".join(lines)


def _identical(path: Path, edits, before_vals, before_prose) -> tuple[bool, str]:
    """Apply `edits` to the clean tree, write, and report whether the rendered
    values AND visible prose are byte-identical. Leaves the file written."""
    text = path.read_text(encoding="utf-8")
    new_text = _apply_cell_edits(text, edits)
    if new_text == text:
        return False, "no-op (stale offsets)"
    new_text = _ensure_percent_import(new_text, {e.line for e in edits})
    path.write_text(new_text, encoding="utf-8")
    av, ap, fail = snapshot_file(path)
    if fail:
        return False, f"exec: {fail[0][:90]}"
    if set(av) != set(before_vals):
        return False, "export set changed"
    vdiff = [k for k in before_vals if before_vals[k] != av.get(k)]
    if vdiff:
        return False, f"value drift: {vdiff[0]} {before_vals[vdiff[0]]!r}->{av.get(vdiff[0])!r}"
    if before_prose != ap:
        return False, "prose drift"
    return True, "byte-identical"


QUEUE_OUT = HERE / "percent_adjudication_queue.txt"


def _queue_bad(path: Path, bad: list[tuple]) -> None:
    rel = str(path).split("contents/")[-1]
    with QUEUE_OUT.open("a", encoding="utf-8") as fh:
        for e, why in bad:
            fh.write(f"{rel}\tL{e.line}\t{e.var}\t{why}\t{e.old}\n")


def process(path: Path) -> tuple[str, str]:
    """Return (verdict, detail). verdict in {PASS, PART, SKIP, FAIL, NOOP}.

    Fast path: apply all edits, accept iff byte-identical. If that fails, bisect
    to the per-edit granularity — keep every edit that is *individually*
    byte-identical, queue the rest (guard trips: negative / >max_ratio percents
    that need a human to add max_ratio= or reclassify). This means one stubborn
    site no longer blocks a chapter's other (provably safe) percent migrations.
    """
    edits, _queue = scan_percent(path)
    if not edits:
        return "NOOP", "no auto-rewritable percent sites"

    before_vals, before_prose, fail = snapshot_file(path)
    if fail:
        return "SKIP", f"chapter does not execute headlessly: {fail[0][:80]}"

    ok, why = _identical(path, edits, before_vals, before_prose)
    if ok:
        pct_viol = [v for v in check_file(path) if v.code in PCT_CODES]
        if pct_viol:
            _revert(path)
            return "FAIL", f"G-contract: {len(pct_viol)} percent dup violation(s): {pct_viol[0].ref}"
        return "PASS", f"{len(edits)} cell rewrites -> fmt_percent, output byte-identical"

    # bisect: test each edit on its own against the clean tree
    _revert(path)
    good, bad = [], []
    for e in edits:
        ind_ok, ind_why = _identical(path, [e], before_vals, before_prose)
        _revert(path)
        (good if ind_ok else bad).append(e if ind_ok else (e, ind_why))

    if not good:
        _queue_bad(path, bad)
        return "FAIL", f"all {len(edits)} sites tripped the guard ({why}); queued for adjudication"

    ok2, why2 = _identical(path, good, before_vals, before_prose)
    if not ok2:
        _revert(path)
        return "FAIL", f"combined good-set not byte-identical ({why2}); needs manual review"
    pct_viol = [v for v in check_file(path) if v.code in PCT_CODES]
    if pct_viol:
        _revert(path)
        return "FAIL", f"G-contract: {pct_viol[0].ref}"
    _queue_bad(path, bad)
    return "PART", f"{len(good)}/{len(edits)} migrated byte-identical; {len(bad)} queued (guard trips)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("qmd", nargs="*")
    ap.add_argument("--all", action="store_true", help="all chapters under contents/")
    ap.add_argument("--write", action="store_true", help="required to actually apply (else lists targets)")
    args = ap.parse_args()

    files = [Path(p) for p in args.qmd]
    if args.all:
        files = sorted(Path("book/quarto/contents").rglob("*.qmd"))
    if not files:
        ap.error("pass qmd file(s) or --all")

    if not args.write:
        print("DRY: would process (use --write):")
        for f in files:
            edits, q = scan_percent(f)
            if edits:
                print(f"  {len(edits):3d} edits  ({len(q)} queued)  {str(f).split('contents/')[-1]}")
        return 0

    results: dict[str, list[str]] = {"PASS": [], "PART": [], "SKIP": [], "FAIL": [], "NOOP": []}
    for f in files:
        if not f.is_file():
            continue
        verdict, detail = process(f)
        results[verdict].append(str(f).split("contents/")[-1])
        if verdict != "NOOP":
            print(f"[{verdict}] {str(f).split('contents/')[-1]}\n    {detail}")
    print("\n=== summary ===")
    for v in ("PASS", "PART", "SKIP", "FAIL"):
        print(f"  {v}: {len(results[v])}")
        for f in results[v]:
            print(f"      {f}")
    return 1 if results["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
