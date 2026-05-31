#!/usr/bin/env python3
"""
run_multiplier_lane.py — apply the PROVABLE multiplier migration with an
automated, byte-level acceptance gate (the auto-verified lane of WS1).

For each chapter this driver:
  1. snapshots the current (== HEAD, clean tree) values + visible prose;
  2. applies codemod_fmt's multiplier rewrite (cell + prose $\\times$);
  3. re-snapshots;
  4. ACCEPTS only if ALL hold, else reverts the file via `git checkout`:
       G-exec   chapter still executes headlessly before and after
       G-value  every changed *_str differs ONLY by a dropped '×'
                (old == new + '×') and no export is added/removed
       G-prose  visible prose is byte-identical (Regime-2 guarantee)
       G-contract  no mult_missing_glyph / mult_literal_x violations remain

This makes the multiplier lane safe to run unattended: a change is kept only
when it is *proven* value-equivalent in the reader's eye.

Usage::

    python3 book/tools/audit/fmt/run_multiplier_lane.py --write <qmd> [<qmd> ...]
    python3 book/tools/audit/fmt/run_multiplier_lane.py --write --all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from assess_equiv import snapshot_file  # noqa: E402
from fmt_prose_contract import check_file  # noqa: E402
from codemod_fmt import scan_file, _apply_cell_edits, _patch_prose  # noqa: E402

GLYPH = "×"


def _revert(path: Path):
    subprocess.run(["git", "checkout", "--", str(path)], check=False)


def process(path: Path) -> tuple[str, str]:
    """Return (verdict, detail). verdict in {PASS, SKIP, FAIL, NOOP}."""
    edits, mult_vars, _ = scan_file(path)
    if not edits:
        return "NOOP", "no provable multiplier sites"

    before_vals, before_prose, fail = snapshot_file(path)
    if fail:
        return "SKIP", f"chapter does not execute headlessly: {fail[0][:80]}"

    text = path.read_text(encoding="utf-8")
    new_text = _apply_cell_edits(text, edits)
    new_text, prose_patches = _patch_prose(new_text, mult_vars)
    path.write_text(new_text, encoding="utf-8")

    after_vals, after_prose, fail2 = snapshot_file(path)
    if fail2:
        _revert(path)
        return "FAIL", f"G-exec: broke execution after rewrite: {fail2[0][:80]}"

    # G-value: only multiplier *_str changed, and only by losing the '×' glyph
    if set(before_vals) != set(after_vals):
        _revert(path)
        added = set(after_vals) - set(before_vals)
        removed = set(before_vals) - set(after_vals)
        return "FAIL", f"G-value: export set changed (+{len(added)} -{len(removed)})"
    bad = []
    for k, ov in before_vals.items():
        nv = after_vals[k]
        if ov == nv:
            continue
        if not (ov == nv + GLYPH):
            bad.append(f"{k}: {ov!r} -> {nv!r}")
    if bad:
        _revert(path)
        return "FAIL", "G-value: unexpected value change(s): " + "; ".join(bad[:3])

    # G-prose: visible prose byte-identical
    if before_prose != after_prose:
        diffs = [k for k in before_prose if before_prose.get(k) != after_prose.get(k)]
        _revert(path)
        return "FAIL", f"G-prose: visible prose changed at {len(diffs)} site(s): {diffs[:2]}"

    # G-contract: no multiplier contract violations remain
    mult_viol = [v for v in check_file(path) if v.code in ("mult_missing_glyph", "mult_literal_x")]
    if mult_viol:
        _revert(path)
        return "FAIL", f"G-contract: {len(mult_viol)} multiplier violation(s): {mult_viol[0].ref}"

    return "PASS", f"{len(edits)} cell + {len(prose_patches)} prose edits, all gates green"


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
            edits, _, _ = scan_file(f)
            if edits:
                print(f"  {len(edits):3d}  {str(f).split('contents/')[-1]}")
        return 0

    results: dict[str, list[str]] = {"PASS": [], "SKIP": [], "FAIL": [], "NOOP": []}
    for f in files:
        if not f.is_file():
            continue
        verdict, detail = process(f)
        results[verdict].append(str(f).split("contents/")[-1])
        if verdict != "NOOP":
            print(f"[{verdict}] {str(f).split('contents/')[-1]}\n    {detail}")
    print("\n=== summary ===")
    for v in ("PASS", "SKIP", "FAIL"):
        print(f"  {v}: {len(results[v])}")
        for f in results[v]:
            print(f"      {f}")
    return 1 if results["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
