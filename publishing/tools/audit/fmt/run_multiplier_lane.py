#!/usr/bin/env python3
"""
run_multiplier_lane.py — retired pre-Design-B multiplier migration helper.

This script used the old convention where ``fmt_multiple`` returned a bare
number and prose supplied ``$\\times$``. That convention is obsolete: Design B
makes ``fmt_multiple`` / ``fmt_multiple_range`` own the glyph and uses
``*_mult_str`` export names. Keep this file only as historical context for the
older typed-fmt migration lanes; do not run it against current chapters.

For each chapter this driver:
  1. snapshots the current (== HEAD, clean tree) values + visible prose;
  2. applies codemod_fmt's multiplier rewrite (cell + prose $\\times$);
  3. re-snapshots;
  4. ACCEPTS only if ALL hold, else reverts the file via `git checkout`:
       G-exec   chapter still executes headlessly before and after
       G-value  every changed *_str differs ONLY by a dropped '×'
                (old == new + '×') and no export is added/removed
       G-prose  visible prose is byte-identical (Regime-2 guarantee)
       G-contract  no obsolete multiplier-glyph violations remain

This makes the multiplier lane safe to run unattended: a change is kept only
when it is *proven* value-equivalent in the reader's eye.

Usage::

    Retired. Use ``codemod_design_b_multipliers.py`` plus
    ``fmt_prose_contract.py`` for the current Design B convention.
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
from codemod_fmt import scan_file, _apply_cell_edits, _patch_prose  # noqa: E402

GLYPH = "×"


def _revert(path: Path):
    subprocess.run(["git", "checkout", "--", str(path)], check=False)


# every glyph form a multiplier string might have carried
GLYPH_FORMS = ("×", "x", "X", " ×", "× ", " x")


def process(path: Path, variants: bool = False) -> tuple[str, str]:
    """Return (verdict, detail). verdict in {PASS, SKIP, FAIL, NOOP}.

    variants=False: byte-identical gate (glyph was in the string, moves to prose
      unchanged). variants=True: transformation-aware gate — the glyph may
      NORMALIZE (x->×, fmt_int rounding made explicit), so we prove the only
      change is: each affected value loses its glyph form, and the visible prose
      equals the original with every old value string rewritten to new+'×'.
    """
    edits, mult_vars, _ = scan_file(path, variants=variants)
    if not edits:
        return "NOOP", "no multiplier sites for this lane"

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

    if set(before_vals) != set(after_vals):
        _revert(path)
        return "FAIL", "G-value: export set changed"

    # G-value: every changed value differs ONLY by a trailing glyph form
    changed: dict[str, tuple[str, str]] = {}
    bad = []
    for k, ov in before_vals.items():
        nv = after_vals[k]
        if ov == nv:
            continue
        changed[k] = (ov, nv)
        if not any(ov == nv + g for g in GLYPH_FORMS):
            bad.append(f"{k}: {ov!r} -> {nv!r}")
    if bad:
        _revert(path)
        return "FAIL", "G-value: unexpected value change(s): " + "; ".join(bad[:3])

    # G-prose
    if variants:
        # expected after = before with each old value string rewritten to new+'×'.
        # Use numeric boundaries so a short value like '3x' does NOT match inside
        # an unrelated longer rendering like '13.3x' (substring-collision guard).
        expected = {}
        for key, bp in before_prose.items():
            s = bp
            for ov, nv in changed.values():
                if not ov:
                    continue
                s = re.sub(r"(?<![\w.])" + re.escape(ov) + r"(?![\w])", nv + "×", s)
            expected[key] = s
        mism = [k for k in expected if expected[k] != after_prose.get(k)]
        if mism:
            _revert(path)
            ex = expected[mism[0]]
            got = after_prose.get(mism[0], "")
            return "FAIL", f"G-prose-transform mismatch at {len(mism)} site(s): expected {ex[:60]!r} got {got[:60]!r}"
    else:
        if before_prose != after_prose:
            diffs = [k for k in before_prose if before_prose.get(k) != after_prose.get(k)]
            _revert(path)
            return "FAIL", f"G-prose: visible prose changed at {len(diffs)} site(s): {diffs[:2]}"

    mult_viol = [v for v in check_file(path) if v.code in ("mult_double_glyph", "mult_suffix")]
    if mult_viol:
        _revert(path)
        return "FAIL", f"G-contract: {len(mult_viol)} multiplier violation(s): {mult_viol[0].ref}"

    return "PASS", f"{len(edits)} cell + {len(prose_patches)} prose edits, all gates green"


def main() -> int:
    print(
        "run_multiplier_lane.py is retired. Design B makes fmt_multiple own "
        "$\\times$ and uses *_mult_str exports; use "
        "codemod_design_b_multipliers.py for current migrations.",
        file=sys.stderr,
    )
    return 2

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("qmd", nargs="*")
    ap.add_argument("--all", action="store_true", help="all chapters under contents/")
    ap.add_argument("--write", action="store_true", help="required to actually apply (else lists targets)")
    ap.add_argument("--variants", action="store_true",
                    help="also convert literal-'x'/' ×'/fmt_int multipliers (transformation-gated)")
    args = ap.parse_args()

    files = [Path(p) for p in args.qmd]
    if args.all:
        files = sorted(Path("book/quarto/contents").rglob("*.qmd"))
    if not files:
        ap.error("pass qmd file(s) or --all")

    if not args.write:
        print("DRY: would process (use --write):")
        for f in files:
            edits, _, _ = scan_file(f, variants=args.variants)
            if edits:
                print(f"  {len(edits):3d}  {str(f).split('contents/')[-1]}")
        return 0

    results: dict[str, list[str]] = {"PASS": [], "SKIP": [], "FAIL": [], "NOOP": []}
    for f in files:
        if not f.is_file():
            continue
        verdict, detail = process(f, variants=args.variants)
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
