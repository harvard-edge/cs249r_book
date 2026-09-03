#!/usr/bin/env python3
"""
run_scale_style_lane.py — apply the approved no-space K/M/B/T count style.

This lane resolves the deferred scale queue after the house-style decision that
scaled counts render as ``70K`` / ``5.3M`` / ``7B``. It rewrites queued
``fmt(..., suffix=" B")`` / ``fmt(..., suffix="k")`` patterns to
``fmt_count(raw, scale=...)`` and accepts only the expected visible style
normalization: remove the space before K/M/B/T and uppercase ``k``.
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
from codemod_fmt import scan_scale_style, _apply_cell_edits  # noqa: E402
from fmt_prose_contract import check_file  # noqa: E402
from run_percent_lane import _ensure_import  # noqa: E402

QUEUE_OUT = HERE / "scale_style_adjudication_queue.txt"


def _revert(path: Path) -> None:
    subprocess.run(["git", "checkout", "--", str(path)], check=False)


def _norm_scale_text(value: str) -> str:
    value = re.sub(r"(?<=\d)\s+([KMBT])\b", r"\1", value)
    value = re.sub(r"(?<=\d)k\b", "K", value)
    return value


def _norm_values_for_targets(d: dict[str, str], targets: set[str]) -> dict[str, str]:
    return {
        k: (_norm_scale_text(v) if k in targets else v)
        for k, v in d.items()
    }


def _prose_key_touched(key: str, targets: set[str]) -> bool:
    refs = {part.split("#", 1)[0] for part in key.split("+")}
    return bool(refs & targets)


def _norm_prose_for_targets(d: dict[str, str], targets: set[str]) -> dict[str, str]:
    return {
        k: (_norm_scale_text(v) if _prose_key_touched(k, targets) else v)
        for k, v in d.items()
    }


def _queue_bad(path: Path, bad: list[tuple]) -> None:
    rel = str(path).split("contents/")[-1]
    with QUEUE_OUT.open("a", encoding="utf-8") as fh:
        for e, why in bad:
            fh.write(f"{rel}\tL{e.line}\t{e.var}\t{why}\t{e.old}\n")


def process(path: Path) -> tuple[str, str]:
    edits, qualified_vars, queue = scan_scale_style(path)
    if queue:
        with QUEUE_OUT.open("a", encoding="utf-8") as fh:
            for q in queue:
                fh.write(f"{q.file}\tL{q.line}\t{q.var}\t{q.action}\t{q.call}\n")
    if not edits:
        return "NOOP", "no approved scale-style sites"

    before_vals, before_prose, fail = snapshot_file(path)
    if fail:
        return "SKIP", f"chapter does not execute headlessly: {fail[0][:80]}"

    text = path.read_text(encoding="utf-8")
    new_text = _apply_cell_edits(text, edits)
    if new_text == text:
        return "FAIL", "no-op (stale offsets)"
    new_text = _ensure_import(new_text, {e.line for e in edits}, "fmt_count")
    path.write_text(new_text, encoding="utf-8")

    after_vals, after_prose, fail = snapshot_file(path)
    if fail:
        _revert(path)
        return "FAIL", f"exec: {fail[0][:90]}"
    if set(after_vals) != set(before_vals):
        _revert(path)
        return "FAIL", "export set changed"
    if _norm_values_for_targets(before_vals, qualified_vars) != after_vals:
        bad_keys = [
            k for k in before_vals
            if (k in qualified_vars and _norm_scale_text(before_vals[k]) != after_vals.get(k))
            or (k not in qualified_vars and before_vals[k] != after_vals.get(k))
        ]
        _revert(path)
        return "FAIL", f"unexpected value drift: {bad_keys[:3]}"
    if _norm_prose_for_targets(before_prose, qualified_vars) != after_prose:
        _revert(path)
        return "FAIL", "unexpected prose drift"
    viol = [v for v in check_file(path) if v.code in ("scale_dup",)]
    if viol:
        _revert(path)
        return "FAIL", f"G-contract: {viol[0].ref}"
    return "PASS", f"{len(edits)} scale sites -> fmt_count, no-space style verified"


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
            edits, _qualified_vars, queue = scan_scale_style(f)
            if edits or queue:
                print(f"  {len(edits):3d} edits  ({len(queue)} queued)  {str(f).split('contents/')[-1]}")
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
