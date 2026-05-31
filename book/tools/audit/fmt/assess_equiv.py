#!/usr/bin/env python3
"""
assess_equiv.py — Equivalence harness for the fmt migration (Gates G1/G4/G6).

Quarto runs a chapter's ```{python}``` cells sequentially in ONE shared kernel,
so we can reproduce the runtime by exec'ing the cells into one namespace and
snapshotting every exported ``*_str`` value (module-level OR class attribute).

This gives the migration its core guarantee: for each rewritten OUTPUT site we
can prove, on the *real data*, that the rendered string is byte-identical
before and after — or flag it for adjudication. No render required.

Modes
-----
  snapshot  : exec a chapter, dump {dotted_name -> rendered_str} as JSON.
  diff      : snapshot two versions (e.g. baseline vs working tree) and report
              every ``*_str`` whose rendered value changed.

Usage
-----
  python3 book/tools/audit/fmt/assess_equiv.py snapshot \\
      --qmd book/quarto/contents/vol1/training/training.qmd \\
      [--json /tmp/fmt_assess/training.values.json] [--show NAME ...]

  python3 book/tools/audit/fmt/assess_equiv.py diff \\
      --before /tmp/before.json --after /tmp/after.json

Exit codes: 0 = clean (or snapshot ok); 3 = diff found; 2 = a cell failed to exec.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# reuse the audited cell extractor so we parse cells identically everywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_fmt_usage import extract_python_cells  # noqa: E402


def _collect_str_exports(ns: dict) -> dict:
    """Return {dotted_name -> str(value)} for every *_str export.

    Captures module-level names AND class attributes (the book's LEGO cells
    assign exports as attributes of a per-cell class, e.g.
    ``TrainingScenarios.foo_str``), matching how prose references them
    (``{python} TrainingScenarios.foo_str``).
    """
    out: dict[str, str] = {}
    for name, val in list(ns.items()):
        if name.startswith("__"):
            continue
        if name.endswith("_str") and isinstance(val, str):
            out[name] = str(val)
        # class defined in the cell: walk its own attributes
        if isinstance(val, type):
            for attr, aval in vars(val).items():
                if attr.endswith("_str") and isinstance(aval, str):
                    out[f"{name}.{attr}"] = str(aval)
    return out


def snapshot(qmd: Path) -> tuple[dict, list]:
    """Exec every code cell of *qmd* in one namespace; return (exports, failures)."""
    text = qmd.read_text(encoding="utf-8", errors="replace")
    ns: dict = {"__name__": "__mlsys_assess__"}
    failures = []
    for fence_line, cell in extract_python_cells(text):
        if not cell.strip():
            continue
        try:
            code = compile(cell, f"{qmd}:cell@{fence_line}", "exec")
            exec(code, ns)  # noqa: S102  (trusted, first-party chapter code)
        except Exception as exc:  # noqa: BLE001
            failures.append({"fence_line": fence_line,
                             "error": f"{type(exc).__name__}: {exc}"})
    return _collect_str_exports(ns), failures


def cmd_snapshot(args) -> int:
    exports, failures = snapshot(Path(args.qmd))
    if failures:
        print(f"[exec] {len(failures)} cell(s) failed:", file=sys.stderr)
        for f in failures:
            print(f"  line {f['fence_line']}: {f['error']}", file=sys.stderr)
    if args.show:
        for name in args.show:
            hits = {k: v for k, v in exports.items() if k.endswith(name) or k == name}
            for k, v in sorted(hits.items()):
                print(f"  {k} = {v!r}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(exports, indent=2, sort_keys=True))
        print(f"[snapshot] {len(exports)} exports -> {args.json}")
    else:
        print(f"[snapshot] {len(exports)} *_str exports "
              f"({len(failures)} cell failures)")
    return 2 if failures else 0


def cmd_diff(args) -> int:
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())
    keys = sorted(set(before) | set(after))
    changed, dropped, added = [], [], []
    for k in keys:
        if k not in after:
            dropped.append(k)
        elif k not in before:
            added.append(k)
        elif before[k] != after[k]:
            changed.append((k, before[k], after[k]))
    if changed:
        print("CHANGED (rendered value differs — must be adjudicated):")
        for k, b, a in changed:
            print(f"  {k}: {b!r} -> {a!r}")
    if dropped:
        print(f"DROPPED exports: {dropped}")
    if added:
        print(f"ADDED exports: {added}")
    if not (changed or dropped or added):
        print("IDENTICAL: every *_str export renders the same. Pure refactor. ✓")
        return 0
    return 3


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--qmd", required=True)
    s.add_argument("--json", default=None)
    s.add_argument("--show", nargs="*", default=None)
    s.set_defaults(fn=cmd_snapshot)
    d = sub.add_parser("diff")
    d.add_argument("--before", required=True)
    d.add_argument("--after", required=True)
    d.set_defaults(fn=cmd_diff)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
