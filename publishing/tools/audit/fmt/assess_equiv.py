#!/usr/bin/env python3
"""
assess_equiv.py — Before/after EQUIVALENCE diff for the fmt migration.

This is the migration's value-equivalence gate. It does NOT reimplement cell
execution — it reuses the existing audit stack:

  * ``cell_exec.py``      — headless shared-namespace exec (matplotlib-safe)
  * ``audit_prose.py``    — exec cells + substitute ``{python}`` refs into prose
                            (the *composite* "value-in-prose" preview)

What it adds on top: a **before vs after diff**, keyed so a refactor can be
*proven* value-preserving on real data without a full Quarto render.

Two equivalence views (see ASSESSMENT.md §0a — the two I1 regimes):

  values : snapshot every ``*_str`` export (module + class attr) keyed by dotted
           name. Regime 1 (string-preserving: usd/percent/pp/count/qty) must be
           byte-identical here.
  prose  : snapshot every inline-ref line's rendered *composite* preview
           (value ⊕ surrounding prose). Regime 2 (fmt_multiple moves ``×`` into
           prose as ``$\\times$``) is proven here, where the string deliberately
           changes but the visible prose does not.

Ground truth remains ``audit_lego_html.py`` (refs vs real rendered HTML); this
harness is the fast pre-render proof that feeds it.

Usage
-----
  # snapshot current working tree
  PYTHONPATH=mlsysim python3 .../assess_equiv.py snapshot --qmd CH.qmd \\
      --json /tmp/fmt_assess/ch.values.json [--prose /tmp/fmt_assess/ch.prose.json]

  # snapshot a git revision of the same file (baseline), then diff
  PYTHONPATH=mlsysim python3 .../assess_equiv.py baseline --qmd CH.qmd \\
      --ref HEAD --out /tmp/fmt_assess/ch.before
  PYTHONPATH=mlsysim python3 .../assess_equiv.py snapshot --qmd CH.qmd \\
      --json /tmp/fmt_assess/ch.after.values.json --prose /tmp/fmt_assess/ch.after.prose.json
  PYTHONPATH=mlsysim python3 .../assess_equiv.py diff \\
      --before /tmp/fmt_assess/ch.before.values.json \\
      --after  /tmp/fmt_assess/ch.after.values.json

Exit: 0 identical/ok · 2 a cell failed to exec · 3 a difference was found.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# reuse the canonical exec + ref machinery (which uses cell_exec.py)
from audit_prose import (  # noqa: E402
    _exec_python_cells, _resolve_ref, INLINE_PY, CELL_START, CELL_END,
)
from visible_text import to_visible  # noqa: E402


def collect_value_exports(ns: dict) -> dict:
    """{dotted_name -> str(value)} for every *_str export (module + class attr)."""
    out: dict[str, str] = {}
    for name, val in list(ns.items()):
        if name.startswith("__"):
            continue
        if name.endswith("_str") and isinstance(val, str):
            out[name] = str(val)
        if isinstance(val, type):
            for attr, aval in vars(val).items():
                if attr.endswith("_str") and isinstance(aval, str):
                    out[f"{name}.{attr}"] = str(aval)
    return out


def collect_prose_previews(lines: list[str], ns: dict) -> dict:
    """{ '<sorted refs>' -> visible composite } for every inline-ref prose line.

    Keyed by the *set of refs* (not line number) so the diff is stable under
    line shifts. The preview substitutes each ref's rendered value into the line
    then normalizes to visible text, so a Regime-2 glyph relocation
    (string ``6×`` → ``6`` + prose ``$\\times$``) compares equal.
    """
    out: dict[str, str] = {}
    in_cell = False
    for line in lines:
        if CELL_START.match(line):
            in_cell = True
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            continue
        if in_cell:
            continue
        refs = INLINE_PY.findall(line)
        if not refs:
            continue
        preview = line
        for ref in refs:
            preview = preview.replace(f"`{{python}} {ref}`", _resolve_ref(ref, ns))
        key = "+".join(sorted(refs))
        # multiple lines can share a ref-set; keep them distinct by appending
        base = key
        i = 1
        while key in out:
            key = f"{base}#{i}"
            i += 1
        out[key] = to_visible(preview)
    return out


def snapshot_file(qmd: Path) -> tuple[dict, dict, list]:
    lines = qmd.read_text(encoding="utf-8").splitlines()
    try:
        ns = _exec_python_cells(lines)
    except RuntimeError as exc:
        return {}, {}, [str(exc)]
    values = collect_value_exports(ns)
    prose = collect_prose_previews(lines, ns)
    return values, prose, []


def cmd_snapshot(args) -> int:
    values, prose, failures = snapshot_file(Path(args.qmd))
    if failures:
        for f in failures:
            print(f"[exec] {f}", file=sys.stderr)
        return 2
    if args.show:
        for name in args.show:
            for k, v in sorted(values.items()):
                if k == name or k.endswith(name):
                    print(f"  {k} = {v!r}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(values, indent=2, sort_keys=True))
        print(f"[values] {len(values)} exports -> {args.json}")
    if args.prose:
        Path(args.prose).parent.mkdir(parents=True, exist_ok=True)
        Path(args.prose).write_text(json.dumps(prose, indent=2, sort_keys=True))
        print(f"[prose ] {len(prose)} inline-ref lines -> {args.prose}")
    if not (args.json or args.prose or args.show):
        print(f"[snapshot] {len(values)} value exports, "
              f"{len(prose)} prose lines")
    return 0


def cmd_baseline(args) -> int:
    """Materialize a git revision of --qmd, snapshot it to <out>.{values,prose}.json."""
    qmd = Path(args.qmd)
    try:
        repo_rel = subprocess.check_output(
            ["git", "ls-files", "--full-name", str(qmd)], text=True).strip()
        blob = subprocess.check_output(
            ["git", "show", f"{args.ref}:{repo_rel}"], text=True)
    except subprocess.CalledProcessError as exc:
        print(f"git error materializing {args.ref}:{qmd}: {exc}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / qmd.name
        tmp.write_text(blob, encoding="utf-8")
        values, prose, failures = snapshot_file(tmp)
    if failures:
        for f in failures:
            print(f"[exec @{args.ref}] {f}", file=sys.stderr)
        return 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.values.json").write_text(
        json.dumps(values, indent=2, sort_keys=True))
    Path(f"{args.out}.prose.json").write_text(
        json.dumps(prose, indent=2, sort_keys=True))
    print(f"[baseline @{args.ref}] {len(values)} values, {len(prose)} prose "
          f"-> {args.out}.{{values,prose}}.json")
    return 0


def _diff_maps(before: dict, after: dict, label: str) -> int:
    keys = sorted(set(before) | set(after))
    changed = [(k, before[k], after[k]) for k in keys
               if k in before and k in after and before[k] != after[k]]
    dropped = [k for k in keys if k in before and k not in after]
    added = [k for k in keys if k in after and k not in before]
    if changed:
        print(f"CHANGED {label} (must be adjudicated):")
        for k, b, a in changed:
            print(f"  {k}:\n    - {b!r}\n    + {a!r}")
    if dropped:
        print(f"DROPPED {label}: {dropped}")
    if added:
        print(f"ADDED {label}: {added}")
    if not (changed or dropped or added):
        print(f"IDENTICAL {label}: no change. ✓")
        return 0
    return 3


def cmd_diff(args) -> int:
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())
    label = "values" if "value" in Path(args.before).name else "prose"
    return _diff_maps(before, after, label)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--qmd", required=True)
    s.add_argument("--json", default=None, help="write value-export snapshot")
    s.add_argument("--prose", default=None, help="write prose-preview snapshot")
    s.add_argument("--show", nargs="*", default=None)
    s.set_defaults(fn=cmd_snapshot)
    b = sub.add_parser("baseline")
    b.add_argument("--qmd", required=True)
    b.add_argument("--ref", default="HEAD")
    b.add_argument("--out", required=True, help="prefix; writes .values.json/.prose.json")
    b.set_defaults(fn=cmd_baseline)
    d = sub.add_parser("diff")
    d.add_argument("--before", required=True)
    d.add_argument("--after", required=True)
    d.set_defaults(fn=cmd_diff)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
