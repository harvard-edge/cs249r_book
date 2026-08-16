#!/usr/bin/env python3
"""Execute every LEGO cell in the book and report every defect, not just the first.

Two guards protect the book's numbers at render time:

  * the precision guard in mlsysim.fmt, which refuses to print a value the
    requested precision would misrepresent (80 GiB shown as "86" decimal GB, or
    a fraction flattened to an integer); and
  * check(), which asserts the narrative invariants the prose states out loud
    ("expected 45 GB", "expected two nodes").

Both abort the cell on the first failure, and Quarto aborts the build on the
first bad cell. So a full render tells you about one defect per 20 minutes and
hides every later one in the same chapter behind it.

This runs the same cells with both guards softened, so one pass over both
volumes reports the complete list with exact file:line. It is the fast gate for
the class of defect that otherwise only a full render finds.

Added 2026-08-16, after a registry change to binary memory units broke both
volumes' builds. Twelve chapters failed; six of those failures were masked
behind an earlier failure in the same chapter and only became visible once the
guards were softened.

Usage:
    check_lego_health.py [--json] [path ...]
Exit status is 1 when any defect is found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]  # book/tools/audit/<file> -> repo root
sys.path.insert(0, str(HERE.parent / "fmt"))
sys.path.insert(0, str(REPO / "mlsysim"))
sys.path.insert(0, str(REPO))

import cell_exec  # noqa: E402

CELL = re.compile(r"^```\{python\}\s*$(.*?)^```\s*$", re.M | re.S)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    import importlib

    F = importlib.import_module("mlsysim.fmt")
    PKG = importlib.import_module("mlsysim")

    cur = {"file": "", "cell": 0, "offset": 0}
    precision: list[dict] = []
    narrative: list[dict] = []

    def qmd_line() -> int | None:
        fr = sys._getframe(2)
        while fr is not None:
            if fr.f_code.co_filename == "<audit-cell>":
                return cur["offset"] + fr.f_lineno
            fr = fr.f_back
        return None

    real_precision = F._check_fmt_precision

    def soft_precision(val, prec, result):
        try:
            real_precision(val, prec, result)
        except Exception as exc:  # noqa: BLE001
            precision.append(
                {
                    "file": cur["file"],
                    "line": qmd_line(),
                    "value": repr(val),
                    "precision": prec,
                    "rendered": result,
                    "message": str(exc).split("\n")[0],
                }
            )

    def soft_check(condition, message):
        if not condition:
            narrative.append(
                {"file": cur["file"], "line": qmd_line(), "message": str(message)}
            )

    F._check_fmt_precision = soft_precision
    F.check = soft_check
    if hasattr(PKG, "check"):
        PKG.check = soft_check

    roots = [Path(p) for p in args.paths] or [REPO / "book/quarto/contents"]
    files: list[Path] = []
    for r in roots:
        files.extend(sorted(r.rglob("*.qmd")) if r.is_dir() else [r])
    files = [f for f in files if "_shelved" not in str(f)]

    errors: list[dict] = []
    for qmd in files:
        try:
            rel = str(qmd.relative_to(REPO))
        except ValueError:
            rel = str(qmd)
        text = qmd.read_text(encoding="utf-8")
        ns = cell_exec.make_exec_namespace()
        for n, m in enumerate(CELL.finditer(text), 1):
            cur["file"], cur["cell"] = rel, n
            cur["offset"] = text[: m.start(1)].count("\n")
            try:
                cell_exec.exec_cell_code(m.group(1), ns)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "file": rel,
                        "cell": n,
                        "message": f"{type(exc).__name__}: {exc}".split("\n")[0][:200],
                        "where": traceback.format_exc(limit=2).splitlines()[-2:],
                    }
                )

    total = len(precision) + len(narrative) + len(errors)

    if args.json:
        print(json.dumps(
            {"precision": precision, "narrative": narrative, "errors": errors},
            indent=1,
        ))
        return 1 if total else 0

    def group(rows):
        out: dict[str, list] = {}
        for r in rows:
            out.setdefault(r["file"], []).append(r)
        return out

    if precision:
        print(f"PRECISION defects ({len(precision)}) "
              "-- a printed number the requested precision misrepresents")
        for fl, items in sorted(group(precision).items()):
            print(f"  {fl}")
            for it in items:
                print(f"    L{it['line']}: value={it['value']} precision={it['precision']}"
                      f" renders as {it['rendered']!r}")
    if narrative:
        print(f"\nNARRATIVE guard failures ({len(narrative)}) "
              "-- the math no longer supports what the prose asserts")
        for fl, items in sorted(group(narrative).items()):
            print(f"  {fl}")
            for it in items:
                print(f"    L{it['line']}: {it['message'][:140]}")
    if errors:
        print(f"\nHARD cell errors ({len(errors)})")
        for e in errors:
            print(f"  {e['file']} cell {e['cell']}: {e['message']}")

    if total == 0:
        print(f"OK: {len(files)} files, every cell executes and every guard holds")
        return 0
    print(f"\n{total} defect(s) across {len(files)} files")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
