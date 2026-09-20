#!/usr/bin/env python3
"""Report graded cells that pass on an empty student notebook.

A region tagged ``role="scaffold"`` ships pre-solved in the student release
(``tito/core/solutions.py``), so a graded cell whose assertions never reach a
stripped region cannot fail no matter what the student does. The release gate
``tests: exceptions cannot turn missing implementations into a pass`` cannot see
this, because the implementation is not missing, it shipped.

This tool measures the hole directly. For each module it builds the real
student-tier source, executes it with no student code, and records which graded
cells still pass.

2026-09-20: first run reported that a large majority of modules award points on an
empty notebook, and that ``01_tensor`` is the only clean module. Two independent
measurements put the total at 553 and 738 of 1897 points; treat this tool's figure
as an UPPER BOUND and the shape of the table, not the exact total, as the finding.

The gap has a cause worth knowing. Several modules import their own export target
(``04_losses`` opens with ``from tinytorch.core.losses import MSELoss, ...``), and
``tinytorch/`` is generated and gitignored, so whatever it currently holds is what
those imports resolve to. Run ``tito dev export --all`` from a clean checkout
before trusting the absolute number, or the locally generated instructor solutions
will satisfy tests that a student's stripped notebook could not.

    python3 tools/check_grading_holes.py            # summary table
    python3 tools/check_grading_holes.py --verbose  # name every offending cell

Exit status is 1 when any graded points are unearnable-by-failing, so this can be
wired into a release gate once the tier assignments are settled.
"""

from __future__ import annotations

import argparse
import io
import pathlib
import re
import sys
import contextlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tito.core.solutions import (  # noqa: E402
    apply_release_tier,
    clear_solution_regions,
    solution_role,
    solution_role_action,
)

CELL_RE = re.compile(r"(?m)^# %%")
POINTS_RE = re.compile(r'"points":\s*(\d+)')
GRADE_ID_RE = re.compile(r'"grade_id":\s*"([^"]+)"')


def module_sources():
    for d in sorted((ROOT / "src").iterdir()):
        if not d.is_dir() or not re.match(r"\d\d_", d.name):
            continue
        py = d / f"{d.name}.py"
        if py.exists():
            yield d.name, py


def split_cells(text: str):
    """Yield (header, body) for each jupytext cell."""
    parts = CELL_RE.split(text)
    # parts[0] is whatever precedes the first cell marker
    for part in parts[1:]:
        header, _, body = part.partition("\n")
        yield "# %%" + header, body


def student_cells(text: str):
    """The module's cells as the student receives them."""
    out = []
    for header, body in split_cells(text):
        src, _ = apply_release_tier(body, "student")
        if "### BEGIN SOLUTION" in src:
            src, _ = clear_solution_regions(src)
        out.append((header, src))
    return out


def audit_module(name: str, py: pathlib.Path, verbose: bool):
    text = py.read_text(encoding="utf-8")
    strips = [
        line
        for line in text.splitlines()
        if "### BEGIN SOLUTION" in line
        and solution_role_action(solution_role(line), "student") == "strip"
    ]
    cells = student_cells(text)

    namespace: dict = {"__name__": "student_module"}
    total = free = 0
    offenders = []

    for header, body in cells:
        is_markdown = "[markdown]" in header
        points_match = POINTS_RE.search(header)

        if is_markdown:
            continue

        if points_match is None:
            # Definitions and given code. Execute so later tests can call them.
            with contextlib.suppress(Exception), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(body, f"{name}:setup", "exec"), namespace)
            continue

        points = int(points_match.group(1))
        total += points
        gid = GRADE_ID_RE.search(header)
        gid = gid.group(1) if gid else "?"

        # Define the test, then call it. A cell that passes here is a cell no
        # amount of missing student work can fail.
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(body, f"{name}:{gid}", "exec"), namespace)
                fn = next(
                    (
                        v
                        for k, v in namespace.items()
                        if k.startswith("test_") and callable(v) and k in body
                    ),
                    None,
                )
                if fn is not None:
                    fn()
        except NotImplementedError:
            continue  # correct: it reached stripped student code
        except Exception:
            continue  # failed for some other reason, which is still a failure
        else:
            free += points
            offenders.append(f"{gid} ({points}p)")

    if verbose and offenders:
        print(f"  {name}: {', '.join(offenders)}")
    return total, free, len(strips), offenders


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true", help="name every offending cell")
    args = ap.parse_args()

    print("Graded points that pass on an EMPTY student notebook")
    print(
        "Upper bound: modules that import their own export target resolve it to\n"
        "whatever tinytorch/ currently holds. Export from a clean checkout first.\n"
    )
    print(f"{'module':18s} {'strips':>6s} {'points':>7s} {'free':>6s} {'%':>5s}")
    print("-" * 48)

    grand_total = grand_free = 0
    clean = []
    for name, py in module_sources():
        total, free, strips, offenders = audit_module(name, py, args.verbose)
        grand_total += total
        grand_free += free
        pct = (100 * free / total) if total else 0.0
        flag = "" if free else "  clean"
        print(f"{name:18s} {strips:6d} {total:7d} {free:6d} {pct:4.0f}%{flag}")
        if not free:
            clean.append(name)

    print("-" * 48)
    pct = (100 * grand_free / grand_total) if grand_total else 0.0
    print(f"{'TOTAL':18s} {'':6s} {grand_total:7d} {grand_free:6d} {pct:4.0f}%")
    print(f"\nclean modules: {', '.join(clean) if clean else 'none'}")

    if grand_free:
        print(
            "\nEach of these awards full marks for code the student never writes.\n"
            "Fix by retagging the region role=core in src/, or by removing the\n"
            "points from a cell that only exercises pre-solved scaffold."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
