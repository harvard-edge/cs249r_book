"""
Solution-region policy shared by ``tito module start`` and ``tito nbgrader``.

Source modules mark every reference implementation with nbgrader delimiters,
optionally tagged with a release role::

    ### BEGIN SOLUTION role="scaffold"
    ...
    ### END SOLUTION

A release tier decides what each role becomes (see NBGRADER_RELEASE_TIERS.md):

* ``strip``   the region is student work, so it is cleared to a stub
* ``keep``    the marker lines go and the code stays (pre-solved scaffold)
* ``remove``  the region goes entirely (instructor-only content)

Clearing follows nbgrader's ClearSolutions preprocessor, stub text included, so
the notebook ``tito module start`` hands a student has the same holes as an
nbgrader student release.

2026-09-15: ``tito module start`` handed students the full reference solutions
(#1684). Both commands now read this one policy so they cannot drift apart.
"""

import re
from typing import Dict, List, Tuple

SOLUTION_BEGIN_MARKER = "### BEGIN SOLUTION"
SOLUTION_END_MARKER = "### END SOLUTION"
SOLUTION_ROLE_RE = re.compile(r"\brole=[\"']?([A-Za-z0-9_-]+)")
# The exercise briefing MODULE_ANATOMY.md section 4 fixes to the docstring tail.
SCAFFOLD_MARKER_RE = re.compile(r"[ \t]*(TODO|APPROACH|EXAMPLE|HINTS?)\b[:\s]")
DOCSTRING_RE = re.compile(r"(?P<q>\"{3}|'{3})(?P<body>.*?)(?P=q)", re.DOTALL)
RELEASE_TIERS = ("student", "challenge", "instructor")
VALID_SOLUTION_ROLES = {"core", "scaffold", "challenge", "instructor"}

# nbgrader's ClearSolutions defaults: code_stub["python"] and text_stub.
CODE_STUB = "# YOUR CODE HERE\nraise NotImplementedError()"
TEXT_STUB = "YOUR ANSWER HERE"


def solution_role(marker_line: str) -> str:
    """Return the role named on a BEGIN SOLUTION line; unannotated means core."""
    match = SOLUTION_ROLE_RE.search(marker_line)
    return match.group(1) if match else "core"


def solution_role_action(role: str, release_tier: str) -> str:
    """Map a region's role to ``strip``, ``keep``, or ``remove`` for one tier."""
    if release_tier == "instructor":
        return "keep"
    if role == "instructor":
        return "remove"
    if release_tier == "challenge":
        return "strip" if role == "challenge" else "keep"
    return "strip" if role == "core" else "keep"


def strip_exercise_scaffold(source: str) -> str:
    """Drop the ``TODO``/``APPROACH``/``EXAMPLE``/``HINT`` tail from docstrings.

    That scaffold exists to brief a student on work they are about to do. In a
    cell whose every region is kept pre-solved, the work is already done, so the
    scaffold reads as an instruction to implement code that sits right beneath
    it. MODULE_ANATOMY.md section 4 fixes the scaffold as the tail of the
    docstring (``TODO``, ``APPROACH``, then optionally ``EXAMPLE`` and
    ``HINT``), so truncating at the first of those markers keeps the
    descriptive summary and removes only the briefing.

    2026-09-20: the student release carried 108 pre-solved cells whose
    docstrings still said "TODO: Implement ...".
    """
    def trim(match: "re.Match[str]") -> str:
        quote, body = match.group("q"), match.group("body")
        body_lines = body.split("\n")
        for i, line in enumerate(body_lines):
            if SCAFFOLD_MARKER_RE.match(line):
                kept = body_lines[:i]
                while kept and not kept[-1].strip():
                    kept.pop()
                if not any(line.strip() for line in kept):
                    # Nothing but scaffold; keep the docstring well-formed.
                    return f"{quote}{quote}"
                closing_indent = re.match(r"[ \t]*", body_lines[-1]).group(0)
                return quote + "\n".join(kept) + "\n" + closing_indent + quote
        return match.group(0)

    return DOCSTRING_RE.sub(trim, source)


def apply_release_tier(source: str, release_tier: str) -> Tuple[str, List[str]]:
    """Apply the release-role policy to one cell's source.

    ``strip`` regions keep their markers so a clearing pass (nbgrader, or
    :func:`clear_solution_regions`) can replace them. ``keep`` regions lose
    their markers but keep their code. ``remove`` regions disappear.

    When every region in the cell is kept, the exercise scaffold in its
    docstrings is removed too; see :func:`strip_exercise_scaffold`.
    """
    if not source or (SOLUTION_BEGIN_MARKER not in source and SOLUTION_END_MARKER not in source):
        return source, []

    lines = source.splitlines()
    out = []
    errors = []
    in_solution = False
    action = "strip"
    saw_strip = False
    saw_keep = False

    for line in lines:
        if SOLUTION_BEGIN_MARKER in line:
            if in_solution:
                errors.append("nested BEGIN SOLUTION marker")
            role = solution_role(line)
            if role not in VALID_SOLUTION_ROLES:
                errors.append(f"unknown solution role '{role}'")
                role = "core"
            action = solution_role_action(role, release_tier)
            in_solution = True
            if action == "strip":
                saw_strip = True
                out.append(line)
            elif action == "keep":
                saw_keep = True
            continue

        if SOLUTION_END_MARKER in line:
            if not in_solution:
                errors.append("END SOLUTION marker without matching BEGIN SOLUTION")
                out.append(line)
                continue
            if action == "strip":
                out.append(line)
            in_solution = False
            action = "strip"
            continue

        if not in_solution or action in {"strip", "keep"}:
            out.append(line)

    if in_solution:
        errors.append("BEGIN SOLUTION marker without matching END SOLUTION")

    result = "\n".join(out)
    if release_tier != "instructor" and saw_keep and not saw_strip:
        result = strip_exercise_scaffold(result)
    if source.endswith("\n"):
        result += "\n"
    return result, errors


def clear_solution_regions(source: str, cell_type: str = "code") -> Tuple[str, List[str]]:
    """Replace each BEGIN/END SOLUTION region with nbgrader's stub.

    The stub takes the indentation of the BEGIN line, so a region inside a
    method body becomes an indented ``raise NotImplementedError()`` and the
    cell still compiles.
    """
    if SOLUTION_BEGIN_MARKER not in source and SOLUTION_END_MARKER not in source:
        return source, []

    stub_lines = (CODE_STUB if cell_type == "code" else TEXT_STUB).split("\n")
    out = []
    errors = []
    in_solution = False

    for line in source.split("\n"):
        if SOLUTION_BEGIN_MARKER in line:
            if in_solution:
                errors.append("nested BEGIN SOLUTION marker")
            in_solution = True
            indent = re.match(r"\s*", line).group(0)
            out.extend(indent + stub_line for stub_line in stub_lines)
        elif SOLUTION_END_MARKER in line:
            if not in_solution:
                errors.append("END SOLUTION marker without matching BEGIN SOLUTION")
            in_solution = False
        elif not in_solution:
            out.append(line)

    if in_solution:
        errors.append("BEGIN SOLUTION marker without matching END SOLUTION")

    return "\n".join(out), errors


def make_student_notebook(notebook: Dict, release_tier: str = "student") -> List[str]:
    """Turn a reference notebook into a release-tier notebook, in place.

    Returns the marker errors found. A caller must not give the notebook to a
    student when the list is non-empty, because a malformed region can leave
    reference code behind.
    """
    if release_tier not in RELEASE_TIERS:
        return [f"Unknown TinyTorch release tier: {release_tier}"]

    errors = []
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if SOLUTION_BEGIN_MARKER not in source and SOLUTION_END_MARKER not in source:
            continue
        source, tier_errors = apply_release_tier(source, release_tier)
        source, clear_errors = clear_solution_regions(source, cell.get("cell_type", "code"))
        errors.extend(f"Cell {index}: {error}" for error in tier_errors + clear_errors)
        cell["source"] = source
    return errors


def cells_with_solution_markers(notebook: Dict) -> List[int]:
    """Return 1-based indices of cells that still contain a solution marker."""
    leftovers = []
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if SOLUTION_BEGIN_MARKER in source or SOLUTION_END_MARKER in source:
            leftovers.append(index)
    return leftovers
