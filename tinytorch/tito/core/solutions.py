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
"""

import re
from typing import List, Tuple

SOLUTION_BEGIN_MARKER = "### BEGIN SOLUTION"
SOLUTION_END_MARKER = "### END SOLUTION"
SOLUTION_ROLE_RE = re.compile(r"\brole=[\"']?([A-Za-z0-9_-]+)")
RELEASE_TIERS = ("student", "challenge", "instructor")
VALID_SOLUTION_ROLES = {"core", "scaffold", "challenge", "instructor"}


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


def apply_release_tier(source: str, release_tier: str) -> Tuple[str, List[str]]:
    """Apply the release-role policy to one cell's source.

    ``strip`` regions keep their markers so a clearing pass (nbgrader, or
    :func:`clear_solution_regions`) can replace them. ``keep`` regions lose
    their markers but keep their code. ``remove`` regions disappear.
    """
    if not source or (SOLUTION_BEGIN_MARKER not in source and SOLUTION_END_MARKER not in source):
        return source, []

    lines = source.splitlines()
    out = []
    errors = []
    in_solution = False
    action = "strip"

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
                out.append(line)
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
    if source.endswith("\n"):
        result += "\n"
    return result, errors
