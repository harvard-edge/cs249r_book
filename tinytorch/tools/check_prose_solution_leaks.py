#!/usr/bin/env python3
"""Detect graded solution code reprinted outside its BEGIN/END SOLUTION region.

Every existing gate is marker-based: apply_release_tier, nbgrader's
ClearSolutions, _write_student_notebook and the release check all reason about
`### BEGIN SOLUTION` ... `### END SOLUTION`. None of them can see a solution
that has been copied into a markdown cell, so a student can be handed the stub
in one cell and the answer in prose a few cells later.

That happened: the 20-point `__iter__` body in src/05_dataloader was reprinted
almost verbatim under "The Shuffle Memory Trap".

This extracts the distinctive lines of every stripped (core) region and looks
for them elsewhere in the same file, outside any solution region.

Usage:  python3 tools/check_prose_solution_leaks.py [src_dir]
Exit:   0 clean, 1 if a solution line appears outside its region.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BEGIN = "### BEGIN SOLUTION"
END = "### END SOLUTION"
ROLE = re.compile(r"\brole=[\"']?([A-Za-z0-9_-]+)")

# Lines too generic to be evidence of a leak. Bare loop and branch headers are
# excluded deliberately: modules scaffold the loop skeleton in the docstring on
# purpose (see the LOOP STRUCTURE block in src/09_convolutions), eliding the
# indexing that is the actual exercise. A skeleton is teaching; a body is a leak.
BORING = re.compile(
    r"^\s*($|#|\"\"\"|'''|return$|pass$|continue$|break$|else:$|try:$|except.*:$"
    r"|for\s+.*\s+in\s+.*:$|while\s+.*:$|if\s+.*:$|with\s+.*:$|def\s+.*:$)"
)
MIN_LEN = 24          # a line shorter than this is rarely distinctive
MIN_HITS = 3          # need several matching lines before calling it a leak


def regions(lines):
    """Yield (role, [body lines]) for each solution region."""
    i = 0
    while i < len(lines):
        if BEGIN in lines[i]:
            m = ROLE.search(lines[i])
            role = m.group(1) if m else "core"
            body, i = [], i + 1
            while i < len(lines) and END not in lines[i]:
                body.append(lines[i])
                i += 1
            yield role, body
        i += 1


def normalize(line: str) -> str:
    """Strip indentation and any trailing comment.

    The leak this was written for had comments bolted onto the copied lines
    ("# Just integers!"), so exact matching missed it entirely. Compare the
    code, not the annotation.
    """
    code = re.sub(r"\s+#.*$", "", line).strip()
    return re.sub(r"\s+", " ", code)


def outside_solution(lines):
    """Return the file's lines with every solution region blanked out."""
    out, inside = [], False
    for line in lines:
        if BEGIN in line:
            inside = True
        if inside:
            out.append("")
        else:
            out.append(line)
        if END in line:
            inside = False
    return out


def main() -> int:
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "src"
    findings = []

    for path in sorted(src.glob("*/*.py")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        elsewhere = outside_solution(lines)
        joined = {normalize(l) for l in elsewhere if normalize(l)}

        for role, body in regions(lines):
            if role != "core":          # only student-core regions are stripped
                continue
            distinctive = [
                normalize(l) for l in body
                if len(normalize(l)) >= MIN_LEN and not BORING.match(l)
            ]
            hits = [l for l in distinctive if l in joined]
            if len(hits) >= MIN_HITS:
                findings.append((path.relative_to(ROOT), hits))

    if not findings:
        print("No graded solution code found reprinted outside its region.")
        return 0

    print("Graded solution code is reprinted outside its BEGIN/END SOLUTION region:\n")
    for path, hits in findings:
        print(f"  {path}")
        for line in hits[:6]:
            print(f"      {line[:88]}")
        if len(hits) > 6:
            print(f"      ... and {len(hits) - 6} more line(s)")
        print()
    print("A student receives the stub and the answer in the same notebook.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
