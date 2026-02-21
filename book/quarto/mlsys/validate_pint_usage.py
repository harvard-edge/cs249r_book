#!/usr/bin/env python3
"""
validate_pint_usage.py
Static analysis for Pint anti-patterns in QMD files.

Scans Python code blocks in .qmd files for patterns that may indicate
unsafe unit handling: bare .magnitude access, hasattr duck-typing, and
.to(unit).magnitude chains that should modernize to .m_as(unit).

Severity levels:
  ERROR   — definite anti-pattern that must be fixed before merge
  WARN    — likely problem; review needed
  INFO    — style improvement opportunity (.to().magnitude → .m_as())

Usage:
    python3 book/quarto/mlsys/validate_pint_usage.py
    python3 book/quarto/mlsys/validate_pint_usage.py --vol1
    python3 book/quarto/mlsys/validate_pint_usage.py --vol2
    python3 book/quarto/mlsys/validate_pint_usage.py --strict
    python3 book/quarto/mlsys/validate_pint_usage.py --file path/to/file.qmd
    python3 book/quarto/mlsys/validate_pint_usage.py --no-info
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

SEVERITY_ERROR = 3
SEVERITY_WARNING = 2
SEVERITY_INFO = 1

SEVERITY_LABELS = {
    SEVERITY_ERROR: "ERROR",
    SEVERITY_WARNING: "WARN ",
    SEVERITY_INFO: "INFO ",
}


class Finding:
    __slots__ = ("severity", "filepath", "line_number", "line_content", "check_name", "message")

    def __init__(self, severity, filepath, line_number, line_content, check_name, message):
        self.severity = severity
        self.filepath = filepath
        self.line_number = line_number
        self.line_content = line_content
        self.check_name = check_name
        self.message = message


# ── Check Definitions ─────────────────────────────────────────────────
#
# Each check is a dict with:
#   name        — short identifier for the check
#   severity    — ERROR / WARN / INFO
#   pattern     — compiled regex; fires when this matches a line
#   safe_pattern (optional) — if this ALSO matches, skip this check
#   description — one-line explanation for humans
#
CHECKS = [
    # ── ERROR ──────────────────────────────────────────────────────────
    {
        "name": "hasattr_magnitude",
        "severity": SEVERITY_ERROR,
        "pattern": re.compile(r"\bhasattr\s*\(.*['\"]magnitude['\"]"),
        "description": (
            "Use isinstance(val, ureg.Quantity) instead of hasattr(val, 'magnitude'). "
            "Duck-typing skips dimension checking entirely."
        ),
    },

    # ── WARNING ────────────────────────────────────────────────────────
    {
        "name": "bare_magnitude",
        "severity": SEVERITY_WARNING,
        # .magnitude not followed by ( — avoids false positive on .magnitude > 0 in __post_init__
        "pattern": re.compile(r"\.magnitude\b"),
        # Safe if the same line also has .to(...).magnitude or .m_as( — explicit unit conversion
        "safe_pattern": re.compile(r"\.to\s*\([^)]+\)\.magnitude|\.m_as\s*\("),
        "description": (
            "Bare .magnitude access without explicit unit — use .m_as(unit) to make the "
            "target unit explicit and enable dimensional safety checking."
        ),
    },

    # ── INFO ───────────────────────────────────────────────────────────
    {
        "name": "to_dot_magnitude",
        "severity": SEVERITY_INFO,
        "pattern": re.compile(r"\.to\s*\([^)]+\)\.magnitude"),
        "description": (
            "Style: .to(unit).magnitude → .m_as(unit) "
            "(single-step extraction; clearer and more concise)."
        ),
    },
]


# ── QMD Parsing ───────────────────────────────────────────────────────

def extract_python_blocks(qmd_content: str) -> List[Tuple[int, str]]:
    """
    Extract (start_line_number, block_content) tuples from QMD Python code blocks.

    Recognises both:
      ```{python}          — regular code block
      ```{python} ...opts  — code block with inline options

    Line numbers are 1-indexed relative to the whole QMD file.
    The returned start_line_number points at the ```{python} line itself;
    block content starts one line below.
    """
    blocks = []
    lines = qmd_content.splitlines()
    in_block = False
    block_start = 0
    block_lines: List[str] = []

    for i, line in enumerate(lines, start=1):
        if not in_block:
            if re.match(r"^```\{python\b", line):
                in_block = True
                block_start = i
                block_lines = []
        else:
            if line.strip() == "```":
                blocks.append((block_start, "\n".join(block_lines)))
                in_block = False
                block_lines = []
            else:
                block_lines.append(line)

    return blocks


# ── Per-Line Checker ──────────────────────────────────────────────────

def check_line(line: str, line_num: int, filepath: str) -> List[Finding]:
    """Run all checks on a single line of Python code inside a QMD cell."""
    findings = []
    stripped = line.strip()

    # Skip blank lines and full-line comments
    if not stripped or stripped.startswith("#"):
        return findings

    for chk in CHECKS:
        if not chk["pattern"].search(line):
            continue

        # If a safe_pattern is defined and also matches, this occurrence is acceptable
        safe = chk.get("safe_pattern")
        if safe and safe.search(line):
            continue

        findings.append(Finding(
            severity=chk["severity"],
            filepath=filepath,
            line_number=line_num,
            line_content=line.rstrip(),
            check_name=chk["name"],
            message=chk["description"],
        ))

    return findings


# ── File Scanner ──────────────────────────────────────────────────────

def scan_file(filepath: Path) -> List[Finding]:
    """Scan a single QMD file; return all findings across all Python blocks."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except OSError as exc:
        return [Finding(
            severity=SEVERITY_ERROR,
            filepath=str(filepath),
            line_number=0,
            line_content="",
            check_name="read_error",
            message=f"Could not read file: {exc}",
        )]

    all_findings: List[Finding] = []
    for block_start, block_content in extract_python_blocks(content):
        for offset, line in enumerate(block_content.splitlines()):
            # +1 because block_start is the ```{python} line, content starts after
            line_num = block_start + offset + 1
            all_findings.extend(check_line(line, line_num, str(filepath)))

    return all_findings


# ── File Discovery ────────────────────────────────────────────────────

def find_qmd_files(root: Path, vol_filter: Optional[str]) -> List[Path]:
    """Find all QMD content files, optionally restricted to vol1 or vol2."""
    all_files = sorted(root.rglob("*.qmd"))
    if vol_filter == "vol1":
        return [f for f in all_files if "/vol1/" in str(f)]
    if vol_filter == "vol2":
        return [f for f in all_files if "/vol2/" in str(f)]
    return all_files


# ── Reporter ──────────────────────────────────────────────────────────

def _rel(filepath: str, repo_root: Path) -> str:
    try:
        return str(Path(filepath).relative_to(repo_root))
    except ValueError:
        return filepath


def report(findings: List[Finding], repo_root: Path, show_info: bool = True) -> None:
    """Print findings grouped and sorted by severity (highest first)."""
    visible = [f for f in findings if show_info or f.severity > SEVERITY_INFO]
    if not visible:
        return
    # Sort: severity descending, then file path, then line number
    visible.sort(key=lambda f: (-f.severity, f.filepath, f.line_number))
    for f in visible:
        label = SEVERITY_LABELS[f.severity]
        print(f"[{label}] {_rel(f.filepath, repo_root)}:{f.line_number}")
        print(f"         check : {f.check_name}")
        print(f"         why   : {f.message}")
        print(f"         line  : {f.line_content.strip()}")
        print()


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    strict = "--strict" in args
    show_info = "--no-info" not in args
    vol_filter: Optional[str] = None
    specific_file: Optional[Path] = None

    if "--vol1" in args:
        vol_filter = "vol1"
    elif "--vol2" in args:
        vol_filter = "vol2"

    if "--file" in args:
        idx = args.index("--file")
        if idx + 1 < len(args):
            specific_file = Path(args[idx + 1])

    # Locate repository root relative to this script
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent  # mlsysbook-vols/
    contents_dir = repo_root / "book" / "quarto" / "contents"

    if specific_file:
        qmd_files = [specific_file.resolve()]
    else:
        qmd_files = find_qmd_files(contents_dir, vol_filter)

    # Scan all files
    all_findings: List[Finding] = []
    for filepath in qmd_files:
        all_findings.extend(scan_file(filepath))

    # Partition by severity
    errors   = [f for f in all_findings if f.severity == SEVERITY_ERROR]
    warnings = [f for f in all_findings if f.severity == SEVERITY_WARNING]
    infos    = [f for f in all_findings if f.severity == SEVERITY_INFO]

    vol_label = f" ({vol_filter})" if vol_filter else ""

    # No findings at all
    if not all_findings:
        print(f"✓ No Pint anti-patterns found in {len(qmd_files)} QMD files{vol_label}")
        return

    # Print findings
    report(all_findings, repo_root, show_info=show_info)

    # Summary line
    visible_count = len(errors) + len(warnings) + (len(infos) if show_info else 0)
    print("─" * 64)
    print(
        f"Scanned {len(qmd_files)} files{vol_label}  |  "
        f"{len(errors)} error(s)  |  {len(warnings)} warning(s)  |  {len(infos)} info"
    )
    if not show_info:
        print("(INFO findings hidden — remove --no-info to see style improvements)")
    print()

    # Exit code
    if strict and (errors or warnings):
        print("FAILED — --strict treats warnings as errors")
        sys.exit(1)
    elif errors:
        print("FAILED — errors must be resolved")
        sys.exit(1)
    elif warnings:
        print("PASSED (warnings present — review recommended)")
    else:
        print("PASSED ✓  (only style suggestions remain)")


if __name__ == "__main__":
    main()
