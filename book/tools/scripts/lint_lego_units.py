#!/usr/bin/env python3
"""LEGO unit discipline linter for QMD cells (warnings + blocking errors)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LintIssue:
    rule: str
    file: str
    line: int
    message: str
    severity: str = "warning"


RULES = (
    "L001", "L002", "L003", "L004", "L006", "L007", "L008", "L009",
    "L014", "L015", "L016", "L019",
)

L014_CLOSED_FMT = re.compile(
    r"^\s*(?P<name>\w+_(?:w|kw|mw|j|mj|wh|kwh|mwh|gwh|gb|tb|gib|ms|s|kg|gbps|tflop|tonnes?)_str)\s*=\s*fmt\s*\(",
    re.M | re.I,
)
OPEN_FMT_ON_SCALAR = re.compile(
    r"^\s*(?P<name>\w+_str)\s*=\s*fmt\s*\(",
    re.M,
)
FMT_QTY_ASSIGN = re.compile(
    r"^\s*(?P<name>\w+_str)\s*=\s*fmt_qty\s*\(",
    re.M,
)
DOMAIN_FMT_ASSIGN = re.compile(
    r"^\s*(?P<name>\w+_str)\s*=\s*(?:fmt_power|fmt_energy|fmt_bandwidth|fmt_memory|fmt_emissions|fmt_latency)\s*\(",
    re.M,
)
MASG_TO_CLOSED = re.compile(
    r"^\s*(?P<name>\w+_(?:w|kw|mw|j|wh|kwh|mwh|gb|tb|ms|s|kg|tonnes?)_str)\s*=.*\.m_as\s*\(",
    re.M | re.I,
)
FMT_QTY_SCALAR = re.compile(
    r"fmt_qty\s*\(\s*\w+\.(?:m_as\s*\(|to\([^)]+\)\.magnitude)"
)
RAW_FMT_SUFFIX = re.compile(
    r"fmt\s*\([^)]*suffix\s*=\s*['\"]\s*(?:GB|TB|MB|kWh|MWh|TFLOP|W|MW|ms|s)\b"
)
ALLOWED_UNIT_LABELS = frozenset({
    "GB", "Gb", "Gb/s", "Tb/s", "TFLOP/s", "PFLOP/s", "TFLOP/s per W",
    "FLOP/byte", "FLOPs", "tons", "tons CO₂", "Wh", "MW", "MB", "KB", "pJ",
    "lux", "TB", "kJ per hour", "Mb/s", "dB",
    "billion FLOPs", "trillion FLOPs",
    "TOPS peak", "TOPS derated",
    "°C", "°C/s",
    "GB per day", "KB of detection summaries",
    "bytes",
})
UNIT_LABEL = re.compile(r"unit_label\s*=")
UNIT_LABEL_VALUE = re.compile(r"""unit_label\s*=\s*['"]([^'"]+)['"]""")
UPPER_TIME = re.compile(r"\b(MS|US|NS)\b")
UREG_ALIAS = re.compile(
    r"ureg\.(millijoule|megawatt|joule|kilowatt_hour|megawatt_hour|kilogram|millisecond|microsecond|minute|watt_hour)\b"
)
PROSE_DUP_UNIT = re.compile(
    r"\{python\}\s*[\w.]+\.(?P<name>\w+_(?:w|kw|mw|wh|kwh|mwh|gb|tb|ms|s|kg)_str)\`\s+(?:W|MW|kWh|MWh|GB|TB|ms|s|kg)\b",
    re.I,
)


def _scan_python_blocks(text: str) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    for match in re.finditer(r"```\{python\}(.*?)```", text, re.S):
        start = text[: match.start()].count("\n") + 1
        blocks.append((start, match.group(1)))
    return blocks


def lint_file(path: Path, root: Path) -> list[LintIssue]:
    path = path.resolve()
    root = root.resolve()
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)
    text = path.read_text(encoding="utf-8")
    issues: list[LintIssue] = []

    for base_line, block in _scan_python_blocks(text):
        for i, line in enumerate(block.splitlines(), start=1):
            lineno = base_line + i
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if FMT_QTY_SCALAR.search(line):
                issues.append(LintIssue(
                    "L001", rel, lineno,
                    "fmt_qty requires Quantity, not .m_as(...) or .to(...).magnitude.",
                ))
            if RAW_FMT_SUFFIX.search(line):
                issues.append(LintIssue("L002", rel, lineno, "Use fmt_qty or domain formatter for physical units."))
            if re.search(r"=\s*\w+\.(?:m_as|to)\(", line) and re.search(
                r"\*\s*(?:GB|TB|watt|MW|joule|second)\b", line
            ):
                issues.append(LintIssue("L003", rel, lineno, "Avoid reattaching units after scalar extraction."))
            if re.search(r"(energy_mwh|carbon|tonnes)\s*=.*(?:/\s*THOUSAND|\*\s*THOUSAND)", line, re.I):
                issues.append(LintIssue("L004", rel, lineno, "Use energy_from_power/carbon_from_energy helpers."))
            if UNIT_LABEL.search(line):
                m = UNIT_LABEL_VALUE.search(line)
                label = m.group(1) if m else ""
                if label not in ALLOWED_UNIT_LABELS:
                    issues.append(LintIssue("L006", rel, lineno, "Prefer domain formatter over unit_label=."))
            if re.search(r"\.to\((?:US|NS|MS)\)", line):
                issues.append(LintIssue(
                    "L007", rel, lineno,
                    "Prefer microsecond/nanosecond/ms over US/NS/MS in .to().",
                ))
            elif re.search(r"\b(?:MS|NS)\b", line) and not re.search(
                r"_fmt_unit_factor\((?:NS|US|MS)\b", line
            ):
                issues.append(LintIssue("L007", rel, lineno, "Prefer ms/nanosecond over MS/NS."))
            if re.search(r"=\s*\d+(?:\.\d+)?\s*\*\s*(TB|GB|TFLOP|MB)\s*/\s*second\b", line) and "(" not in line.split("=")[-1]:
                issues.append(LintIssue("L008", rel, lineno, "Prefer parenthesized rate: 1.9 * (TB / second)."))
            if UREG_ALIAS.search(line):
                issues.append(LintIssue("L009", rel, lineno, "Prefer exported unit alias over ureg.*."))
            if re.search(r"\.m_as\s*\(", line):
                issues.append(LintIssue(
                    "L019", rel, lineno,
                    "Use .to(unit).magnitude instead of .m_as() in LEGO cells.",
                    severity="error",
                ))

        for match in L014_CLOSED_FMT.finditer(block):
            lineno = base_line + block[: match.start()].count("\n")
            issues.append(LintIssue(
                "L014", rel, lineno,
                f"Closed name {match.group('name')} uses open fmt() — use fmt_qty/domain formatter.",
            ))
        for match in MASG_TO_CLOSED.finditer(block):
            issues.append(LintIssue(
                "L016", rel, base_line + match.start() // 80 + 1,
                f"{match.group('name')} assigned from .m_as() scalar.",
            ))
        # L017 retired: fmt_qty closed-auto names (e.g. throughput_str) are valid per lego-units.md.

    for match in PROSE_DUP_UNIT.finditer(text):
        lineno = text[: match.start()].count("\n") + 1
        issues.append(LintIssue(
            "L015", rel, lineno,
            f"Prose repeats unit after closed export {match.group('name')}.",
        ))
    return issues


def _staged_qmd_paths(root: Path) -> list[Path]:
    """Return staged .qmd paths relative to repo root (pre-commit scope)."""
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    paths: list[Path] = []
    for rel in proc.stdout.splitlines():
        if rel.endswith(".qmd"):
            p = (root / rel).resolve()
            if p.exists():
                paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lint LEGO unit discipline in QMD files.")
    parser.add_argument("paths", nargs="*", help="QMD files or directories")
    parser.add_argument("--fail-on", choices=("error", "warning"), default="error")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--baseline", type=Path, help="JSON baseline of allowed warnings")
    parser.add_argument(
        "--staged-only",
        action="store_true",
        help="Lint only git-staged .qmd files (pre-commit default when PRE_COMMIT=1).",
    )
    parser.add_argument(
        "--write-baseline",
        type=Path,
        help="Write all warning-severity issues to JSON baseline and exit 0.",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[3]
    staged_only = args.staged_only or (
        os.environ.get("PRE_COMMIT") == "1" and not args.paths
    )
    qmd_paths: list[Path] = []
    if args.paths:
        for p in args.paths:
            path = (root / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
            if path.is_dir():
                qmd_paths.extend(sorted(path.rglob("*.qmd")))
            elif path.suffix == ".qmd" and path.exists():
                qmd_paths.append(path)
    elif staged_only:
        qmd_paths = _staged_qmd_paths(root)
        if not qmd_paths:
            return 0
    else:
        contents = root / "book" / "quarto" / "contents"
        qmd_paths = sorted(contents.rglob("*.qmd"))

    staged_rels: set[str] = set()
    if staged_only:
        staged_rels = {str(p.relative_to(root)) for p in qmd_paths}

    issues: list[LintIssue] = []
    for path in qmd_paths:
        issues.extend(lint_file(path, root))

    if args.write_baseline:
        warnings_only = [issue for issue in issues if issue.severity == "warning"]
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.write_baseline.write_text(
            json.dumps([issue.__dict__ for issue in warnings_only], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(warnings_only)} warning baseline entries to {args.write_baseline}")
        return 0

    allowed: set[tuple[str, str, str]] = set()
    if args.baseline and args.baseline.exists():
        for entry in json.loads(args.baseline.read_text(encoding="utf-8")):
            key = (entry["rule"], entry["file"], entry["message"])
            if staged_only:
                if entry["file"] in staged_rels:
                    allowed.add(key)
            else:
                allowed.add(key)

    failures: list[LintIssue] = []
    for issue in issues:
        key = (issue.rule, issue.file, issue.message)
        if key in allowed:
            continue
        if args.fail_on == "warning" or issue.severity == "error":
            failures.append(issue)

    if args.format == "json":
        print(json.dumps([issue.__dict__ for issue in failures], indent=2))
    else:
        for issue in failures:
            print(f"{issue.file}:{issue.line}: [{issue.rule}] {issue.message}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
