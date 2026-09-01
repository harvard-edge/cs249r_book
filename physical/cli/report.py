"""Reporting and diagnostic output formatting for Physical AI CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional

# ANSI terminal formatting
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


@dataclass
class LintIssue:
    category: str  # "reference", "citation", "orphan", "formatting", "editorial", "structure", "layout", "overflow"
    severity: str  # "ERROR", "WARNING", "INFO"
    file: str
    line: Optional[int]
    page: Optional[int]
    message: str
    context: Optional[str] = None


@dataclass
class LintReport:
    total_files: int = 0
    passed: bool = True
    issues: List[LintIssue] = field(default_factory=list)

    def add_issue(self, issue: LintIssue):
        self.issues.append(issue)
        if issue.severity == "ERROR":
            self.passed = False

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "WARNING")

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "INFO")

    def to_json(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "total_files": self.total_files,
            "issues": [asdict(i) for i in self.issues],
        }
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def print_terminal_report(report: LintReport, title: str = "PHYSICAL AI BOOK LINTER & AUDITOR"):
    print(f"\n{BOLD}{CYAN}======================================================={RESET}")
    print(f"{BOLD}{CYAN}      {title.center(47)}         {RESET}")
    print(f"{BOLD}{CYAN}======================================================={RESET}\n")

    if not report.issues:
        print(f"{GREEN}{BOLD}✓ ALL NATIVE CHECKS PASSED! Zero errors or layout warnings.{RESET}\n")
        return

    # Group issues by category
    by_category: Dict[str, List[LintIssue]] = {}
    for issue in report.issues:
        by_category.setdefault(issue.category, []).append(issue)

    for cat, issues in by_category.items():
        print(f"{BOLD}{BLUE}▶ {cat.upper()} ISSUES ({len(issues)}){RESET}")
        for iss in issues:
            if iss.severity == "ERROR":
                col = RED
            elif iss.severity == "WARNING":
                col = YELLOW
            else:
                col = CYAN

            loc = f"{iss.file}"
            if iss.line:
                loc += f":{iss.line}"
            if iss.page:
                loc += f" (Page {iss.page})"

            print(f"  {col}[{iss.severity}]{RESET} {BOLD}{loc}{RESET}: {iss.message}")
            if iss.context:
                print(f"    {CYAN}↳ Detail:{RESET} {iss.context}")
        print()

    summary_color = RED if report.error_count > 0 else (YELLOW if report.warning_count > 0 else GREEN)
    print(f"{BOLD}Summary:{RESET} {summary_color}{report.error_count} error(s){RESET}, {YELLOW}{report.warning_count} warning(s){RESET}, {CYAN}{report.info_count} info{RESET} across {report.total_files} chapter files.\n")
