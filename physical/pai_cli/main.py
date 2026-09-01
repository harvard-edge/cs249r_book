#!/usr/bin/env python3
r"""Physical AI CLI (pai).

Unified authoring, building, linting, and layout validation toolkit
for the Physical AI Systems textbook.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

from .builder import BookBuilder
from .checks import CheckRegistry
from .context import BookContext
from .report import (
    BOLD,
    CYAN,
    GREEN,
    LintReport,
    RED,
    RESET,
    YELLOW,
    print_terminal_report,
)


def run_checks(ctx: BookContext, categories: Optional[list[str]] = None) -> LintReport:
    """Executes registered native checks filtered by categories."""
    report = LintReport(total_files=len(ctx.qmd_files))
    all_checks = CheckRegistry.get_checks()

    if categories:
        checks_to_run = [c for c in all_checks if c.category in categories]
    else:
        checks_to_run = all_checks

    print(f"{CYAN}Executing {len(checks_to_run)} native check(s)...{RESET}")
    for check in checks_to_run:
        check.run(ctx, report)

    return report


def main():
    parser = argparse.ArgumentParser(
        prog="pai",
        description=f"{BOLD}Physical AI (PAI) Textbook CLI{RESET} - Unified authoring, build, and validation suite.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # 1. BUILD COMMAND
    build_parser = subparsers.add_parser("build", help="Build the textbook or a single chapter")
    build_parser.add_argument("--to", choices=["pdf", "typst", "html", "all"], default="pdf", help="Target output format (default: pdf)")
    build_parser.add_argument("--chapter", type=str, help="Specific chapter to render (e.g. 01-boundary)")
    build_parser.add_argument("--clean", action="store_true", help="Clean build cache before rendering")

    # 2. CHECK COMMAND
    check_parser = subparsers.add_parser("check", help="Run static semantic, reference, orphan, formatting, and editorial checks")
    check_parser.add_argument("--chapter", type=str, help="Filter to a specific chapter (e.g. 01-boundary)")
    check_parser.add_argument("--json", type=Path, help="Write structured diagnostic JSON report to path")

    # 3. LAYOUT COMMAND
    layout_parser = subparsers.add_parser("layout", help="Run PDF layout, margin bounding box, and LaTeX log checks")
    layout_parser.add_argument("--build", action="store_true", help="Recompile PDF before checking layout")
    layout_parser.add_argument("--sheets", action="store_true", help="Generate spread contact sheets for flagged pages")
    layout_parser.add_argument("--json", type=Path, help="Write structured diagnostic JSON report to path")

    # 4. AUDIT (ALL) COMMAND
    audit_parser = subparsers.add_parser("audit", help="Run end-to-end build, static checks, and visual layout audit")
    audit_parser.add_argument("--build", action="store_true", help="Recompile PDF before auditing")
    audit_parser.add_argument("--chapter", type=str, help="Filter to a specific chapter")
    audit_parser.add_argument("--sheets", action="store_true", help="Generate spread contact sheets for flagged pages")
    audit_parser.add_argument("--json", type=Path, help="Write structured diagnostic JSON report to path")

    # 5. PLAYWRIGHT WEB/BROWSER COMMAND
    web_parser = subparsers.add_parser("playwright", help="Run Playwright browser rendering, broken image, and visual tests on HTML")
    web_parser.add_argument("--chapter", type=str, help="Filter to a specific chapter (e.g. 01-boundary)")
    web_parser.add_argument("--build", action="store_true", help="Recompile HTML before checking")
    web_parser.add_argument("--json", type=Path, help="Write structured diagnostic JSON report to path")

    # 6. PREVIEW COMMAND
    preview_parser = subparsers.add_parser("preview", help="Launch Quarto live preview server")
    preview_parser.add_argument("--port", type=int, default=4200, help="Server port (default: 4200)")

    # 7. CLEAN COMMAND
    clean_parser = subparsers.add_parser("clean", help="Clean build directory and cached LaTeX artifacts")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    builder = BookBuilder(repo_root)
    command = args.command or "check"

    if command == "clean":
        builder.clean()
        sys.exit(0)

    if command == "preview":
        builder.preview(port=args.port)
        sys.exit(0)

    if command == "build":
        if args.chapter:
            success = builder.build_chapter(args.chapter, fmt=args.to)
        else:
            if args.to == "all":
                success = builder.build_book(fmt="pdf", clean_first=args.clean) and \
                          builder.build_book(fmt="html", clean_first=False)
            else:
                success = builder.build_book(fmt=args.to, clean_first=args.clean)
        sys.exit(0 if success else 1)

    if command == "check":
        ctx = BookContext(repo_root, chapter_filter=args.chapter)
        report = run_checks(ctx, categories=["semantic", "orphans", "formatting", "editorial", "structure"])
        print_terminal_report(report, title="PHYSICAL AI STATIC LINTER")
        if args.json:
            report.to_json(args.json)
            print(f"{GREEN}Structured JSON report written to {args.json}{RESET}")
        sys.exit(0 if report.passed else 1)

    if command == "layout":
        if getattr(args, "build", False):
            builder.build_book(fmt="pdf")
        ctx = BookContext(repo_root)
        report = run_checks(ctx, categories=["layout"])
        print_terminal_report(report, title="PHYSICAL AI LAYOUT & MARGIN AUDITOR")
        if getattr(args, "sheets", False) and not report.passed:
            from .checks.layout_margins import render_flagged_contact_sheets
            render_flagged_contact_sheets(ctx, report)
        if args.json:
            report.to_json(args.json)
            print(f"{GREEN}Structured JSON report written to {args.json}{RESET}")
        sys.exit(0 if report.passed else 1)

    if command in ("playwright", "web"):
        if getattr(args, "build", False):
            if getattr(args, "chapter", None):
                builder.build_chapter(args.chapter, fmt="html")
            else:
                builder.build_book(fmt="html")
        ctx = BookContext(repo_root, chapter_filter=getattr(args, "chapter", None))
        report = run_checks(ctx, categories=["playwright"])
        print_terminal_report(report, title="PHYSICAL AI PLAYWRIGHT BROWSER AUDITOR")
        if getattr(args, "json", None):
            report.to_json(args.json)
            print(f"{GREEN}Structured JSON report written to {args.json}{RESET}")
        sys.exit(0 if report.passed else 1)

    if command in ("audit", "all"):
        if getattr(args, "build", False):
            builder.build_book(fmt="pdf")
        ctx = BookContext(repo_root, chapter_filter=getattr(args, "chapter", None))
        report = run_checks(ctx)
        print_terminal_report(report, title="PHYSICAL AI COMPREHENSIVE AUDIT")
        if getattr(args, "sheets", False) and not report.passed:
            from .checks.layout_margins import render_flagged_contact_sheets
            render_flagged_contact_sheets(ctx, report)
        if getattr(args, "json", None):
            report.to_json(args.json)
            print(f"{GREEN}Structured JSON report written to {args.json}{RESET}")
        sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
