"""Markdown and Pandoc callout formatting validation checks."""

from __future__ import annotations

import re

from .base import BaseCheck, CheckRegistry
from ..context import BookContext
from ..report import LintIssue, LintReport


@CheckRegistry.register
class CalloutListFormatCheck(BaseCheck):
    name = "callout_list_formatting"
    description = "Ensures callout blocks include empty lines before lists to prevent collapsed text"
    category = "formatting"

    def run(self, ctx: BookContext, report: LintReport):
        for file_path in ctx.qmd_files:
            rel_path = str(file_path.relative_to(ctx.repo_root))
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()

            in_callout = False
            callout_type = ""
            callout_start = 0

            for idx, line in enumerate(lines, start=1):
                if line.startswith("::: {.callout-"):
                    in_callout = True
                    callout_start = idx
                    callout_type = line.split()[1].strip("{}.")
                    continue

                if in_callout:
                    if line.strip() == ":::":
                        in_callout = False
                        continue

                    # Check for colon followed immediately on next line by a list item without a blank line
                    if line.strip().endswith(":") and idx < len(lines):
                        next_line = lines[idx]  # lines is 0-indexed, idx is line_number of current line
                        if re.match(r"^\s*(\d+\.|\*|\-)\s+", next_line):
                            report.add_issue(LintIssue(
                                category="formatting",
                                severity="ERROR",
                                file=rel_path,
                                line=idx,
                                page=None,
                                message=f"Missing blank line before list in {callout_type}",
                                context="Pandoc collapses list items into an unformatted inline wall of text if not preceded by a blank line."
                            ))
