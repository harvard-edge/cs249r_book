"""Chapter opener structure and pedagogical hierarchy checks."""

from __future__ import annotations

from .base import BaseCheck, CheckRegistry
from ..context import BookContext
from ..report import LintIssue, LintReport


@CheckRegistry.register
class ChapterOpenerStructureCheck(BaseCheck):
    name = "chapter_opener_structure"
    description = "Ensures chapter opener has Title, Locator Figure, Opening Hook, and Objectives before H2"
    category = "structure"

    def run(self, ctx: BookContext, report: LintReport):
        for file_path in ctx.qmd_files:
            try:
                file_path.relative_to(ctx.chapters_dir)
            except ValueError:
                continue
            rel_path = str(file_path.relative_to(ctx.repo_root))
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()

            h1_line = None
            locator_line = None
            obj_line = None
            first_h2_line = None

            for idx, line in enumerate(lines, start=1):
                if line.startswith("# ") and not h1_line:
                    h1_line = idx
                elif "{#fig-locator" in line and not locator_line:
                    locator_line = idx
                elif line.startswith("::: {.callout-objective") and not obj_line:
                    obj_line = idx
                elif line.startswith("## ") and not first_h2_line:
                    first_h2_line = idx

            if h1_line and not locator_line:
                report.add_issue(LintIssue(
                    category="structure",
                    severity="WARNING",
                    file=rel_path,
                    line=h1_line,
                    page=None,
                    message="Missing chapter locator figure ({#fig-locator-...}) after chapter title"
                ))

            if obj_line and first_h2_line and obj_line > first_h2_line:
                report.add_issue(LintIssue(
                    category="structure",
                    severity="ERROR",
                    file=rel_path,
                    line=obj_line,
                    page=None,
                    message="Learning Objectives callout must appear before the first H2 section heading"
                ))
