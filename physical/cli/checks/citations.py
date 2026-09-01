"""BibTeX and bibliography citation validation checks."""

from __future__ import annotations

from .base import BaseCheck, CheckRegistry
from ..context import BookContext
from ..report import LintIssue, LintReport


@CheckRegistry.register
class CitationBibTeXCheck(BaseCheck):
    name = "citation_bibtex"
    description = "Validates that every citation in markdown exists in references.bib and finds unused bib keys"
    category = "semantic"

    def run(self, ctx: BookContext, report: LintReport):
        cited_keys = set()
        for cite_key, file_path, line_idx in ctx.citations:
            cited_keys.add(cite_key)
            if ctx.bib_keys and cite_key not in ctx.bib_keys:
                report.add_issue(LintIssue(
                    category="citation",
                    severity="ERROR",
                    file=str(file_path.relative_to(ctx.repo_root)),
                    line=line_idx,
                    page=None,
                    message=f"Missing bibliography entry: @{cite_key}",
                    context=f"The citation key '@{cite_key}' is not defined in {ctx.bib_path.name}"
                ))

        # Check for unused BibTeX entries in references.bib
        if not ctx.chapter_filter:
            unused_keys = ctx.bib_keys - cited_keys
            for u in sorted(unused_keys):
                report.add_issue(LintIssue(
                    category="citation",
                    severity="INFO",
                    file="book/references.bib",
                    line=None,
                    page=None,
                    message=f"Unused BibTeX entry: '{u}'",
                    context="Key exists in references.bib but is never cited in any chapter"
                ))
