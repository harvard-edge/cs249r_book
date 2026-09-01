"""Cross-reference integrity checks for figures, tables, sections, and equations."""

from __future__ import annotations

from .base import BaseCheck, CheckRegistry
from ..context import BookContext
from ..report import LintIssue, LintReport


@CheckRegistry.register
class CrossReferenceCheck(BaseCheck):
    name = "cross_references"
    description = "Validates that all @fig, @tbl, @sec, and @eq cross-references resolve"
    category = "semantic"

    def run(self, ctx: BookContext, report: LintReport):
        for ref_id, file_path, line_idx in ctx.fig_refs:
            if ref_id not in ctx.fig_defs:
                report.add_issue(LintIssue(
                    category="reference",
                    severity="ERROR",
                    file=str(file_path.relative_to(ctx.repo_root)),
                    line=line_idx,
                    page=None,
                    message=f"Undefined figure cross-reference: @{ref_id}",
                    context=f"No matching {{#{ref_id}}} definition found in any chapter"
                ))

        for ref_id, file_path, line_idx in ctx.tbl_refs:
            if ref_id not in ctx.tbl_defs:
                report.add_issue(LintIssue(
                    category="reference",
                    severity="ERROR",
                    file=str(file_path.relative_to(ctx.repo_root)),
                    line=line_idx,
                    page=None,
                    message=f"Undefined table cross-reference: @{ref_id}",
                    context=f"No matching {{#{ref_id}}} definition found in any chapter"
                ))

        for ref_id, file_path, line_idx in ctx.sec_refs:
            if ref_id not in ctx.sec_defs:
                report.add_issue(LintIssue(
                    category="reference",
                    severity="WARNING",
                    file=str(file_path.relative_to(ctx.repo_root)),
                    line=line_idx,
                    page=None,
                    message=f"Undefined section cross-reference: @{ref_id}",
                    context=f"No matching {{#{ref_id}}} section header found"
                ))

        for ref_id, file_path, line_idx in ctx.eq_refs:
            if ref_id not in ctx.eq_defs:
                report.add_issue(LintIssue(
                    category="reference",
                    severity="ERROR",
                    file=str(file_path.relative_to(ctx.repo_root)),
                    line=line_idx,
                    page=None,
                    message=f"Undefined equation cross-reference: @{ref_id}",
                    context=f"No matching {{#{ref_id}}} equation tag found"
                ))
