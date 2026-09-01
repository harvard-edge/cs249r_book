"""LaTeX build log diagnostics: undefined references, citations, and Overfull boxes."""

from __future__ import annotations

import re

from .base import BaseCheck, CheckRegistry
from ..context import BookContext
from ..report import LintIssue, LintReport


@CheckRegistry.register
class LaTeXLogDiagnosticsCheck(BaseCheck):
    name = "latex_log_diagnostics"
    description = "Parses LaTeX compile logs for unresolved references, citations, and Overfull \\hbox warnings"
    category = "layout"

    def run(self, ctx: BookContext, report: LintReport):
        if not ctx.log_path.exists():
            return

        log_content = ctx.log_path.read_text(encoding="utf-8", errors="ignore")

        # 1. Undefined references
        for match in re.finditer(r"LaTeX Warning: Reference `(.*?)' on page (\d+) undefined on input line (\d+)", log_content):
            ref_name, page, input_line = match.groups()
            report.add_issue(LintIssue(
                category="reference",
                severity="ERROR",
                file=str(ctx.log_path.relative_to(ctx.repo_root)),
                line=int(input_line),
                page=int(page),
                message=f"LaTeX undefined reference: '{ref_name}'",
                context=f"Occurs on PDF page {page}"
            ))

        # 2. Undefined citations
        for match in re.finditer(r"LaTeX Warning: Citation `(.*?)' on page (\d+) undefined on input line (\d+)", log_content):
            cite_name, page, input_line = match.groups()
            report.add_issue(LintIssue(
                category="citation",
                severity="ERROR",
                file=str(ctx.log_path.relative_to(ctx.repo_root)),
                line=int(input_line),
                page=int(page),
                message=f"LaTeX undefined citation: '{cite_name}'",
                context=f"Occurs on PDF page {page}"
            ))

        # 3. Overfull \hbox warnings (> 1.5pt)
        for match in re.finditer(r"Overfull \\hbox \(([0-9.]+)pt too wide\) in paragraph at lines (\d+)--(\d+)", log_content):
            overflow_pt, start_l, end_l = match.groups()
            pt_val = float(overflow_pt)
            if pt_val > 1.5:
                report.add_issue(LintIssue(
                    category="overflow",
                    severity="WARNING" if pt_val < 10.0 else "ERROR",
                    file="Physical-AI.tex",
                    line=int(start_l),
                    page=None,
                    message=f"Overfull \\hbox: {pt_val:.1f}pt ({pt_val * 0.3527:.2f}mm) protruding beyond text margin",
                    context=f"Lines {start_l}--{end_l} in generated TeX"
                ))
