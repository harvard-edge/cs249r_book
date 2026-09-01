"""Comprehensive orphan detection for figures, tables, equations, callouts, and disk assets."""

from __future__ import annotations

from pathlib import Path

from .base import BaseCheck, CheckRegistry
from ..context import BookContext
from ..report import LintIssue, LintReport


@CheckRegistry.register
class OrphanArtifactCheck(BaseCheck):
    name = "orphan_artifacts"
    description = "Detects figures, tables, equations, callout blocks, and disk assets that are unused or uncited"
    category = "orphans"

    def run(self, ctx: BookContext, report: LintReport):
        # 1. Orphan Figures
        cited_figs = {r[0] for r in ctx.fig_refs}
        for fig_id, (file_path, line_idx) in ctx.fig_defs.items():
            if fig_id.startswith("fig-locator"):
                continue
            if fig_id not in cited_figs:
                report.add_issue(LintIssue(
                    category="orphans",
                    severity="WARNING",
                    file=str(file_path.relative_to(ctx.repo_root)),
                    line=line_idx,
                    page=None,
                    message=f"Orphan figure: {{#{fig_id}}} defined but never cited with @{fig_id}",
                    context="Every figure in the textbook must be introduced and discussed in body text."
                ))

        # 2. Orphan Tables
        cited_tbls = {r[0] for r in ctx.tbl_refs}
        for tbl_id, (file_path, line_idx) in ctx.tbl_defs.items():
            if tbl_id not in cited_tbls:
                report.add_issue(LintIssue(
                    category="orphans",
                    severity="WARNING",
                    file=str(file_path.relative_to(ctx.repo_root)),
                    line=line_idx,
                    page=None,
                    message=f"Orphan table: {{#{tbl_id}}} defined but never cited with @{tbl_id}",
                    context="Every table in the textbook must be introduced and discussed in body text."
                ))

        # 3. Orphan Equations
        cited_eqs = {r[0] for r in ctx.eq_refs}
        for eq_id, (file_path, line_idx) in ctx.eq_defs.items():
            if eq_id not in cited_eqs:
                report.add_issue(LintIssue(
                    category="orphans",
                    severity="INFO",
                    file=str(file_path.relative_to(ctx.repo_root)),
                    line=line_idx,
                    page=None,
                    message=f"Uncited numbered equation: {{#{eq_id}}}",
                    context="Equation has an explicit cross-reference label but is not cited with @eq-..."
                ))

        # 4. Orphan Callouts / Laws / Contracts / Autopsies
        all_refs = {r[0] for r in ctx.fig_refs + ctx.tbl_refs + ctx.sec_refs + ctx.eq_refs}
        for callout_id, (file_path, line_idx) in ctx.callout_defs.items():
            if callout_id not in all_refs:
                report.add_issue(LintIssue(
                    category="orphans",
                    severity="INFO",
                    file=str(file_path.relative_to(ctx.repo_root)),
                    line=line_idx,
                    page=None,
                    message=f"Uncited callout identifier: {{#{callout_id}}}",
                    context="Callout block has a custom identifier tag that is not cited in the text."
                ))

        # 5. Orphan Asset Files on Disk
        # Build set of all image paths referenced in markdown files
        referenced_basenames = set()
        for img_str, _, _ in ctx.image_includes:
            p = Path(img_str)
            referenced_basenames.add(p.name)
            referenced_basenames.add(p.stem)  # in case extension differs (svg vs pdf)

        for disk_asset in ctx.disk_assets:
            # Check if asset name or stem is included anywhere
            if disk_asset.name not in referenced_basenames and disk_asset.stem not in referenced_basenames:
                # Exclude pipeline locators or template icons
                if "pipeline_locator" in disk_asset.name or "icon" in disk_asset.name:
                    continue
                report.add_issue(LintIssue(
                    category="orphans",
                    severity="WARNING",
                    file=str(disk_asset.relative_to(ctx.repo_root)),
                    line=None,
                    page=None,
                    message=f"Orphan disk asset: '{disk_asset.name}' is never referenced in any markdown file",
                    context="Asset exists in figures directory but has no matching ![caption](...) reference"
                ))
