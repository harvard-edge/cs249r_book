"""
Native validation commands for MLSysBook Binder CLI.

This module intentionally implements validation logic directly in Binder,
without shelling out to legacy scripts under tools/scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class ValidationIssue:
    file: str
    line: int
    code: str
    message: str
    severity: str = "error"
    context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "context": self.context,
        }


@dataclass
class ValidationRunResult:
    name: str
    description: str
    files_checked: int
    issues: List[ValidationIssue]
    elapsed_ms: int

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "files_checked": self.files_checked,
            "passed": self.passed,
            "issue_count": len(self.issues),
            "elapsed_ms": self.elapsed_ms,
            "issues": [issue.to_dict() for issue in self.issues],
        }


INLINE_REF_PATTERN = re.compile(r"`\{python\}\s+(\w+)`")
CELL_START_PATTERN = re.compile(r"^```\{python\}|^```python")
CELL_END_PATTERN = re.compile(r"^```\s*$")
ASSIGN_PATTERN = re.compile(r"^([A-Za-z_]\w*)\s*=")
GRID_TABLE_SEP_PATTERN = re.compile(r"^\+[-:=+]+\+$")
LATEX_INLINE_PATTERN = re.compile(r"(?<!\\)\$\s*`\{python\}\s+[^`]+`|`\{python\}\s+[^`]+`\s*(?<!\\)\$")
LATEX_ADJACENT_PATTERN = re.compile(r"`\{python\}\s+[^`]+`\s*\$\\(times|approx|ll|gg|mu)\$")

CITATION_REF_PATTERN = re.compile(r"@([A-Za-z0-9_:\-.]+)")
CITATION_BRACKET_PATTERN = re.compile(r"\[-?@[A-Za-z0-9_:\-.]+(?:;\s*-?@[A-Za-z0-9_:\-.]+)*\]")

LABEL_DEF_PATTERNS = {
    "Figure": [
        re.compile(r"\{#(fig-[\w-]+)"),              # {#fig-xyz ...}
        re.compile(r"#\|\s*label:\s*(fig-[\w-]+)"),  # #| label: fig-xyz
        re.compile(r"%%\|\s*label:\s*(fig-[\w-]+)"), # %%| label: fig-xyz (Jupyter)
    ],
    "Table": [
        re.compile(r"\{#(tbl-[\w-]+)"),              # {#tbl-xyz}
        re.compile(r"#\|\s*label:\s*(tbl-[\w-]+)"),  # #| label: tbl-xyz
    ],
    "Section": [
        re.compile(r"\{#(sec-[\w-]+)"),              # {#sec-xyz}
        re.compile(r"^id:\s*(sec-[\w-]+)"),          # YAML id: sec-xyz
    ],
    "Equation": [re.compile(r"\{#(eq-[\w-]+)")],     # {#eq-xyz}
    "Listing": [
        re.compile(r"\{#(lst-[\w-]+)"),              # {#lst-xyz ...}
        re.compile(r"#\|\s*label:\s*(lst-[\w-]+)"),  # #| label: lst-xyz
    ],
}
LABEL_REF_PATTERN = re.compile(r"@((?:fig|tbl|sec|eq|lst)-[\w-]+)")

EXCLUDED_CITATION_PREFIXES = ("fig-", "tbl-", "sec-", "eq-", "lst-", "ch-")


class ValidateCommand:
    """Native `binder check` command group (also available as `binder validate`).

    Groups:
        refs        — inline-python, cross-refs, citations, inline patterns
        labels      — duplicate labels, orphaned/unreferenced labels
        headers     — section header IDs
        footnotes   — placement rules, reference integrity
        figures     — captions/alt-text, float flow, image files
        rendering   — render patterns, indexes, dropcaps, parts
        all         — run every check
    """

    # Maps group name → list of (scope_name, runner_method_name) pairs.
    # This is the single source of truth for the hierarchy.
    GROUPS: Dict[str, List[tuple]] = {
        "refs": [
            ("python-syntax", "_run_python_syntax"),
            ("inline-python", "_run_inline_python"),
            ("cross-refs", "_run_refs"),
            ("citations", "_run_citations"),
            ("inline", "_run_inline_refs"),
            ("self-ref", "_run_self_referential"),
        ],
        "labels": [
            ("duplicates", "_run_duplicate_labels"),
            ("orphans", "_run_unreferenced_labels"),
            ("fig-labels", "_run_fig_label_underscores"),
        ],
        "headers": [
            ("ids", "_run_headers"),
        ],
        "footnotes": [
            ("placement", "_run_footnote_placement"),
            ("integrity", "_run_footnote_refs"),
            ("cross-chapter", "_run_footnote_cross_chapter"),
        ],
        "figures": [
            ("captions", "_run_figures"),
            ("flow", "_run_float_flow"),
            ("files", "_run_images"),
        ],
        "rendering": [
            ("patterns", "_run_rendering"),
            ("python-echo", "_run_python_echo"),
            ("indexes", "_run_indexes"),
            ("dropcaps", "_run_dropcaps"),
            ("parts", "_run_parts"),
            ("heading-levels", "_run_heading_levels"),
            ("duplicate-words", "_run_duplicate_words"),
            ("grid-tables", "_run_grid_tables"),
            ("tables", "_run_table_content"),
            ("ascii", "_run_ascii"),
        ],
        "images": [
            ("formats", "_run_image_formats"),
            ("external", "_run_external_images"),
        ],
        "json": [
            ("syntax", "_run_json_syntax"),
        ],
        "units": [
            ("physics", "_run_unit_tests"),
        ],
        "spelling": [
            ("prose", "_run_spelling_prose"),
            ("tikz", "_run_spelling_tikz"),
        ],
        "epub": [
            ("structure", "_run_epub"),
        ],
        "sources": [
            ("citations", "_run_sources"),
        ],
    }

    def __init__(self, config_manager, chapter_discovery):
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery

    def run(self, args: List[str]) -> bool:
        all_group_names = list(self.GROUPS.keys()) + ["all"]
        parser = argparse.ArgumentParser(
            prog="binder check",
            description="Run quality checks on book content",
            add_help=True,
        )
        parser.add_argument(
            "subcommand",
            nargs="?",
            choices=all_group_names,
            help="Check group to run (refs, labels, headers, footnotes, figures, rendering, all)",
        )
        parser.add_argument("--scope", default=None, help="Narrow to a specific check within a group")
        parser.add_argument("--path", default=None, help="File or directory path to check")
        parser.add_argument("--vol1", action="store_true", help="Scope to Volume I")
        parser.add_argument("--vol2", action="store_true", help="Scope to Volume II")
        parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
        parser.add_argument("--citations-in-code", action="store_true", help="refs: check citations in code fences")
        parser.add_argument("--citations-in-raw", action="store_true", help="refs: check citations in raw blocks")
        parser.add_argument("--check-patterns", action="store_true", default=True, help="refs --scope inline: include pattern hazard checks (default: on)")
        parser.add_argument("--no-check-patterns", action="store_false", dest="check_patterns", help="refs --scope inline: skip pattern hazard checks")
        parser.add_argument("--figures", action="store_true", help="labels: filter to figures")
        parser.add_argument("--tables", action="store_true", help="labels: filter to tables")
        parser.add_argument("--sections", action="store_true", help="labels: filter to sections")
        parser.add_argument("--equations", action="store_true", help="labels: filter to equations")
        parser.add_argument("--listings", action="store_true", help="labels: filter to listings")
        parser.add_argument("--all-types", action="store_true", help="labels: all label types")

        try:
            ns = parser.parse_args(args)
        except SystemExit:
            # argparse uses SystemExit(0) for --help and non-zero for parse errors.
            return ("-h" in args) or ("--help" in args)

        if not ns.subcommand:
            self._print_check_help()
            return False

        root_path = self._resolve_path(ns.path, ns.vol1, ns.vol2)
        if not root_path.exists():
            self._emit(ns.json, {"status": "error", "message": f"Path not found: {root_path}"}, failed=True)
            return False

        runs: List[ValidationRunResult] = []

        if ns.subcommand == "all":
            for group_name in self.GROUPS:
                runs.extend(self._run_group(group_name, None, root_path, ns))
        else:
            group_name = ns.subcommand
            scope = ns.scope
            if scope and not any(s == scope for s, _ in self.GROUPS.get(group_name, [])):
                valid = [s for s, _ in self.GROUPS[group_name]]
                console.print(f"[red]Unknown scope '{scope}' for group '{group_name}'.[/red]")
                console.print(f"[yellow]Valid scopes: {', '.join(valid)}[/yellow]")
                return False
            runs.extend(self._run_group(group_name, scope, root_path, ns))

        any_failed = any(not run.passed for run in runs)
        summary = {
            "status": "failed" if any_failed else "passed",
            "command": ns.subcommand,
            "path": str(root_path),
            "runs": [run.to_dict() for run in runs],
            "total_issues": sum(len(run.issues) for run in runs),
        }

        if ns.json:
            print(json.dumps(summary, indent=2))
        else:
            self._print_human_summary(summary, verbose=ns.verbose)

        return not any_failed

    # ------------------------------------------------------------------
    # Group dispatch
    # ------------------------------------------------------------------

    def _run_group(
        self,
        group: str,
        scope: Optional[str],
        root: Path,
        ns: argparse.Namespace,
    ) -> List[ValidationRunResult]:
        """Run all checks in *group*, or just the one matching *scope*."""
        results: List[ValidationRunResult] = []
        for scope_name, method_name in self.GROUPS[group]:
            if scope and scope != scope_name:
                continue
            method = getattr(self, method_name)
            # Some runners need extra kwargs
            if method_name == "_run_refs":
                checks_code = ns.citations_in_code or (not ns.citations_in_code and not ns.citations_in_raw)
                checks_raw = ns.citations_in_raw or (not ns.citations_in_code and not ns.citations_in_raw)
                results.append(method(root, citations_in_code=checks_code, citations_in_raw=checks_raw))
            elif method_name == "_run_inline_refs":
                results.append(method(root, check_patterns=ns.check_patterns))
            elif method_name in ("_run_duplicate_labels", "_run_unreferenced_labels"):
                results.append(method(root, self._selected_label_types(ns)))
            else:
                results.append(method(root))
        return results

    def _print_check_help(self) -> None:
        """Print a nicely formatted help for the check command."""
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Group", style="cyan", width=14)
        table.add_column("Scopes", style="yellow", width=38)
        table.add_column("Description", style="white", width=32)

        descriptions = {
            "refs": "References, citations, inline Python, self-ref",
            "labels": "Duplicate labels, orphans, fig-label underscores",
            "headers": "Section header IDs ({#sec-...})",
            "footnotes": "Placement, integrity, cross-chapter duplicates",
            "figures": "Captions, float flow, image files",
            "rendering": "Patterns, indexes, dropcaps, headings, typos, tables, ASCII",
            "images": "Image file formats, external URLs",
            "json": "JSON file syntax validation",
            "units": "Physics engine unit conversion tests",
            "spelling": "Prose and TikZ spell checking (requires aspell)",
            "epub": "EPUB file validation",
            "sources": "Source citation analysis and validation",
        }
        for group_name, checks in self.GROUPS.items():
            scopes = ", ".join(s for s, _ in checks)
            desc = descriptions.get(group_name, "")
            table.add_row(group_name, scopes, desc)
        table.add_row("all", "(everything)", "Run all checks")

        console.print(Panel(table, title="binder check <group> [--scope <name>]", border_style="cyan"))
        console.print("[dim]Examples:[/dim]")
        console.print("  [cyan]./binder check refs[/cyan]              [dim]# all reference checks[/dim]")
        console.print("  [cyan]./binder check refs --scope citations[/cyan]  [dim]# only citation check[/dim]")
        console.print("  [cyan]./binder check figures --vol1[/cyan]    [dim]# all figure checks, Vol I[/dim]")
        console.print("  [cyan]./binder check all[/cyan]               [dim]# everything[/dim]")
        console.print()

    # ------------------------------------------------------------------

    def _resolve_path(self, path_arg: Optional[str], vol1: bool, vol2: bool) -> Path:
        if path_arg:
            path = Path(path_arg)
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            return path
        base = self.config_manager.book_dir / "contents"
        if vol1 and not vol2:
            return base / "vol1"
        if vol2 and not vol1:
            return base / "vol2"
        return base

    def _selected_label_types(self, ns: argparse.Namespace) -> Dict[str, List[re.Pattern[str]]]:
        explicit = ns.figures or ns.tables or ns.sections or ns.equations or ns.listings
        if ns.all_types:
            return LABEL_DEF_PATTERNS
        if explicit:
            selected: Dict[str, List[re.Pattern[str]]] = {}
            if ns.figures:
                selected["Figure"] = LABEL_DEF_PATTERNS["Figure"]
            if ns.tables:
                selected["Table"] = LABEL_DEF_PATTERNS["Table"]
            if ns.sections:
                selected["Section"] = LABEL_DEF_PATTERNS["Section"]
            if ns.equations:
                selected["Equation"] = LABEL_DEF_PATTERNS["Equation"]
            if ns.listings:
                selected["Listing"] = LABEL_DEF_PATTERNS["Listing"]
            return selected
        # default: all label types
        return LABEL_DEF_PATTERNS

    def _qmd_files(self, root: Path) -> List[Path]:
        if root.is_file():
            return [root] if root.suffix == ".qmd" else []
        return sorted(root.rglob("*.qmd"))

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="ignore")

    def _relative_file(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.config_manager.book_dir))
        except ValueError:
            return str(path)

    def _run_python_syntax(self, root: Path) -> ValidationRunResult:
        """Compile every ```{python} code block to catch syntax errors."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        block_start_re = re.compile(r"^```\{python\}")
        block_end_re = re.compile(r"^```\s*$")

        for file in files:
            content = self._read_text(file)
            lines = content.split("\n")
            rel = str(file.relative_to(root)) if file.is_relative_to(root) else str(file)

            in_block = False
            block_lines: List[str] = []
            block_start_line = 0

            for i, line in enumerate(lines, start=1):
                if block_start_re.match(line):
                    in_block = True
                    block_lines = []
                    block_start_line = i
                    continue
                if in_block and block_end_re.match(line):
                    in_block = False
                    # Skip YAML-style #| directives before compiling
                    source_lines = [
                        ln for ln in block_lines
                        if not ln.strip().startswith("#|")
                    ]
                    source = "\n".join(source_lines)
                    if not source.strip():
                        continue
                    try:
                        compile(source, f"{rel}:{block_start_line}", "exec")
                    except SyntaxError as exc:
                        err_line = block_start_line + (exc.lineno or 1)
                        issues.append(ValidationIssue(
                            file=rel,
                            line=err_line,
                            code="python_syntax",
                            message=f"Python syntax error: {exc.msg}",
                            severity="error",
                            context=(exc.text or "").strip()[:120],
                        ))
                    continue
                if in_block:
                    block_lines.append(line)

        return ValidationRunResult(
            name="python-syntax",
            description="Validate Python code block syntax (compile check)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_inline_python(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        regex_checks = [
            ("missing_backtick", re.compile(r"(?<!`)(\{python\}\s+\w+`)"), "Missing opening backtick before {python}", "error"),
            ("dollar_as_backtick", re.compile(r"\$\{python\}\s+\w+`"), "Dollar sign used instead of backtick before {python}", "error"),
            ("display_math", re.compile(r"\$\$[^$]*`?\{python\}"), "Inline Python inside $$...$$ display math", "error"),
            # NOTE: $\times$ adjacent to inline Python is the PREFERRED convention.
            # Only flag non-_str variables inside $...$ math (decimal stripping risk).
            ("latex_adjacent_raw", re.compile(r"`\{python\}\s+(?!\w+_str)[^`]+`\s*\$\\(times|approx|ll|gg|mu|le|ge|neq|pm|cdot|div)"), "Non-_str inline Python adjacent to LaTeX operator (decimal stripping risk)", "warning"),
        ]

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code_block = False
            in_grid = False

            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    continue

                for code, pattern, message, severity in regex_checks:
                    for match in pattern.finditer(line):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code=code,
                            message=message,
                            severity=severity,
                            context=match.group(0)[:160],
                        ))

                if LATEX_INLINE_PATTERN.search(line):
                    issues.append(ValidationIssue(
                        file=self._relative_file(file),
                        line=idx,
                        code="python_in_math",
                        message="Inline Python inside $...$ math can render incorrectly",
                        severity="error",
                        context=line.strip()[:160],
                    ))

                if GRID_TABLE_SEP_PATTERN.match(stripped):
                    in_grid = True
                elif in_grid and not stripped.startswith("|") and stripped:
                    in_grid = False

                if in_grid and "`{python}" in line:
                    issues.append(ValidationIssue(
                        file=self._relative_file(file),
                        line=idx,
                        code="grid_table_python",
                        message="Inline Python in grid table; convert to pipe table",
                        severity="error",
                        context=line.strip()[:160],
                    ))

                # Unwrapped {python} — missing backticks entirely
                # Match {python} NOT preceded by ` and NOT at start of #| label line
                if "{python}" in line and not stripped.startswith("#|"):
                    for um in re.finditer(r"(?<!`)\{python\}\s+\w+", line):
                        # Make sure it's not inside a backtick span
                        before = line[:um.start()]
                        if before.count("`") % 2 == 0:  # even backticks = not inside span
                            issues.append(ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code="unwrapped_python",
                                message="Inline Python missing backtick wrapping — will render as literal text",
                                severity="error",
                                context=um.group(0)[:120],
                            ))

                # Inline Python in headings — fragile for TOC/bookmarks/PDF
                if stripped.startswith("#") and not stripped.startswith("#|") and "`{python}" in line:
                    issues.append(ValidationIssue(
                        file=self._relative_file(file),
                        line=idx,
                        code="python_in_heading",
                        message="Inline Python in heading — fragile for TOC, bookmarks, and PDF",
                        severity="warning",
                        context=stripped[:120],
                    ))

        return ValidationRunResult(
            name="inline-python",
            description="Validate inline Python syntax and placement",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_refs(self, root: Path, citations_in_code: bool, citations_in_raw: bool) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        fenced_code_pattern = re.compile(r"```\{([^}]+)\}(.*?)```", re.DOTALL)
        raw_block_pattern = re.compile(r"```\{=(html|latex|tex)\}(.*?)```", re.DOTALL | re.IGNORECASE)
        problematic_classes = {"tikz", "latex", "tex"}

        for file in files:
            content = self._read_text(file)
            if citations_in_code:
                for match in fenced_code_pattern.finditer(content):
                    attrs = match.group(1)
                    code_content = match.group(2)
                    class_match = re.search(r"\.([A-Za-z][A-Za-z0-9_-]*)", attrs)
                    cls = class_match.group(1).lower() if class_match else "unknown"
                    if cls not in problematic_classes:
                        continue
                    for cite_match in CITATION_BRACKET_PATTERN.finditer(code_content):
                        offset = match.start() + len(f"```{{{attrs}}}") + cite_match.start()
                        line_no = content[:offset].count("\n") + 1
                        line = content.splitlines()[line_no - 1] if line_no - 1 < len(content.splitlines()) else ""
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=line_no,
                            code="citation_in_code",
                            message=f"Citation in .{cls} code block will not be processed",
                            severity="error",
                            context=line.strip()[:160],
                        ))

            if citations_in_raw:
                for match in raw_block_pattern.finditer(content):
                    raw_type = match.group(1).lower()
                    block = match.group(2)
                    for cite_match in CITATION_BRACKET_PATTERN.finditer(block):
                        offset = match.start() + cite_match.start()
                        line_no = content[:offset].count("\n") + 1
                        line = content.splitlines()[line_no - 1] if line_no - 1 < len(content.splitlines()) else ""
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=line_no,
                            code="citation_in_raw",
                            message=f"Citation in raw {raw_type} block will not be processed",
                            severity="error",
                            context=line.strip()[:160],
                        ))

        return ValidationRunResult(
            name="refs",
            description="Validate citation/reference placement in raw/code blocks",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _bibliography_for_qmd(self, file: Path) -> Optional[Path]:
        """Resolve the volume backmatter references.bib for a .qmd from its path."""
        try:
            rel = file.relative_to(self.config_manager.book_dir)
        except ValueError:
            return None
        parts = rel.parts
        if "vol1" in parts:
            bib_file = self.config_manager.book_dir / "contents" / "vol1" / "backmatter" / "references.bib"
        elif "vol2" in parts:
            bib_file = self.config_manager.book_dir / "contents" / "vol2" / "backmatter" / "references.bib"
        else:
            return None
        return bib_file if bib_file.exists() else None

    def _run_citations(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        bib_key_pattern = re.compile(r"@\w+\{([^,\s]+)")

        for file in files:
            bib_file = self._bibliography_for_qmd(file)
            if bib_file is None:
                continue

            content = self._read_text(file)
            bib_content = self._read_text(bib_file)
            bib_keys = set(bib_key_pattern.findall(bib_content))
            qmd_content_no_code = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
            qmd_content_no_code = re.sub(r"`[^`]+`", "", qmd_content_no_code)
            refs = set(CITATION_REF_PATTERN.findall(qmd_content_no_code))
            refs = {r.rstrip(".,;:") for r in refs if not r.startswith(EXCLUDED_CITATION_PREFIXES)}
            refs = {r for r in refs if not re.match(r"^\d+\.\d+", r)}
            missing = sorted(refs - bib_keys)
            for key in missing:
                line_no = self._line_for_token(content, f"@{key}")
                issues.append(ValidationIssue(
                    file=self._relative_file(file),
                    line=line_no,
                    code="missing_citation",
                    message=f"Citation key @{key} missing in bibliography",
                    severity="error",
                    context=f"@{key}",
                ))

        return ValidationRunResult(
            name="citations",
            description="Validate citation keys against bibliography files",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_duplicate_labels(self, root: Path, label_types: Dict[str, List[re.Pattern[str]]]) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []
        definitions: Dict[str, List[Tuple[Path, int, str]]] = {}

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                for label_type, patterns in label_types.items():
                    for pattern in patterns:
                        for match in pattern.finditer(line):
                            label = match.group(1)
                            definitions.setdefault(label, []).append((file, idx, label_type))

        for label, locations in definitions.items():
            if len(locations) <= 1:
                continue
            for file, line_no, label_type in locations:
                issues.append(ValidationIssue(
                    file=self._relative_file(file),
                    line=line_no,
                    code="duplicate_label",
                    message=f"Duplicate {label_type.lower()} label: {label}",
                    severity="error",
                    context=label,
                ))

        return ValidationRunResult(
            name="duplicate-labels",
            description="Detect duplicate label definitions",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_unreferenced_labels(self, root: Path, label_types: Dict[str, List[re.Pattern[str]]]) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        defined: Dict[str, Tuple[Path, int, str]] = {}
        references: Dict[str, List[Tuple[Path, int]]] = {}

        for file in files:
            lines = self._read_text(file).splitlines()
            for idx, line in enumerate(lines, 1):
                for label_type, patterns in label_types.items():
                    for pattern in patterns:
                        for match in pattern.finditer(line):
                            defined.setdefault(match.group(1), (file, idx, label_type))

                for match in LABEL_REF_PATTERN.finditer(line):
                    label = match.group(1)
                    references.setdefault(label, []).append((file, idx))

        # unreferenced definitions (skip section defaults, consistent with legacy behavior)
        for label, (file, line_no, label_type) in defined.items():
            if label_type == "Section":
                continue
            if label not in references:
                issues.append(ValidationIssue(
                    file=self._relative_file(file),
                    line=line_no,
                    code="unreferenced_label",
                    message=f"{label_type} label {label} is never referenced",
                    severity="warning",
                    context=label,
                ))

        # unresolved references
        defined_labels = set(defined.keys())
        for label, locations in references.items():
            if label in defined_labels:
                continue
            for file, line_no in locations:
                issues.append(ValidationIssue(
                    file=self._relative_file(file),
                    line=line_no,
                    code="unresolved_reference",
                    message=f"Reference @{label} has no matching label definition",
                    severity="error",
                    context=f"@{label}",
                ))

        return ValidationRunResult(
            name="unreferenced-labels",
            description="Detect unreferenced labels and unresolved references",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_inline_refs(self, root: Path, check_patterns: bool) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        yaml_option_inline = re.compile(r"^#\|\s*(fig-cap|tbl-cap|lst-cap|fig-alt):\s*.*`\{python\}")
        caption_syntax_inline = re.compile(r"^:\s+.*`\{python\}.*\{#(tbl|fig|lst)-")
        inline_fstring = re.compile(r"`\{python\}\s*f\"[^`]+`")
        inline_func_call = re.compile(r"`\{python\}\s*\w+\([^`]+\)`")

        for file in files:
            lines = self._read_text(file).splitlines()
            refs: List[Tuple[int, str]] = []
            compute_vars: Set[str] = set()
            in_cell = False

            for idx, line in enumerate(lines, 1):
                if CELL_START_PATTERN.match(line.strip()):
                    in_cell = True
                    continue
                if in_cell and CELL_END_PATTERN.match(line.strip()):
                    in_cell = False
                    continue
                if in_cell:
                    assign = ASSIGN_PATTERN.match(line.strip())
                    if assign:
                        compute_vars.add(assign.group(1))

                for match in INLINE_REF_PATTERN.finditer(line):
                    refs.append((idx, match.group(1)))

            for line_no, var in refs:
                if var not in compute_vars:
                    issues.append(ValidationIssue(
                        file=self._relative_file(file),
                        line=line_no,
                        code="undefined_inline_ref",
                        message=f"Inline reference `{var}` is not defined in python cells",
                        severity="error",
                        context=f"`{{python}} {var}`",
                    ))

            if check_patterns:
                in_grid = False
                for idx, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if LATEX_INLINE_PATTERN.search(line):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="latex_math_inline_python",
                            message="Inline Python inside LaTeX math can strip decimals",
                            severity="warning",
                            context=stripped[:160],
                        ))
                    if LATEX_ADJACENT_PATTERN.search(line):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="latex_adjacent_inline_python",
                            message="Inline Python adjacent to LaTeX operator is fragile",
                            severity="warning",
                            context=stripped[:160],
                        ))
                    if GRID_TABLE_SEP_PATTERN.match(stripped):
                        in_grid = True
                    elif in_grid and stripped and not stripped.startswith("|"):
                        in_grid = False
                    if in_grid and "`{python}" in line:
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="grid_table_inline_python",
                            message="Inline Python in grid tables is unsupported",
                            severity="error",
                            context=stripped[:160],
                        ))
                    if inline_fstring.search(line):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="inline_fstring",
                            message="Inline f-string should be precomputed in Python cell",
                            severity="warning",
                            context=stripped[:160],
                        ))
                    if inline_func_call.search(line):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="inline_function_call",
                            message="Inline function call should be precomputed in Python cell",
                            severity="warning",
                            context=stripped[:160],
                        ))
                    if yaml_option_inline.search(line):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="yaml_option_inline_python",
                            message="Inline Python in YAML fig/tbl/lst metadata will not render",
                            severity="error",
                            context=stripped[:160],
                        ))
                    if caption_syntax_inline.search(line):
                        issues.append(ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="caption_inline_python",
                            message="Inline Python in caption syntax will not render",
                            severity="error",
                            context=stripped[:160],
                        ))

        return ValidationRunResult(
            name="inline-refs",
            description="Validate inline Python refs and rendering hazard patterns",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Headers  (ported from manage_section_ids.py --verify)
    # ------------------------------------------------------------------

    def _run_headers(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        header_pat = re.compile(r"^(#{1,6})\s+(.+?)(?:\s*\{[^}]*\})?$")
        div_start_pat = re.compile(r"^:::\s*\{\.")
        div_end_pat = re.compile(r"^:::\s*$")
        code_block_pat = re.compile(r"^```[^`]*$")
        sec_id_pat = re.compile(r"\{#sec-[^}]+\}")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            in_div = False

            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if code_block_pat.match(stripped):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                if div_start_pat.match(stripped):
                    in_div = True
                    continue
                if div_end_pat.match(stripped):
                    in_div = False
                    continue
                if in_div:
                    continue

                match = header_pat.match(line)
                if not match:
                    continue

                # Extract existing attributes
                existing_attrs = ""
                if "{" in line:
                    attrs_start = line.find("{")
                    attrs_end = line.rfind("}")
                    if attrs_end > attrs_start:
                        existing_attrs = line[attrs_start : attrs_end + 1]

                if ".unnumbered" in existing_attrs:
                    continue

                if not sec_id_pat.search(line):
                    title = match.group(2).strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="missing_section_id",
                            message=f"Header missing section ID: {title}",
                            severity="error",
                            context=line.strip()[:160],
                        )
                    )

        return ValidationRunResult(
            name="headers",
            description="Verify section headers have {#sec-...} IDs",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Footnote Placement  (ported from check_forbidden_footnotes.py)
    # ------------------------------------------------------------------

    def _run_footnote_placement(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        fn_pat = re.compile(r"\[\^fn-[\w-]+\]")
        inline_fn_pat = re.compile(r"\^\[[^\]]+\]")
        table_sep_pat = re.compile(r"^\|[\s\-:+]+\|")

        for file in files:
            lines = self._read_text(file).splitlines()
            div_depth = 0
            div_start_line = 0

            for idx, line in enumerate(lines, 1):
                stripped = line.strip()

                # Track div nesting
                if re.match(r"^:{3,4}\s*\{", stripped) or re.match(r"^:{3,4}\s+\w", stripped):
                    div_depth += 1
                    if div_depth == 1:
                        div_start_line = idx
                elif re.match(r"^:{3,4}\s*$", stripped):
                    if div_depth > 0:
                        div_depth -= 1
                        if div_depth == 0:
                            div_start_line = 0

                # Check inline footnotes (always forbidden)
                for m in inline_fn_pat.finditer(line):
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="inline_footnote",
                            message=f"Inline footnote syntax; use [^fn-name] reference format",
                            severity="error",
                            context=m.group(0)[:80],
                        )
                    )

                footnotes = fn_pat.findall(line)
                if not footnotes:
                    continue

                # Table cell check
                if stripped.startswith("|") and stripped.count("|") >= 2 and not table_sep_pat.match(stripped):
                    for fn in footnotes:
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code="footnote_in_table",
                                message=f"Footnote {fn} in table cell",
                                severity="error",
                                context=stripped[:80],
                            )
                        )

                # YAML caption check
                if re.match(r"^\s*(fig-cap|tbl-cap):", line):
                    cap_type = "figure" if "fig-cap" in line else "table"
                    for fn in footnotes:
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code=f"footnote_in_{cap_type}_caption",
                                message=f"Footnote {fn} in {cap_type} caption",
                                severity="error",
                                context=stripped[:80],
                            )
                        )

                # Markdown caption check
                if re.match(r"^:\s*\*\*[^*]+\*\*:", line):
                    for fn in footnotes:
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code="footnote_in_markdown_caption",
                                message=f"Footnote {fn} in markdown caption",
                                severity="error",
                                context=stripped[:80],
                            )
                        )

                # Callout title check
                if re.match(r"^:{3,4}\s*\{.*title=", stripped):
                    title_match = re.search(r'title="([^"]*)"', line)
                    if title_match and fn_pat.search(title_match.group(1)):
                        for fn in fn_pat.findall(title_match.group(1)):
                            issues.append(
                                ValidationIssue(
                                    file=self._relative_file(file),
                                    line=idx,
                                    code="footnote_in_callout_title",
                                    message=f"Footnote {fn} in callout title (breaks LaTeX)",
                                    severity="error",
                                    context=stripped[:80],
                                )
                            )

                # Div block check
                if div_depth > 0 and div_start_line != idx:
                    for fn in footnotes:
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code="footnote_in_div",
                                message=f"Footnote {fn} inside div block (started line {div_start_line})",
                                severity="error",
                                context=stripped[:80],
                            )
                        )

        return ValidationRunResult(
            name="footnote-placement",
            description="Check footnotes in forbidden locations",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Footnote Refs  (ported from footnote_cleanup.py --validate)
    # ------------------------------------------------------------------

    def _run_footnote_refs(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        ref_pat = re.compile(r"\[\^([^]]+)\]")
        def_pat = re.compile(r"^\[\^([^]]+)\]:\s*(.+)$", re.MULTILINE)

        for file in files:
            content = self._read_text(file)
            lines = content.split("\n")

            # Collect definitions
            fn_defs: Dict[str, str] = {}
            for m in def_pat.finditer(content):
                fn_defs[m.group(1)] = m.group(2)

            # Collect references (excluding definition lines themselves)
            fn_refs: Dict[str, List[int]] = defaultdict(list)
            for line_num, line in enumerate(lines, 1):
                for m in ref_pat.finditer(line):
                    fn_id = m.group(1)
                    dm = def_pat.match(line)
                    if dm and dm.group(1) == fn_id:
                        continue  # definition line, not a reference
                    fn_refs[fn_id].append(line_num)

            # Undefined references
            for fn_id in sorted(set(fn_refs.keys()) - set(fn_defs.keys())):
                first_line = fn_refs[fn_id][0]
                issues.append(
                    ValidationIssue(
                        file=self._relative_file(file),
                        line=first_line,
                        code="undefined_footnote_ref",
                        message=f"Undefined footnote reference: [^{fn_id}]",
                        severity="error",
                        context=f"[^{fn_id}]",
                    )
                )

            # Unused definitions
            for fn_id in sorted(set(fn_defs.keys()) - set(fn_refs.keys())):
                def_line = self._line_for_token(content, f"[^{fn_id}]:")
                issues.append(
                    ValidationIssue(
                        file=self._relative_file(file),
                        line=def_line,
                        code="unused_footnote_def",
                        message=f"Unused footnote definition: [^{fn_id}]",
                        severity="warning",
                        context=f"[^{fn_id}]:",
                    )
                )

            # Duplicate definitions
            def_counts: Dict[str, int] = defaultdict(int)
            for line in lines:
                dm = re.match(r"^\[\^([^]]+)\]:", line)
                if dm:
                    def_counts[dm.group(1)] += 1
            for fn_id, count in def_counts.items():
                if count > 1:
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=self._line_for_token(content, f"[^{fn_id}]:"),
                            code="duplicate_footnote_def",
                            message=f"Duplicate footnote definition ({count}x): [^{fn_id}]",
                            severity="error",
                            context=f"[^{fn_id}]:",
                        )
                    )

            # Missing blank line before footnote definition
            # Pandoc requires footnote definitions to start a new block.
            # Without a preceding blank line, Pandoc treats the definition
            # as continuation text and renders [^fn-name] as literal text.
            fn_def_line_pat = re.compile(r"^\[\^[^\]]+\]:")
            for idx, line in enumerate(lines):
                if fn_def_line_pat.match(line) and idx > 0:
                    prev = lines[idx - 1]
                    if prev.strip():  # previous line is not blank
                        fn_match = re.match(r"^\[\^([^\]]+)\]:", line)
                        fn_id_str = fn_match.group(1) if fn_match else "?"
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx + 1,
                                code="footnote_missing_blank_line",
                                message=(
                                    f"Footnote definition [^{fn_id_str}] has no blank line before it — "
                                    f"Pandoc will not parse it as a footnote"
                                ),
                                severity="error",
                                context=f"prev: {prev.strip()[:60]}",
                            )
                        )

        return ValidationRunResult(
            name="footnote-refs",
            description="Validate footnote references and definitions",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Figures  (ported from check_figure_completeness.py)
    # ------------------------------------------------------------------

    def _run_figures(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        fig_id_pat = re.compile(r"\{#(fig-[a-zA-Z0-9_-]+)[\s}]")
        md_cap_pat = re.compile(r"!\[(.+?)\]\(")

        for file in files:
            lines = self._read_text(file).splitlines()
            seen_ids: Set[str] = set()

            # Pass 1: attribute-based figures
            for idx, line in enumerate(lines, 1):
                m = fig_id_pat.search(line)
                if not m:
                    continue
                fig_id = m.group(1)
                has_cap = bool(re.search(r'fig-cap="[^"]+', line))
                has_alt = bool(re.search(r'fig-alt="[^"]+', line))

                if "![" in line:
                    md_m = md_cap_pat.search(line)
                    if md_m and md_m.group(1).strip():
                        has_cap = True

                seen_ids.add(fig_id)
                missing = []
                if not has_cap:
                    missing.append("caption")
                if not has_alt:
                    missing.append("alt-text")
                if missing:
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="incomplete_figure",
                            message=f"Figure {fig_id} missing: {', '.join(missing)}",
                            severity="error",
                            context=line.strip()[:120],
                        )
                    )

            # Pass 2: code-cell figures
            in_code = False
            code_start = 0
            cell_opts: Dict[str, str] = {}
            for idx, line in enumerate(lines, 1):
                stripped = line.rstrip()
                if not in_code and re.match(r"^```\{(?:python|r|julia|ojs)", stripped):
                    in_code = True
                    code_start = idx
                    cell_opts = {}
                    continue
                if in_code and stripped == "```":
                    label = cell_opts.get("label", "")
                    if label.startswith("fig-") and label not in seen_ids:
                        cap_val = cell_opts.get("fig-cap", "")
                        alt_val = cell_opts.get("fig-alt", "")
                        missing = []
                        if not cap_val:
                            missing.append("caption")
                        if not alt_val:
                            missing.append("alt-text")
                        if missing:
                            issues.append(
                                ValidationIssue(
                                    file=self._relative_file(file),
                                    line=code_start,
                                    code="incomplete_figure",
                                    message=f"Figure {label} missing: {', '.join(missing)}",
                                    severity="error",
                                    context=f"code-cell figure {label}",
                                )
                            )
                        seen_ids.add(label)
                    in_code = False
                    cell_opts = {}
                    continue
                if in_code:
                    opt_m = re.match(r"^#\|\s*([\w-]+):\s*(.+)$", stripped)
                    if opt_m:
                        val = opt_m.group(2).strip().strip("\"'")
                        cell_opts[opt_m.group(1)] = val

        return ValidationRunResult(
            name="figures",
            description="Check figures have captions and alt-text",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Float Flow  (ported from figure_table_flow_audit.py)
    # ------------------------------------------------------------------

    def _run_float_flow(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        div_def_pat = re.compile(r":::\s*\{[^}]*#((?:fig|tbl)-[\w-]+)")
        img_def_pat = re.compile(r"!\[.*?\]\(.*?\)\s*\{[^}]*#((?:fig|tbl)-[\w-]+)")
        tbl_cap_pat = re.compile(r"^:\s+.*\{[^}]*#((?:fig|tbl)-[\w-]+)")
        ref_pat = re.compile(r"@((?:fig|tbl)-[\w-]+)")

        for file in files:
            lines = self._read_text(file).splitlines()
            defs: Dict[str, int] = {}
            refs: Dict[str, List[int]] = defaultdict(list)
            in_code = False
            in_float = False
            float_label: Optional[str] = None
            code_spans: List[Tuple[int, int]] = []
            code_start = 0
            cell_opts: Dict[str, str] = {}

            for idx, line in enumerate(lines, 1):
                stripped = line.rstrip()

                # Code block tracking
                if not in_code and re.match(r"^```\{", stripped):
                    in_code = True
                    code_start = idx
                    cell_opts = {}
                    continue
                if in_code and stripped == "```":
                    code_spans.append((code_start, idx))
                    label = cell_opts.get("label", "")
                    if label.startswith(("fig-", "tbl-")) and label not in defs:
                        defs[label] = code_start
                    in_code = False
                    cell_opts = {}
                    continue
                if in_code:
                    opt_m = re.match(r"^#\|\s*([\w-]+):\s*(.+)$", stripped)
                    if opt_m:
                        cell_opts[opt_m.group(1)] = opt_m.group(2).strip().strip("\"'")
                    continue

                # Attribute-based definitions
                for pat in [div_def_pat, img_def_pat, tbl_cap_pat]:
                    m = pat.search(line)
                    if m:
                        label = m.group(1)
                        if label not in defs:
                            defs[label] = idx
                        if pat == div_def_pat:
                            in_float = True
                            float_label = label

                # Track float block end
                if in_float:
                    ls = line.strip()
                    if ls.startswith(":::") and not ls.startswith("::: {"):
                        in_float = False
                        float_label = None

                # References
                if "fig-cap=" in line or "fig-alt=" in line:
                    continue
                for m in ref_pat.finditer(line):
                    label = m.group(1)
                    if in_float and label == float_label:
                        continue
                    refs[label].append(idx)

            # Evaluate status
            all_labels = set(defs.keys()) | set(refs.keys())
            for label in sorted(all_labels):
                def_line = defs.get(label)
                ref_lines = refs.get(label, [])
                first_ref = min(ref_lines) if ref_lines else None

                if not def_line:
                    continue  # XREF — informational, skip
                if not first_ref:
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=def_line,
                            code="orphan_float",
                            message=f"{'Figure' if label.startswith('fig-') else 'Table'} {label} defined but never referenced",
                            severity="warning",
                            context=label,
                        )
                    )
                    continue

                # Compute prose gap
                gap = def_line - first_ref
                code_lines = 0
                if gap > 0:
                    for cs, ce in code_spans:
                        os_ = max(first_ref, cs)
                        oe_ = min(def_line, ce)
                        if os_ <= oe_:
                            code_lines += oe_ - os_ + 1
                prose_gap = gap - code_lines

                if prose_gap > 30:
                    # Check closest reference
                    closest = min(ref_lines, key=lambda r: abs(def_line - r))
                    closest_gap = def_line - closest
                    if -5 <= closest_gap <= 30:
                        continue  # OK
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=def_line,
                            code="late_float",
                            message=f"{label} defined at L{def_line}, first referenced at L{first_ref} (too far after mention)",
                            severity="warning",
                            context=label,
                        )
                    )
                elif prose_gap < -5:
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=def_line,
                            code="early_float",
                            message=f"{label} defined at L{def_line}, first referenced at L{first_ref} (appears before mention)",
                            severity="warning",
                            context=label,
                        )
                    )

        return ValidationRunResult(
            name="float-flow",
            description="Audit figure/table placement relative to first reference",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Indexes  (ported from check_index_placement.py)
    # ------------------------------------------------------------------

    def _run_indexes(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        checks = [
            ("index_on_heading", re.compile(r"^#{1,6}\s+.*\\index\{"), "\\index{} on same line as heading"),
            ("index_before_div", re.compile(r"\\index\{[^}]*\}:::"), "\\index{} directly before ::: (div/callout)"),
            ("index_after_div", re.compile(r"^::+\s+\{[^}]*\}\s*\\index\{"), "\\index{} on same line as div/callout"),
            ("index_before_footnote", re.compile(r"^\\index\{[^}]*\}.*\[\^[^\]]+\]:"), "\\index{} before footnote definition"),
        ]

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                if line.strip().startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue

                for code, pattern, message in checks:
                    # Skip fig-cap lines for index_after_div
                    if code == "index_after_div" and "fig-cap=" in line:
                        continue
                    if pattern.search(line):
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code=code,
                                message=message,
                                severity="error",
                                context=line.strip()[:120],
                            )
                        )

        return ValidationRunResult(
            name="indexes",
            description="Check LaTeX \\index{} placement",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Rendering  (ported from check_render_patterns.py)
    # ------------------------------------------------------------------

    def _run_rendering(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        regex_checks = [
            ("missing_opening_backtick", re.compile(r"(?<!`)(\{python\}\s+\w+`)"), "Missing opening backtick on inline Python", "error"),
            ("dollar_before_python", re.compile(r"\$\{python\}\s+\w+`"), "Dollar sign instead of backtick before {python}", "error"),
            ("quad_asterisks", re.compile(r"\*{4,}"), "Quad asterisks — likely malformed bold/italic", "warning"),
            ("footnote_in_table", re.compile(r"^\|.*\[\^fn-[^\]]+\].*\|"), "Footnote in table cell — may break PDF", "warning"),
            ("double_dollar_python", re.compile(r"\$\$[^$]*`\{python\}"), "Inline Python in display math", "error"),
            # Currency: unescaped $ before number can be parsed as math. Use \$ for currency (see book-prose.md).
            # Match: $1,000 (comma), $4.00 (decimal), $50 million/billion/etc.
            # Exclude: $1.5 \times (math), $0.5$ (inline math), $4.6 / (division).
            ("unescaped_currency", re.compile(
                r"(?<!\\)\$[0-9]{1,3}(?:,[0-9]{3})+(?=\s(?!\s*\\times)|,[0-9]|\)|$)"  # $1,000, exclude $25,000 \times
                r"|(?<!\\)\$[0-9]+\.[0-9]+(?=\s(?!\s*\\times)(?!\s*/)(?!\s*-)(?!\s*\+)(?!\s*\\ll)|,[0-9]|\)|$|/)(?!\\$)"  # $4.00, exclude math
                r"|(?<!\\)\$[0-9]+(?=\s+(?:million|billion|thousand|M|B|K|per|each|/))"  # $50 million
            ), "Unescaped dollar before number — use \\$ for currency", "warning"),
        ]

        grid_sep_pat = re.compile(r"^\+[-:=+]+\+$")
        math_span_pat = re.compile(r"(?<!\\)\$(?!\$)(?!`)(.+?)(?<!\\)\$")

        # Lowercase 'x' used as multiplication in prose (should be $\times$).
        # Matches: `...`x word, NUMx word — but NOT hex (0x61), code, fig-alt, or \index.
        # The pattern requires a lowercase letter after x+space, which naturally
        # excludes hardware counts like "8x A100" (uppercase after x).
        lowercase_x_mult_pat = re.compile(
            r"""`x\s+[a-z]"""    # `...`x word  (after inline python)
            r"""|"""
            r"""\dx\s+[a-z]"""   # Nx word  (digit then x then lowercase)
        )
        # Hex literal pattern to exclude matches like 0x61, 0xff
        hex_literal_pat = re.compile(r"0x[0-9a-fA-F]")
        # fig-alt lines to skip
        fig_alt_pat = re.compile(r'fig-alt\s*=\s*"')

        for file in files:
            lines = self._read_text(file).splitlines()
            in_grid = False
            in_code = False

            for idx, line in enumerate(lines, 1):
                stripped = line.strip()

                # Code block tracking
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue

                # Grid table tracking
                if grid_sep_pat.match(stripped):
                    in_grid = True
                elif in_grid and not stripped.startswith("|") and not grid_sep_pat.match(stripped) and stripped:
                    in_grid = False

                if in_grid and "`{python}" in line:
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="grid_table_python",
                            message="Grid table with inline Python — convert to pipe table",
                            severity="error",
                            context=stripped[:120],
                        )
                    )

                # Python inside $...$ math
                for m in math_span_pat.finditer(line):
                    inner = m.group(1)
                    if "{python}" not in inner:
                        continue
                    inner_clean = re.sub(r"\^\{[^}]*`\{python\}[^`]*`[^}]*\}", "", inner)
                    if "{python}" in inner_clean:
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code="python_in_dollar_math",
                                message="Inline Python inside $...$ math block",
                                severity="error",
                                context=m.group(0)[:120],
                            )
                        )

                # Lowercase 'x' used as multiplication in prose
                # Skip fig-alt lines and index entries
                if not fig_alt_pat.search(line) and not stripped.startswith("\\index"):
                    for rm in lowercase_x_mult_pat.finditer(line):
                        # Exclude hex literals like 0x61, 0xff
                        ctx_start = max(0, rm.start() - 1)
                        if hex_literal_pat.match(line[ctx_start : rm.end()]):
                            continue
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code="lowercase_x_multiplication",
                                message="Lowercase 'x' used as multiplication — use $\\times$ instead",
                                severity="warning",
                                context=rm.group(0)[:120],
                            )
                        )

                # Standard regex checks
                for code, pattern, message, severity in regex_checks:
                    for rm in pattern.finditer(line):
                        issues.append(
                            ValidationIssue(
                                file=self._relative_file(file),
                                line=idx,
                                code=code,
                                message=message,
                                severity=severity,
                                context=rm.group(0)[:120],
                            )
                        )

        return ValidationRunResult(
            name="rendering",
            description="Check for problematic rendering patterns",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def _run_python_echo(self, root: Path) -> ValidationRunResult:
        """Ensure every ```{python} block has #| echo: false (code must not appear in output)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        block_start_re = re.compile(r"^```\{python\}")
        block_end_re = re.compile(r"^```\s*$")
        # Quarto chunk option: #| echo: false (with optional whitespace)
        echo_false_re = re.compile(r"#\|\s*echo\s*:\s*false", re.IGNORECASE)

        for file in files:
            lines = self._read_text(file).splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                if not block_start_re.match(line):
                    i += 1
                    continue
                start_line = i + 1
                found_echo_false = False
                j = i + 1
                # Scan option lines: #| key: value, or blank, until we hit code or closing ```
                while j < len(lines):
                    next_line = lines[j]
                    if block_end_re.match(next_line):
                        break
                    stripped = next_line.strip()
                    if echo_false_re.search(stripped):
                        found_echo_false = True
                        break
                    # Option line or blank — keep scanning
                    if stripped.startswith("#|") or not stripped:
                        j += 1
                        continue
                    # Non-option line (actual code or comment) — options are done
                    break
                if not found_echo_false:
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=start_line,
                            code="python_missing_echo_false",
                            message="Python block must include #| echo: false — code must not appear in rendered output",
                            severity="error",
                            context="Add #| echo: false as first line after ```{python}",
                        )
                    )
                # Advance past this block to the line after closing ```
                k = j
                while k < len(lines) and not block_end_re.match(lines[k]):
                    k += 1
                i = k + 1

        return ValidationRunResult(
            name="python-echo",
            description="Check Python blocks have echo: false",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Dropcaps  (ported from validate_dropcap_compat.py)
    # ------------------------------------------------------------------

    def _run_dropcaps(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        chapter_hdr = re.compile(r"^#\s+[^#].*\{#sec-")
        numbered_h2 = re.compile(r"^##\s+[^#]")
        unnumbered_h2 = re.compile(r"^##\s+.*\{.*\.unnumbered.*\}")
        starts_xref = re.compile(r"^\s*@(sec|fig|tbl|lst|eq)-")
        starts_link = re.compile(r"^\s*\[")
        starts_inline = re.compile(r"^\s*`")
        yaml_fence = re.compile(r"^---\s*$")
        code_fence = re.compile(r"^```")
        div_fence = re.compile(r"^:::")
        blank = re.compile(r"^\s*$")
        html_comment = re.compile(r"^\s*<!--")
        raw_latex = re.compile(r"^\s*\\")
        list_item = re.compile(r"^\s*[-*+]|\s*\d+\.")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_fm = False
            in_code = False
            in_div = 0
            found_chapter = False
            found_h2 = False

            for idx, line in enumerate(lines, 1):
                if idx == 1 and yaml_fence.match(line):
                    in_fm = True
                    continue
                if in_fm:
                    if yaml_fence.match(line):
                        in_fm = False
                    continue
                if code_fence.match(line):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                if div_fence.match(line):
                    stripped = line.strip()
                    if stripped == ":::":
                        in_div = max(0, in_div - 1)
                    elif stripped.startswith(":::"):
                        in_div += 1
                    continue
                if in_div > 0:
                    continue

                if chapter_hdr.match(line):
                    found_chapter = True
                    found_h2 = False
                    continue
                if not found_chapter:
                    continue
                if numbered_h2.match(line) and not unnumbered_h2.match(line):
                    if not found_h2:
                        found_h2 = True
                    continue
                if not found_h2:
                    continue
                if blank.match(line) or html_comment.match(line) or raw_latex.match(line) or list_item.match(line):
                    continue
                if line.strip().startswith("#"):
                    continue

                # First paragraph line
                if starts_xref.match(line):
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="dropcap_crossref",
                            message="Drop cap paragraph starts with cross-reference",
                            severity="error",
                            context=line.strip()[:120],
                        )
                    )
                elif starts_link.match(line):
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="dropcap_link",
                            message="Drop cap paragraph starts with markdown link",
                            severity="error",
                            context=line.strip()[:120],
                        )
                    )
                elif starts_inline.match(line):
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="dropcap_inline",
                            message="Drop cap paragraph starts with inline code",
                            severity="error",
                            context=line.strip()[:120],
                        )
                    )
                # Only check first paragraph per file
                break

        return ValidationRunResult(
            name="dropcaps",
            description="Validate drop cap compatibility",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Parts  (ported from validate_part_keys.py)
    # ------------------------------------------------------------------

    def _run_parts(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        part_key_pat = re.compile(r"\\part\{key:([^}]+)\}")

        # Load summaries
        summaries_keys: Set[str] = set()
        possible_paths = [
            self.config_manager.book_dir / "contents" / "parts" / "summaries.yml",
            self.config_manager.book_dir / "contents" / "vol1" / "parts" / "summaries.yml",
            self.config_manager.book_dir / "contents" / "vol2" / "parts" / "summaries.yml",
        ]

        try:
            import yaml
        except ImportError:
            return ValidationRunResult(
                name="parts",
                description="Validate part keys (skipped — pyyaml not installed)",
                files_checked=0,
                issues=[],
                elapsed_ms=int((time.time() - start) * 1000),
            )

        for yml_path in possible_paths:
            if yml_path.exists():
                try:
                    data = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
                    for part in data.get("parts", []):
                        if "key" in part:
                            summaries_keys.add(part["key"].lower().replace("_", "").replace("-", ""))
                except Exception:
                    pass

        if not summaries_keys:
            # No summaries found — skip gracefully
            return ValidationRunResult(
                name="parts",
                description="Validate part keys (skipped — no summaries.yml found)",
                files_checked=0,
                issues=[],
                elapsed_ms=int((time.time() - start) * 1000),
            )

        for file in files:
            content = self._read_text(file)
            for m in part_key_pat.finditer(content):
                key = m.group(1)
                norm = key.lower().replace("_", "").replace("-", "")
                if norm not in summaries_keys:
                    line_no = content[: m.start()].count("\n") + 1
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=line_no,
                            code="invalid_part_key",
                            message=f"Part key '{key}' not found in summaries.yml",
                            severity="error",
                            context=m.group(0),
                        )
                    )

        return ValidationRunResult(
            name="parts",
            description="Validate \\part{{key:...}} against summaries.yml",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Heading levels  (detect skipped heading levels)
    # ------------------------------------------------------------------

    def _run_heading_levels(self, root: Path) -> ValidationRunResult:
        """Detect heading level skips outside of div contexts.

        Headings inside Quarto divs (callouts, panels, columns, etc.) are
        in a separate nesting context and are excluded from the hierarchy
        check.  Only headings at the top-level (div depth 0) are compared
        against each other.
        """
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        heading_pat = re.compile(r"^(#{1,6})\s+")
        code_fence = re.compile(r"^```")
        yaml_fence = re.compile(r"^---\s*$")
        # Div open: ::: or :::: (with optional class/id)
        div_open_pat = re.compile(r"^(:{3,})\s*\{")
        # Div close: bare ::: or :::: on its own line
        div_close_pat = re.compile(r"^(:{3,})\s*$")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            in_yaml = False
            prev_level = 0
            div_depth = 0

            for idx, line in enumerate(lines, 1):
                stripped = line.strip()

                # Track YAML front matter
                if idx == 1 and yaml_fence.match(line):
                    in_yaml = True
                    continue
                if in_yaml:
                    if yaml_fence.match(line):
                        in_yaml = False
                    continue

                # Track code blocks
                if code_fence.match(stripped):
                    in_code = not in_code
                    continue
                if in_code:
                    continue

                # Track div nesting depth
                if div_open_pat.match(stripped):
                    div_depth += 1
                    continue
                if div_close_pat.match(stripped) and div_depth > 0:
                    div_depth -= 1
                    continue

                # Skip headings inside divs — they're in a nested context
                if div_depth > 0:
                    continue

                m = heading_pat.match(line)
                if not m:
                    continue

                level = len(m.group(1))

                # Only flag if we skip a level going deeper
                # (e.g., ## -> #### skips ###)
                if prev_level > 0 and level > prev_level + 1:
                    skipped = ", ".join(
                        f"H{i}" for i in range(prev_level + 1, level)
                    )
                    heading_text = line.lstrip("#").strip()
                    # Truncate at { to remove attributes
                    if "{" in heading_text:
                        heading_text = heading_text[: heading_text.index("{")].strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="heading_level_skip",
                            message=f"Heading jumps from H{prev_level} to H{level} (skips {skipped})",
                            severity="warning",
                            context=heading_text[:80],
                        )
                    )

                prev_level = level

        return ValidationRunResult(
            name="heading-levels",
            description="Detect skipped heading levels (e.g., ## to ####)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Duplicate consecutive words  (detect "the the", "is is", etc.)
    # ------------------------------------------------------------------

    _DUPE_WORD_PAT = re.compile(
        r"\b(\w{2,})\s+\1\b",
        re.IGNORECASE,
    )
    # Known false positives: intentional repetitions
    _DUPE_WORD_ALLOW = frozenset({
        "had", "that", "do", "bye", "bla", "cha", "go",
        "log",  # "log log n" is valid math
    })

    def _run_duplicate_words(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        code_fence = re.compile(r"^```")
        yaml_fence = re.compile(r"^---\s*$")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            in_yaml = False

            for idx, line in enumerate(lines, 1):
                # Track YAML front matter
                if idx == 1 and yaml_fence.match(line):
                    in_yaml = True
                    continue
                if in_yaml:
                    if yaml_fence.match(line):
                        in_yaml = False
                    continue

                # Skip code blocks
                if code_fence.match(line.strip()):
                    in_code = not in_code
                    continue
                if in_code:
                    continue

                # Skip HTML comments, raw LaTeX, div fences, HTML tags
                stripped = line.strip()
                if stripped.startswith("<!--") or stripped.startswith("\\") or stripped.startswith(":::"):
                    continue
                if stripped.startswith("<") and not stripped.startswith("<http"):
                    continue
                # Skip lines that are mostly attributes/metadata
                if stripped.startswith("#|") or stripped.startswith("%%|"):
                    continue

                for m in self._DUPE_WORD_PAT.finditer(line):
                    word = m.group(1).lower()
                    if word in self._DUPE_WORD_ALLOW:
                        continue
                    # Skip if inside a LaTeX command or attribute
                    before = line[: m.start()]
                    if before.rstrip().endswith("\\") or "{" in line[m.start() : m.end() + 5]:
                        continue
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="duplicate_word",
                            message=f'Duplicate word: "{m.group(1)} {m.group(1)}"',
                            severity="warning",
                            context=line.strip()[:120],
                        )
                    )

        return ValidationRunResult(
            name="duplicate-words",
            description="Detect duplicate consecutive words (typos)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Images  (ported from validate_image_references.py)
    # ------------------------------------------------------------------

    def _run_images(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        img_pat = re.compile(r"!\[(?:[^\]]|\[[^\]]*\])*\]\(([^)]+)\)(?:\{[^}]*\})?")
        valid_exts = {".png", ".jpg", ".jpeg", ".gif", ".svg"}

        for file in files:
            content = self._read_text(file)
            for m in img_pat.finditer(content):
                img_path = m.group(1).strip()
                if img_path.startswith(("http://", "https://")):
                    continue
                ext = Path(img_path).suffix.lower()
                if ext not in valid_exts:
                    continue

                resolved = (file.parent / img_path).resolve()
                line_no = content[: m.start()].count("\n") + 1

                if not resolved.exists():
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=line_no,
                            code="missing_image",
                            message=f"Image not found: {img_path}",
                            severity="error",
                            context=img_path,
                        )
                    )
                else:
                    # Case check
                    try:
                        actual = self._realcase(str(resolved))
                        if str(resolved) != actual:
                            issues.append(
                                ValidationIssue(
                                    file=self._relative_file(file),
                                    line=line_no,
                                    code="image_case_mismatch",
                                    message=f"Image case mismatch: ref='{Path(str(resolved)).name}' disk='{Path(actual).name}'",
                                    severity="error",
                                    context=img_path,
                                )
                            )
                    except (FileNotFoundError, OSError):
                        pass

        return ValidationRunResult(
            name="images",
            description="Validate image references exist on disk",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    @staticmethod
    def _realcase(path: str) -> str:
        """Resolve actual case of a path on disk."""
        dirname, basename = os.path.split(path)
        if dirname == path:
            return dirname
        dirname = ValidateCommand._realcase(dirname)
        norm_base = os.path.normcase(basename)
        try:
            for child in os.listdir(dirname):
                if os.path.normcase(child) == norm_base:
                    return os.path.join(dirname, child)
        except OSError:
            pass
        return path

    # ------------------------------------------------------------------
    # Self-referential sections  (ported from check_self_referential_sections.py)
    # ------------------------------------------------------------------

    def _run_self_referential(self, root: Path) -> ValidationRunResult:
        """Detect sections that reference themselves, their parent, or child."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        heading_pat = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+\{#([^}]+)\})?$")
        ref_pat = re.compile(r"@(sec-[a-zA-Z0-9-]+)")

        for file in files:
            lines = self._read_text(file).splitlines()

            # Build heading hierarchy
            headings: List[Dict] = []
            parent_stack: Dict[int, Dict] = {}

            for idx, line in enumerate(lines, 1):
                m = heading_pat.match(line)
                if not m:
                    continue
                level = len(m.group(1))
                title = m.group(2).strip()
                sec_id = m.group(3)
                parent_id = None
                for plevel in range(level - 1, 0, -1):
                    if plevel in parent_stack:
                        parent_id = parent_stack[plevel].get("id")
                        break
                hd = {"level": level, "title": title, "id": sec_id,
                      "line": idx, "parent_id": parent_id}
                headings.append(hd)
                parent_stack[level] = hd
                parent_stack = {k: v for k, v in parent_stack.items() if k <= level}

            # Build section map and children map
            section_map: Dict[str, Dict] = {}
            children_map: Dict[str, List[str]] = defaultdict(list)
            for hd in headings:
                if hd["id"]:
                    section_map[hd["id"]] = hd
                    if hd["parent_id"]:
                        children_map[hd["parent_id"]].append(hd["id"])

            # Check references
            for idx, line in enumerate(lines, 1):
                for m in ref_pat.finditer(line):
                    ref_id = m.group(1)
                    # Find which section this line belongs to
                    current = None
                    for hd in headings:
                        if hd["line"] <= idx:
                            current = hd
                        else:
                            break
                    if not current or not current["id"]:
                        continue

                    cur_id = current["id"]
                    if ref_id == cur_id:
                        issues.append(ValidationIssue(
                            file=self._relative_file(file), line=idx,
                            code="self_reference",
                            message=f"Section '{current['title']}' references itself (@{ref_id})",
                            severity="warning",
                            context=line.strip()[:120],
                        ))
                    elif current["parent_id"] == ref_id:
                        parent = section_map.get(ref_id)
                        ptitle = parent["title"] if parent else ref_id
                        issues.append(ValidationIssue(
                            file=self._relative_file(file), line=idx,
                            code="parent_reference",
                            message=f"Section '{current['title']}' references its parent '{ptitle}' (@{ref_id})",
                            severity="warning",
                            context=line.strip()[:120],
                        ))
                    elif ref_id in children_map.get(cur_id, []):
                        child = section_map.get(ref_id)
                        ctitle = child["title"] if child else ref_id
                        issues.append(ValidationIssue(
                            file=self._relative_file(file), line=idx,
                            code="child_reference",
                            message=f"Section '{current['title']}' references its child '{ctitle}' (@{ref_id})",
                            severity="warning",
                            context=line.strip()[:120],
                        ))

        return ValidationRunResult(
            name="self-referential",
            description="Detect self-referential section references",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Figure label underscores  (ported from check_fig_references.py)
    # ------------------------------------------------------------------

    def _run_fig_label_underscores(self, root: Path) -> ValidationRunResult:
        """Find figure references containing underscores (invalid in Quarto)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        fig_ref_pat = re.compile(r"(?:\{#|@)fig-([^}\s]+)")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                if line.strip().startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                for m in fig_ref_pat.finditer(line):
                    label_suffix = m.group(1)
                    if "_" in label_suffix:
                        issues.append(ValidationIssue(
                            file=self._relative_file(file), line=idx,
                            code="fig_label_underscore",
                            message=f"Figure label contains underscore: fig-{label_suffix} (use hyphens)",
                            severity="error",
                            context=line.strip()[:120],
                        ))

        return ValidationRunResult(
            name="fig-labels",
            description="Detect underscores in figure labels",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # ASCII check  (ported from check_ascii.py)
    # ------------------------------------------------------------------

    def _run_ascii(self, root: Path) -> ValidationRunResult:
        """Find non-ASCII Unicode characters in QMD files."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        non_ascii_pat = re.compile(r"[^\x00-\x7F]")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                # Skip LaTeX raw blocks and HTML comments
                if stripped.startswith("\\") or stripped.startswith("<!--"):
                    continue
                for m in non_ascii_pat.finditer(line):
                    char = m.group(0)
                    col = m.start()
                    context = line[max(0, col - 10):min(len(line), col + 10)]
                    issues.append(ValidationIssue(
                        file=self._relative_file(file), line=idx,
                        code="non_ascii",
                        message=f"Non-ASCII character '{char}' (U+{ord(char):04X})",
                        severity="warning",
                        context=context.strip(),
                    ))

        return ValidationRunResult(
            name="ascii",
            description="Detect non-ASCII characters in QMD files",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Cross-chapter footnote duplicates  (ported from audit_footnotes_cross_chapter.py)
    # ------------------------------------------------------------------

    def _run_footnote_cross_chapter(self, root: Path) -> ValidationRunResult:
        """Find duplicate footnote IDs across chapters."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        fn_def_pat = re.compile(r"\[\^(fn-[^\]]+)\]:\s*(.+?)(?=\n\n|\n\[\^|\Z)", re.DOTALL)

        # Collect all footnotes by ID
        footnotes_by_id: Dict[str, List[Tuple[Path, str]]] = defaultdict(list)

        for file in files:
            content = self._read_text(file)
            for m in fn_def_pat.finditer(content):
                fn_id = m.group(1)
                fn_content = " ".join(m.group(2).split())[:200]
                footnotes_by_id[fn_id].append((file, fn_content))

        # Report duplicates
        for fn_id, occurrences in footnotes_by_id.items():
            if len(occurrences) <= 1:
                continue
            for file, content in occurrences:
                line_no = self._line_for_token(self._read_text(file), f"[^{fn_id}]:")
                issues.append(ValidationIssue(
                    file=self._relative_file(file), line=line_no,
                    code="cross_chapter_footnote",
                    message=f"Footnote [^{fn_id}] also defined in {len(occurrences) - 1} other file(s)",
                    severity="warning",
                    context=content[:80],
                ))

        return ValidationRunResult(
            name="cross-chapter-footnotes",
            description="Detect duplicate footnote IDs across chapters",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    # ------------------------------------------------------------------
    # Table content validation  (delegated to validate_tables.py)
    # ------------------------------------------------------------------

    def _run_table_content(self, root: Path) -> ValidationRunResult:
        """Validate grid table content (bare pipes, fracs, HTML entities, etc.)."""
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "content" / "validate_tables.py"
        )
        args = ["-d", str(root)]
        return self._delegate_script(script, args, "table-content")

    # ------------------------------------------------------------------
    # Spelling checks  (delegated to check_prose_spelling.py / check_tikz_spelling.py)
    # ------------------------------------------------------------------

    def _run_spelling_prose(self, root: Path) -> ValidationRunResult:
        """Spell check prose text (requires aspell)."""
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "content" / "check_prose_spelling.py"
        )
        return self._delegate_script(script, [str(root)], "spelling-prose")

    def _run_spelling_tikz(self, root: Path) -> ValidationRunResult:
        """Spell check TikZ diagram text (requires aspell)."""
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "content" / "check_tikz_spelling.py"
        )
        # check_tikz_spelling.py auto-scans from repo root, so pass no args
        return self._delegate_script(script, [], "spelling-tikz")

    # ------------------------------------------------------------------
    # EPUB validation  (delegated to validate_epub.py)
    # ------------------------------------------------------------------

    def _run_epub(self, root: Path) -> ValidationRunResult:
        """Validate EPUB file structure and content."""
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "utilities" / "validate_epub.py"
        )
        # Find EPUB files in build output directories
        book_dir = Path(__file__).resolve().parent.parent.parent
        epub_files = list(book_dir.rglob("*.epub"))
        if not epub_files:
            return ValidationRunResult(
                name="epub", description="EPUB validation (no .epub files found)",
                files_checked=0, issues=[], elapsed_ms=0,
            )
        # Validate the most recent EPUB
        epub_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return self._delegate_script(script, ["--quick", str(epub_files[0])], "epub")

    # ------------------------------------------------------------------
    # Source citation validation  (delegated to manage_sources.py)
    # ------------------------------------------------------------------

    def _run_sources(self, root: Path) -> ValidationRunResult:
        """Validate source citations (asterisk sources, formatting, etc.)."""
        import subprocess as _sp
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "utilities" / "manage_sources.py"
        )
        # manage_sources.py expects to be run from the quarto root (where contents/ lives)
        quarto_dir = Path(__file__).resolve().parent.parent.parent / "quarto"
        t0 = time.time()
        cmd = ["python3", str(script), "--problems"]
        try:
            result = _sp.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(quarto_dir))
            elapsed = int((time.time() - t0) * 1000)
            if result.returncode == 0:
                return ValidationRunResult(
                    name="sources", description="Source citation validation",
                    files_checked=0, issues=[], elapsed_ms=elapsed,
                )
            output = (result.stdout + result.stderr).strip()
            return ValidationRunResult(
                name="sources", description="Source citation validation",
                files_checked=0, elapsed_ms=elapsed,
                issues=[ValidationIssue(
                    file="(script output)", line=0, code="sources",
                    message=output[:500] if output else f"Script exited with code {result.returncode}",
                    severity="error",
                )],
            )
        except FileNotFoundError:
            elapsed = int((time.time() - t0) * 1000)
            return ValidationRunResult(
                name="sources", description="Source citation validation",
                files_checked=0, elapsed_ms=elapsed,
                issues=[ValidationIssue(
                    file=str(script), line=0, code="sources",
                    message=f"Script not found: {script}", severity="error",
                )],
            )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _line_for_token(self, content: str, token: str) -> int:
        index = content.find(token)
        if index < 0:
            return 1
        return content[:index].count("\n") + 1

    def _print_human_summary(self, summary: Dict[str, Any], verbose: bool = False) -> None:
        runs = summary["runs"]
        total = summary["total_issues"]
        status = summary["status"]

        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Check", style="cyan")
        table.add_column("Files", style="dim")
        table.add_column("Issues", style="yellow")
        table.add_column("Elapsed", style="dim")
        table.add_column("Status", style="white")
        for run in runs:
            table.add_row(
                run["name"],
                str(run["files_checked"]),
                str(run["issue_count"]),
                f'{run["elapsed_ms"]}ms',
                "PASS" if run["passed"] else "FAIL",
            )
        console.print(Panel(table, title="Binder Check Summary", border_style="cyan"))

        if total == 0:
            console.print("[green]✅ All validation checks passed.[/green]")
            return

        # Count errors vs warnings across all runs
        total_errors = 0
        total_warnings = 0
        for run in runs:
            for issue in run["issues"]:
                if issue["severity"] == "error":
                    total_errors += 1
                else:
                    total_warnings += 1

        for run in runs:
            if run["issue_count"] == 0:
                continue
            run_errors = sum(1 for i in run["issues"] if i["severity"] == "error")
            run_warnings = run["issue_count"] - run_errors
            parts = []
            if run_errors:
                parts.append(f"{run_errors} error(s)")
            if run_warnings:
                parts.append(f"{run_warnings} warning(s)")
            label = ", ".join(parts)
            color = "bold red" if run_errors else "bold yellow"
            console.print(f"[{color}]{run['name']}[/{color}] ({label})")
            for issue in run["issues"][:30]:
                line = issue["line"]
                file = issue["file"]
                msg = issue["message"]
                sev = issue["severity"]
                sev_icon = "❌" if sev == "error" else "⚠️"
                console.print(f"  {sev_icon} {file}:{line} {msg}")
                if verbose and issue.get("context"):
                    console.print(f"     [dim]{issue['context']}[/dim]")
            if run["issue_count"] > 30:
                console.print(f"  [dim]... {run['issue_count'] - 30} more[/dim]")
            console.print()

        if status == "failed":
            console.print(f"[red]❌ Validation failed with {total_errors} error(s).[/red]")
        elif total_warnings > 0:
            console.print(f"[yellow]⚠️  Passed with {total_warnings} warning(s).[/yellow]")

    def _emit(self, as_json: bool, payload: Dict[str, Any], failed: bool) -> None:
        if as_json:
            print(json.dumps(payload, indent=2))
            return
        if failed:
            console.print(f"[red]{payload.get('message', 'Operation failed')}[/red]")
        else:
            console.print(f"[green]{payload.get('message', 'Operation succeeded')}[/green]")

    # ------------------------------------------------------------------
    # Delegated checks (call existing scripts via subprocess)
    # ------------------------------------------------------------------

    @staticmethod
    def _delegate_script(script_path: Path, args: List[str], run_name: str) -> ValidationRunResult:
        """Run an external script and convert its exit code to a ValidationRunResult."""
        import subprocess
        t0 = time.time()
        cmd = ["python3", str(script_path)] + args
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            elapsed = int((time.time() - t0) * 1000)
            if result.returncode == 0:
                return ValidationRunResult(
                    name=run_name, description=run_name,
                    files_checked=0, issues=[], elapsed_ms=elapsed,
                )
            # Script failed — report its output as a single error
            output = (result.stdout + result.stderr).strip()
            return ValidationRunResult(
                name=run_name, description=run_name,
                files_checked=0, elapsed_ms=elapsed,
                issues=[ValidationIssue(
                    file="(script output)", line=0, code=run_name,
                    message=output[:500] if output else f"Script exited with code {result.returncode}",
                    severity="error",
                )],
            )
        except FileNotFoundError:
            elapsed = int((time.time() - t0) * 1000)
            return ValidationRunResult(
                name=run_name, description=run_name,
                files_checked=0, elapsed_ms=elapsed,
                issues=[ValidationIssue(
                    file=str(script_path), line=0, code=run_name,
                    message=f"Script not found: {script_path}", severity="error",
                )],
            )
        except subprocess.TimeoutExpired:
            elapsed = int((time.time() - t0) * 1000)
            return ValidationRunResult(
                name=run_name, description=run_name,
                files_checked=0, elapsed_ms=elapsed,
                issues=[ValidationIssue(
                    file=str(script_path), line=0, code=run_name,
                    message="Script timed out after 120s", severity="error",
                )],
            )

    def _run_grid_tables(self, root: Path) -> ValidationRunResult:
        """Check for grid tables (should be converted to pipe tables)."""
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "utilities" / "convert_grid_to_pipe_tables.py"
        )
        qmd_files = [str(f) for f in sorted(root.rglob("*.qmd"))]
        if not qmd_files:
            return ValidationRunResult(name="grid-tables", issues=[])
        return self._delegate_script(script, ["--check"] + qmd_files, "grid-tables")

    def _run_image_formats(self, root: Path) -> ValidationRunResult:
        """Validate image file formats using Pillow."""
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "images" / "manage_images.py"
        )
        image_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.gif"):
            image_files.extend(str(f) for f in sorted(root.rglob(ext)))
        if not image_files:
            return ValidationRunResult(name="image-formats", issues=[])
        return self._delegate_script(script, image_files, "image-formats")

    def _run_external_images(self, root: Path) -> ValidationRunResult:
        """Check for external image URLs in QMD files."""
        script = (
            Path(__file__).resolve().parent.parent.parent
            / "tools" / "scripts" / "images" / "manage_external_images.py"
        )
        return self._delegate_script(
            script, ["--validate", str(root)], "external-images"
        )

    def _run_json_syntax(self, root: Path) -> ValidationRunResult:
        """Validate JSON file syntax."""
        t0 = time.time()
        json_files = sorted(root.rglob("*.json"))
        if not json_files:
            return ValidationRunResult(
                name="json-syntax", description="Validate JSON file syntax",
                files_checked=0, issues=[], elapsed_ms=0,
            )
        issues: List[ValidationIssue] = []
        for fpath in json_files:
            try:
                with open(fpath, "r") as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                issues.append(ValidationIssue(
                    file=str(fpath), line=e.lineno or 0, code="json-syntax",
                    message=f"Invalid JSON: {e.msg}", severity="error",
                ))
            except Exception as e:
                issues.append(ValidationIssue(
                    file=str(fpath), line=0, code="json-syntax",
                    message=f"Cannot read: {e}", severity="error",
                ))
        elapsed = int((time.time() - t0) * 1000)
        return ValidationRunResult(
            name="json-syntax", description="Validate JSON file syntax",
            files_checked=len(json_files), issues=issues, elapsed_ms=elapsed,
        )

    def _run_unit_tests(self, root: Path) -> ValidationRunResult:
        """Run physics engine unit conversion tests."""
        # validate.py is at book/cli/commands/validate.py
        # test_units.py is at book/quarto/mlsys/test_units.py
        book_dir = Path(__file__).resolve().parent.parent.parent  # book/
        script = book_dir / "quarto" / "mlsys" / "test_units.py"
        return self._delegate_script(script, [], "unit-tests")
