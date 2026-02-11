"""
Native validation commands for MLSysBook Binder CLI.

This module intentionally implements validation logic directly in Binder,
without shelling out to legacy scripts under tools/scripts.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
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
        return len(self.issues) == 0

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
    "Figure": [re.compile(r"\{#(fig-[\w:-]+)")],
    "Table": [re.compile(r"\{#(tbl-[\w:-]+)")],
    "Section": [re.compile(r"\{#(sec-[\w:-]+)")],
    "Equation": [re.compile(r"\{#(eq-[\w:-]+)")],
    "Listing": [re.compile(r"\{#(lst-[\w:-]+)")],
}
LABEL_REF_PATTERN = re.compile(r"@((?:fig|tbl|sec|eq|lst)-[\w:-]+)")

EXCLUDED_CITATION_PREFIXES = ("fig-", "tbl-", "sec-", "eq-", "lst-", "ch-")


class ValidateCommand:
    """Native `binder validate` command group."""

    def __init__(self, config_manager, chapter_discovery):
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery

    def run(self, args: List[str]) -> bool:
        parser = argparse.ArgumentParser(
            prog="binder validate",
            description="Run Binder-native validation checks",
            add_help=True,
        )
        parser.add_argument(
            "subcommand",
            nargs="?",
            choices=[
                "inline-python",
                "refs",
                "citations",
                "duplicate-labels",
                "unreferenced-labels",
                "inline-refs",
                "all",
            ],
            help="Validation command to run",
        )
        parser.add_argument("--path", default=None, help="File or directory path to validate")
        parser.add_argument("--vol1", action="store_true", help="Scope validation to Volume I")
        parser.add_argument("--vol2", action="store_true", help="Scope validation to Volume II")
        parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
        parser.add_argument("--citations-in-code", action="store_true", help="refs: check citations in code fences")
        parser.add_argument("--citations-in-raw", action="store_true", help="refs: check citations in raw blocks")
        parser.add_argument("--check-patterns", action="store_true", help="inline-refs: include pattern hazard checks")
        parser.add_argument("--figures", action="store_true", help="duplicate/unreferenced labels: figures")
        parser.add_argument("--tables", action="store_true", help="duplicate/unreferenced labels: tables")
        parser.add_argument("--sections", action="store_true", help="duplicate/unreferenced labels: sections")
        parser.add_argument("--equations", action="store_true", help="duplicate/unreferenced labels: equations")
        parser.add_argument("--listings", action="store_true", help="duplicate/unreferenced labels: listings")
        parser.add_argument("--all-types", action="store_true", help="duplicate/unreferenced labels: all types")

        try:
            ns = parser.parse_args(args)
        except SystemExit:
            # argparse uses SystemExit(0) for --help and non-zero for parse errors.
            return ("-h" in args) or ("--help" in args)

        if not ns.subcommand:
            parser.print_help()
            return False

        root_path = self._resolve_path(ns.path, ns.vol1, ns.vol2)
        if not root_path.exists():
            self._emit(ns.json, {"status": "error", "message": f"Path not found: {root_path}"}, failed=True)
            return False

        runs: List[ValidationRunResult] = []
        if ns.subcommand == "all":
            runs.append(self._run_inline_python(root_path))
            runs.append(self._run_refs(root_path, citations_in_code=True, citations_in_raw=True))
            runs.append(self._run_citations(root_path))
            label_types = self._selected_label_types(ns)
            runs.append(self._run_duplicate_labels(root_path, label_types))
            runs.append(self._run_unreferenced_labels(root_path, label_types))
            runs.append(self._run_inline_refs(root_path, check_patterns=ns.check_patterns))
        elif ns.subcommand == "inline-python":
            runs.append(self._run_inline_python(root_path))
        elif ns.subcommand == "refs":
            checks_code = ns.citations_in_code or (not ns.citations_in_code and not ns.citations_in_raw)
            checks_raw = ns.citations_in_raw or (not ns.citations_in_code and not ns.citations_in_raw)
            runs.append(self._run_refs(root_path, citations_in_code=checks_code, citations_in_raw=checks_raw))
        elif ns.subcommand == "citations":
            runs.append(self._run_citations(root_path))
        elif ns.subcommand == "duplicate-labels":
            runs.append(self._run_duplicate_labels(root_path, self._selected_label_types(ns)))
        elif ns.subcommand == "unreferenced-labels":
            runs.append(self._run_unreferenced_labels(root_path, self._selected_label_types(ns)))
        elif ns.subcommand == "inline-refs":
            runs.append(self._run_inline_refs(root_path, check_patterns=ns.check_patterns))

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
        # default common types
        return {
            "Figure": LABEL_DEF_PATTERNS["Figure"],
            "Table": LABEL_DEF_PATTERNS["Table"],
            "Section": LABEL_DEF_PATTERNS["Section"],
            "Listing": LABEL_DEF_PATTERNS["Listing"],
        }

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

    def _run_inline_python(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        regex_checks = [
            ("missing_backtick", re.compile(r"(?<!`)(\{python\}\s+\w+`)"), "Missing opening backtick before {python}", "error"),
            ("dollar_as_backtick", re.compile(r"\$\{python\}\s+\w+`"), "Dollar sign used instead of backtick before {python}", "error"),
            ("display_math", re.compile(r"\$\$[^$]*`?\{python\}"), "Inline Python inside $$...$$ display math", "error"),
            ("latex_adjacent", re.compile(r"`\{python\}[^`]+`\s*\$\\(times|approx|ll|gg|mu|le|ge|neq|pm|cdot|div)"), "Inline Python adjacent to LaTeX operator", "warning"),
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

    def _run_citations(self, root: Path) -> ValidationRunResult:
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []

        bib_field_pattern = re.compile(r"^bibliography:\s*([^\s]+\.bib)\s*$", re.MULTILINE)
        bib_key_pattern = re.compile(r"@\w+\{([^,\s]+)")

        for file in files:
            content = self._read_text(file)
            bib_match = bib_field_pattern.search(content)
            if not bib_match:
                continue
            bib_file = file.parent / bib_match.group(1)
            if not bib_file.exists():
                issues.append(ValidationIssue(
                    file=self._relative_file(file),
                    line=1,
                    code="missing_bib_file",
                    message=f"Bibliography file not found: {bib_match.group(1)}",
                    severity="error",
                ))
                continue

            bib_content = self._read_text(bib_file)
            bib_keys = set(bib_key_pattern.findall(bib_content))
            qmd_content_no_code = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
            qmd_content_no_code = re.sub(r"`[^`]+`", "", qmd_content_no_code)
            refs = set(CITATION_REF_PATTERN.findall(qmd_content_no_code))
            refs = {r.rstrip(".,;:") for r in refs if not r.startswith(EXCLUDED_CITATION_PREFIXES)}
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
        console.print(Panel(table, title="Binder Validate Summary", border_style="cyan"))

        if total == 0:
            console.print("[green]✅ All validation checks passed.[/green]")
            return

        for run in runs:
            if run["issue_count"] == 0:
                continue
            console.print(f"[bold red]{run['name']}[/bold red] ({run['issue_count']} issues)")
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
            console.print(f"[red]❌ Validation failed with {total} issue(s).[/red]")

    def _emit(self, as_json: bool, payload: Dict[str, Any], failed: bool) -> None:
        if as_json:
            print(json.dumps(payload, indent=2))
            return
        if failed:
            console.print(f"[red]{payload.get('message', 'Operation failed')}[/red]")
        else:
            console.print(f"[green]{payload.get('message', 'Operation succeeded')}[/green]")
