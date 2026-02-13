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
                "headers",
                "footnote-placement",
                "footnote-refs",
                "figures",
                "float-flow",
                "indexes",
                "rendering",
                "dropcaps",
                "parts",
                "images",
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
            runs.append(self._run_headers(root_path))
            runs.append(self._run_footnote_placement(root_path))
            runs.append(self._run_footnote_refs(root_path))
            runs.append(self._run_figures(root_path))
            runs.append(self._run_float_flow(root_path))
            runs.append(self._run_indexes(root_path))
            runs.append(self._run_rendering(root_path))
            runs.append(self._run_dropcaps(root_path))
            runs.append(self._run_parts(root_path))
            runs.append(self._run_images(root_path))
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
        elif ns.subcommand == "headers":
            runs.append(self._run_headers(root_path))
        elif ns.subcommand == "footnote-placement":
            runs.append(self._run_footnote_placement(root_path))
        elif ns.subcommand == "footnote-refs":
            runs.append(self._run_footnote_refs(root_path))
        elif ns.subcommand == "figures":
            runs.append(self._run_figures(root_path))
        elif ns.subcommand == "float-flow":
            runs.append(self._run_float_flow(root_path))
        elif ns.subcommand == "indexes":
            runs.append(self._run_indexes(root_path))
        elif ns.subcommand == "rendering":
            runs.append(self._run_rendering(root_path))
        elif ns.subcommand == "dropcaps":
            runs.append(self._run_dropcaps(root_path))
        elif ns.subcommand == "parts":
            runs.append(self._run_parts(root_path))
        elif ns.subcommand == "images":
            runs.append(self._run_images(root_path))

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
            description="Verify all section headers have {#sec-...} IDs",
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
        ]

        grid_sep_pat = re.compile(r"^\+[-:=+]+\+$")
        math_span_pat = re.compile(r"(?<!\\)\$(?!\$)(?!`)(.+?)(?<!\\)\$")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_grid = False

            for idx, line in enumerate(lines, 1):
                stripped = line.strip()

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
