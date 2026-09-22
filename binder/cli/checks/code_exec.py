"""
Execute Python blocks in QMD files to verify zero runtime errors.

This ensures every code cell in Quarto markdown runs cleanly under the
mlsysim environment without syntax errors, runtime exceptions, Pint unit
dimension mismatch, or failed check() invariant guards.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class CodeExecIssue:
    file: Path
    line: int
    block_index: int
    message: str
    context: str = ""


def check_code_exec(path: Path, text: str | None = None) -> List[CodeExecIssue]:
    """Execute all python blocks in a QMD file sequentially."""
    if text is None:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            return [CodeExecIssue(file=path, line=1, block_index=0, message=f"Failed to read file: {exc}")]

    if "```{python}" not in text:
        return []

    lines = text.splitlines(keepends=True)
    in_cell = False
    cell_start_line = 0
    cell_lines: list[str] = []
    cells: list[tuple[int, str]] = []

    for lineno, line in enumerate(lines, 1):
        if line.startswith("```{python}"):
            in_cell = True
            cell_start_line = lineno
            cell_lines = []
        elif in_cell and line.startswith("```"):
            in_cell = False
            cells.append((cell_start_line, "".join(cell_lines)))
            cell_lines = []
        elif in_cell:
            cell_lines.append(line)

    if not cells:
        return []

    issues: List[CodeExecIssue] = []
    scope: dict = {}
    repo_root = Path(__file__).resolve().parents[3]
    books_dir = repo_root / "books"
    mlsysim_dir = repo_root / "mlsysim"

    import sys
    for p in (str(mlsysim_dir), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import contextlib
    import io
    import warnings

    old_cwd = os.getcwd()

    try:
        os.chdir(books_dir)
        for idx, (lineno, block) in enumerate(cells):
            clean_lines = [line for line in block.splitlines() if not line.strip().startswith("#|")]
            code = "\n".join(clean_lines)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        exec(code, scope)
            except Exception as exc:
                first_code_line = next((l.strip() for l in clean_lines if l.strip()), "")
                issues.append(
                    CodeExecIssue(
                        file=path,
                        line=lineno,
                        block_index=idx,
                        message=f"Block {idx} failed: {type(exc).__name__}: {exc}",
                        context=first_code_line,
                    )
                )
                break  # Subsequent blocks in the file depend on earlier scope
    finally:
        os.chdir(old_cwd)

    return issues
