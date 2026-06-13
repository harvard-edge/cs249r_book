#!/usr/bin/env python3
"""Format Python code blocks in .qmd files using Black.

Also wraps long comments and docstrings that Black doesn't handle.

Two block profiles:
- Display blocks (```python / ```{.python}): Black at 70 chars + comment
  wrapping — narrow snippets shown to the reader.
- Executable LEGO cells (```{python}): Black at 150 chars with string
  normalization off and no comment wrapping — keeps the one-line
  `field_str = fmt_*(...)` export style and never touches `#|` options
  or LEGO box-drawing headers.
"""

import ast
import re
import sys
import textwrap
from pathlib import Path
from typing import List
import subprocess
import tempfile


def is_valid_python(code: str) -> bool:
    """Check if code string is valid Python syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def wrap_comment(line: str, max_length: int = 70) -> List[str]:
    """Wrap a comment line that exceeds max_length.

    Preserves leading indentation and comment marker.
    Returns list of wrapped lines.
    """
    if len(line) <= max_length:
        return [line]

    # Extract indentation
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    # Handle comment lines (# ...)
    if stripped.startswith("#"):
        # Find where the comment text starts (after # and any spaces)
        match = re.match(r"(#+\s*)", stripped)
        if match:
            comment_prefix = match.group(1)
            comment_text = stripped[len(comment_prefix) :]

            # Calculate available width for text
            prefix_len = len(indent) + len(comment_prefix)
            available_width = max_length - prefix_len

            if available_width < 20:
                # Not enough room to wrap meaningfully
                return [line]

            # Wrap the text
            wrapped = textwrap.wrap(
                comment_text,
                width=available_width,
                break_long_words=False,
                break_on_hyphens=False,
            )

            if not wrapped:
                return [line]

            # Reconstruct lines with proper indentation
            result = []
            for i, wrapped_line in enumerate(wrapped):
                result.append(f"{indent}{comment_prefix}{wrapped_line}")
            return result

    return [line]


def wrap_single_line_docstring(line: str, max_length: int = 70) -> List[str]:
    """Convert a long single-line docstring to multi-line.

    Handles both triple-double-quotes and triple-single-quotes.
    Returns list of lines (original if no change needed).
    """
    if len(line) <= max_length:
        return [line]

    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    # Check for single-line docstring patterns
    # Pattern: """...""" or '''...'''
    for quote in ['"""', "'''"]:
        if stripped.startswith(quote) and stripped.endswith(quote):
            # Single-line docstring
            content = stripped[3:-3].strip()
            if not content:
                return [line]

            # Calculate available width for content
            # Account for indent + quotes
            available_width = max_length - len(indent) - 4  # 4 for indent margin

            if available_width < 20:
                return [line]

            # Wrap the content
            wrapped = textwrap.wrap(
                content,
                width=available_width,
                break_long_words=False,
                break_on_hyphens=False,
            )

            if len(wrapped) <= 1:
                # Can't improve by wrapping
                return [line]

            # Convert to multi-line docstring
            result = [f"{indent}{quote}{wrapped[0]}"]
            for wrapped_line in wrapped[1:]:
                result.append(f"{indent}{wrapped_line}")
            result.append(f"{indent}{quote}")
            return result

    return [line]


def wrap_long_lines(code: str, max_length: int = 70) -> str:
    """Wrap long comment and docstring lines in Python code.

    This runs after Black formatting to handle lines that
    Black doesn't wrap (comments and string literals).
    """
    lines = code.split("\n")
    result = []

    for line in lines:
        stripped = line.lstrip()

        # Check if it's a comment line
        if stripped.startswith("#") and len(line) > max_length:
            result.extend(wrap_comment(line, max_length))
        # Check if it's a single-line docstring
        elif (
            (stripped.startswith('"""') and stripped.endswith('"""'))
            or (stripped.startswith("'''") and stripped.endswith("'''"))
        ) and len(line) > max_length:
            result.extend(wrap_single_line_docstring(line, max_length))
        else:
            result.append(line)

    return "\n".join(result)


# Executable LEGO cells: wide one-line exports, preserve quote style.
EXEC_LINE_LENGTH = 150

# Sentinel that shields Quarto `#|` option lines from Black, which would
# otherwise rewrite `#| echo: false` as `# | echo: false` (Black inserts
# a space after `#` unless followed by `!`, `:`, `#`, or `'`).
_QMD_OPT_SENTINEL = "#!__QMD_OPT__"


def run_black(code: str, line_length: int, skip_string_normalization: bool = False) -> str:
    """Run Black on a code string via a temp file; return formatted code.

    Raises subprocess.CalledProcessError / FileNotFoundError on failure.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        cmd = ["black", "--line-length", str(line_length), "--quiet"]
        if skip_string_normalization:
            cmd.append("--skip-string-normalization")
        cmd.append(temp_path)

        subprocess.run(
            cmd,
            check=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )

        with open(temp_path, "r") as f:
            return f.read().rstrip()
    finally:
        Path(temp_path).unlink(missing_ok=True)


def _collapse_cell_blanks(code: str) -> str:
    """Collapse runs of blank lines to a single blank line.

    Lines inside multi-line string literals are left untouched. Keeps
    LEGO cells compact (the corpus convention) and makes the output a
    fixed point for the `book-format-blanks` hook, whose fence tracking
    can desync on commented-out partial code blocks.
    """
    import io
    import tokenize

    string_lines = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.STRING and tok.start[0] < tok.end[0]:
                string_lines.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenizeError, SyntaxError, IndentationError):
        return code

    result = []
    prev_blank = False
    for lineno, line in enumerate(code.split("\n"), 1):
        is_blank = line.strip() == "" and lineno not in string_lines
        if is_blank and prev_blank:
            continue
        result.append(line)
        prev_blank = is_blank
    return "\n".join(result)


def format_exec_cell(code: str) -> str:
    """Format an executable ```{python} cell.

    Shields `#|` option lines, runs Black at EXEC_LINE_LENGTH with quote
    normalization off, collapses multi-blank runs to one line, and skips
    comment wrapping (LEGO headers use box-drawing characters that must
    not be rewrapped).
    """
    shielded = "\n".join(
        line.replace("#|", _QMD_OPT_SENTINEL, 1)
        if line.lstrip().startswith("#|")
        else line
        for line in code.split("\n")
    )

    if not is_valid_python(shielded):
        return code

    try:
        formatted = run_black(
            shielded, EXEC_LINE_LENGTH, skip_string_normalization=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return code

    return _collapse_cell_blanks(formatted.replace(_QMD_OPT_SENTINEL, "#|"))


def format_python_blocks(content: str, line_length: int = 70) -> str:
    """Find and format Python code blocks in markdown using Black.

    Also wraps long comments and docstrings (display blocks only).
    """
    lines = content.split("\n")
    result = []
    in_python_block = False
    is_exec_cell = False
    python_lines = []

    for line in lines:
        # Detect start of Python block
        if re.match(r"^```(\{\.)?python", line) or re.match(r"^```\{python\}", line):
            in_python_block = True
            is_exec_cell = bool(re.match(r"^```\{python\}", line))
            result.append(line)
            python_lines = []
            continue

        # Detect end of Python block
        if in_python_block and line.strip() == "```":
            # Format accumulated Python code with Black
            if python_lines:
                code = "\n".join(python_lines)

                if is_exec_cell:
                    # Executable LEGO cell: wide one-liners, no comment
                    # wrapping, `#|` options and quotes preserved.
                    result.extend(format_exec_cell(code).split("\n"))
                # Validate Python syntax before attempting to format
                elif not is_valid_python(code):
                    # Skip Black for invalid Python, but still wrap comments
                    wrapped_code = wrap_long_lines(code, line_length)
                    result.extend(wrapped_code.split("\n"))
                else:
                    try:
                        formatted = run_black(code, line_length)

                        # Wrap long comments/docstrings that Black didn't handle
                        formatted = wrap_long_lines(formatted, line_length)

                        result.extend(formatted.split("\n"))
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        # If Black fails, still try to wrap comments
                        wrapped_code = wrap_long_lines(code, line_length)
                        result.extend(wrapped_code.split("\n"))

            python_lines = []
            in_python_block = False
            is_exec_cell = False
            result.append(line)
            continue

        # Accumulate Python code
        if in_python_block:
            python_lines.append(line)
        else:
            result.append(line)

    return "\n".join(result)


def main(files: List[str], line_length: int = 70) -> int:
    """Format Python blocks in .qmd files."""
    changed = 0
    for filepath in files:
        path = Path(filepath)
        if path.suffix == ".qmd":
            try:
                content = path.read_text(encoding="utf-8")
                formatted = format_python_blocks(content, line_length)

                if formatted != content:
                    path.write_text(formatted, encoding="utf-8")
                    print(f"Formatted: {filepath}")
                    changed += 1
            except Exception as e:
                print(f"Error processing {filepath}: {e}", file=sys.stderr)
                return 1

    return 0 if changed == 0 else 1  # Return 1 if changes made


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
