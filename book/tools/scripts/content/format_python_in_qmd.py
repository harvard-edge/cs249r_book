#!/usr/bin/env python3
"""Format Python code blocks in .qmd files using Black.

Also wraps long comments and docstrings that Black doesn't handle.
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


def format_python_blocks(content: str, line_length: int = 70) -> str:
    """Find and format Python code blocks in markdown using Black.

    Also wraps long comments and docstrings.
    """
    lines = content.split("\n")
    result = []
    in_python_block = False
    python_lines = []

    for line in lines:
        # Detect start of Python block
        if re.match(r"^```(\{\.)?python", line):
            in_python_block = True
            result.append(line)
            python_lines = []
            continue

        # Detect end of Python block
        if in_python_block and line.strip() == "```":
            # Format accumulated Python code with Black
            if python_lines:
                code = "\n".join(python_lines)

                # Validate Python syntax before attempting to format
                if not is_valid_python(code):
                    # Skip Black for invalid Python, but still wrap comments
                    wrapped_code = wrap_long_lines(code, line_length)
                    result.extend(wrapped_code.split("\n"))
                else:
                    try:
                        # Write to temp file
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".py", delete=False
                        ) as f:
                            f.write(code)
                            temp_path = f.name

                        # Run Black with stderr suppressed
                        subprocess.run(
                            [
                                "black",
                                "--line-length",
                                str(line_length),
                                "--quiet",
                                temp_path,
                            ],
                            check=True,
                            stderr=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                        )

                        # Read formatted code
                        with open(temp_path, "r") as f:
                            formatted = f.read().rstrip()

                        # Wrap long comments/docstrings that Black didn't handle
                        formatted = wrap_long_lines(formatted, line_length)

                        result.extend(formatted.split("\n"))

                        # Cleanup
                        Path(temp_path).unlink()
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        # If Black fails, still try to wrap comments
                        wrapped_code = wrap_long_lines(code, line_length)
                        result.extend(wrapped_code.split("\n"))

            python_lines = []
            in_python_block = False
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
