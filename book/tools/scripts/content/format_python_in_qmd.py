#!/usr/bin/env python3
"""Format Python code blocks in .qmd files using Black."""

import ast
import re
import sys
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


def format_python_blocks(content: str, line_length: int = 70) -> str:
    """Find and format Python code blocks in markdown using Black."""
    lines = content.split('\n')
    result = []
    in_python_block = False
    python_lines = []
    block_indent = ""

    for line in lines:
        # Detect start of Python block
        if re.match(r'^```(\{\.)?python', line):
            in_python_block = True
            result.append(line)
            python_lines = []
            continue

        # Detect end of Python block
        if in_python_block and line.strip() == '```':
            # Format accumulated Python code with Black
            if python_lines:
                code = '\n'.join(python_lines)

                # Validate Python syntax before attempting to format
                if not is_valid_python(code):
                    # Skip formatting for invalid Python (e.g., shell, pseudocode, etc.)
                    result.extend(python_lines)
                else:
                    try:
                        # Write to temp file
                        with tempfile.NamedTemporaryFile(
                            mode='w', suffix='.py', delete=False
                        ) as f:
                            f.write(code)
                            temp_path = f.name

                        # Run Black with stderr suppressed
                        subprocess.run(
                            ['black', '--line-length', str(line_length),
                             '--quiet', temp_path],
                            check=True,
                            stderr=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL
                        )

                        # Read formatted code
                        with open(temp_path, 'r') as f:
                            formatted = f.read().rstrip()

                        result.extend(formatted.split('\n'))

                        # Cleanup
                        Path(temp_path).unlink()
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        # If Black fails or isn't installed, keep original
                        result.extend(python_lines)

            python_lines = []
            in_python_block = False
            result.append(line)
            continue

        # Accumulate Python code
        if in_python_block:
            python_lines.append(line)
        else:
            result.append(line)

    return '\n'.join(result)


def main(files: List[str], line_length: int = 70) -> int:
    """Format Python blocks in .qmd files."""
    changed = 0
    for filepath in files:
        path = Path(filepath)
        if path.suffix == '.qmd':
            try:
                content = path.read_text(encoding='utf-8')
                formatted = format_python_blocks(content, line_length)

                if formatted != content:
                    path.write_text(formatted, encoding='utf-8')
                    print(f"Formatted: {filepath}")
                    changed += 1
            except Exception as e:
                print(f"Error processing {filepath}: {e}",
                      file=sys.stderr)
                return 1

    return 0 if changed == 0 else 1  # Return 1 if changes made


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
