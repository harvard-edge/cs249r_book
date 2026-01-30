#!/usr/bin/env python3
"""
Validate that CLI commands referenced in documentation match actual tito CLI.

This script extracts all `tito X Y` commands from markdown files and validates
them against the actual CLI structure. Runs as a pre-commit hook to catch
documentation drift before it reaches the repo.

Usage:
    python tools/dev/validate_cli_docs.py [--fix] [--verbose]

Exit codes:
    0 - All commands valid
    1 - Invalid commands found
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple

# Directories to scan for markdown files
DOCS_DIRS = ["site", "modules", "tests", "milestones"]

# Files to skip (generated, vendored, etc.)
SKIP_PATTERNS = [".venv", "node_modules", "_build", ".git"]

# Known valid commands from tito --help
# Format: {command_group: [subcommands]}
VALID_COMMANDS: Dict[str, List[str]] = {
    "setup": [],  # No subcommands
    "update": [],  # No subcommands
    "export": [],  # Takes module args, not subcommands
    "test": [],  # Takes module args, not subcommands
    "logo": [],  # No subcommands
    "system": ["info", "health", "jupyter", "update", "logo"],
    "module": ["start", "view", "resume", "complete", "test", "reset", "status", "list"],
    "dev": ["test", "export"],
    "src": ["export", "test"],
    "package": ["reset", "nbdev"],
    "nbgrader": ["init", "generate", "release", "collect", "autograde", "feedback", "status", "analytics", "report"],
    "milestone": ["list", "run", "info", "status", "timeline", "test", "demo"],
    "community": ["login", "logout", "profile", "status", "map"],
    "benchmark": ["baseline", "capstone"],
    "olympics": ["logo", "status"],
    "grade": ["release", "generate", "collect", "autograde", "manual", "feedback", "export", "setup"],
}

# Known INVALID commands that should be flagged
KNOWN_INVALID = {
    "tito checkpoint": "Use 'tito module status' instead",
    "tito milestones": "Use 'tito milestone' (singular) instead",
    "tito system check": "Use 'tito system health' instead",
    "tito system reset": "Command doesn't exist. Use 'tito module reset' for modules",
    "tito community join": "Use 'tito community login' instead",
    "tito community update": "Use 'tito community profile' instead",
    "tito jupyter": "Use 'tito system jupyter' instead",
    "tito notebooks": "Command doesn't exist",
}


def get_valid_command_set() -> Set[str]:
    """Build set of all valid tito commands."""
    valid = set()

    for group, subcommands in VALID_COMMANDS.items():
        valid.add(f"tito {group}")
        for sub in subcommands:
            valid.add(f"tito {group} {sub}")

    return valid


def extract_tito_commands(filepath: Path) -> List[Tuple[int, str]]:
    """Extract all tito commands from a markdown file.

    Returns list of (line_number, command) tuples.
    Only extracts commands that look like actual CLI invocations.
    """
    commands = []

    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return commands

    # Pattern matches tito commands in code blocks or inline code
    # Must start with ` or be at line start (after optional whitespace/comment chars)
    # Excludes title-case words that are clearly prose (e.g., "TITO CLI Reference")
    code_block_pattern = r'`tito\s+([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*)?)'
    line_start_pattern = r'^(?:#\s*)?tito\s+([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*)?)'

    # Words that indicate prose, not commands (case-insensitive check on following word)
    PROSE_INDICATORS = {'cli', 'command', 'commands', 'reference', 'overview', 'guide', 'tool', 'tools'}

    for i, line in enumerate(content.split('\n'), 1):
        # Skip lines that are clearly URLs or links
        if 'http' in line.lower() or 'github.com' in line.lower():
            continue

        # Skip header lines (prose)
        if line.strip().startswith('#') and 'tito' in line.lower() and any(p in line.lower() for p in PROSE_INDICATORS):
            continue

        # Try code block pattern first (most reliable)
        for match in re.finditer(code_block_pattern, line):
            cmd_parts = match.group(1).lower().strip().split()
            # Skip if first word after tito is a prose indicator
            if cmd_parts and cmd_parts[0] in PROSE_INDICATORS:
                continue
            cmd = f"tito {' '.join(cmd_parts)}"
            commands.append((i, cmd))

        # Try line-start pattern for bash code blocks
        for match in re.finditer(line_start_pattern, line.strip()):
            cmd_parts = match.group(1).lower().strip().split()
            # Skip if first word after tito is a prose indicator
            if cmd_parts and cmd_parts[0] in PROSE_INDICATORS:
                continue
            cmd = f"tito {' '.join(cmd_parts)}"
            # Avoid duplicates from the code block pattern
            if (i, cmd) not in commands:
                commands.append((i, cmd))

    return commands


def find_markdown_files(base_dir: Path) -> List[Path]:
    """Find all markdown files in specified directories."""
    files = []

    for docs_dir in DOCS_DIRS:
        search_path = base_dir / docs_dir
        if search_path.exists():
            for md_file in search_path.rglob("*.md"):
                # Skip files in ignored directories
                if any(skip in str(md_file) for skip in SKIP_PATTERNS):
                    continue
                files.append(md_file)

    # Also check root-level markdown files
    for md_file in base_dir.glob("*.md"):
        files.append(md_file)

    return files


def validate_command(cmd: str, valid_commands: Set[str]) -> Tuple[bool, str]:
    """Check if a command is valid.

    Returns (is_valid, error_message).
    """
    # Check against known invalid patterns first
    for invalid, suggestion in KNOWN_INVALID.items():
        if cmd.startswith(invalid):
            return False, suggestion

    # Check if it's a valid base command
    parts = cmd.split()
    if len(parts) < 2:
        return False, "Invalid command format"

    base_cmd = f"{parts[0]} {parts[1]}"

    # Check if group exists
    if parts[1] not in VALID_COMMANDS:
        return False, f"Unknown command group: {parts[1]}"

    # If command has subcommand, validate it
    if len(parts) >= 3:
        full_cmd = f"{parts[0]} {parts[1]} {parts[2]}"
        subcommands = VALID_COMMANDS.get(parts[1], [])

        # If this group has defined subcommands, check them
        if subcommands and parts[2] not in subcommands:
            return False, f"Unknown subcommand: {parts[2]}. Valid: {', '.join(subcommands)}"

    return True, ""


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # Find tinytorch root (script is in tinytorch/tools/dev/)
    script_path = Path(__file__).resolve()
    tinytorch_root = script_path.parent.parent.parent

    # If tinytorch_root is not actually tinytorch (e.g., we're in a different structure),
    # try to find it from current working directory
    if not (tinytorch_root / "bin" / "tito").exists():
        cwd = Path.cwd()
        if (cwd / "tinytorch" / "bin" / "tito").exists():
            tinytorch_root = cwd / "tinytorch"
        elif (cwd / "bin" / "tito").exists():
            tinytorch_root = cwd

    if verbose:
        print(f"Scanning for CLI references in: {tinytorch_root}")

    valid_commands = get_valid_command_set()
    md_files = find_markdown_files(tinytorch_root)

    if verbose:
        print(f"Found {len(md_files)} markdown files to check")

    errors: List[Tuple[Path, int, str, str]] = []

    for md_file in md_files:
        commands = extract_tito_commands(md_file)

        for line_num, cmd in commands:
            is_valid, error_msg = validate_command(cmd, valid_commands)

            if not is_valid:
                rel_path = md_file.relative_to(tinytorch_root)
                errors.append((rel_path, line_num, cmd, error_msg))

    if errors:
        print(f"\n{'='*60}")
        print(f"CLI Documentation Validation FAILED")
        print(f"{'='*60}\n")
        print(f"Found {len(errors)} invalid CLI command reference(s):\n")

        for filepath, line, cmd, msg in errors:
            print(f"  {filepath}:{line}")
            print(f"    Command: {cmd}")
            print(f"    Issue: {msg}")
            print()

        print("Fix these issues before committing.")
        print("Run 'tito --help' to see valid commands.\n")
        return 1

    if verbose:
        print(f"\n All {len(md_files)} markdown files have valid CLI references!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
