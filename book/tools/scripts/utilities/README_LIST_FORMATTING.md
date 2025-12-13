# Markdown List Formatting Checker

## Overview

This tool ensures proper markdown list formatting by enforcing that bullet lists are preceded by an empty line. This is required for correct markdown rendering across different parsers (Quarto, GitHub, etc.).

## The Problem

Markdown requires an empty line before bullet lists for proper rendering:

### ❌ Incorrect (breaks rendering)
```markdown
Configuration:
- Item 1
- Item 2
```

### ✅ Correct (renders properly)
```markdown
Configuration:

- Item 1
- Item 2
```

## Usage

### Check for issues (without fixing)

```bash
# Check a single file
python tools/scripts/utilities/check_list_formatting.py --check file.qmd

# Check all files in a directory
python tools/scripts/utilities/check_list_formatting.py --check --recursive quarto/contents/

# Check all .qmd files in project
python tools/scripts/utilities/check_list_formatting.py --check --recursive quarto/
```

### Fix issues automatically

```bash
# Fix a single file
python tools/scripts/utilities/check_list_formatting.py --fix file.qmd

# Fix all files in a directory
python tools/scripts/utilities/check_list_formatting.py --fix --recursive quarto/contents/

# Fix all .qmd files in project
python tools/scripts/utilities/check_list_formatting.py --fix --recursive quarto/
```

## Pre-commit Integration

This check is automatically run as part of pre-commit hooks in **Phase 1: Auto-formatters**. It will automatically fix any issues before commit.

To run manually:
```bash
pre-commit run check-list-formatting --all-files
```

## What It Checks

The script identifies lines that:
1. End with a colon (`:`)
2. Are immediately followed by a bullet list (`- `)
3. Don't have special context (code blocks, headers, tables, etc.)

## What It Fixes

When run with `--fix`, the script automatically inserts an empty line between the colon line and the bullet list.

## Exit Codes

- `0`: Success (no issues found, or all issues fixed)
- `1`: Issues found (in check-only mode)

## Examples

### Before Fix
```markdown
GPT-2 Configuration:
- Parameters: 1.5B
- Batch size: 32
- Training time: 2 weeks
```

### After Fix
```markdown
GPT-2 Configuration:

- Parameters: 1.5B
- Batch size: 32
- Training time: 2 weeks
```

## Integration with Other Tools

This check complements other formatting tools in the pre-commit pipeline:
- Runs after `mdformat` and `collapse-extra-blank-lines`
- Runs before content validators
- Works with all `.qmd` files in the repository

## Troubleshooting

### False Positives

If the tool incorrectly flags something, check:
1. Is it inside a code block? (should be ignored)
2. Is it a table? (should be ignored)
3. Is it a special Quarto directive? (should be ignored)

If legitimate content is being flagged, update the exclusion rules in the script.

### Script Location

The script is located at:
```
tools/scripts/utilities/check_list_formatting.py
```

### Pre-commit Configuration

The hook is configured in `.pre-commit-config.yaml` under the `local` repo in Phase 1.
