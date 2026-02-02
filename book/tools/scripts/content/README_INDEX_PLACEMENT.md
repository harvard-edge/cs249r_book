# Index Placement Checker

## Overview

The `check_index_placement.py` script validates that LaTeX `\index{}` commands are properly placed in QMD files. When `\index{}` appears on the same line as structural elements (headings, callouts, divs), it breaks Quarto rendering.

## Problem

### Incorrect Patterns (breaks rendering):

```markdown
### Heading {#id}\index{Topic!subtopic}

\index{Topic!subtopic}::: {.callout-warning}

::: {.callout-note} \index{Topic!subtopic}
```

### Correct Pattern:

```markdown
### Heading {#id}

\index{Topic!subtopic}Content starts here...

::: {.callout-warning}

\index{Topic!subtopic}Content starts here...
```

## Usage

### Standalone

```bash
# Check single file
python check_index_placement.py file.qmd

# Check multiple files
python check_index_placement.py file1.qmd file2.qmd

# Check directory recursively
python check_index_placement.py -d book/quarto/contents/vol1/

# Quiet mode (summary only)
python check_index_placement.py -d book/quarto/contents/ --quiet
```

### Pre-commit Hook

Automatically runs via pre-commit:

```bash
# Run all checks
pre-commit run --all-files

# Run only this check
pre-commit run book-check-index-placement --all-files
```

## What It Checks

1. **Headings**: `\index{}` must not appear on same line as `# ## ### ####` headings
2. **Callout openings**: `\index{}` must not appear on same line as `::: {.callout-...}`
3. **Div openings**: `\index{}` must not appear directly before or after `:::`

## Exceptions

- `\index{}` inside code blocks (```...```) are ignored (false positives)
- `\index{}` inside figure captions (`fig-cap="..."`) are allowed

## Exit Codes

- `0`: No issues found
- `1`: Issues found

## Implementation Details

The script:
- Tracks code block boundaries to avoid false positives
- Uses regex patterns to detect problematic placements
- Provides line numbers and context for each issue
- Shows visual indicators (`~~~`) under the problematic `\index{}`

## Related

- Pre-commit config: `.pre-commit-config.yaml`
- Hook ID: `book-check-index-placement`
