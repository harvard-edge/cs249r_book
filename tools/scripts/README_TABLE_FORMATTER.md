# Table Formatter

Automatically formats markdown grid tables in the MLSysBook with proper bolding and alignment.

## Features

### 1. Smart Bolding
- **Always bolds headers** (first row)
- **Intelligently bolds first column** based on table type:
  - ✅ Comparison tables (Criterion, Aspect, Feature, etc.)
  - ✅ Trade-off tables (detected by caption keywords)
  - ✅ Definition tables (descriptive multi-word entries)
  - ❌ Data tables (Year, ID, numeric values)
  - ❌ Simple identifier columns (#, Index, etc.)

### 2. Automatic Width Calculation
- Calculates optimal column widths based on content
- Accounts for Unicode characters (arrows, symbols)
- Accounts for bold markers in width calculations
- Adjusts alignment bars to match content

### 3. Handles Edge Cases
- Empty cells (doesn't add bold markers to empty strings)
- Multi-row cells (spanning rows)
- Already bolded content (doesn't double-bold)
- Tables with mixed content types

## Usage

### Check a single file
```bash
python tools/scripts/format_tables.py --check quarto/contents/core/optimizations/optimizations.qmd
```

### Fix a single file
```bash
python tools/scripts/format_tables.py --fix quarto/contents/core/optimizations/optimizations.qmd
```

### Check all files
```bash
python tools/scripts/format_tables.py --check-all
```

### Fix all files
```bash
python tools/scripts/format_tables.py --fix-all
```

### Verbose mode (see detection decisions)
```bash
python tools/scripts/format_tables.py --check <file> --verbose
```

## Detection Heuristics

The script uses multiple strategies to determine if the first column should be bolded:

### Caption Analysis
Keywords in table caption trigger first column bolding:
- "comparison", "trade-off", "overview", "summary"
- "criteria", "characteristics", "features", "differences"

### Header Analysis
First column header names that trigger bolding:
- "criterion", "aspect", "feature", "technique"
- "method", "approach", "strategy", "type"
- "challenge", "principle", "component"

First column header names that prevent bolding:
- "id", "#", "number", "index"
- "year", "date", "time", "rank"

### Content Analysis
- **Numeric content** (>70%): DON'T bold (data table)
- **Descriptive phrases** (>50% multi-word): Bold (comparison table)
- **Questions** (contains "?"): Bold (FAQ style)

## Examples

### Comparison Table (First Column Bolded)
```markdown
+-----------+----------+----------+
| **Criterion** | **Method A** | **Method B** |
+:=========+:========:+:========:+
| **Accuracy**  |   High   |  Medium  |
+-----------+----------+----------+
```

### Data Table (First Column NOT Bolded)
```markdown
+------+---------+--------+
| **Year** | **Revenue** | **Growth** |
+:====:+:=======:+:======:+
| 2020 |  100M   |  10%   |
+------+---------+--------+
```

## Pre-commit Integration

To add this to pre-commit hooks, add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: format-tables
      name: "Format markdown tables"
      entry: python tools/scripts/format_tables.py --check-all
      language: python
      pass_filenames: false
      files: ^quarto/contents/.*\.qmd$
```

## Testing

Run the test suite:
```bash
python tools/scripts/test_format_tables.py
```

Tests cover:
- Display width calculation (including Unicode)
- Bold text handling
- Row parsing
- Alignment extraction
- Border and separator building
- Cell formatting
- Empty cells
- Already formatted tables

## Design Rationale

### Why Bold Headers?
- Universal textbook standard
- Improves scannability
- Clear visual hierarchy

### Why Selective First Column Bolding?
- **Comparison tables**: First column is categorical labels (should stand out)
- **Data tables**: First column is just data (shouldn't dominate visually)
- **Context matters**: Detection uses caption, headers, and content to decide

### Why Auto-adjust Widths?
- Prevents misalignment when adding bold markers
- Ensures professional appearance
- Accommodates Unicode characters correctly

## Limitations

- Only processes grid-style tables (not pipe tables)
- Assumes well-formed table structure
- Detection heuristics may need tuning for edge cases
- Does not handle nested tables or complex merged cells

## Future Enhancements

Potential improvements:
- Add command-line override for detection (force bold/no-bold)
- Support for pipe-style tables
- More sophisticated content type detection
- Integration with Quarto table generation
- Table style templates
