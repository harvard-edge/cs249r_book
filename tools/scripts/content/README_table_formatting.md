# Table Formatting Guidelines for MLSysBook

## Overview

This directory contains tools and guidelines for formatting grid tables in the MLSysBook project. The goal is to improve readability and consistency across all tables in the textbook.

## Alignment Strategy

For a technical textbook, we use content-based alignment to enhance readability:

### Left-Align (`:===========+`)
- **Descriptive text**: Technique names, method descriptions
- **Use cases**: Application scenarios, explanatory text
- **System requirements**: Hardware/software specifications
- **Default**: When content doesn't clearly fit other categories

### Center-Align (`:==========:`)
- **Categorical values**: High/Medium/Low, Yes/No, On/Off
- **Bit-widths**: 32-bit, 16-bit, 8-bit
- **Short technical terms**: FP32, INT8, TPU
- **Status indicators**: Enabled/Disabled, Fast/Slow

### Right-Align (`===========:`)
- **Numerical comparisons**: 2×, 4×, 8×
- **Percentages**: 95%, 50%, 25%
- **Measurements**: 100ms, 2GB, 50W
- **Performance metrics**: Baseline (1×), 4–8× faster
- **Monetary values**: $1,000, €500

## Examples

### Good Alignment Example
```markdown
+------------------+-----------+------------------+---------------------------+
| Technique        | Bit-Width | Storage Reduction| Use Cases                 |
+:=================+==========:+=================:+===========================+
| FP32             |    32-bit |        Baseline  | Training & inference      |
| FP16             |    16-bit |             2×   | Accelerated training      |
| INT8             |     8-bit |             4×   | Edge AI, mobile devices   |
+------------------+-----------+------------------+---------------------------+
```

### Analysis
- **Column 1** (Technique): Left-aligned descriptive text
- **Column 2** (Bit-Width): Center-aligned categorical technical terms
- **Column 3** (Storage Reduction): Right-aligned numerical comparisons
- **Column 4** (Use Cases): Left-aligned descriptive text

## Tools

### `format_grid_tables.py`
Automated tool for detecting and reformatting grid tables with appropriate alignment.

**Features:**
- Analyzes column content to determine optimal alignment
- Preserves table structure and content
- Supports dry-run mode for safe testing
- Processes individual files or directories

**Usage:**
```bash
# Preview changes (recommended first step)
python3 scripts/content/format_grid_tables.py book/contents/core/ml_systems/ml_systems.qmd --dry-run

# Apply to single file
python3 scripts/content/format_grid_tables.py book/contents/core/ml_systems/ml_systems.qmd

# Apply to entire book content
python3 scripts/content/format_grid_tables.py book/contents --dry-run  # Preview first
python3 scripts/content/format_grid_tables.py book/contents             # Apply
```

## Manual Guidelines

When creating or editing tables manually:

1. **Identify content type** for each column
2. **Choose appropriate alignment** based on guidelines above
3. **Use consistent spacing** for better readability
4. **Test with dry-run** before applying automated formatting

## Content Analysis Patterns

The automated tool uses these patterns to detect content types:

### Numeric Patterns (Right-align)
- `2×`, `4×`, `10.5×` (multiplication factors)
- `50%`, `95%` (percentages)
- `100ms`, `2GB`, `50W` (measurements)
- `$1,000` (currency)

### Categorical Patterns (Center-align)
- `High/Low/Medium`
- `Yes/No/True/False`
- `32-bit`, `16-bit` (bit widths)
- `FP32`, `INT8` (short technical acronyms)

### Default (Left-align)
- Long descriptive text
- Use case descriptions
- System requirements
- Technique names

## Quality Checks

Before committing table changes:

1. ✅ **Content preserved**: All original table data intact
2. ✅ **Alignment appropriate**: Follows content-based rules
3. ✅ **Consistent formatting**: Similar tables use same alignment
4. ✅ **Readable output**: Tables enhance rather than hinder comprehension

## Troubleshooting

### Common Issues
1. **Content missing**: Check parser for multi-line cell handling
2. **Wrong alignment**: Verify content analysis patterns
3. **Broken structure**: Ensure row/column separators are preserved

### Best Practices
1. **Start small**: Test on individual files before batch processing
2. **Use dry-run**: Always preview changes first
3. **Git safety**: Work on feature branches (like this one!)
4. **Manual review**: Check output for edge cases 