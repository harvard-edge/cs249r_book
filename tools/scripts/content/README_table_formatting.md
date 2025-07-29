# Table Formatting Tools for MLSysBook

## Overview

This directory contains tools for formatting grid tables in the MLSysBook project with intelligent, content-based alignment. The tools are designed for both manual use and integration with automated workflows like pre-commit hooks.

## 🎯 Alignment Strategy

For technical content, we use content-based alignment to enhance readability:

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
- **Hardware types**: CPU, GPU, TPU

### Right-Align (`===========:`)
- **Numerical comparisons**: 2×, 4×, 8×
- **Percentages**: 95%, 50%, 25%
- **Measurements**: 100ms, 2GB, 50W
- **Performance metrics**: Baseline (1×), 4–8× faster
- **Monetary values**: $1,000, €500

## 🛠️ Tools

### 1. `format_grid_tables.py` - Main Formatter

**Enhanced Features:**
- 🧠 **Intelligent content analysis** - Automatically detects data types
- 📊 **Table classification** - Identifies table types (precision, performance, etc.)
- 🔍 **Robust parsing** - Handles complex table structures
- ⚡ **Fast processing** - Optimized for large codebases
- 🎯 **Pre-commit ready** - Integrates with CI/CD pipelines

**Command-Line Interface:**
```bash
# Process single file
python3 format_grid_tables.py -f path/to/file.qmd [options]

# Process directory
python3 format_grid_tables.py -d path/to/directory/ [options]

# Process entire book
python3 format_grid_tables.py --check-all [options]

# Options
--dry-run      # Preview changes without applying
--verbose      # Show detailed processing info
--pre-commit   # Exit with error if changes needed (for CI)
```

**Usage Examples:**
```bash
# Preview what would change in a single file
python3 scripts/content/format_grid_tables.py -f book/contents/core/ml_systems.qmd --dry-run

# Format all tables in the optimizations chapter
python3 scripts/content/format_grid_tables.py -d book/contents/core/optimizations/ --verbose

# Check entire book (safe, shows what would change)
python3 scripts/content/format_grid_tables.py --check-all --dry-run

# Format entire book
python3 scripts/content/format_grid_tables.py --check-all
```

### 2. `format_all_tables.sh` - Convenient Wrapper

**Interactive script for safe batch processing:**
```bash
# Interactive mode (asks for confirmation)
./scripts/content/format_all_tables.sh

# Force mode (no confirmation)
./scripts/content/format_all_tables.sh --force
```

**Features:**
- 🛡️ **Safety first** - Always previews changes before applying
- 💬 **Interactive** - Asks for confirmation unless `--force` used
- 🎨 **Colored output** - Clear visual feedback
- ✅ **Error handling** - Graceful failure with helpful messages

## 🔧 Table Analysis Engine

The formatter uses advanced pattern matching to classify content:

### Detected Table Types
- **`precision_comparison`** - Numerical precision formats (FP32, INT8, etc.)
- **`model_comparison`** - Model architectures and specifications
- **`performance_table`** - Benchmarks, latency, throughput data
- **`technique_comparison`** - ML techniques and methods
- **`requirements_table`** - System requirements and constraints
- **`general_table`** - Other structured data

### Content Analysis Patterns

**Numeric Patterns** (→ Right-align):
- Multiplication factors: `2×`, `4×`, `10.5×`
- Percentages: `50%`, `95%`
- Measurements: `100ms`, `2GB`, `50W`
- Currency: `$1,000`
- Baselines: `Baseline (1×)`

**Categorical Patterns** (→ Center-align):
- Levels: `High/Low/Medium`
- Booleans: `Yes/No`, `True/False`
- Hardware: `CPU`, `GPU`, `TPU`
- Technical terms: `FP32`, `INT8`
- Bit widths: `32-bit`, `16-bit`

**Descriptive Patterns** (→ Left-align):
- Technique names and descriptions
- Use case explanations
- System requirements

## 🔗 Integration Options

### Option 1: Pre-commit Hook
Add to `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: format-grid-tables
        name: Format Grid Tables
        entry: python3 tools/scripts/content/format_grid_tables.py
        language: system
        files: '\.qmd$'
        args: ['--check-all', '--pre-commit']
        pass_filenames: false
```

### Option 2: Manual Workflow
```bash
# Before committing changes
./scripts/content/format_all_tables.sh

# Or check specific files
python3 scripts/content/format_grid_tables.py -f path/to/changed.qmd --dry-run
```

### Option 3: CI/CD Integration
```bash
# In CI pipeline - fail if tables need formatting
python3 tools/scripts/content/format_grid_tables.py --check-all --pre-commit --dry-run
```

## 📝 Examples

### Before (Poor Alignment)
```markdown
+----------------+----------+---------------+----------------+
| Model          | Platform | Latency (ms)  | Throughput     |
+================+==========+===============+================+
| MobileNet v2   | CPU      | 25.3          | 39.5 FPS       |
| MobileNet v2   | GPU      | 8.7           | 115.0 FPS      |
+----------------+----------+---------------+----------------+
```

### After (Optimal Alignment)
```markdown
+----------------+----------+---------------+----------------+
| Model          | Platform | Latency (ms)  | Throughput     |
+:===============+==========+==============:+===============:+
| MobileNet v2   |   CPU    |         25.3  |      39.5 FPS  |
| MobileNet v2   |   GPU    |          8.7  |     115.0 FPS  |
+----------------+----------+---------------+----------------+
```

**Analysis:**
- Column 1: Left-aligned (model names)
- Column 2: Center-aligned (platform categories)  
- Column 3: Right-aligned (numerical measurements)
- Column 4: Right-aligned (performance metrics)

## ✅ Quality Assurance

**Automated Checks:**
- ✅ Content preservation - All data intact
- ✅ Structure validation - Proper table format
- ✅ Alignment optimization - Content-appropriate alignment
- ✅ Consistency - Similar tables use same patterns

**Before Committing:**
1. **Test locally**: `--dry-run` first
2. **Review changes**: Check git diff
3. **Verify rendering**: Build and check output
4. **Run full check**: `--check-all` on entire book

## 🐛 Troubleshooting

### Common Issues

**"No data rows found"** warnings:
- Expected for non-standard table structures
- Only affects malformed tables, others process normally

**Content missing after formatting**:
- Check for complex multi-line cells
- Use `--verbose` for detailed parsing info
- Report as bug if data loss occurs

**Wrong alignment detected**:
- Review content analysis patterns
- May indicate need for pattern refinement
- Can be manually adjusted after formatting

### Best Practices

1. **Start small** - Test on individual files first
2. **Use dry-run** - Always preview changes
3. **Git safety** - Work on feature branches
4. **Regular formatting** - Integrate into workflow
5. **Review output** - Check edge cases manually

## 🚀 Performance

- **Fast processing**: ~1000 lines/second
- **Memory efficient**: Processes files incrementally  
- **Scalable**: Handles entire book content efficiently
- **Safe**: Multiple validation layers prevent data loss

## 📋 Quick Reference

```bash
# Most common commands
python3 scripts/content/format_grid_tables.py --check-all --dry-run  # Preview all
python3 scripts/content/format_grid_tables.py -f file.qmd           # Format one
python3 scripts/content/format_grid_tables.py -d directory/         # Format dir
./scripts/content/format_all_tables.sh                              # Interactive
```

For more examples, see `table_formatting_examples.qmd`. 