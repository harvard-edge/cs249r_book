# Part Key Validation System

## Overview

The part key validation system ensures that all `\part{key:xxx}` commands in your `.qmd` files reference valid keys defined in `quarto/contents/parts/summaries.yml`. This prevents build failures and ensures consistent part titles throughout your book.

## How It Works

### 1. Pre-Scan Validation (Lua Filter)

The `quarto/filters/inject_parts.lua` filter includes a pre-scan step that:

- **Scans the entire document** before processing any blocks
- **Validates all keys** against `summaries.yml`
- **Reports issues immediately** with detailed error messages
- **Stops the build** if any invalid keys are found

### 2. Standalone Validation Script

The `tools/scripts/utilities/validate_part_keys.py` script provides:

- **Independent validation** without running the full build
- **Comprehensive reporting** of all issues
- **File and line number** information for each problem
- **Available keys listing** for reference

## Usage

### Pre-commit Hook (Recommended)

The validation runs automatically on every commit:

```bash
# Pre-commit will run validation automatically
git add .
git commit -m "Your commit message"
# If there are invalid keys, the commit will be blocked
```

### Manual Validation

```bash
# Run the validation script from the book directory
python3 tools/scripts/utilities/validate_part_keys.py

# Or use the wrapper script
./tools/scripts/check_keys.sh
```

## Available Keys

The following keys are defined in `quarto/contents/parts/summaries.yml`:

| Key | Title | Type |
|-----|-------|------|
| `frontmatter` | Frontmatter | Division |
| `volume_1` | Volume I: Introduction to ML Systems | Division |
| `vol1_foundations` | ML Foundations | Part |
| `vol1_development` | System Development | Part |
| `vol1_optimization` | Model Optimization | Part |
| `vol1_operations` | System Operations | Part |
| `volume_2` | Volume II: Advanced ML Systems | Division |
| `vol2_scale` | Foundations of Scale | Part |
| `vol2_distributed` | Distributed Systems | Part |
| `vol2_production` | Production Challenges | Part |
| `vol2_responsible` | Responsible Deployment | Part |
| `backmatter` | References | Division |
| `references` | Bibliography | Lab |

## Error Handling

### Lua Filter Errors

When the Lua filter encounters an invalid key:

```
❌ CRITICAL ERROR: UNDEFINED KEY 'invalid_key' not found in summaries.yml
📍 Location: RawBlock processing
🔍 DEBUG: Key 'invalid_key' found in RawBlock
📍 RawBlock content: \part{key:invalid_key}
📍 RawBlock format: latex
🔍 Available keys: frontmatter, volume_1, vol1_foundations, vol1_development, vol1_optimization, vol1_operations, volume_2, vol2_scale, vol2_distributed, vol2_production, vol2_responsible, backmatter, references
💡 Check your .qmd files for \part{key:invalid_key} commands
🛑 Build stopped to prevent incorrect part titles.
```

### Python Script Errors

When the validation script finds issues:

```
❌ ISSUES FOUND:
   📄 quarto/contents/vol1/introduction/introduction.qmd:15
      - Key: 'invalid_key' (normalized: 'invalidkey')
      - Status: NOT FOUND in summaries.yml

💡 To fix these issues:
   1. Add the missing keys to quarto/contents/parts/summaries.yml
   2. Or correct the key names in the .qmd files
   3. Or remove the \part{key:xxx} commands if not needed
```

## Key Normalization

Keys are normalized for comparison by:

1. **Converting to lowercase**
2. **Removing underscores** (`_`)
3. **Removing hyphens** (`-`)

Examples:
- `vol1_foundations` → `vol1foundations`
- `volume_1` → `volume1`
- `front-matter` → `frontmatter`

## Troubleshooting

### Common Issues

1. **Typo in key name**:
   ```qmd
   \part{key:vol1_foundations}  # ✅ Correct
   \part{key:vol1_foundationss}  # ❌ Typo
   ```

2. **Missing key in summaries.yml**:
   ```yaml
   # Add to quarto/contents/parts/summaries.yml
   - key: "new_section"
     title: "New Section"
     description: "Description here"
   ```

3. **Incorrect normalization**:
   ```qmd
   \part{key:vol1_foundations}  # ✅ Will match 'vol1foundations'
   \part{key:vol1-foundations}  # ✅ Will match 'vol1foundations'
   ```

### Debugging

1. **Run validation script**:
   ```bash
   python3 tools/scripts/utilities/validate_part_keys.py
   ```

2. **Check specific file**:
   ```bash
   grep -n "\\part{key:" quarto/contents/**/*.qmd
   ```

---

*Last updated: January 2026*
*Validation script: `tools/scripts/utilities/validate_part_keys.py`*
*Lua filter: `quarto/filters/inject_parts.lua`*
