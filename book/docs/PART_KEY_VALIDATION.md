# Part Key Validation System

## Overview

The part key validation system ensures that all `\part{key:xxx}` commands in your `.qmd` files reference valid keys defined in `quarto/part_summaries.yml`. This prevents build failures and ensures consistent part titles throughout your book.

## How It Works

### 1. Pre-Scan Validation (Lua Filter)

The `config/lua/inject-parts.lua` filter now includes a pre-scan step that:

- **Scans the entire document** before processing any blocks
- **Validates all keys** against `part_summaries.yml`
- **Reports issues immediately** with detailed error messages
- **Stops the build** if any invalid keys are found

### 2. Standalone Validation Script

The `scripts/validate_part_keys.py` script provides:

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
# Run pre-commit manually
pre-commit run validate-part-keys --all-files

# Or run the validation script directly
python3 scripts/validate_part_keys.py

# Or use the wrapper script
./scripts/check_keys.sh
```

### Pre-commit Installation

If you haven't installed pre-commit hooks yet:

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks on existing files
pre-commit run --all-files
```

## Available Keys

The following keys are defined in `quarto/part_summaries.yml`:

| Key | Title | Type |
|-----|-------|------|
| `frontmatter` | Frontmatter | Division |
| `main_content` | Main Content | Division |
| `foundations` | Foundations | Part |
| `principles` | Design Principles | Part |
| `optimization` | Performance Engineering | Part |
| `deployment` | Robust Deployment | Part |
| `trustworthy` | Trustworthy Systems | Part |
| `futures` | ML Systems Frontiers | Part |
| `labs` | Labs | Division |
| `arduino` | Arduino Labs | Lab |
| `xiao` | Seeed XIAO Labs | Lab |
| `grove` | Grove Vision Labs | Lab |
| `raspberry` | Raspberry Pi Labs | Lab |
| `shared` | Shared Labs | Lab |
| `backmatter` | Backmatter | Division |

## Error Handling

### Lua Filter Errors

When the Lua filter encounters an invalid key:

```
❌ CRITICAL ERROR: UNDEFINED KEY 'invalid_key' not found in part_summaries.yml
📍 Location: RawBlock processing
🔍 DEBUG: Key 'invalid_key' found in RawBlock
📍 RawBlock content: \part{key:invalid_key}
📍 RawBlock format: latex
🔍 Available keys: frontmatter, main_content, foundations, principles, optimization, deployment, trustworthy, futures, labs, arduino, xiao, grove, raspberry, shared, backmatter
💡 Check your .qmd files for \part{key:invalid_key} commands
🛑 Build stopped to prevent incorrect part titles.
```

### Python Script Errors

When the validation script finds issues:

```
❌ ISSUES FOUND:
   📄 book/contents/core/example.qmd:15
      - Key: 'invalid_key' (normalized: 'invalidkey')
      - Status: NOT FOUND in part_summaries.yml

💡 To fix these issues:
   1. Add the missing keys to quarto/part_summaries.yml
   2. Or correct the key names in the .qmd files
   3. Or remove the \part{key:xxx} commands if not needed
```

## Key Normalization

Keys are normalized for comparison by:

1. **Converting to lowercase**
2. **Removing underscores** (`_`)
3. **Removing hyphens** (`-`)

Examples:
- `main_content` → `maincontent`
- `trustworthy` → `trustworthy`
- `front-matter` → `frontmatter`

## Troubleshooting

### Common Issues

1. **Typo in key name**:
   ```qmd
   \part{key:trustworthy}  # ✅ Correct
   \part{key:trustworthy}  # ❌ Typo
   ```

2. **Missing key in part_summaries.yml**:
   ```yaml
   # Add to book/part_summaries.yml
   - key: "new_section"
     title: "New Section"
     description: "Description here"
   ```

3. **Incorrect normalization**:
   ```qmd
   \part{key:main_content}  # ✅ Will match 'maincontent'
   \part{key:main-content}  # ✅ Will match 'maincontent'
   ```

### Debugging

1. **Run validation script**:
   ```bash
   python3 scripts/validate_part_keys.py
   ```

2. **Check specific file**:
   ```bash
   grep -n "\\part{key:" book/contents/**/*.qmd
   ```

3. **View available keys**:
   ```bash
   python3 -c "
   import yaml
   with open('quarto/part_summaries.yml') as f:
       data = yaml.safe_load(f)
       for part in data['parts']:
           print(f\"'{part['key']}' -> '{part['title']}'\")
   "
   ```

## Best Practices

1. **Pre-commit hooks catch issues automatically**:
   ```bash
   git add .
   git commit -m "Your changes"
   # Pre-commit will validate and block if issues found
   ```

2. **Add new keys to part_summaries.yml first**:
   ```yaml
   - key: "new_section"
     title: "New Section"
     description: "Description here"
     type: "part"
     numbered: true
   ```

3. **Use consistent key naming**:
   - Use lowercase with underscores
   - Be descriptive but concise
   - Follow existing patterns

4. **Test changes**:
   ```bash
   # Test validation manually
   pre-commit run validate-part-keys --all-files

   # Or test a single file
   quarto render quarto/contents/core/example.qmd --to pdf
   ```

5. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Integration with Git Workflow

The validation is integrated into the pre-commit hooks to catch issues early:

1. **Pre-commit validation** runs before every commit
2. **Fails fast** if any issues are found
3. **Provides detailed error messages** for debugging
4. **Prevents broken commits** from being pushed

This ensures that all commits are consistent and error-free.

### Pre-commit Hook Configuration

The validation is configured in `.pre-commit-config.yaml`:

```yaml
- id: validate-part-keys
  name: "Validate part keys in .qmd files"
  entry: python scripts/validate_part_keys.py
  language: python
  additional_dependencies:
    - pyyaml
  pass_filenames: false
  files: ''
```

---

*Last updated: $(date)*
*Validation script: `scripts/validate_part_keys.py`*
*Lua filter: `config/lua/inject-parts.lua`*
