# Quick Start: Quarto Configuration System

## TL;DR

The Quarto configuration has been refactored to use shared configs. Here's what you need to know:

### Making Changes

**Shared settings** (affects HTML, PDF, EPUB):
```bash
# 1. Edit files in config/shared/
# 2. Regenerate configs
python scripts/generate_config.py
```

**Format-specific settings**:
```bash
# 1. Edit config/_quarto-{format}-overrides.yml
# 2. Regenerate that format
python scripts/generate_config.py html  # or pdf, epub
```

**Important**: Don't edit `_quarto-{format}.yml` files directly - they're auto-generated!

## Common Tasks

### Add a Bibliography File

```bash
# Edit config/shared/bibliography.yml
# Add: - contents/path/to/new.bib
python scripts/generate_config.py
```

### Change Book Title

```bash
# Edit config/shared/book-metadata.yml
# Update: title: "New Title"
python scripts/generate_config.py
```

### Modify HTML Navbar

```bash
# Edit config/_quarto-html-overrides.yml
# Update: website.navbar section
python scripts/generate_config.py html
```

### Add a New Callout Type

```bash
# Edit config/shared/filter-metadata.yml
# Add new group and class definitions
python scripts/generate_config.py
```

## File Locations

- **Shared configs**: `config/shared/`
- **Format overrides**: `config/_quarto-{format}-overrides.yml`
- **Generated configs**: `config/_quarto-{format}.yml` (don't edit!)
- **Generator script**: `scripts/generate_config.py`

## Before Building

Always ensure configs are up to date:

```bash
cd book/quarto
python scripts/generate_config.py
```

Then build as usual:

```bash
quarto render  # Uses HTML config by default
```

## Need More Help?

- **Detailed docs**: `scripts/README.md`
- **Refactoring details**: `CONFIGURATION_REFACTORING.md`
- **Shared configs**: `config/shared/README.md`

