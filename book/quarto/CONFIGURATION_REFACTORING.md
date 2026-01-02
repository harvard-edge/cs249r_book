# Quarto Configuration Refactoring

## Summary

This document describes the refactoring of Quarto configuration files to eliminate code duplication and improve maintainability across HTML, PDF, and EPUB output formats.

## Problem

Previously, the project had significant code duplication across three format-specific configuration files:
- `config/_quarto-html.yml` (~560 lines)
- `config/_quarto-pdf.yml` (~480 lines)
- `config/_quarto-epub.yml` (~400 lines)

**Duplicated sections included:**
- Bibliography lists (identical across all formats)
- Book metadata (title, author, abstract - mostly identical)
- Chapter structure (similar with format-specific variations)
- Cross-reference configuration (identical)
- Diagram/TikZ configuration (identical)
- Filter metadata (mostly identical with minor format-specific differences)

This duplication made maintenance difficult:
- Changes needed to be applied in multiple places
- Risk of inconsistencies between formats
- Difficult to ensure all formats stay in sync

## Solution

A **shared configuration system** was implemented with:

1. **Shared Configuration Files** (`config/shared/`):
   - `bibliography.yml` - Bibliography entries
   - `book-metadata.yml` - Book title, author, abstract
   - `chapters.yml` - Chapter structure
   - `crossref.yml` - Cross-reference settings
   - `diagram.yml` - TikZ/diagram configuration
   - `filter-metadata.yml` - Filter metadata

2. **Format-Specific Overrides** (`config/_quarto-{format}-overrides.yml`):
   - Contains only format-unique settings
   - HTML: Website navigation, sidebar, HTML format options
   - PDF: PDF format options, title page, LaTeX includes
   - EPUB: EPUB metadata, part structure, EPUB format options

3. **Configuration Generator** (`scripts/generate_config.py`):
   - Merges shared configs with format-specific overrides
   - Applies format-specific logic (e.g., icon format, collapse behavior)
   - Generates final `_quarto-{format}.yml` files

## Benefits

1. **Reduced Duplication**: Common settings defined once (~70% reduction in duplicated code)
2. **Easier Maintenance**: Update shared settings in one place
3. **Consistency**: Automatic synchronization across formats
4. **Flexibility**: Format-specific customizations remain easy to manage
5. **Documentation**: Clear separation of shared vs. format-specific settings

## Migration Guide

### For Contributors

**Before making configuration changes:**

1. **Shared settings** (affects all formats):
   - Edit files in `config/shared/`
   - Run: `python scripts/generate_config.py`

2. **Format-specific settings**:
   - Edit `config/_quarto-{format}-overrides.yml`
   - Run: `python scripts/generate_config.py {format}`

**Important**: Do NOT edit the generated `_quarto-{format}.yml` files directly. They are auto-generated and will be overwritten.

### Build Process

The build process remains the same. The main `_quarto.yml` file still points to the generated config files:

```yaml
config/_quarto-html.yml  # Generated from shared configs + HTML overrides
```

Before building, ensure configs are up to date:

```bash
# Generate all configs
python scripts/generate_config.py

# Or generate specific format
python scripts/generate_config.py html
```

## File Structure

```
book/quarto/
├── config/
│   ├── shared/                          # Shared configuration (NEW)
│   │   ├── bibliography.yml
│   │   ├── book-metadata.yml
│   │   ├── chapters.yml
│   │   ├── crossref.yml
│   │   ├── diagram.yml
│   │   └── filter-metadata.yml
│   │
│   ├── _quarto-html-overrides.yml       # HTML-specific (NEW)
│   ├── _quarto-pdf-overrides.yml        # PDF-specific (NEW)
│   ├── _quarto-epub-overrides.yml       # EPUB-specific (NEW)
│   │
│   ├── _quarto-html.yml                 # Generated (auto-created)
│   ├── _quarto-pdf.yml                  # Generated (auto-created)
│   └── _quarto-epub.yml                 # Generated (auto-created)
│
└── scripts/
    └── generate_config.py               # Configuration generator (NEW)
```

## Usage Examples

### Adding a New Bibliography File

**Before** (needed to edit 3 files):
```bash
# Edit config/_quarto-html.yml
# Edit config/_quarto-pdf.yml
# Edit config/_quarto-epub.yml
```

**After** (edit 1 file):
```bash
# Edit config/shared/bibliography.yml
python scripts/generate_config.py
```

### Changing Book Title

**Before** (needed to edit 3 files):
```bash
# Edit config/_quarto-html.yml
# Edit config/_quarto-pdf.yml
# Edit config/_quarto-epub.yml
```

**After** (edit 1 file):
```bash
# Edit config/shared/book-metadata.yml
python scripts/generate_config.py
```

### Modifying HTML Navbar

**Before** (edit HTML config):
```bash
# Edit config/_quarto-html.yml
```

**After** (edit HTML override):
```bash
# Edit config/_quarto-html-overrides.yml
python scripts/generate_config.py html
```

## Technical Details

### Deep Merge Algorithm

The configuration generator uses a deep merge algorithm:

- **Dictionaries**: Recursively merged (override values take precedence)
- **Lists**: Completely replaced by override (no merging)
- **Primitive values**: Override replaces base value

### Format-Specific Logic

The generator automatically applies format-specific settings:

- **Icon format**: `png` for HTML/EPUB, `pdf` for PDF
- **Collapse behavior**: Format-specific defaults for quiz callouts
- **Cross-reference files**: Different JSON files for HTML/EPUB vs. PDF
- **Diagram output**: SVG for HTML, native TikZ for PDF/EPUB

## Testing

After refactoring, verify:

1. **HTML build** works correctly:
   ```bash
   cd book/quarto
   python scripts/generate_config.py html
   quarto render --to html
   ```

2. **PDF build** works correctly:
   ```bash
   cd book/quarto
   python scripts/generate_config.py pdf
   quarto render --to pdf
   ```

3. **EPUB build** works correctly:
   ```bash
   cd book/quarto
   python scripts/generate_config.py epub
   quarto render --to epub
   ```

## Future Enhancements

Potential improvements:

- [ ] Pre-commit hook to auto-regenerate configs
- [ ] Configuration validation against Quarto schema
- [ ] Diff tool to see config changes
- [ ] CI/CD integration for auto-generation
- [ ] Support for environment-specific overrides

## Questions?

See `scripts/README.md` for detailed usage instructions and troubleshooting.

