# Quarto Configuration Management

This directory contains scripts for managing Quarto book configurations and reducing code duplication across output formats (HTML, PDF, EPUB).

## Overview

The Quarto book project uses a **shared configuration system** that eliminates duplication by:

1. **Shared Configuration Files** (`config/shared/`): Common settings used across all formats
2. **Format-Specific Overrides** (`config/_quarto-{format}-overrides.yml`): Unique settings for each format
3. **Configuration Generator** (`generate_config.py`): Merges shared configs with format-specific overrides

## Directory Structure

```
book/quarto/
├── config/
│   ├── shared/                    # Shared configuration files
│   │   ├── bibliography.yml       # Bibliography entries (all formats)
│   │   ├── book-metadata.yml      # Book title, author, abstract, etc.
│   │   ├── chapters.yml           # Chapter structure
│   │   ├── crossref.yml           # Cross-reference settings
│   │   ├── diagram.yml            # TikZ/diagram configuration
│   │   └── filter-metadata.yml    # Filter metadata (quizzes, callouts, etc.)
│   │
│   ├── _quarto-html-overrides.yml # HTML-specific overrides
│   ├── _quarto-pdf-overrides.yml  # PDF-specific overrides
│   ├── _quarto-epub-overrides.yml # EPUB-specific overrides
│   │
│   ├── _quarto-html.yml           # Generated HTML config (DO NOT EDIT)
│   ├── _quarto-pdf.yml            # Generated PDF config (DO NOT EDIT)
│   └── _quarto-epub.yml           # Generated EPUB config (DO NOT EDIT)
│
└── scripts/
    ├── generate_config.py         # Configuration generator script
    └── README.md                  # This file
```

## Usage

### Generating Configuration Files

To generate configuration files from shared configs:

```bash
# Generate all formats (HTML, PDF, EPUB)
python scripts/generate_config.py

# Generate specific format
python scripts/generate_config.py html
python scripts/generate_config.py pdf
python scripts/generate_config.py epub
```

### Making Changes

**To modify shared settings** (affects all formats):
1. Edit files in `config/shared/`
2. Regenerate configs: `python scripts/generate_config.py`

**To modify format-specific settings**:
1. Edit `config/_quarto-{format}-overrides.yml`
2. Regenerate that format: `python scripts/generate_config.py {format}`

**Important**: Do NOT edit the generated `_quarto-{format}.yml` files directly. They are auto-generated and will be overwritten.

## Configuration Files

### Shared Configuration Files

#### `bibliography.yml`
Contains all bibliography entries used across all formats. To add or remove bibliography files, edit this file.

#### `book-metadata.yml`
Contains book metadata:
- Title, subtitle, author information
- Abstract
- Cover image settings
- Page footer configuration

#### `chapters.yml`
Defines the chapter structure. Note: EPUB uses a different structure with parts, which is defined in `_quarto-epub-overrides.yml`.

#### `crossref.yml`
Cross-reference configuration (appendix settings, custom reference types).

#### `diagram.yml`
TikZ and diagram engine configuration. Format-specific output settings (e.g., SVG for HTML) are applied automatically.

#### `filter-metadata.yml`
Filter metadata including:
- Quiz configuration
- Cross-reference injection settings
- Part summaries
- Custom numbered blocks (callouts) configuration

Format-specific settings (like icon format, collapse behavior) are automatically applied by the generator.

### Format-Specific Override Files

These files contain only the unique settings for each format:

- **HTML**: Website navigation, sidebar, navbar, HTML format options
- **PDF**: PDF-specific format options, title page configuration, LaTeX includes
- **EPUB**: EPUB metadata, part structure, EPUB format options

## How It Works

The `generate_config.py` script:

1. **Loads shared configs**: Reads all files from `config/shared/` and merges them
2. **Loads format overrides**: Reads format-specific override file
3. **Merges configurations**: Deep merges shared config with format overrides
4. **Applies format-specific logic**: Automatically sets format-specific values (e.g., icon format, collapse behavior)
5. **Writes generated config**: Creates the final `_quarto-{format}.yml` file

### Deep Merge Behavior

- **Dictionaries**: Recursively merged (override values take precedence)
- **Lists**: Completely replaced by override (no merging)
- **Primitive values**: Override replaces base value

## Benefits

1. **Reduced Duplication**: Common settings defined once in shared configs
2. **Consistency**: Changes to shared settings automatically apply to all formats
3. **Maintainability**: Easier to update and maintain configuration
4. **Flexibility**: Format-specific customizations remain easy to manage

## Workflow Examples

### Adding a New Bibliography File

1. Edit `config/shared/bibliography.yml`
2. Add the new `.bib` file path
3. Run: `python scripts/generate_config.py`
4. All formats (HTML, PDF, EPUB) now include the new bibliography

### Changing Book Title

1. Edit `config/shared/book-metadata.yml`
2. Update the `title` field
3. Run: `python scripts/generate_config.py`
4. All formats now use the new title

### Modifying HTML Navbar

1. Edit `config/_quarto-html-overrides.yml`
2. Update the `website.navbar` section
3. Run: `python scripts/generate_config.py html`
4. HTML config is updated

### Adding a New Custom Callout Type

1. Edit `config/shared/filter-metadata.yml`
2. Add the new callout group and class definitions
3. Run: `python scripts/generate_config.py`
4. All formats now support the new callout type

## Troubleshooting

### Generated config doesn't match expected output

1. Check that shared configs are correctly formatted YAML
2. Verify format-specific overrides are properly structured
3. Run with verbose output to see merge process

### Format-specific setting not applied

1. Check if the setting should be in shared config or format override
2. Verify the override file is correctly named and located
3. Ensure the generator's format-specific logic handles the setting

### YAML parsing errors

1. Validate YAML syntax using a YAML validator
2. Check for indentation issues (YAML is sensitive to spacing)
3. Verify all list items are properly formatted

## Future Improvements

Potential enhancements to the configuration system:

- [ ] Validation of generated configs against Quarto schema
- [ ] Automatic regeneration on shared config changes (pre-commit hook)
- [ ] Support for environment-specific overrides (dev, staging, production)
- [ ] Configuration diff tool to see what changed between generations
- [ ] Integration with CI/CD to auto-generate configs on changes

## Related Documentation

- [Quarto Project Configuration](https://quarto.org/docs/projects/)
- [Quarto Book Format](https://quarto.org/docs/books/)
- [Quarto Website Format](https://quarto.org/docs/websites/)
