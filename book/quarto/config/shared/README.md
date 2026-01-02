# Shared Configuration Files

This directory contains shared configuration files that are used across all Quarto output formats (HTML, PDF, EPUB).

## Files

- **`bibliography.yml`**: Bibliography entries used in all formats
- **`book-metadata.yml`**: Book title, author, abstract, cover image, etc.
- **`chapters.yml`**: Chapter structure (note: EPUB uses parts structure defined in overrides)
- **`crossref.yml`**: Cross-reference configuration
- **`diagram.yml`**: TikZ and diagram engine configuration
- **`filter-metadata.yml`**: Filter metadata (quizzes, callouts, cross-references, etc.)

## Usage

**Do NOT edit these files directly unless you understand the impact on all formats.**

After editing shared configs, regenerate the format-specific configs:

```bash
cd book/quarto
python scripts/generate_config.py
```

## Making Changes

### Adding a Bibliography File

Edit `bibliography.yml` and add the new `.bib` file path to the list.

### Changing Book Metadata

Edit `book-metadata.yml` to update title, author, abstract, etc.

### Modifying Chapter Structure

Edit `chapters.yml` to add, remove, or reorder chapters.

**Note**: EPUB uses a different structure with parts. See `../_quarto-epub-overrides.yml` for EPUB-specific chapter structure.

### Updating Filter Metadata

Edit `filter-metadata.yml` to modify:
- Quiz configuration
- Cross-reference settings
- Custom callout types and styling

## Format-Specific Overrides

Format-specific settings are defined in:
- `../_quarto-html-overrides.yml` - HTML/website specific
- `../_quarto-pdf-overrides.yml` - PDF specific
- `../_quarto-epub-overrides.yml` - EPUB specific

See the main [README.md](../scripts/README.md) for more information.

