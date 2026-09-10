# Citation Validation

## Overview

The citation validation script ensures that all `@key` citation references in `.qmd` files have corresponding entries in their associated `.bib` files. This prevents Quarto build failures caused by missing bibliography entries.

## Automatic Validation (Pre-commit Hook)

The validation runs automatically when you commit `.qmd` files:

```bash
git add myfile.qmd
git commit -m "Update chapter"
```

If there are missing citations, the commit will fail with output like:

```
❌ CITATION VALIDATION FAILED

The following .qmd files reference citations that are missing from their .bib files:

📄 quarto/contents/core/conclusion/conclusion.qmd:
   ❌ @koomey2011web
   ❌ @han2015deep
   ❌ @openai2023gpt4
   ❌ @vaswani2017attention
```

## Manual Validation

You can also run the validation manually:

### Validate specific files
```bash
python tools/scripts/content/validate_citations.py chapter.qmd
```

### Validate all files in a directory
```bash
python tools/scripts/content/validate_citations.py -d quarto/contents/core/
```

### Quiet mode (only show errors)
```bash
python tools/scripts/content/validate_citations.py -d quarto/contents/ --quiet
```

## How It Works

1. **Extracts bibliography file**: Reads the `bibliography:` field from the `.qmd` file's YAML frontmatter
2. **Finds citations**: Scans the `.qmd` file for all `@key` patterns
3. **Filters false positives**: Excludes cross-reference labels like `@fig-`, `@tbl-`, `@sec-`, etc.
4. **Validates**: Checks that each citation key exists in the `.bib` file
5. **Reports**: Lists any missing citations

## Fixing Missing Citations

When validation fails, you have several options:

### Option 1: Copy from another chapter's .bib file

1. Search for the citation key in other `.bib` files:
   ```bash
   grep -r "@article{koomey2011web" quarto/contents/
   ```

2. Copy the BibTeX entry to your chapter's `.bib` file

### Option 2: Search online databases

Use tools like:
- [Google Scholar](https://scholar.google.com/) - Export BibTeX
- [Semantic Scholar](https://www.semanticscholar.org/)
- [DBLP](https://dblp.org/) for computer science papers

### Option 3: Remove the citation

If the citation is no longer needed, remove the `@key` reference from the `.qmd` file.

## Supported Citation Formats

The validator recognizes various citation formats:

- `[@key]` - Standard citation
- `@key` - Inline citation
- `[@key1; @key2]` - Multiple citations
- `[-@key]` - Suppress author citation
- `[@key, p. 123]` - Citation with page numbers

## Common Issues

### False Positives

The validator filters out cross-references to figures, tables, sections, etc. These are **not** citations:
- `@fig-architecture` ✓ (ignored)
- `@tbl-results` ✓ (ignored)
- `@sec-introduction` ✓ (ignored)
- `@eq-formula` ✓ (ignored)

### Bibliography File Not Found

If you see:
```
❌ Bibliography file not found: chapter.bib
```

Make sure:
1. The `.bib` file exists in the same directory as the `.qmd` file
2. The `bibliography:` field in the YAML frontmatter is correct

## Integration with Build Process

This validation runs **before** the Quarto build, catching issues early:

```
Pre-commit → Citation Validation → Git Commit → Quarto Build
```

This saves time by preventing failed builds due to missing citations.

## Examples

### Successful validation
```bash
$ python tools/scripts/content/validate_citations.py conclusion.qmd
✅ All citations validated successfully (1 files checked)
```

### Failed validation
```bash
$ python tools/scripts/content/validate_citations.py conclusion.qmd

❌ CITATION VALIDATION FAILED

📄 quarto/contents/core/conclusion/conclusion.qmd:
   ❌ @smith2020deep
   ❌ @jones2021ml

To fix these issues:
1. Find the citation entry in another chapter's .bib file
2. Copy the BibTeX entry to the appropriate .bib file
3. Or remove the citation reference if it's no longer needed
```

## See Also

- `clean_bibliographies.py` - Remove unused entries from `.bib` files
- `fix_bibliography.py` - Update citation key formats
- `.pre-commit-config.yaml` - Pre-commit hook configuration
