# Scripts Directory

Automation scripts and tools for the Machine Learning Systems textbook.

## Deprecation Note

For workflows now exposed by Binder, prefer `./book/binder ...` commands over direct script execution.

- Validation checks: use `./book/binder validate ...`
- Maintenance utilities: use `./book/binder maintain ...`

Scripts remain available as internal utilities, but direct invocation is soft-deprecated for Binder-covered tasks.

## Directory Structure

```
scripts/
├── common/           Shared base classes, config, logging, validators
├── content/          Content validation, formatting, and editing tools
├── docs/             Script documentation
├── genai/            AI-assisted tools (quizzes, footnotes, dash fixes)
├── glossary/         Glossary generation and consolidation
├── images/           Image processing, compression, validation
├── infrastructure/   CI/CD and Docker utilities
├── maintenance/      Repo health, image casing, build artifact cleanup
├── publish/          MIT Press release builder, figure extraction, deployment
├── socratiQ/         SocratiQ integration
├── testing/          Debug builds, test runners, linters
└── utilities/        Footnote analysis, ref auditing, JSON/EPUB validation
```

## Key Scripts by Task

### Content Editing
- `content/format_blank_lines.py` - Normalize blank lines in .qmd files
- `content/format_tables.py` - Format Quarto tables
- `content/section_splitter.py` - Split chapters into sections for processing
- `content/relocate_figures.py` - Move figures closer to first reference
- `content/manage_section_ids.py` - Manage `@sec-` cross-reference IDs

### Validation
- `content/check_duplicate_labels.py` - Find duplicate labels
- `content/check_fig_references.py` - Validate figure references
- `content/check_unreferenced_labels.py` - Find unused labels
- `content/validate_citations.py` - Check citation formatting
- `utilities/validate_epub.py` - Validate EPUB output
- `utilities/validate_json.py` - Validate JSON files

### Publishing
- `publish/mit-press-release.sh` - Build MIT Press PDFs (regular or copy-edit)
- `publish/extract_figures.py` - Extract figure lists for MIT Press submission
- `publish/publish.sh` - Full release workflow with versioning
- `publish/render_compress_publish.py` - Render, compress, and publish

### Images
- `images/compress_images.py` - Compress images for web/PDF
- `images/validate_image_references.py` - Check image references
- `images/convert_svg_to_png.py` - SVG to PNG conversion

### Glossary
- `glossary/build_global_glossary.py` - Build master glossary from chapters
- `glossary/consolidate_similar_terms.py` - Merge near-duplicate terms

### AI Tools
- `genai/quizzes.py` - Generate quiz questions
- `genai/footnote_assistant.py` - AI-assisted footnote writing

## Usage

All Python scripts use `python3`. Most support `--help` for options.

```bash
python3 book/tools/scripts/content/format_blank_lines.py path/to/file.qmd
python3 book/tools/scripts/publish/extract_figures.py --vol 1
./book/tools/scripts/publish/mit-press-release.sh --vol1
```
