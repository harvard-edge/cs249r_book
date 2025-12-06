# TinyTorch PDF Book Generation

This directory contains the configuration for generating the TinyTorch course as a PDF book.

## Building the PDF

To build the PDF version of the TinyTorch course:

```bash
# Install Jupyter Book if not already installed
pip install jupyter-book

# Build the PDF (from the docs/ directory)
jupyter-book build . --builder pdflatex

# Or from the repository root:
jupyter-book build docs --builder pdflatex
```

The generated PDF will be in `docs/_build/latex/tinytorch-course.pdf`.

## Structure

- `_config_pdf.yml` - Jupyter Book configuration optimized for PDF output
- `_toc_pdf.yml` - Linear table of contents for the PDF book
- `cover.md` - Cover page for the PDF
- `preface.md` - Preface explaining the book's approach and philosophy

## Content Sources

The PDF pulls content from:
- **Module ABOUT.md files**: `../modules/XX_*/ABOUT.md` - Core technical content
- **Site files**: `../site/*.md` - Introduction, quick start guide, resources
- **Site chapters**: `../site/chapters/*.md` - Course overview and milestones

All content is sourced from a single location and reused for both the website and PDF, ensuring consistency.

## Customization

### PDF-Specific Settings

The `_config_pdf.yml` includes PDF-specific settings:
- Disabled notebook execution (`execute_notebooks: "off"`)
- LaTeX engine configuration
- Custom page headers and formatting
- Paper size and typography settings

### Chapter Ordering

The `_toc_pdf.yml` provides linear chapter ordering suitable for reading cover-to-cover, unlike the website's multi-section structure.

## Dependencies

Building the PDF requires:
- `jupyter-book`
- `pyppeteer` (for HTML to PDF conversion)
- LaTeX distribution (e.g., TeX Live, MiKTeX)
- `latexmk` (usually included with LaTeX distributions)

## Troubleshooting

**LaTeX errors**: Ensure you have a complete LaTeX distribution installed
**Missing fonts**: Install the required fonts for the logo and styling
**Build timeouts**: Increase the timeout in `_config_pdf.yml` if needed

## Future Enhancements

Planned improvements for the PDF:
- Custom LaTeX styling for code blocks
- Better figure placement and captions
- Index generation
- Cross-reference optimization
- Improved table formatting
