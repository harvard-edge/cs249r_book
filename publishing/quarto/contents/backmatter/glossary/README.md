# Glossary Maintenance

The book has two independent glossaries:

- Volume 1: `book/quarto/contents/vol1/backmatter/glossary/glossary.qmd`
- Volume 2: `book/quarto/contents/vol2/backmatter/glossary/glossary.qmd`

These QMD files are the source of truth and the published artifacts. The PDF,
HTML, and EPUB configurations render these files directly.

The previous JSON-based glossary pipeline has been retired. Do not add
chapter-level or regenerated JSON files for glossary data. Those files created a
stale parallel source of truth and were not read directly by the book build.

## Update Workflow

1. Edit the appropriate volume's `glossary.qmd`.
2. Keep Volume 1 and Volume 2 terms separate.
3. Add only terms that are useful to readers of that volume.
4. Rebuild the affected volume.

Useful command:

```bash
./binder fix glossary paths
```

This prints the canonical glossary files that the book uses.
