# Binder: Native vs script delegation

Binder is the canonical CLI. This doc lists which commands are **fully self-contained** (no dependency on `book/tools/scripts/`) and which still **delegate** to scripts.

## Fully native (no tools/scripts)

| Command | Implementation |
|--------|----------------|
| `validate references` | `book/cli/commands/reference_check.py` — all logic in CLI |
| Build, preview, doctor, bib (betterbib), render, info | Use Quarto, system tools, or in-process logic only |
| `maintain images compress` | Uses ImageMagick via subprocess; no scripts |
| `maintain repo-health` | Uses git commands; no scripts |
| Unit-tests check | Runs `book/quarto/mlsys/test_units.py` (book package, not tools/scripts) |

## Still delegate to book/tools/scripts

| Command / check | Script path |
|-----------------|-------------|
| **validate** table-content | `tools/scripts/content/validate_tables.py` |
| **validate** spelling-prose | `tools/scripts/content/check_prose_spelling.py` |
| **validate** spelling-tikz | `tools/scripts/content/check_tikz_spelling.py` |
| **validate** epub | `tools/scripts/utilities/validate_epub.py` |
| **validate** sources | `tools/scripts/utilities/manage_sources.py` |
| **validate** grid-tables | `tools/scripts/utilities/convert_grid_to_pipe_tables.py` |
| **validate** image-formats | `tools/scripts/images/manage_images.py` |
| **validate** external images | `tools/scripts/images/manage_external_images.py` |
| **clean** (build artifacts) | `tools/scripts/maintenance/cleanup_build_artifacts.py` |
| **format** python | `tools/scripts/content/format_python_in_qmd.py` |
| **format** tables | `tools/scripts/content/format_tables.py` |
| **format** divs | `tools/scripts/content/format_div_spacing.py` |
| **format** prettify | `tools/scripts/utilities/prettify_pipe_tables.py` |
| **debug** (section splitter) | Imports `tools/scripts/content/section_splitter.py` |

## Goal

Over time, these should be migrated into native CLI modules (under `book/cli/commands/`) so Binder is fully self-contained and does not depend on `book/tools/scripts/`.
