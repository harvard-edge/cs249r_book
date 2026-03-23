# MLSysBook CLI (`binder`) — Architecture & Reference

The `./book/binder` CLI is the single entry point for building, validating, and formatting the MLSysBook. All pre-commit hooks route through binder.

## Architecture

```
cli/
├── main.py                  # CLI router, help system (Rich UI)
├── core/
│   ├── config.py            # Quarto config management (symlinks, switching)
│   └── discovery.py         # Chapter/file discovery, fuzzy matching
├── commands/
│   ├── build.py             # Build HTML/PDF/EPUB (full book or chapters)
│   ├── preview.py           # Live dev server with hot reload
│   ├── validate.py          # All validation checks (check group)
│   ├── formatting.py        # Auto-formatters (format group)
│   ├── reference_check.py   # Bibliography verification (Crossref/DOI)
│   ├── bib.py               # Bibliography management
│   ├── info.py              # Stats, figures, concepts, headers, acronyms
│   ├── render.py            # Plot rendering to PNG gallery
│   ├── newsletter.py        # Newsletter drafts and publishing
│   ├── clean.py             # Build artifact cleanup
│   ├── debug.py             # Failing chapter/section finder
│   ├── doctor.py            # Health check
│   └── maintenance.py       # Image compression, repo health
├── formats/                 # Format-specific handlers
└── utils/                   # Shared utilities
```

## Command Groups

### Build & Preview

```bash
./binder build pdf --vol1          # Build Volume I PDF
./binder build html --vol2         # Build Volume II website
./binder build pdf intro           # Single chapter PDF
./binder preview                   # Live dev server (full book)
./binder preview intro             # Live dev server (single chapter)
```

### Check (Validation)

All validation is in `validate.py`. Each check belongs to a **group** and has a **scope**.

```bash
./binder check <group>                    # Run all scopes in a group
./binder check <group> --scope <name>     # Run one specific scope
./binder check all                        # Run everything
./binder check all --vol1                 # Volume I only
```

#### Check Groups & Scopes

| Group | Scope | What it checks |
|-------|-------|----------------|
| **refs** | `python-syntax` | Python code block syntax errors |
| | `inline-python` | Inline `{python}` reference validity |
| | `cross-refs` | `@sec-`, `@fig-`, `@tbl-` reference targets exist |
| | `citations` | `[@key]` citation keys exist in `.bib` |
| | `inline` | Inline ref patterns, scope analysis |
| | `self-ref` | Self-referential cross-references |
| **labels** | `duplicates` | Duplicate `#fig-`, `#tbl-`, `#lst-` labels |
| | `orphans` | Defined labels never referenced |
| | `fig-labels` | Underscore in figure label IDs |
| **headers** | `ids` | All `##` sections have `{#sec-...}` IDs |
| **footnotes** | `placement` | Footnotes not in tables/captions |
| | `integrity` | Every `[^fn-...]` reference has a definition |
| | `cross-chapter` | Duplicate footnote IDs across chapters |
| **figures** | `captions` | Figures have `fig-cap` and `fig-alt` |
| | `div-syntax` | Figures use div syntax (not markdown-image) |
| | `flow` | Figures placed near first reference |
| | `files` | Referenced image files exist on disk |
| **rendering** | `patterns` | LaTeX+Python rendering hazards |
| | `python-echo` | Python blocks have `echo: false` |
| | `indexes` | `\index{}` not inline with headings |
| | `dropcaps` | Drop cap compatibility |
| | `parts` | Part key validation |
| | `heading-levels` | No skipped heading levels (##→####) |
| | `duplicate-words` | Repeated consecutive words |
| | `grid-tables` | Warn about grid tables (prefer pipe) |
| | `tables` | Table content validation |
| | `ascii` | Non-ASCII characters in prose |
| | `percent-spacing` | No space between value and `%` |
| | `unit-spacing` | Space between number and unit (`100 ms`) |
| | `binary-units` | Use `GB`/`TB` not `GiB`/`TiB` |
| | `contractions` | No contractions in body prose |
| | `unblended-prose` | No split paragraphs |
| | `times-spacing` | Space after `$\times$` before word |
| | `times-product-spacing` | Space before `$\times$` after inline code |
| | `purpose-unnumbered` | Purpose sections have `{.unnumbered}` |
| | `div-fences` | Malformed `:::` / `::::` fences |
| **images** | `formats` | Image file format validation |
| | `external` | No external image URLs |
| | `svg-xml` | SVG XML well-formedness |
| **json** | `syntax` | JSON file syntax |
| **units** | `physics` | Pint unit consistency in `mlsys/` |
| **spelling** | `prose` | Spell check prose content |
| | `tikz` | Spell check TikZ labels |
| **epub** | `structure` | EPUB structure validation |
| **sources** | `citations` | Source citation verification |
| **references** | `hallucinator` | Bibliography entry verification (Crossref/DOI) |
| **content** | `tree` | Content tree structure |

### Format (Auto-formatters)

```bash
./binder format blanks             # Collapse extra blank lines
./binder format python             # Format Python blocks (Black, 70 chars)
./binder format lists              # Fix list spacing
./binder format divs               # Fix div/callout spacing
./binder format tables             # Format grid tables
./binder format prettify           # Align pipe table columns
./binder format all                # Run all formatters
```

### Other Commands

```bash
./binder info stats --vol1         # Word counts, figure counts
./binder info figures --vol1       # Extract figure list
./binder info concepts --vol1      # Extract concept list
./binder bib sync --vol1           # Sync bibliography
./binder render plots --vol1       # Render matplotlib plots to PNG
./binder clean                     # Remove build artifacts
./binder doctor                    # Comprehensive health check
./binder debug pdf --vol1          # Find failing chapter in PDF build
```

## Pre-commit Integration

Every pre-commit hook routes through `./book/binder`. The pattern is:

```yaml
- id: book-check-<name>
  entry: ./book/binder check <group> --scope <scope>
```

No inline bash scripts. One CLI, one validation framework, consistent output.

## Script Delegation

Some commands still delegate to scripts under `book/tools/scripts/` (spelling, epub, image formats, table formatting, Python formatting). These will be migrated to native CLI modules over time.
