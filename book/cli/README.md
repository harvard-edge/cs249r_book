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
| **epub** | `hygiene` | Fast SVG/BibTeX source invariants (pre-commit, <1s) |
| | `epubcheck` | W3C epubcheck on built EPUBs under `_build/epub-vol*/` (CI, ~7s per volume) |
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

## EPUB Checks — Two Layers, One CLI Surface

EPUB validation exists in two layers that both sit inside the binder CLI. Developers and CI exercise identical code paths, so there is no drift between "what pre-commit runs" and "what CI enforces."

```
./binder check epub                       # all scopes (default)
./binder check epub --scope hygiene       # layer 1 — source-level, pre-commit grade
./binder check epub --scope epubcheck     # layer 2 — post-build W3C validator
./binder check epub --scope epubcheck \
    --max-fatal 0 --max-errors 0          # tighten thresholds (CI-style)
./binder check epub --json                # machine-readable output (tools, CI)
```

Implementation: `book/cli/commands/validate.py` (`_run_epub_hygiene`, `_run_epubcheck` native methods) and the shared primitives in `book/cli/commands/_epub_checks.py`.

### Layer 1 — hygiene (source-level, fast)

Runs in <1s across all SVG and BibTeX files under `book/quarto/contents/`. No EPUB build required. Wired into the pre-commit hook `book-epub-hygiene`.

Catches the four source-level patterns that historically broke builds in April 2026 (issues [#1014](https://github.com/harvard-edge/cs249r_book/issues/1014), [#1052](https://github.com/harvard-edge/cs249r_book/issues/1052), [#1148](https://github.com/harvard-edge/cs249r_book/issues/1148)):

| Code | What it catches | Symptom at epubcheck time |
|---|---|---|
| `svg-c0` | C0 control characters (U+00–U+1F except TAB/LF/CR) inside SVG `aria-label="..."` values. Produced by matplotlib rendering titles that contain raw-bytes representations. | `FATAL(RSC-016)` — EPUB fails to load in Kindle and ClearView. |
| `svg-dupe-marker` | Duplicate `<marker id="X"/>` inside a single SVG's `<defs>`. Produced by copy-paste during figure editing. | `ERROR(RSC-005)` — figure may not render. |
| `bib-url-escape-underscore` | `\_` inside a `url=` or `http`-bearing `doi=` BibTeX field. BibTeX's LaTeX escape leaks through citeproc. | `ERROR(RSC-020)` — Kindle's URL validator rejects the href. |
| `bib-url-escape-percent` | `\%` inside the same URL fields. | `ERROR(RSC-020)`. |
| `bib-url-raw-angle` | Raw `<` or `>` inside URL fields (legal in SICI DOIs but forbidden by strict URI syntax). | `ERROR(RSC-020)`. |

If one of these triggers during commit, the fix is almost always at source:

- **SVG C0 chars** — open the matplotlib script that produced the plot and find the `plt.title(...)` / `ax.set_xlabel(...)` call with a `repr(bytes)` or similar. Replace with the decoded string.
- **Duplicate SVG markers** — open the SVG in a text editor and delete the duplicate `<marker>` element. The compact one-line form and the multi-line expanded form are two common duplicates; keep one.
- **BibTeX URL escapes** — edit the `.bib` entry: replace `\_` with `_`, `\%` with `%`, `<`/`>` with `%3C`/`%3E`. These escapes are LaTeX-only; BibTeX `url = { ... }` fields do not require them.

### Layer 2 — epubcheck (built-EPUB, post-build)

Runs the W3C `epubcheck` validator against every EPUB discovered under `book/quarto/_build/epub-vol*/`. Emits `ValidationIssue` records with file, line, column, RSC/OPF code, severity, and human-readable message. When running under GitHub Actions, also emits `::error file=...,line=...::` annotations so findings show up inline on PR diffs.

Requires the `epubcheck` Python package (pinned in `book/tools/dependencies/requirements.txt`) plus a Java runtime (JRE 8+). If neither `python -m epubcheck` nor the `epubcheck` system binary is available, the check emits an `epubcheck-missing` issue with install instructions rather than silently passing.

Thresholds (from the command line, environment variables, or the CI workflow):

- `--max-fatal N` — fail the run if FATAL count exceeds N (default: 0). Kindle and ClearView reject any EPUB with FATAL, so 0 is the correct production threshold.
- `--max-errors N` — fail the run if ERROR count exceeds N (default: unlimited while the RSC-005 / RSC-012 baselines stabilize). Tighten to 0 once the baseline is clean.

### Defense in depth

The two layers exist because neither alone is sufficient:

- **Hygiene alone** is a whitelist of known patterns. It cannot catch a category of error epubcheck invents in a future version, nor renderer-emitted markup (bare `<br>`, `--` in TikZ comments) that only appears after Quarto + post-process have run.
- **Epubcheck alone** takes ~7 seconds per volume and requires a full EPUB build. Developers under time pressure will find ways around it.

The rendered-EPUB layer also gets belt-and-suspenders support from `book/quarto/scripts/epub_postprocess.py`, a Quarto post-render hook that sanitizes XHTML/SVG/OPF in the built EPUB (strips `--` from HTML comments, closes bare `<br>` tags, strips C0 chars from SVG aria-labels, normalizes URL escapes, declares the `mathml` OPF property). So a regression caught at source by hygiene, missed there, rescued by post-process, and missed again is still caught by epubcheck in CI. Three independent nets.

## Script Delegation

Some commands still delegate to scripts under `book/tools/scripts/` (spelling, image formats, table formatting, Python formatting). These will be migrated to native CLI modules over time. EPUB checks have already been migrated: the `hygiene` and `epubcheck` scopes live in `book/cli/commands/_epub_checks.py` as pure Python, not subprocess-to-script.
