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

All validation lives in `validate.py`. Each check belongs to a **group** and has a **scope**. Each scope carries a **`default`** flag that controls whether it runs in the curated set (pre-commit / CI) or only on demand.

```bash
./binder check <group>                  # default=True scopes only
./binder check <group> --scope <name>   # one specific scope
./binder check <group> --all-scopes     # every scope, including default=False
./binder check all                      # every group's default=True scopes
./binder check all --vol1               # ... scoped to Volume I
./binder check <group> help             # per-group scope listing
```

#### How scopes are discovered

`./binder check` (no args) prints the full group catalogue. Scopes marked with `*` are `default=False` — opt-in only, reachable via `--scope` or `--all-scopes`. Per-group help (`./binder check <group> help`) prints the same catalogue with one row per scope, plus the runner method name and a one-line note.

The authoritative list is the `GROUPS` dict in `book/cli/commands/validate.py`. That dict is the only place a new scope needs to be wired; pre-commit, CI, and ad-hoc invocations all read from it.

#### Default vs. opt-in: the contract

A scope is `default=True` when it should run on every commit. Reasons to mark a scope `default=False`:

1. It currently fails on dev (corpus debt; opt in to find the cleanup work, then flip the flag).
2. It is intentionally heavy — slow (`math/render-audit`, ~10 min) or external (`references/hallucinator`, network calls; `spelling/*` requires aspell; `epub/epubcheck` requires JRE).
3. It is a manual-stage check (`render-audit` is also wired to pre-commit's `[manual]` stage).

Flipping `default=False → default=True` once dev is clean is a one-line edit in `validate.py`. No YAML change needed.

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

Every `book-*` pre-commit hook routes through `./book/binder`. The hook tree mirrors the binder command tree one-to-one: there is one hook per check group (`book-check-refs`, `book-check-prose`, `book-check-footnotes`, ...). Each hook runs the curated default scopes for its group; opt-in scopes are reachable on the CLI via `--scope` or `--all-scopes`.

```yaml
- id: book-check-<group>
  entry: ./book/binder check <group>
  language: system
  pass_filenames: false
  files: ^book/quarto/contents/.*\.qmd$
```

Two structural exceptions, both documented inline in `.pre-commit-config.yaml`:

- **Per-scope split for `labels`** — the `orphans` scope is currently clean only on Vol I (Vol II has forward references to unwritten chapters). The split lets the universally-clean `duplicates` scope cover both volumes while `orphans` stays vol1-only:

  | Group | Scope hook | Vol coverage |
  |---|---|---|
  | labels  | `book-check-labels-orphans` | vol1 only (vol2 forward refs to unwritten chapters) |
  | labels  | `book-check-labels-duplicates` | both vols (fig+tbl+lst by binder default) |

  Collapse back to a single `book-check-labels` once Vol II's chapter list is complete.

- **Format-vs-content split** (`book-check-tables` runs `binder check tables` for content; `book-check-tables-format` runs `binder format tables --check` for whitespace) — these are different binder commands.

No inline bash scripts. One CLI, one validation framework, one error format. Retired-hook history lives in `.pre-commit-history.md` so the live config is not a graveyard.

## Adding a new check

The flow that a future maintainer should mimic:

1. **Implement the runner** in `book/cli/commands/validate.py` as `_run_<scope>(self, root: Path) -> ValidationRunResult`. For per-file regex checks, walk `self._qmd_files(root)`. For graph / corpus checks, build the model once and emit `ValidationIssue` entries. For external scripts you want to keep callable standalone, wrap with `self._delegate_script(script_path, args, run_name)`.
2. **Register the scope** by adding a `Scope("<name>", "_run_<scope>", default=...)` entry to the right group in the `GROUPS` dict at the top of the file. Mark `default=False` if the scope still fails on dev or is intentionally opt-in.
3. **Surface any new flags** in the argparse block in `ValidateCommand.run()`, and dispatch them in `_run_group` if the runner needs them as kwargs.
4. **Stop.** Pre-commit picks up the new scope automatically because the existing `book-check-<group>` hook already runs every default-True scope in the group. You only add a new pre-commit hook for a brand-new group, or to pass a scope-specific flag (the vol1 / format-check exceptions above).

When you flip a scope from `default=False` → `default=True`, that single edit in `validate.py` ships the scope to every developer's pre-commit and to CI on the next push.

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

### Layer 2 — smoke (built-EPUB, no Java)

Runs in <1s against every EPUB under `_build/epub-vol*/`. Catches reader-compatibility issues that epubcheck does not, because it enforces EPUB 3 spec conformance while readers enforce a stricter subset:

| Code | What it catches | Symptom |
|---|---|---|
| `smoke-css-custom-property-decl` | `--var-name: value;` declarations in packaged CSS. | ClearView / Tolino (pre-2023 firmware) silently fail to load the EPUB. |
| `smoke-css-custom-property-use` | `var(--x)` references. | Same readers render the default fallback, producing visual regressions. |
| `smoke-external-resource` | `src="https://..."` / `<link href="https://...">` in any XHTML. | EPUB readers do not fetch remote assets; image shows as broken-icon on every reader. |

**Fix at source** in every case — there is no legitimate reason a packaged EPUB should reference external assets, and CSS custom properties should be inlined (or post-processed away) for the packaged stylesheet.

Useful when Java is not installed locally and `epubcheck` is unavailable; the other two layers still provide real coverage.

### Layer 3 — epubcheck (built-EPUB, full spec)

Runs the W3C `epubcheck` validator against every EPUB discovered under `book/quarto/_build/epub-vol*/`. Emits `ValidationIssue` records with file, line, column, RSC/OPF code, severity, and human-readable message. When running under GitHub Actions, also emits `::error file=...,line=...::` annotations so findings show up inline on PR diffs.

Requires the `epubcheck` Python package (pinned in `book/tools/dependencies/requirements.txt`) plus a Java runtime (JRE 8+). If neither `python -m epubcheck` nor the `epubcheck` system binary is available, the check emits an `epubcheck-missing` issue with install instructions rather than silently passing.

Thresholds (from the command line, environment variables, or the CI workflow). Two modes:

- **Flat thresholds** (for local use or simple CI):
  - `--max-fatal N` — fail if FATAL count exceeds N (default: 0). Kindle and ClearView reject any EPUB with FATAL, so 0 is the correct production threshold.
  - `--max-errors N` — fail if ERROR count exceeds N (default: unlimited). Tighten to 0 once the baseline is clean.

- **Baseline ratchet** (preferred in CI):
  - `--baseline PATH` — fail only if per-volume counts **increase** over what is recorded at `PATH` (JSON). Doesn't cliff-fail during incremental cleanup.
  - `--update-baseline` — rewrite the baseline file to the current counts. Run after a cleanup lands; commit the updated file in the same PR. This is how you lower the ceiling.
  - When `--baseline` is supplied, the flat thresholds are ignored.

The canonical baseline lives at `book/tools/audit/epubcheck-baseline.json`. The CI workflow (`book-validate-dev.yml`) uses the ratchet by default — a PR cannot land if epubcheck counts regress past that file. Initial state (April 2026) is `0/0/0` for both volumes, so *any* new FATAL, ERROR, or WARNING blocks the PR.

### Defense in depth

The three layers exist because none alone is sufficient:

- **Hygiene alone** is a whitelist of known patterns. It cannot catch a category of error epubcheck invents in a future version, nor renderer-emitted markup (bare `<br>`, `--` in TikZ comments) that only appears after Quarto + post-process have run.
- **Epubcheck alone** takes ~7 seconds per volume and requires a full EPUB build. Developers under time pressure will find ways around it.

The rendered-EPUB layer also gets belt-and-suspenders support from `book/quarto/scripts/epub_postprocess.py`, a Quarto post-render hook that sanitizes XHTML/SVG/OPF in the built EPUB (strips `--` from HTML comments, closes bare `<br>` tags, strips C0 chars from SVG aria-labels, normalizes URL escapes, aligns the nav item's `mathml` OPF property with the rendered nav content). So a regression caught at source by hygiene, missed there, rescued by post-process, and missed again is still caught by epubcheck in CI. Three independent nets.

## EPUB — How This Integrates with the Book Workflow

Three layers of defense, each invoked automatically at the moment it adds value:

### Author workflow (writing and committing)

1. **Write normally.** No new constraints on QMD content. The EPUB checks only fire on SVG and BibTeX source — prose changes pass through untouched.
2. **Commit.** The `book-epub-hygiene` pre-commit hook runs `./binder check epub --scope hygiene` if you touched SVG, BibTeX, or the check implementation. Fails in <1s if an SVG has duplicate `<marker>` ids or a bib URL has `\_` escapes. The failure message tells you to auto-repair with `./binder check epub --scope hygiene --fix`.
3. **Build locally.** `./binder build epub --vol1` runs:
   - Hygiene preflight (fast-fail on source issues, ~100ms).
   - Quarto render (~2 minutes).
   - Post-render sanitizer (fixes renderer-emitted issues).
   - Smoke + epubcheck post-flight (~7s) — verifies the final EPUB against the same CI baseline.
4. **Push.** CI builds both volumes and re-runs `./binder check epub --scope epubcheck --baseline …`. Any regression past the recorded counts blocks the PR.

### Release workflow (preparing an EPUB for readers)

1. **Build:** `./binder build epub --all` (or per-volume). Post-flight already validates — if the final line is `✓ smoke` and `✓ epubcheck`, the file is ready.
2. **Visual spot-check:** open the file in Sigil or Calibre Editor. Structural checks don't guarantee aesthetic quality.
3. **Ship:** upload the file from `book/quarto/_build/epub-vol*/`.

### When CI (or local post-flight) blocks you

The failure line names the volume, severity, and delta:

```
epubcheck: regression against baseline (2 FATAL, 15 ERROR total)
  • vol1 FATAL: 2 > baseline 0 (+2)
```

Two dispositions:

- **Real regression:** fix the underlying issue. `./binder check epub help` maps every error code to a source-level fix. `./binder check epub --scope hygiene --fix` auto-repairs the four mechanical classes.
- **Accepted increase** (e.g., a Quarto upgrade introduced a new warning class you're OK living with): `./binder check epub --scope epubcheck --baseline book/tools/audit/epubcheck-baseline.json --update-baseline` — commit the updated JSON in the same PR so the audit trail is clear.

### Escape hatches

Rare but real cases where you need to build anyway:

- `./binder build epub --vol1 --skip-hygiene` — bypass pre-render hygiene (e.g., debugging a source-level issue that's triggering the check you're trying to investigate).
- `./binder build epub --vol1 --skip-validate` — bypass post-render validation (e.g., iterating on a known-broken build; no Java locally).

Both are loud about being bypassed — the skipped step prints a yellow warning so the bypass is never silent.

### Onboarding

A new contributor runs `./binder doctor` once. The EPUB section reports Java availability, epubcheck availability, existing EPUB artifacts, and source hygiene in one table, with install commands inline if anything is missing. No separate EPUB setup document to hunt for.

## Script Delegation

Some commands still delegate to scripts under `book/tools/scripts/` (spelling, image formats, table formatting, Python formatting). These will be migrated to native CLI modules over time. EPUB checks have already been migrated: the `hygiene` and `epubcheck` scopes live in `book/cli/commands/_epub_checks.py` as pure Python, not subprocess-to-script.
