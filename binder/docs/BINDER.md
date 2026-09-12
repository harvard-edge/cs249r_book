# Book Binder CLI

The **Book Binder** (`./binder/binder`) is the single entry point for building, checking, fixing, and formatting the MLSysBook.

## Author workflow (typical)

Most chapter work needs only **build** (and optionally **preview**). You do not need to run `check` or `fix` by hand — **pre-commit runs those on every commit** via the same `./binder/binder check …` commands documented below.

```bash
# From repository root — one-time per clone
./binder/binder setup

# Day to day: build what you're editing
./binder/binder build html --vol1 vol1/ml_systems      # fast HTML chapter
./binder/binder build pdf --vol1 vol1/ml_systems       # PDF chapter
./binder/binder build epub --vol1 vol1/ml_systems      # EPUB chapter
./binder/binder build html --vol1                      # whole Volume I site
./binder/binder build pdf --vol1                       # whole Volume I PDF

# Optional: live reload while editing
./binder/binder preview vol1/ml_systems

# When a commit is blocked, pre-commit prints the failing binder command.
# Re-run that command locally to see details, or:
./binder/binder check refs --path books/vol1/ml_systems/ml_systems.qmd
```

Commit as usual; hooks handle validation. Run `./binder/binder check all --vol1` only when you want a full local sweep before pushing.

For fast chapter-by-chapter PDF layout iteration with full-volume numbering,
see [Mapped Chapter Layout Builds](LAYOUT_CHAPTER_BUILDS.md).

## Public API policy

Binder is the **single source of truth** for book automation in this repository.

- **Checks:** `./binder/binder check <group> [--scope …]` — every `book-check-*` pre-commit hook dispatches here.
- **Fixes:** `./binder/binder fix <topic> <action>` — maintenance and content repair (alias: `maintain`).
- **Formats:** `./binder/binder format <target>` — auto-formatters (some pre-commit hooks use `format … --check`).
- Use `./binder/binder …` from the **repository root**. If your shell is already in `book/`, `./binder/binder …` is equivalent.
| Implementation | Where check logic lives |
|----------------|-------------------------|
| **Preferred** | `binder/cli/checks/*.py` — imported by `validate.py` |
| **Shared CLI primitives** | `binder/cli/core/*.py` — reusable command logic such as bibliography fixes and artifact cleanup |
| **Inline** | `binder/cli/commands/validate.py` — small regex/graph checks |
| **Transitional** | importlib/subprocess into `binder/tools/` (being migrated) |

Scripts under `binder/tools/` are not the public API. Pre-commit and CI call Binder subcommands, not scripts directly.

**Command taxonomy**

| Verb | Ownership rule |
|------|----------------|
| `build`, `preview` | Render book outputs or run the live authoring server. |
| `check` | Read-only validation. This is the pre-commit/CI surface. |
| `format` | Deterministic source formatting; supports `--check` where useful for hooks. |
| `fix` | Targeted source repair or maintenance actions that may write files. |
| `bib` | Bibliography lifecycle: mechanical fixes, normalize, clean, update, sync. |
| `clean` | Remove generated artifacts and local build state only. |
| `reset` | Restore build YAML manifests after scoped builds. |
| `info` | Read-only reports and inventories. |
| `render` | Generate derived assets such as plot galleries. |
| `audit` | Heavier or ledgered audits that are not normal commit checks. |
| `layout` | PDF visual/layout diagnostics: whitespace, margin overflow, header/footer collisions, and table-only PDF audits. |
| `doctor`, `status`, `list`, `setup`, `switch`, `debug`, `headings` | Environment, state, diagnostics, or specialized maintenance. |

Compatibility aliases are intentionally thin: `validate` routes to `check`, and `maintain` routes to `fix`. Prefer the canonical verbs in docs, hooks, and new automation.

**Documentation map**

| Audience | Document |
|----------|----------|
| Authors & daily use | This file (`binder/docs/BINDER.md`) — command reference |
| Check/fix implementation | [`binder/cli/README.md`](../cli/README.md) — architecture, adding scopes, EPUB layers |
| Scope registry (code) | `binder/cli/commands/validate.py` → `GROUPS` dict |
| Pre-commit wiring | `.pre-commit-config.yaml` — one hook per check group (or explicit `--scope`) |

Run `./binder/binder check` with no arguments to print the live group/scope catalogue (authoritative; docs may lag).

## Quick start (full reference)

```bash
# First time setup (from repository root)
./binder/binder setup
./binder/binder doctor

# Build & preview — primary author commands (see BUILD.md)
./binder/binder build html --vol1 vol1/training
./binder/binder preview vol1/introduction

# Checks — usually pre-commit only; run locally when debugging a failed hook
./binder/binder check cli
./binder/binder check all --vol1
./binder/binder check refs --path books/vol1/01_introduction/01_introduction.qmd

# Fixes — maintenance / repair (rare in daily chapter work)
./binder/binder fix repo-health
./binder/binder fix headers add --vol1 --dry-run

# Live command reference
./binder/binder help
./binder/binder check refs help    # per-group scopes and error codes
```

## Installation

The binder lives at `binder` (Python entry point → `binder/cli/`). Ensure it is executable:

```bash
chmod +x binder
```

Requires Python 3.10+ and project dependencies (Rich, etc.). Run `./binder/binder doctor` to verify Quarto, Java/epubcheck, and other tooling.

---

## Check — validation (`check <group>`)

> **Authors:** you usually skip this section. Pre-commit invokes these automatically on commit. Use it when a hook fails and you need the full error output, or when maintaining the check suite.

### Command shape

```bash
./binder/binder check <group> [--scope <name>] [--vol1|--vol2] [--path PATH] [--json]
./binder/binder check all [--vol1|--vol2]          # every group's curated scopes
./binder/binder check <group> --all-scopes         # include opt-in / heavy scopes
./binder/binder check <group> help                 # scopes + error codes for one group
```

`validate` is a backward-compatible alias for `check` (same parser). Prefer **`check`**.

**Important:** the first argument after `check` must be a **group name** (e.g. `refs`, `labels`), not a legacy flat name. Scopes such as `inline-python` or `duplicates` require `--scope`.

### Check groups

| Group | What it validates | Common scopes |
|-------|-------------------|---------------|
| `cli` | Public Binder command/help contract | `contract` |
| `refs` | Cross-refs, citations, inline `{python}` refs | `cross-refs`, `citations`, `scaffold-citations`, `inline`; opt-in: `inline-python`, `self-ref` |
| `labels` | Duplicate and orphan `@fig-` / `@tbl-` / … labels | `duplicates`, `orphans` |
| `headers` | Section IDs (`{#sec-…}`) and headline case | `ids`, `case` |
| `bib` | Bibliography hygiene, metadata style, citation integrity | `hygiene`, `style`, `integrity`; opt-in: `orphans`, `key-content` |
| `footnotes` | Definition shape, placement, integrity | `definition-shape`, `placement`, `integrity` |
| `figures` | Captions, div syntax, alt text, label-required | default set in `check figures help` |
| `markup` | Low-level markup (patterns, div fences, callouts) | `patterns`, `div-fences`, `callouts` |
| `prose` | Contractions, duplicate words, above/below, … | see `check prose help` |
| `punctuation` | Em-dash, slash, vs., e.g./i.e., en-dash ranges | |
| `numbers` | Unit spacing, binary units, percent rules | |
| `math` | `\times` spacing, attribute LaTeX leaks, LEGO fmt/suffix canonical, multiplier prose style | `canonical`; opt-in: `multiplier-style`, `render-audit` |
| `structure` | Heading levels, parts, Purpose sections | |
| `code` | Python `echo: false`, `_str` LaTeX leaks, LEGO dead code | `lego-dead-code` |
| `tables` | Grid→pipe, content hygiene, caption-required | |
| `listings` | `#lst-` divs carry `lst-cap` | |
| `index` | `\index{}` placement, anti-patterns, xrefs | |
| `images` | Formats, external URLs, SVG XML | |
| `json` | JSON syntax in book tree | |
| `units` | mlsysim physics unit tests | |
| `notation` | Iron-law symbol consistency | |
| `spelling` | aspell on prose / TikZ | opt-in (needs aspell) |
| `epub` | Source hygiene; opt-in: smoke, epubcheck | `hygiene --fix` auto-repairs source |
| `pdf` | Built-PDF cross-ref scans, plus LaTeX-log gates (overfull boxes, missing glyphs) | post-build, requires artifact; see [Log-based PDF gates](#log-based-pdf-gates) |
| `registry` | Constants-to-registry migration gates | `sources`, `tests`, `appendix` |
| `sources` | Source-note / citation formatting | |
| `references` | External .bib verification (hallucinator) | opt-in, network |
| `content` | Content tree structure | opt-in |

### Examples

```bash
# Pre-commit-equivalent: all curated checks on Volume I
./binder/binder check all --vol1

# Binder command surface contract (also always runs in pre-commit)
./binder/binder check cli

# Single file, inline Python execution
./binder/binder check refs --scope inline-python --path books/vol1/training/training.qmd

# Inline `{python}` variable references
./binder/binder check refs --scope inline --path books/vol1/01_introduction/01_introduction.qmd

# LEGO fmt / suffix discipline (also runs as part of `check math` on commit)
./binder/binder check math --scope canonical --path books/vol1/training/training.qmd

# Body-prose multiplier style (opt-in while existing chapters are cleaned up)
./binder/binder check math --scope multiplier-style --path books/vol1/training/training.qmd

# Label hygiene
./binder/binder check labels --scope duplicates --vol1
./binder/binder check labels --scope orphans --vol1

# External bibliography audit (optional dependency)
./binder/binder check references --scope hallucinator -f books/references.bib --limit 10

# Machine-readable output (CI / automation integration)
./binder/binder check refs --json --quiet
```

### Example-rich check docs

New or migrated Binder checks should show concrete bad/good examples in two places:

1. The checker module docstring, so an implementer or automated repair pass can open the code and immediately see the intended pattern.
2. The CLI documentation, so authors can understand the failure without reverse-engineering the regex.

`./binder/binder check cli` catches command-surface drift:

| Error code | Bad command surface | Canonical fix |
|------------|---------------------|---------------|
| `cli_contract_exit` | `./binder/binder reset` exits nonzero, or `./binder/binder pdf reset --vol1` exits zero | Help paths return `0`; removed commands and parse errors return `1`. |
| `cli_contract_missing_output` | `./binder/binder check` no longer lists `cli` / `contract`, or reset help omits `reset pdf --vol1` | Update help text and docs so pre-commit failures show the command users should rerun. |
| `cli_contract_unexpected_output` | `./binder/binder build --help` still advertises `build reset` | Remove stale help and keep reset under `./binder/binder reset <fmt\|all>`. |
| `cli_contract_timeout` | A help or migration command starts a build/render path | Keep CLI contract commands fast, read-only, and independent of Quarto builds. |

For example, `./binder/binder check math --scope multiplier-style` catches these patterns:

| Error code | Bad source pattern | Canonical fix |
|------------|--------------------|---------------|
| `body_multiplier_suffix` | `speedup_str = fmt(speedup, suffix="x")` or `suffix="×"` for body prose | Use `speedup_mult_str = fmt_multiple(speedup, ...)`; the formatter owns `×`, so prose uses `` `{python} speedup_mult_str` `` by itself. |
| `mult_double_glyph` | `` `{python} speedup_mult_str`$\times$ `` | Remove the prose glyph; `fmt_multiple` / `fmt_multiple_range` already emit `×`. |
| `unicode_times_in_prose` | `A100 × H100` in normal Quarto prose | Use `A100 $\times$ H100` in prose. Raw `×` is only for non-LaTeX contexts such as alt text, Matplotlib labels, code fences, and ASCII diagrams. |
| `times_product_spacing` | `$n$$\times$$m$` or `` `{python} n_str`$\times$`{python} m_str` `` | Put spaces around arithmetic products: `$n$ $\times$ $m$` or `` `{python} n_str` $\times$ `{python} m_str` ``. Computed prose multipliers use `*_mult_str` instead of a separate prose glyph. |
| `fmt_sci_math_context` | `flops_math = fmt_sci(flops)` or `MarkdownStr(f"${fmt_sci(flops)}$")` | Treat `fmt_sci()` as plain-text output. For prose math, use `fmt_math(sci_latex(...))` or another LaTeX-first helper. |

### Log-based PDF gates

Some defects are only visible to LaTeX, not to the source or the finished PDF.
`binder check pdf --scope verify` therefore reads the LuaLaTeX build log in
addition to the PDF text:

| Gate | Fires on | Blocking |
|---|---|---|
| `quarto-crossref-warning` | `Unable to resolve crossref` | yes |
| `overfull-hbox` | Horizontal overflow >= 20pt | yes |
| `overfull-vbox` | Vertical / margin overflow >= 20pt | yes |
| `missing-glyph` | `Missing character:` — a character the font could not render | yes |

`missing-glyph` has no severity threshold, unlike the overfull gates. An
overfull box is a judgment about how much overflow is tolerable; a dropped
character is always a defect, because the text is simply absent from the printed
page with no visible marker. Warnings group by (character, font), and the report
names the worst offenders so one bad glyph in a widely used face is a single
actionable row.

**Finding the log.** The PDF configs set `latex-clean: false`, so the log
survives beside the generated `.tex` as
`books/Machine-Learning-Systems-Vol{1,2}.log` (both suffixes are
gitignored; this also preserves the `.aux` that mapped chapter layout builds
need). The check discovers it automatically:

```bash
./binder/binder build pdf --vol1          # writes the log
./binder/binder check pdf --scope verify --vol1   # reads it, no --log needed
./binder/binder check pdf --scope verify --vol1 --log /path/to/other.log  # override
```

A discovered log older than the PDF is ignored, because a log left behind by an
earlier or failed run would otherwise vouch for a build it never described — the
gates then report as *skipped* rather than passing on stale evidence. An explicit
`--log` is always trusted. If no log is found, the log-based gates skip; the
PDF-text gates still run.

### Diagnostic shape

Binder check output is designed to be actionable from the terminal and from `--json` automation. Each issue includes:

| Field | Meaning |
|-------|---------|
| `file` + `line` | The exact source location to edit. |
| `code` | Stable error code, useful for documentation and automated repair prompts. |
| `message` | Short diagnosis of the problem. |
| `context` | The offending source snippet; human output labels this as `source:`. |
| `suggestion` | Optional canonical rewrite guidance; human output labels this as `fix:`. |

For automated repair or structured review, prefer:

```bash
./binder/binder check math --scope multiplier-style --path books/vol1/training/training.qmd --json --quiet
```

Exit codes: `0` = passed, `1` = failures or command error.

### Pre-commit ↔ binder mapping

Every `book-check-*` hook in `.pre-commit-config.yaml` calls `./binder/binder check …`. The hook ID mirrors the group; scopes are embedded in the `entry` when needed.

| Pre-commit hook | Binder command |
|-----------------|----------------|
| `book-check-cli-contract` | `check cli` |
| `book-check-headers` | `check headers` |
| `book-check-structure` | `check structure` |
| `book-check-labels-orphans` | `check labels --scope orphans` |
| `book-check-labels-duplicates` | `check labels --scope duplicates` |
| `book-check-refs` | `check refs` |
| `book-check-footnotes` | `check footnotes` |
| `book-check-figures` | `check figures` |
| `book-check-images` | `check images` |
| `book-check-tables` | `check tables` |
| `book-check-listings` | `check listings` |
| `book-check-tables-format` | `format tables --check` |
| `book-check-markup` | `check markup` |
| `book-check-code` | `check code` |
| `book-check-prose` | `check prose` |
| `book-check-punctuation` | `check punctuation` |
| `book-check-numbers` | `check numbers` |
| `book-check-math` | `check math` (includes `canonical` scope for LEGO fmt discipline) |
| `book-check-notation` | `check notation` |
| `book-check-index` | `check index` |
| `book-check-sources` | `check sources` |
| `book-check-units` | `check units` |
| `book-check-epub` | `check epub` (scope `hygiene`) |
| `book-check-bib` | `check bib` |
| `book-check-registry-sources` | `check registry --scope sources` |
| `mlsysim-check-registry-gates` | `check registry --scope tests` |
| `book-check-math-render-audit` | `check math --scope render-audit` (manual stage) |

To reproduce a hook locally:

```bash
pre-commit run book-check-refs --files books/vol1/01_introduction/01_introduction.qmd
# equivalent:
./binder/binder check refs --path books/vol1/01_introduction/01_introduction.qmd
```

### Bibliography (`.bib` + pre-commit)

Committed `.bib` files go through pre-commit in this order:

1. **`bib-apply-mechanical`** — `./binder/binder bib mechanical --pre-commit` on staged `.bib` only
2. **`bibtex-tidy`** — layout
3. **`./binder/binder check bib`** — curated bibliography gate:
   `hygiene` for new hard BibTeX errors, `style` for new warning/info
   metadata debt, and `integrity` for volume-scoped citation resolution.
   Baselines: `binder/tools/bib_lint_baseline.json` and
   `binder/tools/bib_lint_style_baseline.json`.

Normalize the whole tree by hand:

```bash
./binder/binder bib mechanical refs.bib    # safe field-level fixes for selected files
./binder/binder bib normalize              # all git-tracked *.bib
./binder/binder bib normalize --vol1
```

Metadata refresh: `./binder/binder bib update` (betterbib sync + citekey propagation).

---

## Fix — maintenance (`fix <topic> <action>`)

> **Authors:** rarely needed day to day. Pre-commit and `./binder/binder fix …` overlap only for optional housekeeping (repo health, image compression, section IDs).

Canonical namespace for repairs and housekeeping. `maintain` is an alias for `fix`.

| Topic | Actions | Example |
|-------|---------|---------|
| `glossary` | `paths` | `./binder/binder fix glossary paths [--vol1\|--vol2]` |
| `images` | `compress` | `./binder/binder fix images compress --all --smart-compression [--apply]` |
| `repo-health` | `check` (optional) | `./binder/binder fix repo-health [--json] [--min-size-mb N]` |
| `headers` | `add`, `repair`, `list`, `remove` | `./binder/binder fix headers add --vol1 --dry-run` |
| `footnotes` | `cleanup`, `reorganize`, `remove` | `./binder/binder fix footnotes cleanup --vol1 --dry-run` |

**Related commands (not under `fix`):**

- `./binder/binder headings check|dry-run|apply` — headline-case enforcement (also runs as `check headers --scope case`)
- `./binder/binder check epub --scope hygiene --fix` — auto-repair SVG/BibTeX EPUB source issues
- `./binder/binder bib mechanical|normalize|sync|clean|update` — bibliography tooling
- `./binder/binder layout tables --vol1|--vol2` — render a table-only PDF audit plus contact sheets under `binder/.layout/tables/`

### Layout diagnostics

| Command | Purpose |
|---------|---------|
| `layout --vol1\|--vol2` | High-level auto-layout planner: build/reuse the volume PDF, scan main-flow whitespace and margin geometry, and emit one strategy-routed plan. |
| `layout check <pdf>` | Flag pages with excessive bottom whitespace and likely next-page culprits. |
| `layout margins <pdf>` | Gate margin figures/notes that overflow into the footer or off the page. |
| `layout collisions <pdf>` | Find body content that invades running header/footer bands. |
| `layout tables --vol1\|--vol2` | Render only source tables using production PDF geometry, emit JSON/CSV metrics, and create contact sheets for fast visual review. |

Recommended release-polish entrypoint:

```bash
./binder/binder build pdf --vol1 --layout
./binder/binder build pdf --vol2 --layout
```

Use `./binder/binder layout --vol1 --no-build` when the PDF was already built.

#### Auto-layout contract for structured repair

Use the high-level planner unless debugging one scanner. It is the stable
machine contract:

```bash
./binder/binder layout --vol1 --no-build --json /tmp/layout-plan.json
```

The JSON plan has:

| Field | Meaning |
|-------|---------|
| `volume`, `pdf`, `pages_scanned`, `page_count` | Render target metadata. |
| `workflow.next_phase` | The phase to repair first: `1-main-flow`, `2-margin-calibration`, or `clean`. |
| `workflow.phase_order` | The required order: main prose flow first, margin calibration second. |
| `counts.by_phase` | Split between prose-flow and margin-calibration findings. |
| `counts.by_channel` | Split between `main-flow` and `margin-geometry`. |
| `counts.by_strategy` | Routing count by repair strategy. |
| `items[]` | Ordered work queue, back-to-front within chapters where page shifts matter. |

Each `items[]` row has the common fields `phase`, `channel`, `strategy`,
`confidence`, `automatable`, `deferred`, `ready`, `chapter`, `sheet`, `label`,
`source_file`, `source_line`, `section`, and `suggested_fix`. Main-flow rows
also include `gap_pct`, `culprit`, and `detail`; margin rows include `issue`,
`side`, `snippet`, and rendered geometry detail.

Route by `channel` and `strategy`, not by free-form prose:

| Strategy | Channel | Meaning |
|----------|---------|---------|
| `callout-tcbbreak` | `main-flow` | High-confidence callout gap. Insert or move `{=latex}` `\tcbbreak` at the semantic boundary named in `suggested_fix`, then rebuild. |
| `source-flow-callout-adjacent` | `main-flow` | Rendered symptom is a callout/box, but source localization landed outside the callout. Inspect adjacent table/listing/lead-in/heading before editing the callout. |
| `table-source-flow`, `figure-source-flow`, `paragraph-source-flow` | `main-flow` | Move source flow first; use sizing/spacing only after a rebuild confirms source-flow did not solve it. |
| `margin-offset` | `margin-geometry` | Apply or adjust a footnote/sidenote `[offset=...]`. |
| `margin-vspace` | `margin-geometry` | Apply or adjust in-block `.column-margin` `\vspace*{...}`. |
| `margin-stack-solve` | `margin-geometry` | Multi-object margin packing problem; solve offsets/vspace together or send to visual review. |
| `accept-*` | either | Structural whitespace; do not edit unless a human explicitly asks. |
| `manual-review`, `margin-geometry-review`, `callout-localize` | either | Low-confidence source mapping; inspect visually before applying. |

Safe automation loop:

1. Build or reuse the PDF: `build pdf --volN --layout` or `layout --volN --no-build`.
2. Work only on rows where `phase == workflow.next_phase`.
3. Apply only rows where `ready=true`, `automatable=true`, and `confidence=high`.
4. Rebuild the volume PDF.
5. Re-run the planner and repeat until no ready rows remain in the current phase.
6. When `workflow.next_phase` becomes `2-margin-calibration`, turn on the margin guides/debug frames for visual calibration before accepting margin edits.
7. Leave low-confidence rows as plan output for visual review.

Phase 1 repairs main prose flow: callout splits, source-flow moves around
tables/figures, paragraph adjustments, and structural accepts. Do not tune
margin offsets while Phase 1 has active rows; those values are unstable until
the prose page breaks stop moving.

Phase 2 repairs margin geometry after prose flow is stable. For margin rows,
turn on the LaTeX guides in `books/shared/tex/header-includes.tex` by changing
`\MarginDebugfalse` to `\MarginDebugtrue`, rebuild the affected PDF, inspect the
red margin-note frames, then restore `\MarginDebugfalse` before committing.
Use `[offset=...]` for footnote/sidenote rows and in-block `.column-margin`
`\vspace*{...}` for margin figure/caption rows.

Low-level commands still expose `layout_strategy`:

- `layout check --csv` for main-flow whitespace only.
- Native `layout margins --csv` for margin geometry only.
- `layout collisions` for header/footer band debugging.

---

## Build & preview (summary)

| Command | Description | Example |
|---------|-------------|---------|
| `build [html\|pdf\|epub] [chapter[,…]] --volN` | Build a volume, or selected chapters of one volume | `./binder/binder build pdf intro --vol1` |
| `build <fmt[,fmt…]> [chapters] --volN\|--all --parallel [N]` | Run several builds at once, each in its own git worktree | `./binder/binder build html,pdf --all --parallel 4` |
| `debug <fmt> --volN [--chapter X] [--parallel N]` | Find the chapter, then the section, that breaks a build | `./binder/binder debug pdf --vol1 --parallel 4` |
| `preview [chapter]` | Live dev server | `./binder/binder preview vol1/intro` |
| `reset [html\|pdf\|epub\|all] [--vol1\|--vol2]` | Recover configs that older binder versions left partly commented out | `./binder/binder reset pdf --vol1` |

See [BUILD.md](BUILD.md) and [DEVELOPMENT.md](DEVELOPMENT.md) for full build workflows, and
[Chapter, parallel, and debug builds](#chapter-parallel-and-debug-builds) below.

### Management commands

| Command | Description |
|---------|-------------|
| `setup` | Configure environment and pre-commit |
| `clean` | Remove build artifacts |
| `switch <format>` | Copy a format's config to the active `_quarto.yml` |
| `list` / `status` | Chapters and config status |
| `doctor` | Tooling health check |
| `help` | Command reference |

**Note:** `publish` is not a Binder subcommand. Release publishing uses GitHub Actions and scripts under `binder/tools/scripts/publish/`.

---

## Chapter names

Chapters can be referenced by their short names. Common examples:

- `intro` → Introduction chapter
- `ml_systems` → Machine Learning Systems chapter
- `nn_computation` → Neural Computation chapter
- `training` → Training chapter
- `ops` → MLOps chapter

Use `./binder/binder list` to see all available chapters.

## Build Outputs

All output lands under `books/_build/`, which is gitignored:

| Build | Output location |
|-------|-----------------|
| One volume, one format | `books/_build/<format>-<volume>/` (for example `pdf-vol1/Machine-Learning-Systems-Vol1.pdf`) |
| Selected chapters | `books/_build/<format>-<volume>/chapters/<chapter>/` |
| `--parallel` run | `books/_build/parallel/<run-id>/<job>/` holding `build.log` and `output/`, plus `summary.json` for the run |
| `debug` run | `books/_build/debug/<volume>/<format>/<run-id>/` with `phase1/` (chapter scan) and `phase2/<chapter>/` (bisection) |
| Last renderer output | `books/_build/last-build.log` |

## Publishing

Release publishing is **not** a Binder subcommand. Use GitHub Actions and scripts under `binder/tools/scripts/publish/`. The sections below describe historical `publish` behavior and may be outdated — see your team's release runbook.

<details>
<summary>Legacy publish documentation (historical)</summary>

The former `publish` CLI command has been removed from Binder.

### 1. Interactive Mode (Default)

When called without arguments, `publish` runs the interactive wizard:

```bash
# Interactive publishing wizard
./binder/binder publish
```

**What interactive mode does:**

1. **🔍 Pre-flight checks** - Verifies git status and branch
2. **🧹 Cleans** - Removes previous builds
3. **📚 Builds HTML** - Creates web version
4. **📄 Builds PDF** - Creates downloadable version
5. **📦 Copies PDF** - Moves PDF to assets directory
6. **💾 Commits** - Adds PDF to git
7. **🚀 Pushes** - Triggers GitHub Actions deployment

### 2. Command-Line Trigger Mode

When called with arguments, `publish` triggers the GitHub Actions workflow directly:

```bash
# Trigger GitHub Actions workflow
./binder/binder publish "Description" [COMMIT_HASH]

# With options
./binder/binder publish "Add new chapter" abc123def --type patch --no-ai
```

**What command-line mode does:**

1. **🔍 Validates environment** - Checks GitHub CLI, authentication, branch
2. **✅ Validates commit** - Ensures the dev commit exists (if provided)
3. **🚀 Triggers workflow** - Uses GitHub CLI to trigger the publish-live workflow
4. **📊 Provides feedback** - Shows monitoring links and next steps

**Options:**
- `--type patch|minor|major` - Release type (default: minor)
- `--no-ai` - Disable enhanced release notes
- `--yes` - Skip confirmation prompts

**Requirements:**
- GitHub CLI installed and authenticated (`gh auth login`)
- Must be on main or dev branch
- Dev commit must exist (if provided)

### Publishing Workflow:

```bash
# Development workflow
./binder/binder preview intro          # Preview a chapter
./binder/binder build                  # Build complete HTML
./binder/binder build pdf              # Build complete PDF
./binder/binder publish                # Publish to the world
```

### After Publishing:

- **🌐 Web version**: Available at https://harvard-edge.github.io/cs249r_book
- **📄 PDF download**: Available at https://harvard-edge.github.io/cs249r_book/assets/downloads/Machine-Learning-Systems.pdf
- **📈 GitHub Actions**: Monitors build progress at https://github.com/harvard-edge/cs249r_book/actions

### Requirements:

- Must be on `main` branch
- No uncommitted changes
- Git repository properly configured

</details>

## Advanced Features

### Chapter, parallel, and debug builds

#### Selected chapters

`./binder/binder build <fmt> <chapter>[,<chapter>…] --volN` renders the volume's
generated `index.qmd` plus the selected chapters into
`books/_build/<format>-<volume>/chapters/<chapter>/`. Binder writes a temporary
`_quarto.yml` for the build and restores the previous `_quarto.yml` and
`index.qmd` afterwards, even when the render fails or is interrupted. The
canonical configs under `books/config/` are never edited. Chapters named without
`--volN` must all belong to one volume. See
[Iterate on one chapter](../README.md#iterate-on-one-chapter) for name matching
and validation notes.

#### Parallel builds

A build rewrites files at the Quarto project root, so two builds cannot share a
checkout. `--parallel [N]` runs several builds at once by giving each worker its
own disposable git worktree:

```bash
# Several chapters of one volume, each as its own build
./binder/binder build pdf intro,training --vol1 --parallel

# Every chapter of a volume on its own, four at a time
./binder/binder build pdf --vol1 --each-chapter --parallel 4

# Every volume and format at once
./binder/binder build html,pdf,epub --all --parallel 6
```

How it works:

- Binder snapshots the working tree: staged and unstaged edits through
  `git stash create`, which leaves the shared stash list alone, plus copies of
  untracked files. Every worktree builds what is on disk, not only what is
  committed.
- Each worker checks the snapshot out under the system temporary directory
  (`binder-workspaces/`) and runs `./binder/binder build` there, reusing its
  worktree for every job it picks up.
- Each job's log and output move to `books/_build/parallel/<run-id>/<job>/`, and
  `summary.json` lists every job. The worktrees and their temporary run
  directory are removed at the end; `--keep-workspaces` leaves them for
  inspection.
- Workers running side by side each get a private `XDG_CACHE_HOME`, because the
  diagram filter's cache is not safe for concurrent writers.
- Ctrl-C stops every running build (SIGTERM, then SIGKILL after 15 seconds) and
  removes the worktrees.
- The default worker count is a quarter of the CPU cores, clamped to 1–4. Each
  worker renders a whole Quarto project and each worktree is a full checkout
  (about 1 GB), so raise `N` with care.
- `--skip-validate`, `--skip-hygiene`, `--json`, and the whole-volume PDF flags
  `--no-cover` and `--print-marks` pass through to the builds. `--layout` does
  not, and output is not opened automatically.

#### Debugging a failing build

`./binder/binder debug <fmt> --volN` finds what breaks a build in two phases:

1. **Chapter scan.** Every chapter in the volume's PDF order is built on its own
   with `--skip-validate`, since references to omitted chapters would otherwise
   always fail. `--parallel N` builds N chapters at a time.
2. **Section bisection.** Each failing chapter is rebuilt from its preamble alone,
   then with growing numbers of `##` sections, until the first breaking section
   is found. `--chapter <name>` skips the scan and bisects one chapter.

Debug builds run in worktrees through the same machinery as `--parallel`, so the
truncated chapters never touch your files. Logs land under
`books/_build/debug/<volume>/<format>/<run-id>/`.

```bash
./binder/binder debug pdf --vol1 --parallel 4
./binder/binder debug html --vol2 --chapter training
```

### Configuration Management

The binder automatically manages Quarto configurations:

- **`config/_quarto-html-volN.yml`**: Website build configuration
- **`config/_quarto-pdf-volN.yml`**: Academic PDF build configuration
- **`config/_quarto-epub-volN.yml`**: EPUB build configuration
- **`_quarto.yml`**: Generated copy of the active configuration; its first line names the source file

**Important**: Binder copies the chosen configuration to `_quarto.yml` and generates `index.qmd` before each volume build. Volume IV sources live inside `vol4/`: `vol4/index.qmd` supplies the HTML homepage, and `vol4/frontmatter/about.qmd` supplies the PDF/EPUB preface. Volumes I–III retain `index-volN.qmd`. Both root copies are regenerated on every build and gitignored; edit the canonical sources, never the copies.

Use `./binder/binder switch <format>` to change the active configuration.

## Development Workflow

### Typical Chapter Development

```bash
# 1. Start development on a chapter
./binder/binder preview intro

# 2. Make edits, save files (auto-rebuild in preview mode)

# 3. Build the chapters you touched
./binder/binder build pdf intro,ml_systems --vol1 --skip-validate

# 4. Build the whole volume before committing
./binder/binder build pdf --vol1
```

### Before Committing

```bash
# Clean up any build artifacts
./binder/binder clean

# Run health check
./binder/binder doctor

# Build every volume and format to ensure everything works
./binder/binder build html,pdf,epub --all --parallel 4
```

## Troubleshooting

### Common Issues

**"Chapter not found"**
- Use `./binder/binder list` to see available chapters
- Check that the chapter QMD file exists
- Verify the chapter path in configuration files

**"Build artifacts detected"**
- Run `./binder/binder clean` to remove temporary files
- Use `./binder/binder doctor` to verify system health

**"Wrong or missing `_quarto.yml`"**
- Check which configuration is active: `head -1 books/_quarto.yml`
- Regenerate it: `./binder/binder switch html` (or `pdf`, `epub`)
- Never edit `books/_quarto.yml` directly; the next build overwrites it

**Leftover build worktrees**
- An interrupted `--parallel` or `debug` run that was killed outright can leave
  worktrees in the system temporary directory; `git worktree prune` clears
  registrations whose directories are gone, and `git worktree list` shows the rest

### Performance Tips

- Build selected chapters (`./binder/binder build pdf intro --vol1 --skip-validate`) while iterating
- Use `--parallel` to build several chapters, volumes, or formats at once
- Use `./binder/binder debug <fmt> --volN --parallel N` to find a failing chapter quickly
- Preview mode auto-rebuilds on file changes

## Further reading

- [BUILD.md](BUILD.md) — build instructions
- [DEVELOPMENT.md](DEVELOPMENT.md) — development setup
- [`binder/cli/README.md`](../cli/README.md) — check/fix architecture, adding scopes, EPUB layers
