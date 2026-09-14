# Shared Quarto Config

Single source of truth for reusable Quarto YAML fragments.

## Scope

This folder is only for Quarto YAML metadata fragments (filters, crossref, format settings).
It is not for CSS/SCSS, TeX templates, or other rendering assets.

## Active config: how `_quarto.yml` is selected

Quarto looks for `_quarto.yml` at the project root (`books/`). That file
is a **generated copy** the `binder` CLI writes every time you build, preview,
or switch volume/format:

    books/_quarto.yml   <-  config/_quarto-{html,pdf,epub}-{vol1,vol2,vol3,vol4}.yml

The first line of the copy names its source. Volume builds also generate
`books/index.qmd` from the entry-point sources below. Both copies are gitignored.

**Do not edit `books/_quarto.yml` directly.** Edit the canonical file
under `config/` instead. The next build overwrites the copy.

The copy is written by [`binder/cli/core/config.py`](../../../binder/cli/core/config.py)
(`ConfigManager.activate_config`). To switch manually, use the `binder` CLI
or copy by hand:

    cd books/
    cp config/_quarto-html-vol2.yml _quarto.yml

### Per-volume landing pages

Volume IV keeps its authored entry-point content inside the volume:

| Format | Source copied to `books/index.qmd` |
|---|---|
| HTML | `books/vol4/index.qmd` |
| PDF / EPUB | `books/vol4/frontmatter/about.qmd` |

The HTML site renders the preface separately from the homepage. PDF and EPUB
include it only through `index.qmd`, with no second `about.qmd` chapter.
Volumes I–III retain `books/index-volN.qmd` as their source for every format.

Local Binder and Linux/Windows CI call the same generator,
`binder/cli/core/volume_index.py`. It creates a fresh copy rather than a symlink,
and removes legacy symlinks before copying so a build cannot overwrite their
targets. Root-relative asset paths keep the Volume IV homepage valid when
rendered through the generated root entry point.

**Never commit or edit `books/index.qmd`** (symlink *or* regular file). The
canonical source is the only editing location; the next build replaces the
ignored root copy. The matching render list includes only the generated
homepage, avoiding a second render of the source to a competing index page.

## Layout

- `base/`: defaults shared by multiple formats.
- `html/`: HTML-only shared metadata.
- `pdf/`: PDF-only shared metadata.
- `epub/`: EPUB-only shared metadata (create when needed).

## Include Order

In each `_quarto-<format>-<vol>.yml`:

1. `base/*`
2. `<format>/*` (e.g., `html/*`, `pdf/*`, `epub/*`)
3. `vol1/*` or `vol2/*` only for unavoidable differences (e.g., content paths)

Later files override earlier files.

## Policy

- Prefer zero volume-specific overrides; keep them minimal when required.
- Add volume-specific fragments only when content cannot be shared safely.
- Keep chapter lists, nav/sidebar, bibliography, and book title local to top-level `_quarto-*` configs.

## Future direction: shrinking the 8-config matrix

There are currently **8 top-level book configs** (`_quarto-{html,pdf,pdf-copyedit,epub}-{vol1,vol2}.yml`),
totalling ~1,500 lines. Each one already pulls many fragments from
`shared/`, but each also redeclares a fair amount of per-format and
per-volume metadata.

A future refactor could compress this further using YAML anchors or
`metadata-files:` composition along these lines:

```
config/
├── shared/
│   ├── base/                       # truly format-agnostic (already exists)
│   ├── html/                       # HTML-only (already exists)
│   ├── pdf/                        # PDF-only (already exists)
│   ├── epub/                       # EPUB-only (already exists)
│   ├── vol1/                       # vol-specific overlays (titles, OG, sidebar prefix)
│   └── vol2/
└── _quarto-<format>-<vol>[-variant].yml   # ~30 lines each: just metadata-files: + format-specific overrides
```

Effort/risk trade-off: the 8 explicit configs are easy to read and the
duplication is mostly metadata (titles, descriptions, sidebar entries) that
must remain per-volume anyway. Defer this refactor until either (a) we add a
9th config (e.g. accessible-PDF), (b) a structural change needs to land in
all 8 at once, or (c) drift across volumes becomes a recurring bug source.

Until then: when editing one config, grep for the same key in its sibling
configs and confirm whether the change should propagate.

## Current Shared Files

- `base/crossref-labels.yml`
- `base/custom-numbered-blocks.yml`
- `base/diagram.yml`
- `base/execute-env.yml`
- `html/announcement.yml`
- `html/filters.yml`
- `html/filter-metadata.yml`
- `epub/filters.yml`
- `epub/filter-metadata.yml`
- `pdf/filters.yml`
- `pdf/filter-metadata.yml`
- `pdf/custom-numbered-blocks-overrides.yml`
- `pdf/titlepage-theme-common.yml`
- `pdf/titlepage-pdf-common.yml`
- `pdf/titlepage-pdf-copyedit-common.yml`
- `pdf/build-production-common.yml`
- `pdf/build-copyedit-common.yml`
- `pdf/copyedit-watermark.yml`
- `vol1/filter-metadata-paths.yml`
- `vol2/filter-metadata-paths.yml`
