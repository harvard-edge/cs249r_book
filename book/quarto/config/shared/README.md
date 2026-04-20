# Shared Quarto Config

Single source of truth for reusable Quarto YAML fragments.

## Scope

This folder is only for Quarto YAML metadata fragments (filters, crossref, format settings).
It is not for CSS/SCSS, TeX templates, or other rendering assets.

## Active config: how `_quarto.yml` is selected

Quarto looks for `_quarto.yml` and `index.qmd` at the project root
(`book/quarto/`). Both files there are **symlinks** that the CLI rewrites every
time you build, preview, or switch volume/format:

    book/quarto/_quarto.yml   ->  config/_quarto-{html,pdf,epub}-{vol1,vol2}.yml
    book/quarto/index.qmd     ->  contents/{vol1,vol2}/index.qmd

The actual files live in `config/`; the symlink at the project root is just a
pointer to the currently-active configuration.

**Do not edit `book/quarto/_quarto.yml` or `book/quarto/index.qmd` directly.**
Edit the canonical file under `config/` (or `contents/<vol>/index.qmd`) instead.
The symlink target may be overwritten on the next build.

The symlink is managed by [`book/cli/core/config.py`](../../../cli/core/config.py)
(`ConfigManager.setup_symlink` / `_setup_index_symlink`). To switch manually,
use the `binder` CLI commands or relink by hand:

    cd book/quarto
    ln -sf config/_quarto-html-vol2.yml _quarto.yml
    ln -sf contents/vol2/index.qmd index.qmd

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

- `base/crossref-video.yml`
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
