# Shared Quarto Config

Single source of truth for reusable Quarto YAML fragments.

## Scope

This folder is only for Quarto YAML metadata fragments (filters, crossref, format settings).
It is not for CSS/SCSS, TeX templates, or other rendering assets.

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

## Current Shared Files

- `base/crossref-video.yml`
- `base/custom-numbered-blocks.yml`
- `base/diagram.yml`
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
