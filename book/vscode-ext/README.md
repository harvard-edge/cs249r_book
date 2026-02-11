# MLSysBook VS Code Extension

The MLSysBook extension provides editor-first workflows for building, debugging, and validating the book without leaving VS Code.

## Binder-First Integration

This extension treats `./book/binder` as the operational backend.

- Build and debug commands execute Binder subcommands.
- Validation actions execute `binder validate ...`.
- Maintenance actions execute `binder maintain ...`.

## Work vs Configuration

The extension separates daily work from configuration:

- Work views: Build, Debug, Recent Runs, Navigator, Validate, Publish/Maintenance
- Configuration view: execution mode, navigator preset, visual preset, and extension settings

This keeps daily actions focused while preserving easy access to tuning controls.

## Why This Matters

- One public CLI contract for terminal, extension, and CI.
- Reduced drift between script behavior and editor behavior.
- Better diagnostics with optional machine-readable output (`--json`) from Binder validation.

## Core Workflows

- Build chapter: `./book/binder build html <chapter>` / `./book/binder build pdf <chapter>`
- Debug failures: `./book/binder debug <pdf|html|epub> --vol1|--vol2`
- Validate content: `./book/binder validate all`
- Maintenance checks: `./book/binder maintain repo-health`

## Quarto Visual Customization

The extension includes QMD-focused visual highlighting to improve scanability in long Quarto documents.

Recommended baseline:

- `mlsysbook.enableQmdChunkHighlight = true`
- `mlsysbook.qmdVisualPreset = balanced`
- `mlsysbook.highlightInlineReferences = true`
- `mlsysbook.highlightLabelDefinitions = true`
- `mlsysbook.highlightDivFenceMarkers = true`

What gets emphasized:

- Inline references like `@fig-...`, `@tbl-...`, `@sec-...`
- Label definitions like `{#fig-...}` and `#| label: ...`
- Quarto div fence markers (`:::` / `::::`) and region blocks (callouts/divs/code fences)

## Notes

- Legacy script entrypoints still exist for compatibility in some areas.
- For Binder-covered tasks, direct script invocation is soft-deprecated.
