# MLSysBook Tools

`book/tools` contains production tooling for the MLSysBook source tree. These
tools are part of how the book is built, checked, generated, audited, and
released. Some are called by humans, some by CI, some by Binder commands, and
some by Quarto-adjacent generation workflows.

Do not delete, rename, or move anything under this tree only because it looks
like a one-off script. First audit actual use in QMD files, Binder commands, CI,
release scripts, documentation, and generated-asset workflows.

## Production Policy

- Treat `book/tools` as book infrastructure, not scratch space.
- Before relocating or removing a tool, search for call sites with `rg`, update
  callers and documentation, and run the relevant validation command.
- Prefer stable modules or Binder-facing entrypoints for new book dependencies.
  Scripts may remain as compatibility entrypoints when they are already part of
  production workflows.
- Generated assets should record their source tool in the nearby README or
  comments so future editors can regenerate rather than hand-edit them.
- Tooling that depends on external packages should document the runtime command
  and any special environment setup.

## Current Areas

| Area | Purpose |
|---|---|
| `scripts/` | Python and shell entrypoints for generation, maintenance, release, and Quarto-adjacent workflows. |
| `audit/` | Durable audit records and audit helpers used to track book quality work. |
| `dependencies/` | Dependency and environment documentation. |
| `git-hooks/` | Repository hook documentation and related setup. |

## Figure Tool Ownership

Historically, the book used `mlsysim.viz` for two different kinds of plotting:

- Simulator visualizations that understand `mlsysim` objects, such as roofline,
  distributed roofline, and evaluation scorecard plots.
- Book publication style helpers, such as the shared palette, matplotlib style,
  and one-line `setup_plot()` helper used by many QMD figure blocks.

The simulator-aware plotting functions should stay in `mlsysim`. The book
publication style helpers are owned by Book Tools because they are book
production policy rather than simulator domain logic. Current QMD figure-style
imports should use `book.tools.figures.style`.

Current figure-tool structure:

```text
book/tools/
  figures/
    style.py        # COLORS, set_book_style(), setup_plot(), web DPI
    margin/         # margin SVG devices and inline-ready drawing helpers
  scripts/
    margin_figures/ # generation, inventory, insertion, QA script entrypoints
```

Recommended next migration:

```text
book/tools/
  figures/margin/
    inline.py       # optional Quarto-facing margin_svg() helper
```

Compatibility policy:

1. New and migrated book code imports book style from `book.tools.figures.style`.
2. `mlsysim.viz` may continue to re-export the style helpers only as a
   compatibility bridge for older snippets and standalone simulator tutorials.
3. Add an inline margin helper only if the book starts rendering margin figures
   from executable cells instead of committed SVG assets.

This keeps simulator-object plotting in `mlsysim` while keeping book publication
policy in Book Tools.
