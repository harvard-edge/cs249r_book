# MLSysBook Tools

`binder/tools` contains production tooling for the MLSysBook source tree. These
tools are part of how the book is built, checked, generated, audited, and
released. Some are called by humans, some by CI, some by Binder commands, and
some by Quarto-adjacent generation workflows.

Do not delete, rename, or move anything under this tree only because it looks
like a one-off script. First audit actual use in QMD files, Binder commands, CI,
release scripts, documentation, and generated-asset workflows.

## Production Policy

- Treat `binder/tools` as book infrastructure, not scratch space.
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

The matplotlib figure style (palette, font stack, font sizes, figure size, and
the `setup_plot()` helper) has one source: `mlsysim/mlsysim/viz/style.py`,
alongside the simulator-aware plotters in `mlsysim.viz`. Every figure, whether
a chapter cell or an `mlsysim` plotter, inherits the same sizes from it.

```text
mlsysim/mlsysim/viz/
  style.py        # COLORS, set_book_style(), setup_plot(), BODY_FIGSIZE, MIN_FONT_PT
  fonts/          # bundled TeX Gyre Heros faces registered by set_book_style()
binder/tools/figures/
  style.py        # re-export of mlsysim.viz.style (the import path chapter cells use)
```

Chapter cells may import either `binder.tools.figures.style` or `mlsysim.viz`;
both resolve to the same objects. Define nothing style-related in binder.

`./binder/binder check figures --scope print-scale --volN` renders every
matplotlib body figure and checks it against the limits in `viz/style.py`
(print scale, minimum font size, figures created without the style).
