# Figure Tools

Production figure helpers for MLSysBook live here. These modules are importable
Book Tools, not scratch scripts.

## Modules

- `style.py`: shared matplotlib palette, font stack, setup helpers, and simple
  book-owned chart helpers such as `bar_compare()`.
- `margin/devices.py`: stable margin-figure device vocabulary and drawing
  helpers used by the SVG generation workflow.

Script entrypoints that generate, insert, inventory, or render figures remain
under `book/tools/scripts/`. Keep reusable drawing policy here and keep scripts
thin where practical.
