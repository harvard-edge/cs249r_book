# Margin Figure Devices

`devices.py` is the canonical implementation of the reusable margin-figure
visual language. It owns device geometry, semantic colors, label-size bands, and
SVG export behavior.

Production scripts under `book/tools/scripts/margin_figures/` import this module
to generate committed chapter SVGs. The legacy `margin_devices.py` script module
is only a compatibility wrapper.

## What To Edit

Most editorial tweaks should happen in one of two places:

- Per-figure data, labels, captions, and asset names:
  `book/tools/scripts/margin_figures/generate_margin_figures.py`
- Shared device geometry, fonts, colors, label placement, and SVG export:
  `book/tools/figures/margin/devices.py`

Do not hand-edit generated SVG files. Regenerate them from Python so HTML and
PDF stay reproducible.

## Editor Workflow

From the repository root:

```bash
MPLCONFIGDIR=/tmp/mplconfig python3 book/tools/scripts/margin_figures/generate_margin_figures.py
```

Then inspect the output:

```bash
python3 book/tools/scripts/margin_figures/render_margin_contact_sheet.py \
  --svg book/quarto/contents/<volume>/<chapter>/images/svg/<asset>.svg \
  --output /tmp/mlsysbook-margin-check.png
```

The QMD `.column-margin` block owns the prose anchor, caption, and alt text. The
Python owns how the SVG is drawn. If a figure is in the wrong paragraph, edit the
QMD placement. If a label overlaps, a value changes, or the shape is wrong, edit
the Python and regenerate.

## Device Choice

Pick the device by the relationship the reader should see:

| Relationship | Device |
|---|---|
| Magnitude span or gap | `ladder()` |
| Threshold, wall, or cliff | `knee()` |
| Trend or divergence | `sparkline()` |
| Dominant term in a total | `ironbar()` |
| Memory-bound vs compute-bound regime | `roofline()` |
| Framework axis locator | `dam()` |
| Category or state | `taxonomy()` |
| One source affecting many dependents | `blast()` |
| Budget or matched-capacity envelope | `budget_envelope()` |
| Short ordered window | `sequence_strip()` |
| Dependency or feedback relation | `causal_chain()` |
| All-to-all peer coupling | `all_to_all_topology()` |

If a relationship does not fit one of these devices, first consider prose or a
numbered body figure. Add a new margin device only when the shape will recur
across the book and remains readable at 1.25 inches wide.

## Inline Figures

The current production model is executable Python -> committed SVG -> QMD margin
block. That gives editors a single Python place to tweak while keeping Quarto
HTML/PDF builds stable.

Embedding raw matplotlib code directly inside every `.column-margin` block has
not been migrated and would be harder to maintain: it would duplicate imports,
style setup, and SVG hygiene across chapters. If inline execution becomes a hard
requirement, add a small `book.tools.figures.margin.inline` helper first, then
call that helper from QMD cells.
