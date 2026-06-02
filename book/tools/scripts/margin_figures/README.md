# Margin Figure Tooling

This directory is the source of truth for generated MLSysBook margin-figure SVGs.
Generated assets live beside their chapters under
`book/quarto/contents/<volume>/<chapter>/images/svg/`.

## Files

- `generate_margin_figures.py` creates the committed margin SVG assets.
- `margin_devices.py` owns the canonical device vocabulary and visual style.
- `insert_curated_margin_figures.py` inserts accepted curated candidates into QMD.
- `render_margin_contact_sheet.py` renders referenced margin SVGs into contact sheets for visual QA.

Related editorial rules and records:

- `.claude/rules/margin-figures.md`
- `.claude/rules/figure-visual-language.md`
- `book/tools/audit/margin_figure_opportunities.yml`
- `book/tools/audit/margin_figure_decisions.yml`
- `book/tools/audit/margin_figure_style_audit.md`

## Generate

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/mplconfig python3 book/tools/scripts/margin_figures/generate_margin_figures.py
```

Do not hand-edit generated SVGs. Change the Python source, regenerate, then check
the diff.

`margin_devices.py` fixes the SVG hash salt and omits per-run date metadata.
Keep those settings in place so regeneration changes only intentional geometry,
labels, or style decisions rather than timestamps and random clip-path ids.

## Render And Inspect

Render a volume contact sheet:

```bash
python3 book/tools/scripts/margin_figures/render_margin_contact_sheet.py \
  --volume vol1 \
  --output /tmp/mlsysbook-vol1-margin-sheet.png
```

Render a chapter or a few explicit SVGs:

```bash
python3 book/tools/scripts/margin_figures/render_margin_contact_sheet.py \
  --chapter vol2/inference \
  --output /tmp/mlsysbook-inference-margin-sheet.png
```

```bash
python3 book/tools/scripts/margin_figures/render_margin_contact_sheet.py \
  --svg book/quarto/contents/vol2/network_fabrics/images/svg/network_fabrics_physical_reach_ladder.svg \
  --svg book/quarto/contents/vol2/responsible_ai/images/svg/responsible_ai_representation_tax_ladder.svg \
  --output /tmp/mlsysbook-margin-focused-sheet.png
```

The renderer uses `rsvg-convert` and Pillow. Inspect the contact sheet at normal
size and zoomed in. Check that text is legible at margin scale, labels do not
collide, line weights are clean, red is reserved for danger/limits, and the image
does not look like a miniaturized body plot.

## SVG Hygiene

Margin SVGs should have outlined text so the HTML and PDF builds do not depend on
font availability:

```bash
rg -n '<text|font-family|font-size' book/quarto/contents/vol*/**/images/svg/*.svg
```

For generated margin SVGs, this command should not report live text/font-family
residue. A direct `font-size` hit usually means a renderer escaped the
`svg.fonttype='path'` contract.

## Scale Honesty

If geometry encodes a number, it must be visually to scale on a declared scale.
Use these defaults:

- `ladder()` for magnitude spans. It is linear for small spans and log-scaled for
  large spans.
- `ironbar()` or `simple_bar()` for fractional composition where segment widths
  sum to a total.
- `pipeline_rows()` for Amdahl-style before/after bars where every row shares
  the same absolute time denominator.
- Dots, arrows, formulas, or equal-weight labels for symbolic relationships.

Do not add minimum visual widths to quantitative bars. If a true value would be
too small to see on a linear scale, use the ladder's log scale or draw a
schematic that does not pretend to be quantitative.

Schematic physical-scale spines are allowed when the prose names ordered
operating domains rather than asking for proportional measurement. In that case,
keep the levels evenly spaced, make the caption/alt text categorical, and record
that the spacing is not a quantitative axis.

## Placement Ownership

The generator decides how an asset is drawn. It does not decide the final prose
anchor. Placement belongs in the QMD `.column-margin` block and must satisfy the
spatial-contiguity rule in `.claude/rules/margin-figures.md`.

For curated/generic figures, the durable bookkeeping is:

- `margin_figure_opportunities.yml`: proposed chapter, idea, device, labels, and
  rationale.
- `margin_figure_decisions.yml`: accepted/rejected decision and final device.
- QMD: exact paragraph placement, caption, and alt text.

The asset id convention is stable: candidate id with hyphens replaced by
underscores, written under the candidate's `chapter` directory.
