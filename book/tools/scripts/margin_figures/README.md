# Margin Figure Tooling

This directory is the source of truth for generated MLSysBook margin-figure SVGs.
Generated assets live beside their chapters under
`book/quarto/contents/<volume>/<chapter>/images/svg/`.

## Files

- `generate_margin_figures.py` creates the committed margin SVG assets.
- `margin_devices.py` owns the canonical device vocabulary and visual style.
- `insert_curated_margin_figures.py` inserts accepted curated candidates into QMD.
- `inventory_margin_figures.py` inventories actual QMD margin placements.
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

## Device Catalog

The production margin kit is intentionally small and repeatable. Choose the
device by the relationship the reader should see, not by the chapter topic:

| Reader needs to see | Device | Default use |
|---|---|---|
| A magnitude span or gap | `hierarchy-ladder` | Orders of magnitude, capacity, bandwidth, energy, latency. |
| A threshold, cliff, or regime change | `scale-anchor` | Queueing knees, utilization cliffs, phase changes, limits. |
| Growth, decay, divergence, or saturation | `sparkline-trend` | One or two simple curves over time/scale. |
| Which term dominates a total | `iron-law-bar` | Data/compute/latency/resource decomposition. |
| Memory-bound vs compute-bound placement | `thumbnail-roofline` | Roofline regime locator. |
| Which D/A/M or D/A/I axis matters | `dam-locator` | Framework locator, not a numeric chart. |
| A category or state selection | `taxonomy-mini` | Two-axis quadrant or short status list. |
| One source affecting many dependents | `blast-radius` | Correlated failure or propagation. |

`other-new` is an audit flag, not a normal production device. If a proposed
visual does not fit the catalog, first try to rewrite it as one of the devices
above. Otherwise promote it to a numbered body figure, leave it as prose, or
reject it. Add a new device only when the visual concept recurs across chapters,
is readable at 1.25in width, and has a stable meaning not covered by the kit.

## Caption Discipline

Draft the caption before drawing. The caption is the editorial takeaway; the
`fig-alt` is the objective accessibility description. Do not copy one into the
other.

Good margin captions are short declarative phrases that reinforce the paragraph
beside them:

| Device | Caption pattern |
|---|---|
| `hierarchy-ladder` | `X dwarfs Y` or `the constraint spans N orders`. |
| `scale-anchor` | `Past T, Y becomes the constraint`. |
| `sparkline-trend` | `X outpaces or falls behind Y as scale grows`. |
| `thumbnail-roofline` | `The workload sits in or crosses into regime R`. |
| `iron-law-bar` | `Term X dominates the total`. |
| `dam-locator` | `This paragraph turns on axis X`. |
| `taxonomy-mini` | `This case lands in quadrant/state X`. |
| `blast-radius` | `One source perturbs many dependents`. |

Reject captions that are just titles, legends, footnotes, implementation notes,
or generic prompts like "why this matters." If the caption says "dominates,"
the marks must visibly show dominance; if it says "cliff," the curve must
visibly cliff.

## Inventory Placements

List the actual SVG margin figures currently placed in the book:

```bash
python3 book/tools/scripts/margin_figures/inventory_margin_figures.py
```

Useful targeted checks:

```bash
python3 book/tools/scripts/margin_figures/inventory_margin_figures.py \
  --chapter vol2/data_storage
```

```bash
python3 book/tools/scripts/margin_figures/inventory_margin_figures.py \
  --untracked-only
```

```bash
python3 book/tools/scripts/margin_figures/inventory_margin_figures.py \
  --format csv \
  --output /tmp/mlsysbook-margin-figures.csv
```

The inventory is intentionally based on QMD `.column-margin` blocks rather than
only on the audit YAML. Use it to confirm the real placement line, caption,
alt text, asset existence, and any matching opportunity/decision metadata.

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
- `ratio_annotation_ladder()` for measured tiers plus a derived multiplier,
  percentage, or symbolic span. The annotation is drawn as a thin leader between
  the compared bar endpoints; it is not a third tier and should never float as
  bare text. If a ladder includes context tiers beyond the pair named by the
  ratio, pass the compared tier indexes explicitly so the leader points to the
  right endpoints.
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
