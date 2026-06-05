# Margin Figure Tooling

This directory is the source of truth for generated MLSysBook margin-figure SVGs.
Generated assets live beside their chapters under
`book/quarto/contents/<volume>/<chapter>/images/svg/`.

Production status: these tools are book production dependencies. The book
depends on the generated SVGs, and those SVGs depend on this generation workflow
plus the Book Tools modules under `book/tools/figures/`. Do not delete, move, or
rewrite these files without updating the generation workflow, QMD placements,
audit records, and this README.

## Files

- `generate_margin_figures.py` creates the committed margin SVG assets.
- `margin_devices.py` is a compatibility wrapper for the canonical device module.
- `insert_curated_margin_figures.py` inserts accepted curated candidates into QMD.
- `inventory_margin_figures.py` inventories actual QMD margin placements.
- `audit_margin_caption_alignment.py` extracts each placed margin figure with
  nearby prose so captions can be audited against the narrative.
- `render_margin_reader_link_audit.py` renders an inspectable markdown packet
  with source excerpts, embedded figures, captions, alt text, and local prose
  anchors for editor/LLM review.
- `render_margin_contact_sheet.py` renders referenced margin SVGs into contact sheets for visual QA.

Current organization: shared book figure style lives in
`book/tools/figures/style.py`, and stable margin drawing devices live in
`book/tools/figures/margin/devices.py`. This directory retains script
entrypoints for generation, insertion, inventory, rendering, and compatibility.
That keeps book-owned plotting policy out of `mlsysim` while preserving simple
CLI commands for production work.

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

`book.tools.figures.margin.devices` fixes the SVG hash salt and omits per-run
date metadata.
Keep those settings in place so regeneration changes only intentional geometry,
labels, or style decisions rather than timestamps and random clip-path ids.

## Tweak An Existing Figure

1. Find the SVG asset in the QMD `.column-margin` block.
2. Search for the asset stem in `generate_margin_figures.py`.
3. Change the figure-specific values, labels, or device call there.
4. If the same label-placement or geometry issue affects a whole device family,
   change `book/tools/figures/margin/devices.py` instead.
5. Regenerate the SVGs and inspect a contact sheet.
6. Update the QMD caption or `fig-alt` only when the prose takeaway or objective
   accessibility description changes.

For example, if `model_compression_quantization_roofline.svg` has a label that
sits on a line, edit the per-figure roofline call or the shared `roofline()`
device. If the caption no longer matches the paragraph, edit the QMD margin
block rather than the Python.

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

Label placement checks are part of visual QA. Labels must stay inside the visual
frame, must not sit directly on a line, marker, arrow, or bar edge, and must not
bleed beyond the image crop. Move labels toward nearby whitespace before shrinking
the font. For roofline devices, keep memory-bound labels near the dot and off the
slope; keep compute-bound labels below the plateau/dot when possible.

Default SVG label sizes are tuned for the 1.25in margin column:

- Base matplotlib font size: `5.5`.
- Ordinary labels: `5.0-5.5`.
- In-bar numeric labels: `4.7-5.2`, bold.
- Dense sequence-strip cell labels: `4.6`.
- Ratio/bracket labels: `4.8-5.2`, bold.
- Tiny scale cues: `3.9-4.6`.
- Emphasis labels: at most `6.0`, unless a single letter is itself the mark.

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
| A finite budget, quota, or matched-rate envelope | `budget-envelope` | Budget burn-down, capacity limit, or co-design envelope. |
| A short ordered window or phase strip | `sequence-strip` | A few windows/phases where order or span is the point. |
| A compact dependency or feedback relation | `causal-chain` | Cause-effect chain or residual feedback loop. |
| Many endpoints coupled to all peers | `all-to-all-topology` | Tiny mesh for many-to-many peer coupling. |

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
| `budget-envelope` | `X crosses or stays inside budget Y`. |
| `sequence-strip` | `Window X spans phases A-B`. |
| `causal-chain` | `A choice propagates to B`. |
| `all-to-all-topology` | `Each peer couples to every other peer`. |

Reject captions that are just titles, legends, footnotes, implementation notes,
or generic prompts like "why this matters." If the caption says "dominates,"
the marks must visibly show dominance; if it says "cliff," the curve must
visibly cliff.

Audit caption/prose alignment from the repository root:

```bash
python3 book/tools/scripts/margin_figures/audit_margin_caption_alignment.py \
  --markdown /tmp/mlsysbook-caption-alignment.md \
  --csv /tmp/mlsysbook-caption-alignment.csv
```

To focus the editorial pass on likely issues:

```bash
python3 book/tools/scripts/margin_figures/audit_margin_caption_alignment.py \
  --review-only \
  --markdown /tmp/mlsysbook-caption-alignment-review.md
```

For a stricter "does the narrative, figure, and caption click together" pass,
raise the review threshold:

```bash
python3 book/tools/scripts/margin_figures/audit_margin_caption_alignment.py \
  --review-threshold 0.30 \
  --review-only \
  --markdown /tmp/mlsysbook-caption-alignment-strict-review.md
```

The overlap score is only a triage signal. Equation-heavy prose, tables, and
captions that use a clearer synonym can score low while still being correct.
Read the caption with the paragraph before and after the margin block before
changing prose, captions, or figure placement.

For an inspectable reader-link packet that answers "where is this figure placed,
what prose is it supporting, and what does the visual encode?", run:

```bash
python3 book/tools/scripts/margin_figures/render_margin_reader_link_audit.py
```

The default output is
`book/tools/audit/margin_figure_reader_link_audit.md`. Each entry embeds the SVG,
shows the exact QMD `.column-margin` source excerpt, records the caption and
`fig-alt`, and quotes the nearest prose before and after the margin block. This
is the preferred evidence artifact for editor and LLM-style review.

The corresponding author-facing verdict record is
`book/tools/audit/margin_figure_reader_alignment_verdicts.md`. It summarizes the
same 224 placements with a pass/fix reader-alignment verdict and a compact prose
anchor for each figure.

For a browser-readable version of the same audit, run:

```bash
python3 book/tools/scripts/margin_figures/render_margin_reader_alignment_html.py
```

The default output is
`book/tools/audit/margin_figure_reader_alignment.html`. Open this file directly
in a browser to review each figure as a card with the SVG, caption, source QMD
line, strongest prose anchor, `fig-alt`, and expandable placement context. The
preview SVGs are embedded into the HTML so the page works from `file://` without
a dev server. The page includes search, chapter filtering, strict-reviewed
filtering, and a prose bridge filter so authors and editors can scan the full
audit without reading the markdown packet linearly.

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
  large spans. Log-scaled ladders carry a small in-figure `log scale` baseline
  cue; do not remove it unless the figure is converted to a purely schematic
  non-quantitative device.
- Use `wall=True` only when the red ceiling is a semantic wall, threshold, fault,
  or hard limit named by the prose. Do not use a red top rule for an ordinary
  largest tier in an "A dwarfs B" comparison.
- Keep ladder labels in sentence style: capitalize the first word, preserve
  acronyms (`HBM`, `DRAM`, `NVMe`, `GPU`), and separate numbers from units
  (`100 ms`, `5.7 yr`, `640 pJ`).
- Use separated bars for two- or three-tier magnitude comparisons. Use
  `style="staircase"` for dense ordered hierarchies where the point is the level
  sequence and vertical compactness helps the reader scan.
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
- `budget_envelope()` for finite budgets, quotas, capacity limits, or matched
  rates. Use `style="burn"` when crossing the red limit is the point and
  `style="matched"` when two capacities must be co-designed.
- `sequence_strip()` for a few ordered windows/phases with an optional red
  bracketed span. Do not use it for a full process diagram.
- `causal_chain()` for compact dependency or residual-feedback sketches. If the
  arrows need more than a few labels, use a body figure.
- `all_to_all_topology()` only when many-to-many peer coupling is the local
  teaching job; otherwise prefer a body topology figure or prose.
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
