# Margin Figure Style Audit

Created: 2026-05-30

Scope: the 127 curated margin illustrations inserted on `feat/margin-elements`, generated from `book/tools/audit/margin_figure_decisions.yml`.

Terminology: `D-A-M` is the Volume I Data-Algorithm-Machine framing. `D-A-I` is the Volume II specialization of the same triad: Data-Algorithm-Infrastructure. It is not a separate framework.

Distribution rule: do not tune the mix of figure types to hit a target histogram. The distribution is a diagnostic. If one device dominates, check whether the local prose really contains that many relationships of that type; change the device only when the paragraph's teaching job changes.

## Current Device Use

| Device | Count | Current variants in use | Editorial read |
|---|---:|---|---|
| `hierarchy-ladder` | 43 | filled bars | Strong coverage for magnitude gaps. We are underusing lollipop/staircase variants. |
| `sparkline-trend` | 27 | filled gap 18, end dots 8, inflection 1 | Good range. Inflection is rare because few candidates were true turning-point claims. |
| `scale-anchor` | 15 | shaded zone 7, dashed threshold 4, two-tone regime 4 | Good range after adding explicit words (`SLO`, `rho=1`, `accept`, etc.). |
| `taxonomy-mini` | 11 | list+dots only | Under-varied. Some candidates should become 2x2 quadrants or dot-cells when the prose has two axes or on/off states. |
| `iron-law-bar` | 9 | stacked only | Adequate for composition/dominance, but trio/columns should appear when the prose compares separate terms directly. |
| `thumbnail-roofline` | 8 | single-regime dot 4, memory-to-compute transition 3, two-regime points 1 | Better after the follow-up pass. We now show compute-bound plateau cases where the text supports them. |
| `blast-radius` | 8 | fan 4, tree 4 | Good for correlated failure and cascades. Rings remain unused. |
| `other-new` | 5 | nested system box, all-to-all topology, Pareto frontier, feedback loop, epsilon budget | Useful exceptions, but should remain rare. Prefer canonical devices when honest. |
| `dam-locator` | 1 | triangle | Underused relative to the book's D-A-M/D-A-I spine. Add more only where the prose explicitly invokes the framework axis/coupling. |

## Device Selection Method

Pick the device by the paragraph's one teaching job:

| Local teaching job | Use this device | Do not use it when... |
|---|---|---|
| Span or scale gap | `hierarchy-ladder` | the prose is about a turning point, not magnitude. |
| Threshold, cliff, knee, limit, SLO boundary | `scale-anchor` | the curve shape is not actually accelerating or thresholded. |
| Change over time/scale, divergence, decay, saturation | `sparkline-trend` | the point is a static comparison. |
| Compute-bound vs memory-bound placement | `thumbnail-roofline` | there is no roofline/regime claim in the prose. |
| Which Iron Law term dominates | `iron-law-bar` | the terms are not Data/Compute/Latency-style components. |
| Which Data-Algorithm-Machine or Data-Algorithm-Infrastructure axis is active | `dam-locator` | the paragraph does not explicitly use that framework or axis shift. |
| Classification, status set, two-axis taxonomy | `taxonomy-mini` | the items are a sequence, magnitude, or causal chain. |
| One fault/source affects many downstream nodes | `blast-radius` | the relationship is not propagation or shared-fate failure. |
| A simple concept does not honestly fit the kit | `other-new` | a canonical device can express the same idea without distortion. |

For new figures, the desired workflow is: read the local paragraph, name the teaching job, choose the matching device, choose the simplest variant, then check redundancy and margin space. A global device-count target should never override that local decision.

Use the current distribution as a prompt for targeted review:

- Many ladders are acceptable if the prose repeatedly asks students to compare orders of magnitude.
- More D-A-M/D-A-I locators are worth adding only at genuine framework-return moments.
- More taxonomy quadrants/dot-cells are worth adding only where the local prose has true axes or state cells.
- More iron-law trio/columns are worth adding only where separate Data/Compute/Latency terms are directly compared.

## Variants Not Yet Used In The Curated Set

| Available variant | Why it is not used yet | When to use it |
|---|---|---|
| ladder `lollipop` | Most current ladder candidates are magnitude spans, not positions on a common scale. | Arithmetic intensity or utilization positions where the point location matters more than bar area. |
| ladder `staircase` | Few candidates are true ordered levels rather than quantities. | Deployment tiers, memory hierarchy levels, abstraction levels. |
| iron-law `trio` | Current curated candidates mostly encode composition. | Direct side-by-side Data/Compute/Latency term comparison. |
| iron-law `columns` | No current narrow candidate needed vertical columns. | Compact comparison where vertical magnitude reads cleaner than a stacked total. |
| D-A-M `boxes` / `pills` | Only one curated D-A-M locator survived; it was a coupled-triad moment. | Short locator moments: "this paragraph is Data", "this section is Infrastructure". |
| taxonomy `quadrant` | The generic renderer currently defaults taxonomy candidates to list+dots. | Two independent axes: high/low entropy x high/low gravity, severity x frequency, etc. |
| taxonomy `dotcells` | No current candidate was explicitly a 2x2 on/off state matrix. | Active/inactive status cells, attack/control matrices. |
| blast `rings` | No audited paragraph clearly described severity by distance from a source. | Fault domains, isolation zones, privacy leakage radius, deployment blast radius by boundary. |

## Generation Model

The current branch generates committed SVG files from a central Python generator:

- `book/tools/scripts/margin_figures/generate_margin_figures.py`
- `book/tools/scripts/margin_figures/margin_devices.py`
- `book/tools/audit/margin_figure_decisions.yml`

Each QMD contains only the margin block, SVG reference, alt text, and italic margin note. This is intentional for the first committed pass: the prose stays readable, SVG output is deterministic enough for review, and the book does not rerun 127 tiny plotting cells during every render.

Inline code in each QMD is possible, but it has costs:

- It would add a large amount of plotting code to prose files.
- It would slow and complicate full-book renders.
- It would make global visual-language fixes harder because each chapter could drift.
- It would increase the chance that a margin figure accidentally becomes a numbered figure or emits format-specific output.

The best next model is not raw inline plotting everywhere. It is a hybrid:

1. Keep canonical renderers centralized in `margin_devices.py`.
2. Keep committed SVGs as the production artifact.
3. Move per-figure intent/parameters closer to the text when a figure needs local customization: either a nearby compact spec comment/block in the QMD or a chapter-local YAML sidecar.
4. For data-bearing figures, connect parameters to the chapter LEGO/MLSysIM source of truth rather than parsed label strings.

That gives us context-specific figures without scattering renderer code across 127 prose locations.

## Editorial Next Pass

Recommended next improvements:

1. Add more D-A-M/D-A-I locators where the prose explicitly shifts axis or returns to the book framework.
2. Convert taxonomy list+dots into quadrants/dotcells where the local prose has a true two-axis structure.
3. Use iron-law trio/columns for direct term comparisons.
4. Use ladder lollipop/staircase when the prose is about positions or ordered levels rather than magnitude bars.
5. Keep blast rings held until there is an honest severity-by-distance concept.
6. Promote any margin figure that needs axes, legends, or more than one idea into a numbered body figure instead.
