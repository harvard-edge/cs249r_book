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

## 2026-06 Editorial Illustration Pass

This pass should be run as a textbook-editor and illustrator pass, not a coverage pass.
The current committed SVGs are mostly visually coherent: labels are outlined paths rather
than live font text, and the device vocabulary is stable. The next improvement is narrative
fit plus small production polish: lighter rule lines, less roofline labeling, unclipped knee
curves, and eventually fixed-width SVG canvases so `width="100%"` gives consistent effective
font size.

For each figure, answer:

1. What turn in the local argument does this mark?
2. What relationship does the image make visible faster than prose?
3. Would the figure become weaker if moved two pages earlier or later?
4. Is this a first introduction of the shape, or a callback to a shape the reader already knows?

If the answers are weak, cut or move the figure even if the graphic itself is attractive.

### Uploaded Physical-Scale Mockup

The new vertical mockup showing "Nanometer / Rack scale / Cluster scale / Planet Scale /
Extraterrestrial" has a strong underlying idea: it makes physical scale feel like a sequence
of constraint boundaries. It should not be inserted as-is.

Editorial verdict:

- The icon-spine style is more decorative than the current margin grammar. It uses five
  detailed pictorial boxes where the book's margin language normally uses simple geometric
  devices.
- The level names mix unlike categories: chip feature scale, rack/building distance, cluster
  geography, planet-scale deployment, and orbital deployment. That mix is only justified if
  the adjacent paragraph explicitly teaches that cross-domain span.
- "Planet scale" and "Extraterrestrial" are seductive details unless the local prose actually
  discusses global or orbital deployment. Current likely homes in `compute_infrastructure`
  and `network_fabrics` discuss transistor-to-data-center and rack/pod/network reach, not
  extraterrestrial deployment.
- If used, translate the idea into the canonical grammar: a slim `hierarchy-ladder` /
  staircase or a justified `other-new` physical-scale spine with minimal line icons, parallel
  labels, and a caption that states the learning purpose.

Best current treatment:

- Do not force the full chip-to-orbit ladder into `network_fabrics`: that chapter already has
  a well-placed slow-link blast-radius margin figure and dense body figures for the five-level
  model, bandwidth hierarchy, alpha-beta crossover, topology, and congestion.
- A scoped physical-reach ladder does fit `network_fabrics` at `### Optical interconnects`,
  because the adjacent prose explicitly compares package-scale CPO, copper cable reach, active
  optics, and 100-meter data-center fiber. This should stay visually simple: a thin reach
  spine, four dots, and paired medium/distance labels rather than five pictorial icon boxes.
- Consider a revised physical-stack margin in `compute_infrastructure` only if it is anchored
  to the chapter's existing "Accelerator / Node / Rack / Pod" narrative. The labels should be
  infrastructure levels, not planet/orbit levels: for example "Die", "Node", "Rack", "Pod",
  "Facility". The figure should make the chapter's central move visible: every boundary adds
  a new wall.
- Otherwise keep the mockup as a design prompt for a future chapter or section that explicitly
  spans chip-to-global/orbital deployment.

### Source-of-Truth Workflow

Do not hand-edit exported SVGs. Make visual changes in:

- `book/tools/scripts/margin_figures/margin_devices.py` for reusable device style.
- `book/tools/scripts/margin_figures/generate_margin_figures.py` for per-figure composition.
- `book/tools/audit/margin_figure_decisions.yml` only when changing the curated figure set.

Then regenerate with:

```bash
MPLCONFIGDIR=/tmp/mplconfig python3 book/tools/scripts/margin_figures/generate_margin_figures.py
```

After regeneration, inspect rendered SVG output, not only diffs. Font cleanliness means no
live `<text>` nodes in committed margin SVGs; rule-line cleanliness means strokes remain
visible at 1.25in without overpowering the labels.

### Production Polish Backlog

High-confidence cleanup items from the 2026-06 SVG/style audit:

1. Keep font outlining. The referenced margin SVGs use outlined glyph paths rather than live
   `<text>` nodes, which is the right production choice for print and HTML consistency.
2. Stabilize the SVG canvas width in a dedicated pass. Current tight-bbox export creates
   variable SVG widths that all QMDs render at `width="100%"`, so apparent label size varies.
   This is a broad generated-artifact change and should be reviewed with contact sheets.
3. Keep red sacred for danger, fault, and true limits. Use `SEL` crimson for selected taxonomy
   cells and neutral/list colors for categories.
4. Strip margin rooflines to the relationship: two regime strokes, a ridge guide, and at most
   the operating-point label(s). The words `memory`, `compute`, and `ridge` are too much at
   margin scale.
5. Promote any bespoke diagram that needs more than three labels or more than one teaching job
   into a body figure or cut it.

### 2026-06 Autonomous Audit Decisions

Implemented in the margin-figures worktree:

1. `benchmarking_confidence_detectability.svg` at the statistical-confidence trap. The local
   prose asks the reader to compare a 1K test set with the sample size needed to see a
   one-point change; a two-mark detectability threshold offloads that comparison without
   duplicating a nearby body figure.
2. `vol2_collective_communication_margin_002.svg` was redrawn in place. The previous FSDP
   loop was a small process diagram; the revised figure shows the actual relationship the
   paragraph teaches: standard data parallelism pays one collective per step, while FSDP pays
   two per layer.
3. Several generic curated figures were replaced with source-pinned custom renderers because
   their captions promised specific values or mechanisms:
   - `vol1_benchmarking_margin_001.svg`: 3x component speedup becomes about 1.2x end-to-end.
   - `vol1_introduction_margin_004.svg`: 60/45/25 ms Amdahl pipeline becomes 60/15/25 ms.
   - `vol1_ml_systems_margin_002.svg`: 18.7 GB/s camera ingest versus a 1.25 GB/s 10G link.
   - `vol1_ml_systems_margin_004.svg`: 100/60/40 ms camera pipeline becomes 100/6/40 ms.
   - `vol1_training_margin_001.svg`: GPT-2 activations exceed a V100 HBM capacity anchor.
   - `vol1_training_margin_002.svg`: storage, DRAM, and V100 HBM bandwidth are ordered correctly.
   - `vol1_training_margin_003.svg`: 64 MB full attention matrix versus 64 KB SRAM tile.
   - `vol2_compute_infrastructure_margin_003.svg`: 50,000h per-GPU MTTF becomes 50h at 1K GPUs and 5h at 10K GPUs.
4. Misleading or stale mechanism figures were redrawn in place:
   - `fleet_orchestration_dependency_cascade.svg` now shows priority inversion as a wait-for chain, not a root-failure tree.
   - `ops_scale_cross_model_blast.svg` now shows an embedding/model-dependency update source, not a shared-infrastructure fault.
   - `inference_decode_roofline.svg` now shows decode moving toward the ridge through parallel verification, not batching as the only lever.
   - `vol2_sustainable_ai_margin_002.svg` now shows PUE as IT base plus infrastructure overhead.
   - `vol2_conclusion_margin_002.svg` and `vol2_conclusion_margin_003.svg` are now matched-rate and gain-composition diagrams rather than stale taxonomy/list fallbacks.
5. Two margin figures were retargeted and moved:
   - `model_compression_dam_locator.svg` now highlights the Machine axis next to INT8/INT4 hardware discussion.
   - `ml_ops_drift_threshold_knee.svg` now visualizes drift-detection delay (17 minutes versus 10 days) next to the sample-rate notebook rather than duplicating the rotting-asset body figure.
6. Placement-only fixes moved figures after their local derivations or away from competing sidenotes:
   - Vol. 2 coordination tax moved into the GPT-3 synchronization notebook after the 4%/2% values are introduced.
   - Responsible AI unlearning cost moved into the cost-of-forgetting notebook.
   - Robust AI robustness tax moved from the chapter definition to the adversarial-training notebook.
   - Robust AI transferability and Huber-loss figures moved after long explanatory footnotes.
   - Ops-scale freshness ladder moved beside the batch-versus-streaming comparison.
   - Model-serving traffic-adaptive batching moved below the table it summarizes.
   - Responsible Engineering Goodhart and automation-paradox figures moved after their mechanisms are named.
7. Added `responsible_ai_representation_tax_ladder.svg` near the representation-tax notebook. This was accepted because the notebook asks readers to multiply subgroup count, images per subgroup, and medical-labeling cost; the margin ladder offloads a real magnitude relationship.

Reviewed and deliberately held:

1. `model_serving` high-utilization cliff near the fallacies section. The idea is strong, but
   the chapter already has a body tail-latency explosion figure and a utilization-latency table.
   Add a late margin callback only after rendered layout confirms the section needs it.
2. `hw_acceleration` multi-chip Amdahl ceiling. The local footnote is useful, but Amdahl's Law
   is already introduced with a theorem, equation, lighthouse callout, and existing margin
   motifs in the chapter. A new margin figure here risks repeating a known shape.
3. `nn_computation` diminishing-returns scaling curve near the fallacies section. The candidate
   is pedagogically sound, but it would stack immediately after an existing roofline margin
   figure. Keep the prose-only shape unless that section is redesigned.
4. `responsible_engr` fairness-metric taxonomy near the confusion matrices. The tables and
   footnote already occupy the reader's attention; a margin taxonomy would likely crowd the
   same conceptual moment.
5. Compute-infrastructure precision dotcells and data-storage staging taxonomy were cut after
   the per-chapter audit. In both cases the local prose already made the relationship clear,
   and the margin graphic was adding labels rather than reducing a real learning burden.
6. Additional suggested figures in data engineering, data selection, frameworks, hardware
   acceleration, neural architectures, network PAUSE propagation, and robust/responsible/sustainable
   recap sections were held unless they solved a nonredundant local learning burden. The accepted
   representation-tax addition is the exception because it carries real arithmetic, not coverage.

### 2026-06 Per-Chapter Agent Audit Round

The stricter pass used one read-only audit agent per chapter QMD, with structural front/back
matter and Purpose-section stack macros exempt. The agents inspected existing `.column-margin`
placements, rendered chapter contact sheets when useful, checked whether SVG text remained
outlined, and proposed additions only when a paragraph exposed a nonredundant local relationship.

Scope completed:

- Vol. 1 chapter files: introduction, ml_systems, ml_workflow, data_engineering,
  nn_computation, nn_architectures, frameworks, training, data_selection, model_compression,
  hw_acceleration, benchmarking, model_serving, ml_ops, responsible_engr, conclusion.
- Vol. 2 chapter files: introduction, compute_infrastructure, network_fabrics,
  collective_communication, data_storage, distributed_training, edge_intelligence,
  fault_tolerance, fleet_orchestration, inference, ops_scale, performance_engineering,
  responsible_ai, robust_ai, security_privacy, sustainable_ai, conclusion.

Accepted changes from this round:

- Repaired generic-parser failures where derived ratios or fallback numbers became fake bars.
  Ratios such as 40x, 18x, 32x, 80x, 200x, and 30x are now annotations unless the ratio itself
  is the measured quantity.
- Replaced inverted or misleading Vol. 2 figures with source-pinned custom renderers:
  distributed-training rho/barrier/energy/bandwidth, collective payload shrink, edge memory
  and radio-savings contrasts, fault checkpoint/downtime, fleet scheduling/capacity/failure
  cadence, inference MoE memory, ops-scale statistics curves, performance-engineering energy
  and roofline figures, responsible-AI trends/risk, security/privacy SGX memory, and
  sustainable-AI carbon/radio-energy ladders.
- Cut weak margin figures rather than filling space: compute precision dotcells and the
  data-storage staging taxonomy.
- Kept most proposed additions out. The agents repeatedly found that body figures, tables,
  footnotes, or the prose itself already carried the moment; adding another margin graphic
  would be decorative or redundant.

Held for a later PDF-layout/SSOT pass:

- Some existing figures still deserve deeper source binding to nearby LEGO classes rather than
  hard-coded values, especially sustainable-AI PUE/energy-per-byte and conclusion tail/gain/power
  figures.
- A full Quarto PDF build should still check vertical collisions with footnote sidenotes. This
  pass inspected QMD anchors and SVG/contact-sheet rendering, not final page breaks.
