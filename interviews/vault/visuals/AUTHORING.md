# Authoring visual questions

StaffML questions can optionally attach a diagram. The practice page renders
it between the scenario prose and the answer textarea, and the `question`
field always stays sticky at the top — reading flow is **scenario →
diagram → answer**.

Use visuals sparingly. A good visual earns its place; a bad one is
noise that slows the reader down.

## When a visual earns its place

A visual earns its place when **all three** hold:

1. **The ask requires reading the diagram.** If the question can be
   answered from the scenario alone, the visual is decorative — omit it.
2. **The visual encodes information that text cannot.** Topology,
   memory layouts, roofline curves, pipeline timelines, dataflow —
   things where spatial structure *is* the payload.
3. **A static image suffices.** Animation, interactivity, and
   multi-step reveals are out of scope.

### High-value candidate topics

Target these first; each one repeats across many chain positions, so
one good diagram earns its keep across dozens of questions.

- Ring / tree AllReduce topologies (show the ring, ask for latency)
- Parameter-server vs. AllReduce dataflow
- Roofline diagrams with workload points plotted
- KV-cache growth vs. sequence length
- Pipeline parallelism bubble / Gantt charts
- Memory hierarchy: HBM + SRAM + host DRAM + disk
- TinyML MCU memory map (flash + SRAM + model footprint)
- Systolic array dataflow (weight-stationary vs. output-stationary)
- Attention computation graph (Q·Kᵀ then softmax then ·V)
- MoE all-to-all shuffle topology

## Authoring workflow

1. **Draft the SVG** following `.claude/rules/svg-style.md` (the book's
   SVG system). Non-negotiables:
   - `viewBox="0 0 680 460"` default (widen only when content demands).
   - `font-family="Helvetica Neue, Helvetica, Arial, sans-serif"` on
     the root `<svg>`.
   - Semantic palette — compute blue `#cfe2f3`/`#4a90c4`, data green
     `#d4edda`/`#3d9e5a`, routing orange `#fdebd0`/`#c87b2a`, error
     red `#f9d6d5`/`#c44`, MIT red accent `#a31f34`.
   - Orthogonal routing (no diagonals except genuine topology diagrams).
   - Arrows anchor at box edges, route around obstacles with ≥10px
     clearance.
   - Integer coordinates on a 10-px grid.
   - Text in `<text>` elements (not paths) — selectable + accessible.
2. **Save** to `interviews/vault/visuals/<track>/<id>.svg` where
   `<track>` matches the question's track (cloud, edge, mobile, tinyml,
   global) and `<id>` matches the question's YAML id. Bare filename
   only — no subdirectories, no path traversal.
3. **Add the `visual:` block** to the matching YAML:
   ```yaml
   visual:
     kind: svg
     path: <id>.svg
     alt: >-
       Full accessibility description — objective, concrete, ≤400
       chars. Describe what the diagram SHOWS, not why it matters.
     caption: "Short caption rendered below the figure. ≤120 chars. Optional."
   ```
4. **Build** — `vault build --legacy-json`. This copies the SVG to
   `interviews/staffml/public/question-visuals/<track>/` and surfaces
   the `visual` metadata in the corpus bundle.
5. **Preview** at `/practice?q=<id>` on the dev server.

## Reference exemplar

`cloud/cloud-visual-001.yaml` + `cloud/cloud-visual-001.svg` — Ring
AllReduce on 4 ranks. Diagram shows the ring topology + chunk labels +
bandwidth annotations; scenario gives concrete numbers; question asks
for the total AllReduce time. Solution walks through the
2(N−1)/N × data / bw formula. Copy this pattern.

## Accessibility requirements (non-negotiable)

- **`alt` is required** and is enforced by the schema. A visual
  without alt fails Pydantic validation; `vault build` will reject it.
- Colour is never the sole semantic channel — pair colour with label
  text, line style, or shape. A colour-blind reader must still be able
  to parse the diagram.
- Text in SVG `<text>` elements (not baked into paths) so it's
  selectable and screen-reader friendly.
- WCAG AA contrast on any label over a coloured fill: `#333` text on
  `#cfe2f3` compute-blue passes; `#999` text on the same fill fails.

## Anti-patterns to reject

- Inline `<svg>` markup inside YAML. The schema's `path` field is a
  bare filename — path traversal is rejected. Use the file-reference
  pattern.
- Mermaid source for anything other than simple node-edge graphs. The
  renderer MVP supports SVG only; `kind: mermaid` is reserved for a
  future inline-text path and is currently a no-op.
- Label text that duplicates the scenario prose. The diagram should
  encode *additional* information, not restate the scenario.
- Decorative gradients, fake 3-D, beveled edges, drop shadows. Every
  mark earns its place (Tufte's data-ink principle — see the SVG
  style guide for the full chart-junk policy).
- Icons or emoji in the SVG. Use neutral shapes + labels.
- Watermarks, signatures, tool branding. The caption handles attribution.

## Build artifacts

The directory `interviews/staffml/public/question-visuals/` is a
build output (written by `copy_visual_assets` during `vault build
--legacy-json`). Source files live under `interviews/vault/visuals/`.
Do not edit the build output directly — changes will be overwritten
on the next build.
