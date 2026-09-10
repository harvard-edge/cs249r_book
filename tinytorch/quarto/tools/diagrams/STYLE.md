# TinyTorch diagram style guide

Visual convention for the per-module "What You'll Build" diagrams in
`../../assets/images/diagrams/`. The SVGs are the **source-of-truth**
artifacts — edit them directly. There is no generator to round-trip
through.

> **Workflow:** edit `*.svg` → `make figures` (rsvg → `*.pdf`) →
> `make pdf` (Quarto renders). The Lua filter at `svg-to-pdf.lua`
> rewrites `<img>.svg` → `.pdf` for the PDF format only; HTML keeps
> the SVG.

---

## Design philosophy

Restrained. Greyscale base, single accent color used surgically.
References: Stripe Press, Hennessy & Patterson, Tufte. The principle:
*~85% of the ink is greyscale; color is reserved for the one element
the reader's eye should land on first.* Categorical color-coding
(blue=compute, green=data, …) requires the reader to internalize a
legend — we don't do it.

A diagram should *land* on the page in under two seconds. If a reader
has to trace which box an arrow came from, the layout is wrong.

## Palette

| Role                     | Fill       | Stroke     | Text       |
|--------------------------|------------|------------|------------|
| Standard node            | `#f4f5f7`  | `#9ca3af`  | `#1f2937`  |
| **Accent** (max 1 / dia) | `#fff1e8`  | `#ff8246`  | `#1f2937`  |
| Outer panel ("frame")    | `#f8f9fa`  | `#e5e7eb`  | `#1f2937`  |
| Subtitle text            | —          | —          | `#6b7280`  |
| Arrow / connector        | —          | `#9ca3af`  | —          |
| Page background          | `#ffffff`  | —          | —          |

`#ff8246` is `flameorange` from the TinyTorch brand palette (also
defined in `pdf/_quarto.yml`). It's the only chromatic color allowed in
a diagram. Don't introduce sage/navy/red/etc.

## Strokes & geometry

- **Border weight**: uniform 1pt on every box, panel, and arrow. Mixing
  weights makes diagrams look noisy.
- **Corner radius**: `rx="2"` on every rectangle. Sharp 90° reads as
  schematic; rx≥8 reads as marketing card.
- **Arrowheads**: defined once per SVG via a `<marker>`, reused on
  every connector. Don't draw bespoke arrowheads with paths.
- **Dashed strokes** (`stroke-dasharray="3,2"`): use sparingly, only
  for "implied / optional / background" relationships. Never for a
  primary information path.

## Typography

- **Font stack**: `Helvetica Neue, Helvetica, Arial, sans-serif`. Sans
  inside diagrams pairs with the body's serif (`texgyrepagella`) per
  the typographic standard for figures-in-prose.
- **Sizes** (tuned for the LaTeX `0.75\linewidth` cap — a 680-wide
  canvas renders at ~4.9in wide on the page):
  - Panel eyebrow ("Your X System"): 11pt bold
  - Node title: 10pt bold
  - Node subtitle: 9pt regular
- **Color**: titles `#1f2937` (dark slate, not pure black), subtitles
  `#6b7280` (slate). Same colors regardless of accent — the *box*
  carries the highlight, not the words inside it.

## Canvas

- **viewBox width**: always `680`. The PDF clamps to `0.75\linewidth`
  no matter what; standardizing the width keeps every diagram the same
  on-page size.
- **viewBox height**: **size to content + ~10–20px breathing room.**
  Do NOT default to 360 if your diagram only needs 240. Empty
  whitespace below the last row reads as "this figure is unfinished."
  Common heights in this set: 240, 260, 280, 300, 320.
- **Outer panel rect**: must enclose every node with ~20px padding on
  all sides. The panel `height` attribute follows from the lowest
  node's bottom edge + padding. Don't leave a tall panel with sparse
  content.
- **No in-SVG title or footer text**. The Quarto `fig-cap=` on the qmd
  side does the labeling. Putting a title inside the figure AND letting
  Quarto add a caption beneath it duplicates the same words in two
  visual styles and wastes vertical space.
- **No outer page border**. The white background is the diagram's
  edge.

> **Sizing recipe.** Place all your nodes first using the coordinates
> below. Look at the lowest `y + height`. Set the panel `height` so
> panel ends ~20px below that. Set viewBox `height` so it ends ~10–20px
> below the panel.

## When to use the orange accent

Apply to **at most one node** per diagram. The chosen node should be
the single most important element — the thing the chapter is teaching
you to build, the conceptual heart, or the produced output. Some
diagrams have **no accent** — that's correct when no element is more
central than the others (e.g. flat overviews of components).

Working rules per diagram type:

| Diagram type                      | Accent target                          |
|-----------------------------------|----------------------------------------|
| Flat component overview           | None                                   |
| Process / pipeline → output       | The output node                        |
| Hierarchy with a base abstraction | The base abstraction                   |
| N inputs → 1 result               | The result node (widened — see below)  |
| Multiple equivalent alternatives  | The default / recommended one          |

The accent is editorial, not decorative. If you can't articulate *why*
a particular node is the accent, drop the accent.

## Layout & topology rules

These were codified the hard way after a visual audit found a dozen
"kludgy" connectors that the eye trips on. **Follow them.**

### 1. Arrows are straight

- **Horizontal:** `M{x1} {y} H{x2}` — only when both endpoints have
  the same `y`.
- **Vertical:** `M{x} {y1} V{y2}` — only when both endpoints have the
  same `x`.
- **Single diagonal:** `M{x1} {y1} L{x2} {y2}` — acceptable only for
  short fan-in/fan-out (e.g. 5 inputs converging into one node), where
  the diagonal angle communicates the topology.

### 2. No L-shaped paths, no bezier feedback loops

These patterns are **forbidden**:

```text
<!-- L-shape (horizontal then vertical) -->
M250 130 H180 V220

<!-- Bezier feedback / wrap-around -->
M510 130 Q540 130 540 220
M510 130 Q540 130 540 190 T510 250 H190
```

If you find yourself drawing one, the **layout is wrong**, not the
arrow. Restructure so the arrow can be a single H/V/diagonal segment.

### 3. Widen the target box for N→1 fan-in

When several upstream boxes feed into one downstream box, **widen the
downstream box to span all the upstream columns** so each arrow drops
straight down. Don't leave it narrow and bend the arrows in.

```text
✗ wrong                         ✓ right
[A]  [B]  [C]  [D]              [A]  [B]  [C]  [D]
   \  |  /  /                     |    |    |    |
    \ | / /                       v    v    v    v
     [out]                      [          out          ]
```

The N→1 result node is also a natural accent target.

### 4. Spread arrowheads at convergence

If 3+ arrows arrive at the same node, distribute their endpoints
across the node's edge — don't pile them on a single point.

```text
✗ all → (540, 220)              ✓ stacked
                                  (540, 200), (540, 210), (540, 220),
                                  (540, 230), (540, 240)
```

### 5. Don't draw paths through other shapes

A connector should not visually overlap an unrelated node, label, or
panel border. If the only way to route an arrow is through another
box, **the layout is wrong** — re-place the boxes.

### 6. Topology should be planar

Most module diagrams have ≤8 nodes. Aim for a layout where edges don't
cross at all. If they must cross, do so at 90° and at a single point
(not a tangle).

### 7. Outer panel encloses everything

The outer `Your X System` panel rect must contain every node. No
floating boxes outside the frame, no half-in-half-out.

### 8. Allowed exceptions

These three patterns may break some of the rules above when the
content genuinely calls for them:

- **Feedback / cycle indicator.** A short labeled path of the form
  `V H V` (down, across, up) under or above a row of boxes is allowed
  when it represents a loop iteration (e.g. "Next Batch" in a training
  loop). Keep the wrap-around band tight (≤40px from the box edge),
  label it in italic 9pt grey.
- **Panel-less computation example.** A diagram that shows *one*
  arithmetic operation (e.g. broadcasting `Matrix + Vector → Result`)
  may omit the outer panel and the "Your X System" eyebrow. The
  diagram is the equation, not a system overview.
- **Layered stack.** A vertical stack of equally-sized boxes
  (Code → Library → Backend → Hardware) may omit the outer panel —
  the stack itself is the visual structure. Each box should still
  follow the standard palette and stroke rules.

Use these exceptions sparingly. Don't invent new ones.

## Centering & sizing in the PDF

Already wired up in `pdf/_quarto.yml`:

- `fig-align: center` — every figure is `\centering`-ed.
- A custom `\renewcommand{\includegraphics}` clamps to
  `0.75\linewidth × 0.45\textheight`. Diagrams will never blow up the
  page no matter what your viewBox says.

You don't need to set `width=` on individual figures. Don't.

## Coordinate-system convention

Every SVG in this set wraps content in `<g transform="translate(0,-50)">`.
This is a leftover from an earlier generator that used a 50px top
margin. New SVGs may keep it (so coordinates stay consistent with the
existing set) or drop it (and shift all `y` values down by 50). Don't
introduce a *different* transform.

## How to add a new diagram

1. Create a new SVG at
   `../../assets/images/diagrams/NN_module-diag-K.svg`. Easiest
   starting point: copy an existing diagram of similar topology and
   edit it.
   - In Inkscape: `File → Document Properties → Custom size 680×{H}`
     (where `{H}` is the height you computed from the sizing recipe),
     `File → Save As → Plain SVG`.
   - In a text editor: copy a sibling, change the body.
2. Use the palette and conventions above. Don't introduce new colors.
3. Place nodes first; choose viewBox/panel height *afterwards* per
   the sizing recipe.
4. Audit visually before committing:

   ```sh
   rsvg-convert -w 1200 -b white \
     ../../assets/images/diagrams/NN_module-diag-K.svg \
     -o /tmp/preview.png
   open /tmp/preview.png   # or any image viewer
   ```

   Check against the rules in *Layout & topology* above.
5. Reference it from the appropriate qmd:

   ```qmd
   ::: {#fig-NN_module-diag-K fig-env="figure" fig-pos="htb" fig-cap="**Concise Title**: One-sentence explanation." fig-alt="Sentence describing the visual content for screen readers."}

   ![](../assets/images/diagrams/NN_module-diag-K.svg)

   :::
   ```

   Note: the path is `.svg`. The `svg-to-pdf.lua` filter rewrites it
   to `.pdf` for the PDF format only.
6. From `tinytorch/quarto/`: `make pdf`. Verify visually.

## How to edit an existing diagram

1. Open the `.svg` in your tool of choice (Inkscape and Affinity
   Designer both round-trip cleanly; a text editor is fine for
   coordinate or color tweaks).
2. Save. Don't run any generator — there isn't one anymore.
3. From `tinytorch/quarto/`: `make pdf`. The Makefile pattern
   rule will reconvert only the changed SVG to PDF and re-render.
4. Re-render to PNG (`rsvg-convert`) and eyeball it. The PDF builds
   from PNG-equivalent data — if it looks wrong as a PNG, it'll look
   wrong in the PDF.

## Files in this directory

| File             | Purpose                                                         |
|------------------|-----------------------------------------------------------------|
| `STYLE.md`       | This document.                                                  |
| `svg-to-pdf.lua` | Quarto Lua filter that rewrites `*.svg` → `*.pdf` for PDF only. |

That's it. No generator, no theme module, no Python.
