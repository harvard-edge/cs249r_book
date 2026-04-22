# TinyTorch diagram style guide

Visual convention for the per-module "What You'll Build" diagrams in
`../../assets/images/diagrams/`. The SVGs are the source-of-truth
artifacts — edit them directly. There is no generator to round-trip
through.

---

## Design philosophy

Restrained. Greyscale base, single accent color used surgically.
References: Stripe Press, Hennessy & Patterson, Tufte. The principle:
*~85% of the ink is greyscale; color is reserved for the one element
the reader's eye should land on first.* Categorical color-coding
(blue=compute, green=data, …) requires the reader to internalize a
legend — we don't do it.

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
- **Sizes** (tuned for the LaTeX `0.75\linewidth` cap — a 680×360
  canvas renders at ~4.9in wide on the page):
  - Panel eyebrow ("Your X System"): 11pt bold
  - Node title: 10pt bold
  - Node subtitle: 9pt regular
- **Color**: titles `#1f2937` (dark slate, not pure black), subtitles
  `#6b7280` (slate). Same colors regardless of accent — the *box*
  carries the highlight, not the words inside it.

## Canvas

- **viewBox**: `0 0 680 360` (landscape, ~1.89:1).
- **No in-SVG title or footer text**. The Quarto `fig-cap=` on the qmd
  side does the labeling. Putting a title inside the figure AND letting
  Quarto add a caption beneath it duplicates the same words in two
  visual styles and wastes vertical space.
- **No outer page border**. The white background is the diagram's
  edge.

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
| Multiple equivalent alternatives  | The default / recommended one          |

The accent is editorial, not decorative. If you can't articulate *why*
a particular node is the accent, drop the accent.

## Centering & sizing in the PDF

Already wired up in `pdf/_quarto.yml`:

- `fig-align: center` — every figure is `\centering`-ed.
- A custom `\renewcommand{\includegraphics}` clamps to
  `0.75\linewidth × 0.45\textheight`. Diagrams will never blow up the
  page no matter what your viewBox says.

You don't need to set `width=` on individual figures. Don't.

## How to add a new diagram

1. Create a new SVG at
   `../../assets/images/diagrams/NN_module-diag-K.svg`. Easiest
   starting point: copy an existing diagram of similar topology and
   edit it.
   - In Inkscape: `File → Document Properties → Custom size 680×360`,
     `File → Save As → Plain SVG`.
   - In a text editor: copy a sibling, change the body.
2. Use the palette and conventions above. Don't introduce new colors.
3. Reference it from the appropriate qmd:
   ```qmd
   ::: {#fig-NN_module-diag-K fig-env="figure" fig-pos="htb" fig-cap="**Concise Title**: One-sentence explanation." fig-alt="Sentence describing the visual content for screen readers."}

   ![](../assets/images/diagrams/NN_module-diag-K.svg)

   :::
   ```
   Note: the path is `.svg`. The `svg-to-pdf.lua` filter rewrites it
   to `.pdf` for the PDF format only.
4. From `tinytorch/site-quarto/`: `make pdf`. Verify visually.

## How to edit an existing diagram

1. Open the `.svg` in your tool of choice (Inkscape and Affinity
   Designer both round-trip cleanly; a text editor is fine for
   coordinate or color tweaks).
2. Save. Don't run any generator — there isn't one anymore.
3. From `tinytorch/site-quarto/`: `make pdf`. The Makefile pattern
   rule will reconvert only the changed SVG to PDF and re-render.

## Files in this directory

| File             | Purpose                                                         |
|------------------|-----------------------------------------------------------------|
| `STYLE.md`       | This document.                                                  |
| `svg-to-pdf.lua` | Quarto Lua filter that rewrites `*.svg` → `*.pdf` for PDF only. |

That's it. No generator, no theme module, no Python.
