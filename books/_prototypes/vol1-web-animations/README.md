# Volume 1 web animation prototype

## Current status: memory figure integrated in Quarto source

The tensor-memory-layout figure now uses the shared `figure-explore` extension
in its normal chapter source. Build it with binder; do not run the old memory
post-render injector. The convolution and other standalone examples remain
review prototypes. See the production
[authoring guide](../../_extensions/mlsysbook-ext/figure-explore/README.md).

```sh
./binder/binder build html 07_frameworks --vol1 --skip-validate
```

The extension ships its own shared dependencies and preserves static PDF/EPUB
figures. Baseline comparisons verified unchanged PDF text, a pixel-identical
PDF memory-figure page, and unchanged canonical EPUB content/static assets
(ignoring execution-cell IDs and the inert opt-in attribute).

A chapter-context ETL/ELT animation and a passage-level opportunity map for Volume 1. These are proposals for web-only figures, not integrated textbook changes.

## Preview

From the repository root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory books/_prototypes/vol1-web-animations
```

Open http://127.0.0.1:8765/etl-chapter-preview.html. Alternatively, open the HTML file directly; it has no external dependencies. Choose another port if 8765 is already occupied.

The figure plays once when it enters view. Pause and replay are available. Reduced-motion preferences switch playback to manual steps.

## Contents

- `etl-chapter-preview.html`: the existing ETL/ELT prose surrounding a five-stage animated record example.
- `opportunities.txt`: candidate insertion passages, storyboards, accuracy constraints, and priorities across all 16 chapters. Source paths are relative to the repository; line numbers reflect the source inspected on October 14, 2026.

The original Quarto chapters, static figures, and PDF build are unchanged. This directory is a standalone prototype and is not wired into the website navigation or chapter rendering.

## Validation and follow-up

The preview was visually inspected at desktop width and in a 320-pixel viewport. The illustration uses a hypothetical event, not measured performance data. Both ETL and ELT retain upstream raw logs in the example; ELT additionally retains raw input in the warehouse.

Before integrating, convert the prototype into a reusable web-only figure component, preserve a static print alternative, and check keyboard behavior, screen-reader announcements, reduced-motion playback, and offscreen pause behavior. The current chapter excerpts are snapshots, not synchronized includes.

## Additional previews (October 16)

Open `living-figures.html` for three narrated, five-stage concept previews:
kernel fusion, activation checkpointing, and affine quantization. They start
paused and offer manual steps, playback, restart, and an expanded overlay.
The earlier ETL example is linked from the gallery.

`series-animation-plan.txt` describes the teaching contract, Quarto integration,
and candidate mechanisms across the four volumes. Volume II–IV candidates are
a preliminary chapter-level proposal, not a full manuscript audit.

The gallery was rendered and visually inspected at desktop and narrow widths;
control checks covered all three end states, restart, expansion/Escape, and
reduced-motion stepping. No production chapter or website configuration changed.
The overlay needs production dialog focus management; theme inheritance, full
screen-reader testing, and real Quarto embedding remain integration work.

## Small chapter-context revision

`convolution-in-context.html` is a single figure placed between the existing
convolution setup and a proposed explanatory closing paragraph. It retains
the 7×7 input / 3×3 filter / 5×5 output geometry from `fig-cnn`.

The six-item sequence menu, Play/Pause, Back, Next, and Restart support both
conceptual stages and individual spatial steps through the final scan. The
example uses declared toy integers, a fixed filter, and computed outputs.

Three independent auditors reviewed pedagogy, numerical/source accuracy, and
UX. Findings incorporated: a playable final scan, per-cell stepping, an
accessible completed-output description, explicit contrast interpretation,
and row labels for the arithmetic. Desktop/mobile renders were visually
inspected. Browser checks covered all menu choices, complete playback, pause,
restart, correct output values, reduced motion, and overflow at 320–1100 px.
This remains a local reading prototype; the production chapter is untouched.

## Feedback iteration and two additional reading examples

The convolution revision pairs each input value with its matching filter
weight, highlights the corresponding product, and builds a running sum.
Manual stepping covers all nine pairs. Controls are above the diagram and
stay visible while scrolling; a mobile cue keeps the current operation beside
them. The grids are smaller and all three pages link to one another.

- `fusion-in-context.html`: two elementwise operations, following the same
  representative element through separate/fused paths; idealized HBM transfer
  counts distinguish intermediate writes/reads from arithmetic.
- `quantization-in-context.html`: a declared 3-bit example separates real input,
  code coordinate, stored integer, reconstruction, rounding, and clipping.
- `reading-figures.css` and `reading-player.js`: shared presentation/playback
  for these two new examples; no network dependencies.

Reviewers checked numerical/source accuracy, visual teaching, and interaction.
Feedback incorporated matched pairs, accessible controls above the diagram,
correct axis labels, and both quantization cases in the final comparison.
Browser verification covered complete playback, pause/replay, every menu item,
numerical endpoints, reduced-motion rendering, and widths320–1100 px. Desktop
and mobile screenshots were visually inspected. No production chapter, theme,
or public website was changed.

## Actual website chapter preview

The animation can now be embedded directly into the original chapter build,
with its real Quarto theme, sidebar, TOC, typography, prose, and caption.
From the repository root:

```sh
./binder/binder build html 06_nn_architectures --vol1 --skip-validate
python3 books/_prototypes/vol1-web-animations/embed-in-chapter.py
python3 -m http.server 8766 --bind 127.0.0.1 --directory books/_build/html-vol1/chapters/06_nn_architectures
```

Open `http://127.0.0.1:8766/vol1/06_nn_architectures/06_nn_architectures.html#fig-cnn`.
The script only modifies generated HTML, retaining a static build copy next
to it. Its CSS/JS is component-scoped; light/dark surfaces inherit the book's
theme. The original `fig-cnn` anchor and caption remain. The static image
remains the no-JavaScript and print fallback.

Verified complete playback, numerical outputs, step menu, original navigation,
print and no-JavaScript fallback, and mobile overflow. Light/dark/mobile
screenshots were inspected and the integrated views audited. This is a
selective local chapter build; other chapter pages are not built by it.
Production source-level embedding remains a separate integration step.

## Diagram / Explore standard and memory-layout pilot

Both actual-chapter previews now default to the original Diagram and offer an
Explore tab under the same figure anchor/caption. Shared `figure-views.js` and
`figure-views.css` provide progressive enhancement, keyboard tabs, selected
state, per-figure events, and print fallback. Switching to Diagram pauses the
lesson; returning to Explore preserves its step without autoplay.

Build and prepare both from the repository root (one build at a time):

```sh
./binder/binder build html 06_nn_architectures --vol1 --skip-validate
python3 books/_prototypes/vol1-web-animations/embed-in-chapter.py
./binder/binder build html 07_frameworks --vol1 --skip-validate
python3 -m http.server 8767 --bind 127.0.0.1 --directory books/_build/html-vol1/chapters
```

- Convolution: `/06_nn_architectures/vol1/06_nn_architectures/06_nn_architectures.html#fig-cnn`
- Tensor memory: `/07_frameworks/vol1/07_frameworks/07_frameworks.html#fig-tensor-memory-layout`

Memory Explore lets readers select any tensor cell, follow its element offset,
and compare row-major/column-major strides. It explicitly distinguishes
element offsets from byte offsets rather than assuming an element width.

This pilot modifies generated local HTML only. Production source-level Quarto
shortcode/filter packaging and full PDF/EPUB builds remain future integration.

## All-volume figure enhancement catalog

`animation-catalog/index.html` is a searchable proposal for every numbered body
figure found in the four canonical volume configurations (including appendices
and generated-index sources). `catalog.json` contains the same proposals,
source IDs/locations, captions, three-stage sequences, appearance constraints,
and systems learning goals. The catalog distinguishes animation, interaction,
annotation, and static treatments; 16 entries form a systems-first shortlist.

Serve from the repository's `books` directory to preserve theme/pilot links:

```sh
python3 -m http.server 8768 --bind 127.0.0.1 --directory books
```

Open `http://127.0.0.1:8768/_prototypes/vol1-web-animations/animation-catalog/index.html`.
To refresh metadata and rebuild after reviewing source changes, run
`animation-catalog/extract-inventory.py`, update the matching proposal files,
then `animation-catalog/build-catalog.py` (using their full paths from repo root).
The builder checks exact key coverage and required proposal fields.
`review-notes.txt` records review scope, corrections, and implementation gates.
These are planning proposals based on captions/context, not 707 implemented
or visually approved animations. The two existing chapter pilots remain local.
