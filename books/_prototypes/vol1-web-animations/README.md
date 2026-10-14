# Volume 1 web animation prototype

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
