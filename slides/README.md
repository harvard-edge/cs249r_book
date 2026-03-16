# ML Systems Lecture Slides

Beamer slide decks for the ML Systems textbook lecture series.

## Quick Start

```bash
cd slides/vol1/01_introduction
xelatex 01_introduction.tex   # Compile (run twice for navigation)
```

## Structure

```
slides/
├── README.md                   # This file
├── assets/
│   ├── beamerthememlsys.sty    # Shared Beamer theme (DO NOT MODIFY per-chapter)
│   ├── beamerthememlsys-midnight.sty  # Dark mode variant
│   ├── beamerthememlsys-minimal.sty   # Ultra-clean variant
│   ├── beamerthememlsys-nature.sty    # Earth tones variant
│   └── img/
│       ├── logo-mlsysbook.png  # Shared logos
│       └── logo-harvard.png
├── _build/
│   ├── vol1/                   # Compiled PDFs
│   └── theme-previews/         # Theme comparison PDFs
├── vol1/                       # Volume I decks (16 chapters)
│   ├── 01_introduction/
│   │   ├── 01_introduction.tex
│   │   └── images/             # All images for this deck
│   └── ...
└── vol2/                       # Volume II decks (future)
```

**Key rule**: Each chapter is **self-contained**. All images live in `<chapter>/images/`.
The only shared assets are the Beamer theme and logos. `\graphicspath` should be `{images/}` only.

## Theme Variants

Swap themes by changing one line in the `.tex` file:

| Theme | File | Character |
|:------|:-----|:----------|
| **Crimson** (default) | `beamerthememlsys.sty` | Harvard crimson, white bg, branded footer |
| **Midnight** | `beamerthememlsys-midnight.sty` | Dark navy bg, cyan accents |
| **Minimal** | `beamerthememlsys-minimal.sty` | No header, steel blue, maximum content space |
| **Nature** | `beamerthememlsys-nature.sty` | Forest green accents, warm earth tones |

All themes share the same API (`\mlsystitle`, `\mlsysfocus`, `\mlsyscard`, etc.).

## Visual Assets

Each chapter's `images/` folder contains all visual assets for that deck:
- **SVG-derived PDFs** — custom diagrams following the book's SVG style guide
- **Photos** — real-world hardware, data centers, devices (CC-licensed)
- **Book figures** — copied into the chapter folder (self-contained)

## Contributing

Agent instructions and production guidelines live in `.claude/docs/slides/`.

## License

These slides accompany the ML Systems textbook and follow the same license terms.
