<!-- DEV-BANNER-START -->
<div align="center">
<table>
<tr><td>
<h3>🚧 Under Active Development</h3>
<p>This component is being built on the <code>dev</code> branch and is <b>not yet available</b> on the live site.<br>
Content may be incomplete or change without notice. The published curriculum lives at <a href="https://mlsysbook.ai"><b>mlsysbook.ai</b></a>.</p>
<p>
<a href="https://github.com/harvard-edge/cs249r_book/tree/dev"><img src="https://img.shields.io/badge/branch-dev-orange?logo=git&logoColor=white" alt="dev branch"></a>
<a href="https://mlsysbook.ai"><img src="https://img.shields.io/badge/live_site-mlsysbook.ai-blue?logo=safari&logoColor=white" alt="live site"></a>
</p>
</td></tr>
</table>
</div>
<!-- DEV-BANNER-END -->

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
