# MLSys·im Paper Figures

This folder contains the vector graphics (`.svg`) used in the academic manuscript (`paper.tex`).

## Visual Language and Color Palette
To ensure consistency across the paper and the documentation site, all figures use the following semantic color scheme:

- **Cyan (`#e0f2fe` background, `#0284c7` stroke):** Represents **Solvers (Layer E)** or analytical math in action.
- **Orange (`#fff7ed` background, `#d97706` stroke):** Represents **Demand / Workloads (Layer A)** (e.g., parameter counts, FLOPs).
- **Purple (`#f5f3ff` or `#ddd6fe` background, `#7c3aed` stroke):** Represents **Supply / Hardware (Layers B-D)** (e.g., silicon, networks).
- **Green (`#ecfdf5` background, `#10b981` stroke):** Represents **Outcomes** (e.g., predicted latency, bottleneck identification, cost).
- **Light Gray (`#f8fafc` background, `#cbd5e1` stroke):** Represents external inputs or documentation (e.g., datasheets, literature).
- **Crimson (`#a31f34`):** Represents binding constraints or critical alerts (e.g., hitting a "Wall").

## Format Guidelines
- **Sharp Corners Only:** All nodes in Mermaid diagrams must use sharp corners (`L` path commands, not `C` cubic beziers, and no `rx` radii) to maintain an academic, block-diagram aesthetic.
- **PDF Compilation:** The `Makefile` in the parent `paper/` directory automatically runs `rsvg-convert` to transform these `.svg` files into `.pdf` files prior to LaTeX compilation. **Do not commit `.pdf` artifacts**; they are generated on the fly.
