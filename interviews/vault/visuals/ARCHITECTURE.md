# Visual question architecture

**Status:** active (proposed 2026-04-25, supersedes the SVG-only path described in `AUTHORING.md` v1)

This document is the design contract for how StaffML attaches diagrams to
questions. It supersedes the implicit "all visuals are hand-authored SVG"
approach that shipped with `cloud-visual-001`.

---

## The core idea — three layers, decoupled

| Layer | Concern | Owns |
|---|---|---|
| **Website (practice page)** | Render the static SVG as `<img>` and surface its alt + caption | `visual.kind`, `visual.path`, `visual.alt`, `visual.caption` |
| **Build pipeline** | Compile DOT or matplotlib source into the same SVG asset the website ships | `visual.source_format` + naming convention |
| **Authoring** | Decide whether a diagram earns its place; pick the format whose layout engine fits | `AUTHORING.md` rules |

The website **only** reads the static SVG asset. It does not know — and
should not know — whether the SVG was hand-drawn, compiled from DOT, or
rendered by a Python script. Build provenance is metadata for
maintainers and tooling, not a runtime concern.

---

## YAML schema (the contract)

```yaml
visual:
  kind: svg                              # what the website ships — always svg
  path: <id>.svg                         # the static SVG asset (the only thing the practice page loads)
  alt: <text, ≤250 chars>                # screen-reader description, no interpretation
  caption: <text, optional>              # small caption shown below the diagram
  source_format: dot | matplotlib | hand # OPTIONAL build metadata — default 'hand'
```

### Why `kind` is always `svg`

`kind` was originally framed as the *authoring* format (svg / dot /
matplotlib). That was a mistake: the website only ever renders SVG, and
exposing the authoring format in `kind:` confused the website schema
with the build-tool schema. The fix is to fix `kind:` at `svg` (the
output format) and add `source_format:` (the input format) as separate,
optional build metadata.

### File layout — naming convention does the work

For each visual, two files may coexist:

```
interviews/vault/visuals/<track>/<id>.svg     # the asset the website ships (always present)
interviews/vault/visuals/<track>/<id>.dot     # iff source_format=dot     — build input
interviews/vault/visuals/<track>/<id>.py      # iff source_format=matplotlib — build input
```

The renderer doesn't need a separate `source:` field in the YAML — it
infers the source filename from the SVG basename and the
`source_format` hint. If `source_format` is `hand` (or absent), no
build step runs; the SVG is the source.

---

## Three supported source formats

We pick the format whose layout engine fits the content. **Don't
hand-author SVG when an auto-layout tool will do it for you.**

| `source_format` | Use for | Layout effort | Tool |
|---|---|---|---|
| `dot` | Topology, graphs, dataflow, network fabrics | Auto | `dot -Tsvg` (graphviz ≥ 2.40) |
| `matplotlib` | Curves, plots, Gantt charts (`barh`), heatmaps | Programmatic | `python3 <script>` then `savefig` |
| `hand` | Custom layouts that don't fit either above (memory-page diagrams, mixed annotations) | Manual | text editor + book SVG style guide |

We considered Mermaid for sequence/Gantt diagrams. Decision: matplotlib's
`barh` plus annotation API covers the same ground without adding a
Node-based dependency. If a future archetype genuinely needs Mermaid
(e.g., declarative state machines), we revisit.

### When to pick which

| Visual archetype | source_format | Why |
|---|---|---|
| Ring AllReduce / Tree AllReduce | `dot` | Nodes + directed edges, layout-engine-perfect |
| Pipeline parallelism / network fabric topology | `dot` | Same |
| Pipeline bubble Gantt / prefill-decode interleave | `matplotlib` (`barh`) | Time on x-axis, lanes on y-axis — programmatic |
| Roofline plot / queueing hockey-stick / scaling curve | `matplotlib` | Continuous functions plus annotations |
| KV-cache page layout / memory hierarchy data path | `hand` | Custom spatial composition |
| Duty-cycle timeline | `matplotlib` | Programmatic time-series |
| Checkpoint/recovery RPO/RTO | `matplotlib` (`barh`) | Same Gantt-like layout |

Default: try DOT first for graph-shaped content; try matplotlib first
for time/quantity-shaped content; fall back to `hand` only when
neither fits.

---

## Build pipeline

`interviews/vault/scripts/render_visuals.py` is the single entry point.
It scans every question YAML's `visual:` block and dispatches by
`source_format`:

| `source_format` | Pipeline |
|---|---|
| `hand` (or absent) | No-op. The SVG IS the source. Just confirm it exists. |
| `dot` | `dot -Tsvg <id>.dot -o <id>.svg` |
| `matplotlib` | `python3 <id>.py` (the script reads `os.environ["VISUAL_OUT_PATH"]` and `savefig` to it) |

After rendering, every output SVG passes a normalization step:

1. Strip XML comments and metadata that vary by renderer version.
2. Set `<svg ... font-family="Helvetica Neue, Helvetica, Arial, sans-serif">`
   to match book style.
3. Confirm dimensions are reasonable (`viewBox` width 600–800).
4. Confirm no embedded raster (`<image href="data:...">`).

Normalization keeps DOT-rendered, matplotlib-rendered, and hand-authored
SVGs visually consistent in the practice-page rendering.

---

## Generation: Gemini in the loop

The corpus has historically been Claude-heavy. To reduce single-model
bias and to scale visual coverage, we add a Gemini-driven generation
pipeline mirroring the existing `gemini_cli_math_review.py`:

`interviews/vault/scripts/gemini_cli_generate_questions.py`:

1. **Targets** weak coverage cells from `portfolio_balance_loop.py`
   output (which track × topic × zone × level slot is under-populated).
2. **Generates** candidate question YAMLs, including a `visual:` block
   when the topic appears in the visual-archetype catalog.
3. **For visual-eligible items**, also generates the source artifact
   (DOT text, or a small matplotlib script) and writes it next to the
   YAML's expected SVG asset path. `render_visuals.py` then compiles it.
4. **Outputs as drafts** (`status: draft`, `provenance:
   gemini-3.1-pro-preview`). Never auto-publishes.

### Cross-model validation

Gemini-generated drafts are reviewed by `gemini_cli_math_review.py`,
which can also be run with a Claude model to provide cross-model
agreement. The default flow:

- Generation: Gemini 3.1 Pro Preview drafts the question.
- Math review pass: Gemini 3.1 Pro re-checks arithmetic, units, and
  hardware specs against the same constants reference used during
  generation. (A Claude-pinned re-run is the natural follow-up for
  release-grade verification.)
- Visual review: a separate visual-fidelity check confirms the rendered
  SVG matches the scenario's claimed quantities (e.g., 4 ranks in the
  ring → 4 nodes in the DOT graph).

A draft promotes only if the math pass returns CORRECT and the visual
check passes. Disagreements escalate to maintainer review.

---

## Coverage goal

Not "N visual questions". Instead:

- Cover every topic in the visual-archetype catalog with at least one
  exemplar (10 archetypes today; the catalog can grow).
- For each track with applicable archetypes, ship at least 2 visual
  exemplars so the "Visual questions only" filter has substantive
  content.
- Cap visual enrichment at the level where it stops earning its place
  (per AUTHORING.md three-condition test).

We expect 30–80 visual-enriched published questions across the corpus
once Gemini-driven generation runs against the catalog. This is a
ceiling, not a target — the test is always "does this diagram earn its
place".

---

## Migration path

This document supersedes the AUTHORING.md assumption that all visuals
are hand-authored SVG. AUTHORING.md remains the authority on *when* a
visual earns its place; this document is the authority on *how* the
visual is encoded and rendered.

Existing `cloud-visual-001.yaml` continues to work unchanged — its
`kind: svg, path: cloud-visual-001.svg` schema is the new default, and
`source_format` is omitted (treated as `hand`).

---

## Open questions

- Should the rendered SVG be normalized so DOT + matplotlib + hand SVG
  all use the same color palette? (Probably yes — palette consistency
  matters more than rendering-tool fidelity.)
- Should we allow a question to attach *multiple* visuals (e.g., a
  before/after comparison)? Schema supports an array easily; UI
  doesn't yet.
- For Gemini-generated visuals, do we need a "diagram review" model
  pass beyond the math review? A multimodal LLM could verify a
  rendered SVG matches the YAML's claims. (Worth prototyping; not yet
  built.)
