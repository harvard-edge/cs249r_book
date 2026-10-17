# Figure Explore

Opt-in Diagram / Explore views inside an existing numbered Quarto figure.
The static figure, caption, alt text, numbering, and cross-reference anchor
remain owned by the chapter. No post-render HTML patch is required.

## Series rollout

The four-volume registry is generated from reviewed `lessons/*.json` files.
Two treatments share the same bounded Diagram / Explore frame:

- Guided annotations preserve the original artwork and focus authored regions.
  They do not simulate execution or invent values. Dense diagrams have an
  Enlarge view with the same state, controls, and readable narration.
- Specialized lessons expose changing ownership, storage, partial results,
  scheduling, branches, or permission state. Their assumptions and accounting
  boundaries remain explicit.

Every canonical catalog entry has a reviewed treatment or an explicit static /
source-conflict reason. A source conflict is not an implemented enhancement.
The catalog reports the original proposal separately from the delivered form.

The reviewed rollout contains 686 enhancements: 213 in Volume I, 257 in
Volume II, 89 in Volume III, and 127 in Volume IV. Of these, 670 are guided
annotations and 16 are specialized stateful lessons. Seventeen cataloged
figures remain static by design; four remain unenrolled because their source
captions or labels conflict with the artwork. Coverage is not an animation
quota or a claim that every figure benefits from changing machine state.

The local validation includes a 686-player runtime sweep, a refined
four-profile projection/capture pass (2,744 screenshots, with recorded retries),
a separate 16-lesson runtime recheck, and actual-chapter visual review by the
owning authors. The 24 focused tests cover compiler enrollment, static format
behavior, deterministic rendering, and surrounding filter compatibility.
The catalog review record distinguishes these checks from semantic judgment
and records the remaining source conflicts.

The final book-code snapshot also passed all eight full-volume PDF/EPUB builds.
All PDF extracted text and page counts match the earlier static baseline;
six figure pages were visually inspected, including four pixel-identical memory
and state examples. EPUBs retain static figures without Explore controls,
runtime assets, or broken image references. The two intentionally seeded
Volume I SVGs match the reviewed HTML artwork exactly. The existing malformed
Volume I dependency-map target remains part of the four held source conflicts.
This is representative visual verification, not an audit of every PDF page.

Source and artwork freshness guards fail back to Diagram if either changes.
The source digest ignores enrollment attributes only. SVG digests ignore
metadata and generated identifier names, preserving geometry, text, and colors.
Raster digests cover all bytes. This detects accidental drift; it is not a
security boundary. Rebuild artwork, inspect mappings, and recompile after a
source edit rather than disabling the guards.

Quarto cross-reference tooltips show static-only figure previews. They must not
clone live controls or IDs. Whole-book figures keep their frame in the reading
column, even when the surrounding Quarto layout includes margin material.

### Review and rebuild

```sh
# First build the canonical artwork through the normal book pipeline.
./binder/binder build html --all --parallel 4 --skip-validate
# Index the successful build tree for visual mapping and freshness checks.
python3 binder/tools/scripts/build_figure_art_index.py --build-root books/_build/parallel/RUN --output /tmp/mlsys-figure-art
# Enroll only reviewed lessons; static and conflicted figures stay static.
python3 binder/tools/scripts/compile_figure_explore.py --require-complete --enroll
python3 binder/tools/scripts/compile_figure_explore.py --require-complete --check
# Rebuild to package the generated volume bundles and registered assets.
./binder/binder build html --all --parallel 4 --skip-validate
```

`--require-complete` checks catalog coverage and schema; it does not certify
blocked entries or replace visual and runtime review. The explicit compilation
report distinguishes ready, intentionally static, and blocked source conflicts.
Keep the artwork index and screenshots outside version control. The authored
manifests, generated registry, and volume JavaScript bundles are versioned.

Run the Quarto format and compiler tests together with
`python3 -m pytest binder/tests/test_figure_explore.py binder/tests/test_figure_explore_compiler.py binder/tests/test_diagram_determinism.py -q`.
The browser QA utility is `binder/tools/scripts/check_figure_explore_browser.cjs`;
its `--help` describes the build tree, Playwright module, and output options.
Its report separates automated checks from human visual and pedagogical review.

## Authoring

Enable the extension in the HTML filter list (the book does this centrally in
`config/shared/html/filters.yml`), then opt in on the existing figure wrapper:

```markdown
::: {#fig-tensor-memory-layout data-explore="tensor-memory-layout" fig-env="figure" fig-cap="...unchanged..." fig-alt="...unchanged..."}

...unchanged static diagram...

:::
```

Do not add another figure wrapper, caption, or number. Do not paste generated
HTML or inline the runtime into the chapter. Diagram is the initial view;
Explore initializes paused and preserves its step when views change.

The generated registry binds each lesson to its exact volume, source, figure ID,
and reviewed artwork. The two original tensor-layout and matrix-tiling lessons
also support explicit legacy registration. Unknown names or mismatched figure
IDs retain the static figure. Register each new lesson explicitly.

## Format behavior

- HTML: Quarto packages the controller and lesson JS/CSS into the rendered
  site's dependency directory. The browser initializes only opted-in figures.
- PDF / LaTeX: the HTML configuration does not load this filter; if explicitly
  loaded elsewhere, the `html:js` guard still leaves the static AST unchanged.
- EPUB: same static behavior. Quarto's `html` alias includes EPUB, so the filter
  intentionally uses `html:js`, not `html`.
- Browser print: static Diagram is visible even when Explore was selected.
- JavaScript disabled, missing lesson, or initialization error: static Diagram
  remains usable; no inactive controls are presented.

The filter handles both ordinary Div and Quarto FloatRefTarget nodes, because
Quarto parses numbered figure wrappers before running user filters. It changes
only an HTML enrollment attribute; figure content and caption AST are unchanged.

## Adding a lesson

1. Author its manifest in `lessons/`, then compile the exact figure binding.
   Specialized factories also declare their script and stylesheet there.
2. Supply a factory under `window.MLSysFigureLessons[lessonName]`. It receives
   an empty, uniquely identified, initially detached Explore panel. It builds
   scoped semantic controls/diagram and returns `{pause, destroy}`.
3. `pause()` must stop timers and interpolation without resetting the state.
   `destroy()` removes lesson listeners/observers and releases resources.
4. Do not initialize tabs or change the original figure/caption inside a lesson.
   The shared controller mounts transactionally and handles view switching.
5. A lesson must pause on document hiding and leaving the viewport; it must
   never autoplay on initialization, view activation, or theme changes.
6. Derive narration, highlights, equations, and counters from one state. Expose
   changing quantities in readable text, not only color or movement.
7. Scope classes and DOM queries. Inherit chapter typography and theme surfaces.
   Preserve the original composition and existing color roles.
8. Add only declared toy data or chapter-owned numerical exports. Real specs
   and calculations continue to use the book's MLSysIM/LEGO data sources.

The dependency manager emits shared assets once per page, and the browser
controller is idempotent. Additional registered lessons receive their own
named dependency, without duplicating shared code. Its DOMContentLoaded hook
runs after the deferred lesson factories have loaded.

## Verification

Run `python3 -m pytest binder/tests/test_figure_explore.py -q` from the repo root.
These tests render actual Quarto HTML, LaTeX, and EPUB, verify shipped assets,
and check unknown/mismatched enrollment fails closed. Full chapter builds and
browser checks cover the actual diagram, theme, keyboard tabs, no-JS/print
fallback, all tensor mappings, playback, and independent state.

Build the pilot normally:

```sh
./binder/binder build html 07_frameworks --vol1 --skip-validate
./binder/binder build pdf 07_frameworks --vol1 --skip-validate
./binder/binder build epub 07_frameworks --vol1 --skip-validate
```

Source API reference: [Quarto Lua API](https://quarto.org/docs/extensions/lua-api.html).

The tiling lesson uses a declared small worked case on the original L-shaped
composition. It distinguishes backing matrices, operand copies, retained partial
sums, and a completed output write. Its counters count scheduled scalar-element
copies and MACs, not measured DRAM traffic or elapsed time.
