# Figure Explore

Opt-in Diagram / Explore views inside an existing numbered Quarto figure.
The static figure, caption, alt text, numbering, and cross-reference anchor
remain owned by the chapter. No post-render HTML patch is required.

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

The extension currently registers `tensor-memory-layout` for
`fig-tensor-memory-layout` and `matrix-tiling` for `fig-tiling-diagram`. Unknown names or mismatched figure IDs emit a
warning and retain the static figure. Register each new lesson explicitly
rather than attaching an unrelated laboratory to a figure.

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

1. Register its exact figure ID, script, and stylesheet in `figure-explore.lua`.
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
