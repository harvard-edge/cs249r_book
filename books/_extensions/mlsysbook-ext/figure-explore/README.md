# Figure Explore

Opt-in Diagram / Explore views inside an existing numbered Quarto figure.
The static figure, caption, alt text, numbering, and cross-reference anchor
remain owned by the chapter. No post-render HTML patch is required.

## Current original-art implementation

The accepted convolution prototype establishes the reference presentation:
animate the recognizable source diagram, put values in its objects, explain
one causal action per step, and pause after a short guided example. Numerical
details support the drawing rather than replace it. Static Diagram remains the
default and the complete print/no-JavaScript fallback.

`mechanisms/implementations.json` selects the original-art factories for
computation, memory, communication, coordination, and physical action across
the four volumes. The current manifest contains 100 original-art implementations:
86 newly authored modules and 14 retained implementations. The other 637
cataloged figures remain static. There is no animation quota. The convolution
prototype retains its separate accepted-prototype label; the remaining reviewed
implementations use hash-bound records in the original-art catalog.
Enrollment alone is not acceptance. Source, artwork, module, shared-runtime,
and timing changes invalidate the relevant current review evidence.
Historical tests below apply only to their recorded snapshots.

The selected set includes two native original-art factories: asynchronous
host–accelerator timing and the two-link arm’s singularity comparison. These
factories use their reviewed controller and calculation exports directly;
unused generic mechanism scenes are not required. Native registration validates
source, artwork, and runtime freshness. Independent calculation and rendered
production reviews remain separate requirements. The arm’s generated data is
packaged as its existing supported data bundle; shared browser code is unchanged.

`original-player.js` provides shared stepping, playback, visible keyboard hints,
and expanded inspection for the new factories. Convolution retains its reviewed
guided-scan controller. Both read build-time timing from the single
`books/config/shared/html/figure-explore.yml`, imported by all HTML volumes.
`movement-ms` controls illustrative movement; `reading-pause-ms` controls the
pause between actions during Play. Neither represents machine execution time.
Reduced-motion navigation commits the next state immediately, without an
intermediate animation clock that a visibility pause could interrupt.
Invalid or missing settings leave the static figure available.

Readers can temporarily choose **0.5×, 1×, 1.5×, or 2×** in the visible Speed
control, or use **− / +** while the figure has focus. The initial rate is 1×;
the YAML durations remain the defaults. This per-figure override scales movement
and automatic reading pauses, preserves position when changed, and does not
resume a paused action. Restart preserves the chosen rate; reloading starts at
1×. N/P still advances or reverses one step. Speed never changes the modeled
quantities or the ordered endpoints.

**Expand** opens the same live component in a full-window dialog. Opening and
closing pause playback, retain the current step and speed, and preserve the
explanation and controls. Close remains visible while scrolling; Escape returns
to the chapter and restores focus. Print and EPUB continue to use Diagram.

The shared layout keeps the sequence menu and current explanation above the
playback controls, with the controls and visible N/P/K/R hints beside the
changing diagram. Only the compact toolbar and step cue remain sticky. The
native **Example and assumptions** disclosure retains the complete case
description without pushing the diagram below the controls. Keep essential
case identity visible in the diagram or its live readout; for example, a
single-query worked case must not appear to execute a complete matrix.
Module-specific case selectors belong before the shared toolbar. Diagram-view
selectors may remain with the artwork.

While an action is moving or paused, stored values and completed results must
still describe the last committed state. Instructions may describe the next
endpoint only when they clearly identify it as pending. Keep read-only operand
selection distinct from a completed accumulation, allocation, or publication.

The compiler binds each replacement to the existing source and artwork hashes,
its runtime assets, data bundle where present, and the shared player. The browser
passes the verified original SVG to the factory and isolates inline SVG IDs so
multiple figures cannot reuse each other's glyphs or paint servers.

### Full-chapter author fixtures (QA only)

Build a candidate inside its original chapter without enrolling it or copying
an entire rendered site:

```sh
python3 binder/tools/scripts/build_original_figure_fixture.py \
  --entry books/_prototypes/vol1-web-animations/original-v1-kv-cache-growth/entry.json
```

The JSON result contains the fixture path and URL. Serve `books` on port 8882,
or pass `--base-url` for the existing local books server. `--art-index` selects
an artwork index (default `/tmp/figures-animation-art-current/index.json`).
`--output` selects an ignored directory under `books/_build`; the default is
`books/_build/original-rollout-fixtures/<entry-directory>/index.html`.

The helper preserves the full chapter and its static figure, disables all old
Explore enrollments, and mounts only the candidate factory. Existing relative
styles and images resolve through one ignored source-site symlink shared by
all fixtures from that site. Current shared-player and factory assets load from
the extension. The production `bundle` loads first. Additional data modules may be declared as `data_script` or
`data_scripts` in the entry's `implementation`, or supplied with repeatable
`--data-script FILE.js`; they load before the factory. Repeated filenames are loaded once, in first-declared order.

Explore opens paused. `window.fixturePlayer` and `window.fixtureRoot` expose
the candidate controller and root for local checks; `window.fixtureError`
records initialization failure. Diagram, print, and failed initialization retain
the original static figure and caption. The fixture uses the loader's current
SVG fragment-isolation algorithm and the same PNG wrapper coordinates and
metadata, but deliberately bypasses production freshness/enrollment checks.
It is author QA, not an integrated build or review acceptance. Production
validation must still use a freshly compiled chapter with the real loader.

### Verified PNG artwork for original-art factories

`context.verifiedArtwork` is also available for registered PNG artwork. The
controller first verifies the exact fetched bytes against the registered SHA-1,
checks the PNG header, and decodes those same bytes. It returns an SVG wrapper
with `viewBox="0 0 W H"` and matching `width` and `height` in intrinsic PNG pixels.
The wrapper has `data-mlsys-artwork-kind="png"`,
`data-mlsys-artwork-width="W"`, and `data-mlsys-artwork-height="H"`. Its sole initial
child is an SVG `image` at `(0, 0)`, sized `W × H`, whose `href` embeds the verified
bytes as a PNG data URL. It performs no second artwork network request.

A factory may clone this SVG and add authored overlays in its original pixel
coordinates. The raster itself contains no SVG objects to select or move;
inspect the original PNG before defining any regions or alternative interactive
layout. Keep the static diagram unchanged. Do not replace the source drawing
with a generic mechanism diagram. SVG artwork continues to arrive as its
original parsed SVG with fragment IDs namespaced before the factory runs.

If optional `data-mlsys-art-width` or `data-mlsys-art-height` attributes are supplied,
they must match the PNG header and decoded dimensions. Hash, fetch, dimension,
decode, or source-URL drift failures prevent the factory from running and leave
the original static figure and caption intact. No controller or player API
changes are required; factories that do not use `verifiedArtwork` are unaffected.

Before extending the manifest, independently review the source artwork and
surrounding section, compute added values in a reproducible calculation, test
the mechanism's invariants, and inspect actual chapter renders on desktop and
phone in both themes. An independent reviewer checks technical meaning and
learner readability; this does not constitute student testing. A source-art
mapping must preserve the illustrated workload and spatial relationships.
Alternative toy cases must remain explicit supporting examples.

The complete four-volume HTML baseline is
`books/_build/parallel/20261020-161134-58948`; all four genuine full-volume jobs
passed. Final Vol1 chapter delivery uses the genuine follow-up run
`books/_build/parallel/20261020-164427-53372` after the convolution playback-label and im2col phone-layout corrections.
Vol2–4 retain their unchanged baseline outputs. The actual review routes name
these runs explicitly; byte-identical copies are preserved in the permanent
books build directory. All 100 selected figures are enrolled exactly
once in their owning chapters, with matching packaged controller, data, source,
and artwork hashes. The fresh canonical artwork index covers all 737 figures;
the other 637 figures retain their static presentation.

The final volume audits read the selected source sections and captions against
their drawings and controllers. They check the illustrated workload, causal
order, ownership and lifetime rules, and the distinction between pending work
and a committed result. Bounded narration fixes keep in-transit operations
separate from stored values. Reproducible mechanism diagnostics check the
reported finite examples and invariants; unchanged mathematical models retain
their earlier numerical scope rather than receiving a new blanket arithmetic
approval. Earlier focused quantization refinements are included in this full
build.

The 86 newer modules have current technical and learner-view role records.
Each records the relevant aspect of the final volume agent audit, the current
artifact fingerprint, finite rendered coverage, and preserved prior review
history. Volume agents review both technical and learner-view aspects; the
prior independent role reviews remain a separate part of the evidence. The
original-art catalog binds all 100 reviewed
implementations to source, artwork, runtime, and actual owning chapter HTML;
stale records return to awaiting review.

Fresh controls checks cover every implementation in Chromium and WebKit, on
desktop and phone, in both themes: 800 passing profiles against these final per-volume outputs.
They verify speed choices, paused diagram identity, scoped shortcuts, restart,
full-window bounds, sticky controls, and Escape focus restoration. No UI
profile is inherited from an earlier build. Mechanism-specific endpoint and
progressed-pause captures complement this controls sweep; each volume report
states its finite cases and limitations. These checks do not prove continuous
correctness at every instant or replace numerical diagnostics.

The shared player and timing contract are unchanged in this pass. The current
full build includes the earlier convolution expansion restoration, FSDP resize
scheduling, and paused HTML-value geometry corrections for tensor layout,
block-sparse GEMM, and im2col.

A separate WebKit phone orientation/repeated-expansion stress trace can still
emit a ResizeObserver delivery warning. The warning also occurs with the prior
tensor and im2col controllers; equivalent prior-controller reproduction was not
established for block-sparse GEMM. This stress limitation remains recorded.
The ordinary packaged controls profiles pass without page errors.

Chromium and WebKit reports record endpoint traversal, local SVG references,
keyboard navigation, and static print fallback in desktop/narrow light/dark
profiles. These finite browser checks do not establish numerical validity or
continuous correctness at every instant. Technical role reviews, reproducible
calculations, and mechanism-specific invariants remain separate evidence.

Static output verification covers 14 representative figures across five
chapters in PDF and EPUB. It is distinct from the complete four-volume HTML
build and does not assert a fresh four-volume PDF/EPUB review. Existing
cross-reference and MathML conversion warnings remain in the build logs;
successful outputs do not establish full-book publication readiness.

The evidence is independent agent and browser review, not actual student
participation. No human students were recruited and no learning gains were
measured. Each record retains its reviewed scope and limitations.

Recheck the shared controller and packaged chapter endpoints with:

```sh
python3 binder/tools/scripts/check_original_player.py
python3 binder/tools/scripts/check_original_player_speed.py
python3 binder/tools/scripts/check_original_figures.py --build-root books/_build/parallel/RUN
python3 binder/tools/scripts/check_original_convolution.py --url CHAPTER_URL --engine webkit
```

The endpoint checker opens the targeted Explore view without autoplay, visits
every state, checks local SVG references, keyboard navigation, narrow layouts,
and browser-print fallback. Figure-specific checks also verify numerical,
ownership, and evidence invariants. These checks cover the rebuilt chapters;
they do not assert a new full-volume PDF or EPUB build.

The memory lessons distinguish mappings from payload copies and allocation
from initialization. KV paging preserves other requests while E resolves its
scattered blocks and releases them after its last consumer. Copy-on-write
reserves a private tail, copies initialized entries, changes only B's table,
then writes the new position. FSDP adds a fixed current-residency view beside
the original phase history, keeps saved activations through parameter release,
and separates backward-state release from gradient reduction. Capture
buffers compare alternative storage paths and preserve converted pages B when
capture pages A become reusable. Future source panels and phase columns are
reference views, not additional simultaneous allocations.

Activation checkpointing shows forward replay from retained checkpoints and
releases each reconstructed activation after its backward consumer finishes.
Checkpoint staging distinguishes an immutable captured generation from live
training state and publishes the restore manifest only after durable writes
and verification. Prefix invalidation preserves token identity while showing
that the same trailing token needs new derived KV state after an earlier
change. Intersample overshoot integrates the original acceleration profile
along the original velocity curves; its normalized example changes the
endpoint reserve rather than the acceleration waveform.

## Selection: changing state must teach something

### Bring the original diagram to life

The primary web experience follows the original figure's objects and connections:
values travel, representations transform, storage persists, and ownership changes.
Preserve the technical relationships and semantic colors, while allowing a new
layout, focused scenes, or a different visual analogy when it teaches better.
The printed layout is not a constraint on the web illustration. A moving value must have a
recognizable source, destination, and consequence; a traveling dot alone does not
explain a mechanism. Show copies as copies, reads as reads, and release only after
the last consumer. Logical dataflow must not imply measured memory traffic.

Keep one short explanation beside the action. Make numerical derivations an
optional inspection layer attached to the current operation. A separate arithmetic
worksheet is useful supplementary material, but it does not substitute for an
animated version of the source diagram. The Diagram view remains complete and
unchanged for print. The older diagram-life review describes a historical
16-lesson snapshot and its proposed revisions. It is planning history, not the
current implementation roster or current production acceptance evidence.

Keep a figure static unless interaction makes its mechanism easier to understand.
An Explore view earns its place when a reader can follow an actual change in
values, location, ownership, lifetime, dependencies, resource use, or permission,
or vary a declared parameter and observe a meaningful consequence. The caption
and surrounding prose remain sufficient for the static textbook.

Do not add a player merely to highlight parts of an unchanged diagram while
repeating its description. Narrated slideshows and blue focus boxes do not
constitute an animated mechanism. Taxonomies, topologies, conceptual triangles,
and chapter maps normally remain static. In particular, the DAM triad remains
a static conceptual diagram. A topology deserves interaction only when traffic,
queueing, failures, or another relevant state actually changes.

Selection requires four answers:

1. What precise systems question does the figure teach?
1. What changes visibly, and how does the state demonstrate cause and effect?
1. What can the reader learn by stepping or varying a parameter that the static
   figure and prose do not already convey as clearly?
1. Does that learning value justify the controls and added visual complexity?

Preserve the original technical meaning, labels, assumptions, palette, and
caption; the web geometry may change. State any worked-case simplification. Reading-time playback must not
imply measured machine latency. Do not invent counters or hardware effects to
make a conceptual diagram appear dynamic. There is no animation quota.

## Historical learner-centered pilot iteration

Three simulated learner-perspective reviewers examined convolution, quantized
arithmetic, and KV paging. No students were recruited, and no learning gains
were measured. Their observations motivated three focused changes:

- Convolution gives detailed arithmetic more reading time, offers paused jumps
  to an edge example and completed output, and keeps local stepping and current
  input/output coordinates near calculations on narrow screens.
- Quantization compares both declared scales side by side, including computed
  integer sums, reconstructed results, errors, and unchanged operand bytes.
  Local cues connect representation changes to the current result.
- KV paging lets a reader inspect an allocated logical-page lookup and check
  whether a request's pages are reusable before or after its final consumer.

Prediction prompts and self-check feedback are learning supports, not measured
assessment. Whether these changes improve understanding remains an untested
hypothesis. The catalog review record separates simulated observations from
that hypothesis and supplies a five-task protocol for actual learner sessions.
No external contact or analytics was added. The 16-lesson roster, chapter prose,
and static artwork are unchanged.

All four HTML jobs passed on final build 20261018-100612-17815. Convolution and
paging passed four-profile runtime, targeted task, and visual checks on the
preceding build 094548, with their code unchanged in the final build.
Quantization passed those checks again in four profiles on final build 100612
after visual feedback prompted a mobile table-fit and dark-caption repair.
All 12 pilot task/profile cases pass across this scoped combination. Final
loading, ID, and freshness checks passed for all 16 lessons across 15 chapter
pages. Full-runtime proof for the 13 unchanged lessons remains historical.
Reviewers inspected the task captures; these checks do not establish student
understanding or learning gains.
The final catalog resolves 707 chapter anchors and exactly 16 Explore markers,
with 691 static entries and clean desktop-light/narrow-dark layouts.

## Historical selective rollout

The counts, builds, and tests in this section describe the earlier selective
rollout. They do not replace the current 100-implementation scope above.

All 674 highlight-only players have been withdrawn. Two of their figures now
have substantive convolution and quantization lessons; the other 672 return to
static presentation. Archived manifests and original proposals are planning
and review evidence, not implemented animations. The catalog
defaults to the implemented stateful subset and can show all retained figures.

The retention audit keeps 14 existing mechanisms and withdraws the dataloader
plot-cursor and release-record flowchart players. Their current controls do not
expose enough changing system state to justify a player. Convolution and
quantized arithmetic add two numerical worked cases, giving 16 registered
lessons: 5 / 4 / 4 / 3 by volume. The other 691 cataloged figures remain static;
no guided-annotation players are enrolled. Fresh final-build loading, ID, and freshness checks passed for all 16. An
absence scan of 146 HTML files confirmed the former image tours and DAM triad
remain static.

Convolution computes products and partial sums as a shared filter moves between
input windows and completes output values. Quantized arithmetic follows float
operands through INT8 representation, corrected INT32 products/accumulation,
and requantized output; changing scale exposes rounding error. Both state their
worked-case assumptions. Each retained lesson's specific dynamic teaching value
appears in the catalog and review record.

Fresh four-profile runtime proof covers 15 unchanged factories on build 182142
and convolution on final build 183145 after a responsive-layout fix. The two new
lessons passed 172 state/profile captures with independent numeric and overflow
checks; quantization also passed eight states at an alternate scale. Visual
review inspected convolution window contacts and representative final chapter
renders. This is a scoped combination of checks, not an assertion that all 16
full runtime tests were rerun on the final build or a blanket semantic approval.

Historical coverage QA belongs to the withdrawn rollout: the 686-player sweep,
2,744 refined captures, later 690-entry integrity check, and four added guided
players establish facts about that snapshot. They do not validate the selective
set or justify keeping a slideshow. The review record retains those results as
history and records fresh mechanism, runtime, and visual checks separately.

PDF and EPUB retain static figures. Fresh focused PDF/EPUB builds of the two
new lessons preserve their original artwork and captions. Their EPUB figure
assets match the earlier full-volume static output byte for byte and contain no
enhancement runtime, controls, or broken image references. Both PDF figure pages
were rendered and inspected. This verifies two chapter targets, not a new
whole-book static rebuild. Previously approved manuscript corrections remain
valid independently of animation enrollment.

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

The compiler rejects `ready` lessons without a declared stateful implementation
script. Archived highlight-only manifests must remain static; changing their
status cannot silently reenroll them. This guard enforces enrollment structure,
not teaching quality.

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
1. Supply a factory under `window.MLSysFigureLessons[lessonName]`. It receives
   an empty, uniquely identified, initially detached Explore panel. It builds
   scoped semantic controls/diagram and returns `{pause, destroy}`.
1. `pause()` must stop timers and interpolation without resetting the state.
   `destroy()` removes lesson listeners/observers and releases resources.
1. Do not initialize tabs or change the original figure/caption inside a lesson.
   The shared controller mounts transactionally and handles view switching.
1. A lesson must pause on document hiding and leaving the viewport; it must
   never autoplay on initialization, view activation, or theme changes.
1. Derive narration, highlights, equations, and counters from one state. Expose
   changing quantities in readable text, not only color or movement.
1. Scope classes and DOM queries. Inherit chapter typography and theme surfaces.
   Preserve the original composition and existing color roles.
1. Add only declared toy data or chapter-owned numerical exports. Real specs
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
