# Timing across an asynchronous boundary

This QA prototype enhances `vol1:fig-benchmark-async-timing` in the benchmark-harness section. The complete original SVG supplies the lanes, labels, enqueue block, work block, dependency arrows, and three timer intervals. Added traces grow along those intervals while one cursor advances through the same time coordinates in both lanes. No separate timing experiment or numeric dashboard replaces the figure.

The lesson has six endpoints: establish the boundaries; return from enqueueing; reach the device start event; execute bracketed work; reach the device stop event; observe completion on the host. The call timer stops before the device starts. At the device stop event, the host is still waiting. Only the final step releases that barrier and stops the synchronized host timer.

The drawing is schematic. Animation duration does not represent measured hardware latency. Device events measure the elapsed interval between their stream positions; they do not measure all accelerator activity. The assumption that earlier device work has completed comes from the paragraph immediately following the figure.

`calculations.py` reproduces boundary coordinates from the canonical SVG transform and checks their ordering against the runtime constants. It also exports the expected categorical timer trace. The runtime retains a committed endpoint independently of an interpolated cursor position; narration identifies a pending action without declaring its endpoint complete.

The shared player supplies manual stepping, sequence selection, pause/resume, replay, speed, keyboard controls, reduced motion, and Expand. On phones, a responsive rendering retains the same horizontal event boundaries, proportions, two lanes, dependency links, and three timing traces. Short labels sit above their rows, with each timer status beside its interval. The complete timeline fits without horizontal panning. Both renderings use one state and one interpolated cursor coordinate; opening Expand does not create another simulation. The unchanged source drawing remains available in Diagram.

## Reproduce the QA fixture

```sh
python3 books/_prototypes/vol1-web-animations/original-v1-benchmark-async-timing/calculations.py
python3 binder/tools/scripts/build_original_figure_fixture.py \
  --entry books/_prototypes/vol1-web-animations/original-v1-benchmark-async-timing/entry.json \
  --art-index books/_build/animation-candidate-art/index.json
```

The fixture uses the actual rendered benchmarking chapter from build `20261020-104811-83480`; it is a local review artifact. This entry does not enroll the figure in production or change the chapter, caption, TikZ, or static assets. PDF and EPUB continue using the existing figure.

If the generated artwork index is unavailable, rebuild it from the successful
four-volume HTML run before creating the fixture:

```sh
python3 binder/tools/scripts/build_figure_art_index.py \
  --build-root books/_build/parallel/20261020-104811-83480 \
  --output books/_build/animation-candidate-art
```

The index and fixture are ignored build artifacts. Rebuilding the HTML run may
require updating the run path; neither artifact is a published chapter.
