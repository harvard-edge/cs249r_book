# Two-link singularity prototype

Owning section: Volume IV, Chapter 11, “Mapping tool motion to joint limits.”
Figure: `vol4:fig-11-two-link-singularity`.

This QA prototype animates the original arm’s geometry. The shoulder, elbow, tool,
link proportions, fixed tool bearing, reach arc, velocity-ellipse scale, and
semantic colors follow the original plotted figure. A faint bent arm provides a
persistent comparison while the current arm opens. The original complete image
remains in the Diagram tab.

The reader first identifies the arm, then adds the tool-velocity ellipse before
changing posture. Opening the elbow narrows the ellipse and increases the
joint-rate demand for the same tool-speed magnitude. The queried direction is
the current weakest direction and changes with posture. This is a sequence of
kinematic configuration comparisons, not integration of the displayed joint
commands or a physically timed trajectory. The direction arrow has fixed drawing
length; it does not encode the speed magnitude.

`calcs.py` executes the chapter’s `calc-planning-two-link-arm` LEGO cell verbatim,
which loads the running machine’s canonical MLSysIM values. It exports a table
of complete configurations. At each animation frame the renderer selects one
complete table row; it never interpolates Jacobians, singular values, and joint
rates independently. The exporter checks forward geometry, Jacobian determinant,
`J qdot = v`, singular-value/norm consistency, monotonic demand, and the scaled
individual joint-speed bounds. The checked endpoint norms match the chapter’s
printed rounded values. Singular values have units m/rad, tool speeds m/s, and
joint rates rad/s.

The final step compares the two joint-rate components with the canonical
individual drive limit and computes the maximum tool speed along the current
weakest direction that meets both instantaneous velocity limits. That result
does not establish acceleration, torque, collision, or other safety feasibility.

Regenerate numerical data:

```sh
python3 books/_prototypes/vol1-web-animations/original-v4-two-link-singularity/calcs.py
```

Build the full-chapter QA fixture:

```sh
python3 binder/tools/scripts/build_original_figure_fixture.py \
  --entry books/_prototypes/vol1-web-animations/original-v4-two-link-singularity/entry.json \
  --art-index books/_build/animation-candidate-art/index.json
```

The factory is explicitly selected for the production chapter. Building this
fixture does not alter enrollment, chapter prose, caption, source artwork, or
static PDF/EPUB output. The shared player supplies playback,
manual stepping, speed, keyboard controls, and the expanded view.

If the generated artwork index is unavailable, rebuild it from the successful
four-volume HTML run before creating the fixture:

```sh
python3 binder/tools/scripts/build_figure_art_index.py \
  --build-root books/_build/parallel/20261020-104811-83480 \
  --output books/_build/animation-candidate-art
```

The index and fixture are ignored build artifacts. Rebuilding the HTML run may
require updating the run path; neither artifact is a published chapter.
