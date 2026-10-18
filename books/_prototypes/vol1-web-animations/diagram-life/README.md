# Diagram-first animation review

This pass reorients the 16 retained mechanisms toward the original diagrams:
movement, transformation, retained state, and ownership. It does not enroll the
691 static figures or treat animation count as a goal. The per-lesson series
review distinguishes present behavior from proposed revisions.

## One working review example

Run `python3 build-preview.py` from this directory after the existing chapter
build is available. Serve `books` locally, open
`_prototypes/vol1-web-animations/diagram-life/quantization-flow.html#fig-weight-activations-quantization`,
select Explore, and press Play or Next. The generated page is an isolated copy of
the actual themed chapter. The canonical build, chapter source, static artwork,
caption, and production lesson remain unchanged. This is not a published update.

The prototype retains both input branches and the compute, INT32, requantization,
activation, and output chain. Labeled tokens travel along connections. The
accumulator retains earlier contributions until the next product arrives. Its
sum changes 0 → 24 → 44 → 84 at the default scale; the final INT8 value 17
reconstructs to 0.42. Optional arithmetic exposes the current transformation.
Back and menu jumps settle immediately; Next illustrates the transition. Pause
freezes motion as well as progression. Reduced motion keeps inspectable states.
Narrow views stack the same connected blocks vertically in a contained diagram
and follow the active block. Desktop retains the source's horizontal chain.
This responsive layout is a prototype tradeoff to evaluate with actual learners.

## Software decision

Use inline SVG for crisp, addressable diagram objects, semantic HTML for controls
and explanations, scoped CSS for the book theme, and ordinary JavaScript for
deterministic state. The browser's Web Animations API moves value tokens and
supports pause/resume without adding a runtime dependency:
[Element.animate](https://developer.mozilla.org/en-US/docs/Web/API/Element/animate)
and [Web Animations usage](https://developer.mozilla.org/en-US/docs/Web/API/Web_Animations_API/Using_the_Web_Animations_API).

GSAP has coordinated [timeline controls](https://gsap.com/docs/v3/GSAP/gsap.timeline/)
and is a candidate if later diagrams demonstrate a need for complex shared
choreography. It is unnecessary for this small controller. Manim's
[rendered output workflow](https://docs.manim.community/en/stable/tutorials/output_and_config.html)
fits narrated clips better than this inspectable, theme-aware figure. Neither
video nor GIF should replace the primary stateful website interaction.

Keep canonical static artwork for PDF/EPUB. Do not assume an animation's paused
frame is a publication-quality print figure. Verify visual fidelity, numbers,
playback, reduced motion, dark mode, narrow layouts, and static fallback before
production integration. No actual learner study or learning gain is claimed.
