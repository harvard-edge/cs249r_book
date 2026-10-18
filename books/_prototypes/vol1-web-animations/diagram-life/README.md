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

The third prototype uses larger teaching scenes rather than reproducing the
printed layout. A small route preserves context while the active mechanism has
space to explain itself:

- A real-value number line shows the original marker and the nearest level.
  `0.31` snaps to `0.30`; a separate aligned code row identifies stored code `6`.
  All three indexed inputs remain visible as converted codes are written.
- A multiply/retain scene preserves both indexed operand arrays, makes logical
  zero-point removal visible, and keeps the old register value during travel.
  The arrived product produces `24 + 20`, then the register settles on `44`.
- Output conversion distinguishes the wide integer sum, its represented real
  quantity, and the final INT8 code. The finer case adds output rounding; the
  coarse case is already on the output grid. Connected ReLU passage preserves
  `17`, with stored zero `−4` labeled separately.

Selecting an input inspects its approximation; changing spacing compares the
same input at the same operation. Clicking a calculation block pauses playback
and opens the current arithmetic. Back/menu jumps settle immediately; Play
animates the selected operation. Next illustrates one operation and holds its
result; pressing Next during motion finishes that operation without advancing.
Replay restarts the story. No scene autoplays on opening. Reduced motion keeps the
same inspectable states. Phone scenes fit the reading column naturally, with
no internal panning, clipped diagram viewport, or automatic scrolling.

Each operation has an explicit action cue: Read → Round → Store, or Read pair →
Multiply → Add. Continuous playback holds completed results for 3.5 seconds.
On phones, compact controls and the current action stay visible while the scene
scrolls. Mobile viewport height changes preserve an in-progress animation.

The default products remain 24, 20, 40; the sum is 84, and final code 17
represents 0.42. Finer activation spacing produces sum 163, represented sum
0.4075, and final code 16 representing 0.40. Neither scale clips these inputs.
Clipping examples or negative output examples require new grid endpoint and
pre-ReLU illustrations; this prototype does not claim to cover them.

## Feedback and verification

Two rounds of simulated novice, visual-design, and technical reviews motivated
this revision. They are not real student participation or learning evidence.
The review record separates observed interface issues from learning hypotheses.

`learner-session.html` and `learner-session.txt` provide a blank, optional actual
learner session: predict a transformation, trace the next pair, and explain why
the wider sum persists. The local notebook sends nothing automatically and can
export notes only on request. A click or completed playback is not mastery.
Actual learner feedback remains to be collected and incorporated when available.
See `verification.txt` for the scoped runtime, visual, and fallback checks.

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
