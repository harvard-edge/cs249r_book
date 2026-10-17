# Original-art guided figures

`guided-figure.js` registers the `guided-figure` factory. This is an **annotation**
treatment, not a numerical simulation or animated execution model. It keeps the
canonical image intact and gives each authored state a visible region focus.
Use specialized mechanism lessons when values, data ownership, object lifetimes,
or causal execution must change.

The controller supplies `{outer, figure, staticNodes, lesson}`. A lesson declares
`key`, `figure_id`, `source`, `title`, `treatment: 'annotated'`, `question`, `scope`,
`regions`, and `steps`. Each region is `{x,y,w,h,label}` in normalized image
coordinates from the top left. Each step is `{title,description,focus:[regionIds]}`.
All regions must occur in at least one step; all steps must focus a valid region.
Coordinates must be finite and entirely within the original image. Exactly one
canonical image must exist among `staticNodes`; otherwise initialization fails
without replacing Diagram. Never guess coordinates from caption text.

Inspect the actual artwork before mapping regions. Preserve its full extent,
orientation, labels, connections, units, legend, and semantic colors. Keep boxes
slightly inside the image edges so borders remain visible. Favor component or
panel boundaries over boxes that cross text. A state description explains the
highlighted relationship, and explicitly identifies alternatives rather than
presenting mutually exclusive branches as chronological events.

The shared player copies image data only, overlays outlines without dimming or
cropping the artwork, and supplies clickable image regions plus accessible
numbered region controls. Playback starts paused and ends once. State selection
and manual stepping pause playback. Tab changes are paused by the outer runtime;
document hiding, printing, and offscreen visibility also pause it. Narration
remains available until the reader chooses another state. Time between states
is reading time, not modeled machine latency. There is no interpolated movement,
so reduced-motion users receive the same complete states.

Before enrollment, validate asset identity/hash in the compiler, inspect every
authored state in the real chapter, and check mobile, both themes, keyboard,
tab-switch pause, replay, and exact menu jumps. Small dense source labels remain
small on mobile: shorten narration, avoid adding more labels, and use a genuinely
specialized responsive lesson when the original cannot be followed at that size.
