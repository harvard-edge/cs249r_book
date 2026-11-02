# Quantization: stored values and pending work

The quantized pipeline in Volume I separates low-precision operands from the
higher-precision accumulator. This refinement keeps that distinction visible
during movement, not only at the sequence endpoints.

Previously, the second product could still be moving while the equation and
step description reported a completed sum. The multiplier also retained the
previous operand pair. On a phone, manually panning to the output could leave
the next operation outside the visible part of the diagram.

The revised animation keeps the committed register value while a product is
in transit. A readout beside the diagram identifies that stored value and the
incoming product; a dashed destination marks work that has not arrived. The
multiplier selects the current pair without committing the next sum. Arrival
updates the register and reveals the completed equation together. Starting an
action recenters its source and destination; a paused learner can still pan
freely to inspect another part of the drawing.

The source TikZ diagram, caption, surrounding prose, numerical exporter, and
shared player are unchanged. The readout supports the diagram rather than
substituting a different exercise. The illustrated movement is not hardware
timing, and this worked example does not establish a performance improvement.

## Reproduce the author fixture

From the repository root, provide a canonical artwork index from a rendered
chapter or volume:

```sh
python3 binder/tools/scripts/build_original_figure_fixture.py \
  --entry books/_prototypes/vol1-web-animations/quantization-clarity/entry.json \
  --art-index books/_build/ARTWORK_INDEX/index.json
```

This command creates an author fixture; it does not establish production
acceptance. Final review uses the actual selectively rebuilt chapter, including
its packaged assets, prose, figure anchor, and Diagram/Explore controls.

Independent technical and learner-view reviewers check the original problem,
the corrected operand-feed transition, all endpoint values, genuinely
progressed paused movement, mobile panning, speed changes, expansion, keyboard
stepping, and reduced motion. Their reports distinguish agent inspection from
student feedback. Local evidence remains under
`books/_build/quantization-refinement`; participant research is not claimed.

`check.py --url ACTUAL_CHAPTER_URL --engine chromium` provides a focused
regression check for pending operands, retained sums, arrival commits, paused
speed changes, and phone action centering. Expected arithmetic comes from the
existing canonical calculation. Repeat with `--engine webkit` for that engine.
