# Where does batching move the classifier on the H100 roofline?

This QA prototype enhances `vol1:fig-hw-acceleration-h100-roofline-lighthouses` in the memory-bandwidth-bounds section of the hardware acceleration chapter. The verified matplotlib PNG (3315 × 2476, SHA-1 `d4348c048b6e…`) stays intact: axes, bandwidth slope, both ceilings, both ridges, legend, every printed point and arrow. The factory adds overlays in the PNG's own pixel coordinates. These are a movable FP16 classifier marker (filled dark dot with a yellow halo), a faded dot and dashed ring at the printed batch-32 position once the marker leaves it, a translucent highlight on the binding term (blue slope or orange FP16 ceiling), and a small value plate in the empty area under the legend. Below the plot, an HTML readout and a bytes-moved bar repeat the committed values at every width.

## Question and storyboard

The static figure freezes the classifier at batch 32. The chapter argues that batching moves a bandwidth-bound dense layer up the roofline until it crosses the ridge (the batch-intensity equation, the dense-layer notebook, and the optimization-regime table). Only a changing batch can show that path, the crossing, and the flat ceiling after it. Six endpoints, one batch change each:

1. Printed point: batch 32, 30.5 FLOP/byte, 102.3 TFLOP/s (10.3% of peak), bandwidth bound. The marker sits on the printed dot.
2. Batch 1: 1.0 FLOP/byte, 3.3 TFLOP/s. Weights (4.10 MB) are nearly all the traffic. The marker lands on GPT-2 XL batch-1 decode, which also reads every FP16 weight once per output.
3. Batch 256: 185.4 FLOP/byte, 621.0 TFLOP/s. FLOPs grow while weight bytes stay fixed, so the point is still on the slope.
4. Batch 512: 290.6 FLOP/byte, not 512, because activation bytes grow with the batch. Still below the 295.2 FLOP/byte ridge, at 98.4% of peak.
5. Batch 527: the first batch past the FP16 ridge (295.3, while 526 gives 295.0). The binding term switches to the 989 TFLOP/s ceiling.
6. Batch 1,024: 405.7 FLOP/byte, but the bound stays 989 TFLOP/s. For this layer intensity can only approach 671.9 FLOP/byte. The cost is waiting for a batch to form, stated qualitatively with no latency number.

## Fidelity guard

Only the classifier marker moves, and only along the FP16 roofline. The INT8 ceiling stays visible but does not apply to this marker, because the printed figure plots the classifier at FP16 only. No slider is attached to GPT-2 decode, since its batched traffic would involve KV-cache bytes that the chapter does not model here. Attainable throughput is the roofline bound `min(peak, bandwidth × intensity)`, not a measured rate. Motion speed represents no time. While a step is moving or paused, the marker follows the interpolated position. The plate (dimmed), the readout, `data-ledger`, and the narration still describe the last committed batch, and the narration marks the destination "On arrival".

## Calculations

`books/_extensions/mlsysbook-ext/figure-explore/original-v1-h100-roofline-batch.calc.py` reads every value from MLSysIM through the same objects as the chapter's LEGO cells:

- H100 dense FP16 peak, INT8 peak, and HBM bandwidth come from `Hardware.Cloud.H100`, as in `ResNet50Int8Crossing`. The script asserts that `compute.peak_flops` equals `fp16_dense`, since `DenseLayerAnalysisCalc` uses the former.
- Intensity, FLOPs, and weight and activation bytes at each batch come from `Scenarios.ResNet50ClassifierHeadBatch32.variant(batch_size=B)`. The batch-32 stop is asserted equal to `DenseLayerAnalysisCalc.dense_ai` (30.5) and 10.3% of peak.
- The crossing batch is `ceil(batch_for_traffic_intensity(ridge))`, checked on both sides.
- The limit `in·out/(in+out)` is the batch-intensity equation as the batch grows without bound.

The script also re-measures the PNG axes frame (spine centers x 373–3256, y 59–2138) and asserts that the printed classifier dot's centroid lies within 2.5 px of the computed batch-32 position (measured offset 0.2 px). `--write` embeds the data block in the factory, and `--check` asserts that the embedded block is current. The browser formats no lesson value. It only maps interpolated intensities onto the log axes during motion.

```sh
PYTHONPATH=mlsysim ../MLSysBook/.venv/bin/python books/_extensions/mlsysbook-ext/figure-explore/original-v1-h100-roofline-batch.calc.py --check
```

## Responsive view

Below 700 px the value plate is hidden because its text would be unreadable at that scale. The marker and halo grow 2.3× so the moving point stays visible, and the readout becomes a two-column grid. The readout and the bytes-moved bar carry the same committed values at every width and in both themes. The plot keeps a white backing in dark mode, as the original PNG does.

## Review check

```sh
python3 books/_prototypes/vol1-web-animations/original-v1-h100-roofline-batch/check.py [--engine chromium|webkit] [--width 1440|390]
```

The check works as follows:

- It runs `calc.py --check` with `../MLSysBook/.venv/bin/python` (override with `RB_CALC_PYTHON`) and confirms that `expected_final` is current.
- It builds the owning-chapter fixture through `build_original_figure_fixture.py`, with the verified PNG wrapper and shared player, on a private port.
- It runs Chromium and WebKit at 1440 and 390 px, in light and dark.

Each profile checks the following:

- No autoplay.
- Every endpoint's `data-ledger` matches calc.py, and the marker sits exactly on the computed pixel endpoint.
- Exactly one binding highlight is shown, and it matches the bound. The bound agrees with the ridge comparison, and weight bytes stay constant.
- The ghost ring appears only away from batch 32.
- The readout text matches the committed values.
- Plate text fits inside the plate, and there is no stage overflow or clipped cells.
- `expected_final` matches.
- Menu jumps are independent of history.
- P/N/R and K keys work, and reduced motion commits immediately.
- The chapter does not overflow horizontally.
- Expand/Escape works.
- In the light profiles, a normal-motion pause inside Expand freezes the marker, ledger, readout, and narration strictly between endpoints. Resume then completes the step.
- The Diagram tab shows the static figure.
- Print hides the player.
- There are no page errors. The only tolerated exception is WebKit's `ResizeObserver loop completed` notice under print emulation, the same allowance as in sibling checks.
- With JavaScript disabled, the static image and caption remain.

Screenshots of every state, the paused motion, the expanded view, and the static view go to `/tmp/original-v1-h100-roofline-batch-review/`.

Before pressing K, the check scrolls to the top and back. The shared player's IntersectionObserver otherwise sometimes keeps a stale "not visible" flag at 390 px after the endpoint sweep, and Play is then ignored. The coordinated-omission sibling check uses the same step.

This is a local review artifact. It establishes neither production enrollment nor PDF/EPUB correctness, and no student testing is claimed.
