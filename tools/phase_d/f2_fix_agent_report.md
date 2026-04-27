# Phase D F.2 — Second-Pass Fix Agent Report

Date: 2026-04-25

## Totals

- Manifest size: 33 items
- Rewritten (substantive revision): 33
- Archived: 0
- Skipped: 0
- All YAMLs validated cleanly via `yaml.safe_load` and required-field check
- All `competency_area` values within the closed enum
- All `(zone, bloom_level)` pairs within ZONE_BLOOM_AFFINITY
- All visual blocks preserved (kind, path, alt, caption unchanged)
- All `status: draft` preserved (no promotions)

## Strategy notes

- Visual block treated as immutable per task contract. When the diagnostic was
  visual-alignment (10 items), the fix path was to rewrite scenario / question /
  realistic_solution so the existing visual alt aligned, typically by framing
  the visual as the "naive baseline" being challenged in the rewritten question
  (edge-2357, edge-2367, edge-2364 secondary, mobile-1891, mobile-1903,
  mobile-1982, tinyml-1562) or by reshaping the scope of the question to match
  what the visual actually shows (mobile-1897 narrowed to uplink-only,
  mobile-1881 added NPU as a downstream stage so the Cloud->Phone->NPU alt
  fits, mobile-1896 reframed to use the visual's stacked bar as the "wrong
  answer" the student must correct).
- For DROP items, applied substantive rewrites (new scenario or new angle on
  the same topic) to surface a viable testable concept.
- For NEEDS_FIX items where the prior fix was insufficient, deepened the
  cycle-accurate / mathematical / hardware-specific content rather than
  cosmetic edits.

## Per-item summary

### edge-2357 (NEEDS_FIX, visual_alignment WARN)
Reframed scenario to start FROM the fragmentation visual as the naive view,
then escalate the question to TLB pressure and page-walk overhead on Orin's
~512-entry iGPU MMU. Visual now reads as the launching point, not a
contradiction.

### edge-2364 (NEEDS_FIX, visual_alignment WARN)
Repositioned the saturated-switch visual as the actual binding constraint by
specifying a 1 GbE TOR switch (4-camera aggregate of 480 MB/s saturates 125
MB/s switch 3.8x). Solution now layers in host-memory ingestion as the next
constraint after a network upgrade — matches both the visual and the original
diagnostic intent.

### edge-2367 (DROP, visual_alignment ERROR)
Visual showed overlapping pipeline; solution is serial. Reframed question to
explicitly use the overlapping-pipeline figure as the textbook upper-bound
(66.6 FPS) being contrasted against the actual single-fabric multi-context
serial result (36.5 FPS, gap = 45%).

### edge-2390 (DROP, scenario_realism ERROR)
Template-injection glitch already cleaned in prior pass. Substantive rewrite:
specified i.MX 8M Plus host paired with Hailo-8, computed raw-frame DRAM
traffic (1.49 GB/s one-way, ~4.5 GB/s pipeline) vs the trivial 25 Mbps H.265
bitrate, and grounded the design in software vs hardware video decode.

### edge-2401 (NEEDS_FIX, uniqueness WARN)
Hard-grounded in Hailo-8 DFC toolchain semantics — per-tensor symmetric INT8,
no per-channel activation quantization, SiLU outliers (max 84 vs 99.9th
percentile 12) — and computed exact INT8 scale and effective-bits per
calibrator. No longer generic Min-Max-vs-Entropy.

### edge-2402 (NEEDS_FIX, uniqueness WARN — too similar to edge-2370)
Pivoted from "power-mode bandwidth selection" to "thermal throttle floor for
roadside SLA". Question now integrates BPMP DVFS clamp (32% sustained) with
LPDDR5 efficiency degradation and recommends heatpipe vs model trim.
Different lever from edge-2370.

### edge-2406 (DROP, math_correct ERROR)
Math was utilization-divided instead of utilization-multiplied. Rewrote
solution: PCIe-bound 317 FPS, compute-bound 364 cold / 255 throttled. The
binding constraint flips from PCIe to compute at the 8-min thermal mark — a
more interesting result than the original (which had no flip). Common_mistake
now explicitly calls out the 26000/50/0.7 = 743 division error.

### edge-2416 (DROP, uniqueness ERROR — duplicate of edge-2363)
Pivoted from M/D/1 vs M/M/1 to a Coral TPU static-batching analysis (B=1, 2,
4) with fill-time penalty for batches. Different question structure entirely;
demonstrates that largest batch is not always best despite higher throughput
ceiling.

### edge-2424 (NEEDS_FIX, uniqueness WARN)
Grounded specifically in TensorRT 8.6 toolchain on Orin: which calibrator
(IInt8MinMaxCalibrator vs IInt8EntropyCalibrator2 vs IInt8LegacyCalibrator),
plus per-layer precision overrides, FP16 mixed-precision fallback, DLA
offload via --useDLACore. No longer generic QAT vs PTQ.

### mobile-1870 (DROP, math_correct ERROR — judge said K=11 yields 0.0077)
Recomputed M/M/1/K blocking probabilities. Min K = 11 (P_11 = 0.00770), not
12 as solution claimed. Updated solution with full sweep K=9..12 and corrected
napkin math.

### mobile-1881 (NEEDS_FIX, visual_alignment WARN)
Visual shows Cloud->Phone->NPU. Added the NPU mapped-ingest stage (6 GB/s) as
the fourth pipeline stage, expanding the question to a 4-stage analysis. The
NPU stage doesn't change the binding constraint (download still wins) but
makes the visual align.

### mobile-1891 (DROP, visual_alignment ERROR)
Visual showed 75 ms; correct answer 400 ms. Reframed question so the figure's
75 ms is explicitly the naive bandwidth-only baseline being challenged by
serialization tax + UFS contention + parallelization analysis.

### mobile-1896 (NEEDS_FIX, visual_alignment WARN)
Stacked bar chart was misleading for a harmonic-mean question. Reframed
question to use the bar chart's arithmetic-mean reading (92 GB/s) as the
junior-engineer wrong answer the student must correct (88.2 GB/s).

### mobile-1897 (NEEDS_FIX, visual_alignment WARN)
Visual shows uplink-only star topology. Narrowed question scope to client->host
uplink only, dropping the downlink limb that the visual didn't show.

### mobile-1903 (DROP, visual_alignment ERROR)
Visual showed sequential Gantt; solution requires double-buffered overlap.
Reframed to use the figure's sequential 30.3 FPS as the v1 baseline that
double-buffering must improve upon (62.5 FPS = 2.13x uplift).

### mobile-1918 (DROP, math_correct ERROR — energy units wrong)
Fixed energy units throughout: ms*W = mJ, never mWh for sub-second windows.
Reworked the policy comparison: continuous (100 mJ/window, 33 ms latency) vs
batched (15.85 mJ/window, 115 ms worst-case latency). Energy ratio ~6.3x.

### mobile-1929 (DROP, cell_fit WARN, uniqueness WARN)
Pivoted from "duty-cycle vs always-on" (overlaps mobile-1918) to "wake/sleep
break-even rate analysis": full charge-up energy model with 2 ms wake at 1 W
+ 0.8 ms compute + 0.5 ms sleep flush, sweeping {1, 5, 10, 30} Hz. Break-even
vs always-on at 303 Hz; recommends 10 Hz matching UX (42.7 mJ/s, 23x better
than always-on).

### mobile-1948 (DROP, scenario_realism ERROR)
Original scenario implied placing entire LLM KV in NPU SRAM (impossible).
Rewrote to a tiered scheme: 96 MB KV in LPDDR5 vs sliding-window in 8 MB
Hexagon TCM. Hit rate ~8% gives only ~8% energy savings; INT4 K/V quantisation
is the bigger lever.

### mobile-1949 (DROP, scenario_realism WARN, uniqueness WARN)
Replaced generic CPU vs NPU template with hard numbers: Cortex-X4 NEON SDOT
(27 GOPS @ 2 W = 13.5 GOPS/W peak) vs Hexagon NPU (12.86 GOPS/W peak) and
realistic 10 GOPS demand-point efficiency (10 GOPS/W vs 12.8 GOPS/W). Shows
the NPU's lead is only 1.3x at this demand, widens dramatically as workload
scales.

### mobile-1982 (DROP, visual_alignment ERROR)
Visual claimed 300 ms drain; correct answer 2720 ms. Reframed so the figure's
linear 300 ms drain is the naive baseline (15 frames * 20 ms) and the question
asks for the realistic worst-case under cold-cache + variance + open-loop
arrivals.

### mobile-1995 (DROP, math_correct ERROR — ignored 800us preempt + 18% throttle)
Rewrote solution to honour both costs: FIFO tail = 50 ms cold / 61 ms
throttled (both fail 16.6 ms); priority+preempt tail = 5.8 ms cold / 6.9 ms
throttled (both pass). 800 us tax is 4.8% of budget.

### mobile-2025 (NEEDS_FIX, cell_fit WARN — L6+ but trivial math)
Upgraded from "25M*4 - 25M*0.5 = 87.5 MB savings" to a full mixed-precision
SLC-resident allocation: per-tier weight footprint (4M FP16 embedding + 3.2M
INT8 outer attention + 17.8M INT4 bulk = 20.1 MB) plus 4 MB activation
double-buffer plus 6 MB OS reservation, yielding 73.4 effective TOPS. Three
constraints to balance simultaneously now match L6+ optimization complexity.

### mobile-2028 (DROP, math_correct ERROR — never computed fallback budget)
Added the missing fallback computation: 0.7 * 25 + 0.3 * 25 * 1.4 = 28
MFLOPs/frame -> 3.36 GFLOPs/s (12% increase over INT8 nominal 3.0 GFLOPs/s).
Throughput is trivially under-provisioned; thermal envelope at 1.4 W is the
real binding constraint.

### tinyml-1562 (NEEDS_FIX, visual_alignment WARN)
Visual shows 1000 MB/s raw line rate; correct effective is 886 MB/s. Reframed
question so the figure's optimistic 1000 MB/s view is the input the student
must derate by 128b/130b + TLP overhead (886 MB/s effective).

### tinyml-1634 (DROP, math_correct WARN, scenario_realism WARN)
Replaced unrealistic 100 KB / 5 ms (20 MB/s) flash speed with realistic 8 KB
/ 80 ms (100 KB/s) STM32L4 numbers. Built full energy-budget comparison: 5-min
cadence wins 273 mJ/hr vs 691 mJ/hr (2.5x), with analytic optimum at 2.31 s.

### tinyml-1652 (DROP, scenario_realism WARN, uniqueness WARN)
Pivoted from "Ring AllReduce on UART-daisy-chained Cortex-M4" to a realistic
"BLE mesh federated embedding-reduce on nRF52840 nodes" with 3 uJ/byte radio
TX energy. Same 800 B per-node math, now grounded in a real TinyML federated-
sensor scenario.

### tinyml-1661 (NEEDS_FIX, math_correct WARN)
Added the brownout-voltage threshold (1.8 V BOR) to the energy formula and
showed both the charge-budget approach (3.2 mC required vs 2.4 mC available)
and the energy approach (5.76 mJ usable vs 7.68 mJ required). Both views
agree the original 2 mF cap fails by ~25%; recommended C_min = 2.67 mF.

### tinyml-1681 (NEEDS_FIX, cell_fit WARN — L6+ but 1.2x heuristic)
Replaced the 1.2x MAC heuristic with a true cycle-accurate cost model:
T_total = T_load + T_MAC + T_spill + T_pad + T_store, decomposed for
Cortex-M4 SMLAD with explicit register-pressure analysis (16 in-channels in
8 packed pairs forces 2-pair spill). Final answer: 5912 cycles vs heuristic
5760 (2.6% under), with explicit regimes where the heuristic loses fidelity.

### tinyml-1716 (DROP, math_correct ERROR — ignored 8% DMA contention)
Reworked solution to honour the 8% DMA-contention tax: ideal 100 fps drops to
94 fps (1.88x speedup vs ideal 2.0x). Identifies that single-bank chip
already gets 94% of the dual-bank benefit, justifying or rejecting the
dual-bank cost premium based on tail-sensitivity.

### tinyml-1721 (DROP, math_correct ERROR — ignored leakage and lifetime)
Added the constant 3 uA leakage (correctly small at 0.3% of budget) and the
non-linear discharge derate from 250 mAh nameplate to 150 mAh effective:
naive lifetime 247 hr vs realistic 148 hr (40% shorter). Active compute
dominates at 98.8% of total current.

### tinyml-1723 (DROP, math_correct ERROR — single-buffer sum instead of overlap)
Rewrote to a true triple-buffer (three-frames-in-flight) accounting:
3 * (2 + 12 + 0.5) KB = 43.5 KB (~17% of 256 KB SRAM), vs ping-pong's 29 KB
(~11%). Sensitivity analysis shows that a 64 KB activation arena pushes
triple-buffering to 78% of SRAM (infeasible), forcing ping-pong fallback.

### tinyml-1724 (DROP, math_correct ERROR — ignored which retention mode wins)
Full charge-budget comparison of RAM retention vs cold-boot under both 50 ms
and 500 ms cadences. RAM retention wins both regimes (2.81 vs 3.05 mA
unbatched; 2.09 vs 2.11 mA batched). Computed break-even sleep period =
12 uC / 5 uA = 2.4 s, above which cold-boot becomes the right choice.

### tinyml-1732 (NEEDS_FIX, cell_fit WARN — didn't address half-transfer interrupt mechanics)
Replaced high-level "DMA frees the CPU" answer with full HTIF/TCIF circular-
DMA mechanics: pointer-swap on each interrupt, deadline = N / sample_rate
(32 ms at 16 kHz, N=512), and explicit silent-overrun failure mode if
compute_time > per-half deadline.

## Concerns / caveats for re-judge

1. **Some YAMLs needed plain-scalar -> single-quoted scalar conversions** to
   avoid `mapping values not allowed here` errors when `: ` patterns appear
   in technical prose (edge-2416, mobile-1897, tinyml-1723). Quoted scalars
   verified to parse cleanly.
2. **mobile-1897** had an editing artifact (duplicate `details:` key) which
   was caught and corrected mid-pass. Final structure verified.
3. **mobile-2025** is L6+ optimization; the rewrite expands the cognitive
   load substantially (multi-constraint mixed-precision allocation with
   throughput maximisation), which may push expected_time_minutes higher
   than the current 5; left unchanged per scope rules but flagged for the
   reviewer.
4. **edge-2367** (L1 recall) — reframing the visual as a "textbook baseline"
   is consistent with the visual's content but requires the reader to do a
   gap-analysis (66.6 FPS upper bound vs 36.5 FPS realised), which is
   slightly above pure recall. The bloom_level is `remember` and the zone
   is `recall` per the locked schema axes; the reviewer may want to verify
   this still fits L1.
5. **tinyml-1652** kept the 3-node ring topology to align with the visual,
   but moved from "UART-daisy-chained Cortex-M4" to "BLE mesh on nRF52840"
   to land a realistic federated-sensor scenario. The visual alt
   ("three-node cyclic ring topology") still aligns.
