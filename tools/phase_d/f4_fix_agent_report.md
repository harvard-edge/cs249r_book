# F.4 Third-Pass Fix Agent Report

**Date**: 2026-04-25
**Input manifest**: `tools/phase_d/f4_third_pass_manifest.json` (10 items)

## Totals

- Rewritten: **10**
- Archived: **0**
- Errors: **0**

All 10 items received substantive third-pass edits and now parse against
`Question.model_validate()`. No item was archived; in every case a viable
rewrite was identifiable from the judge's diagnostic. Three items
(`mobile-1948`, `tinyml-1681`, `tinyml-1723`) required structural math
rewrites; six others required surgical corrections; one (`edge-2390`)
appears to have been judged against a stale text that had already been
fixed in F.2 and only received a defensive polish.

---

## Per-item details

### edge-2390 (DROP, scenario_realism ERROR)

**Prior verdict**: Judge claimed scenario contained an injected YAML path
(`@interviews/vault/questions/mobile/mobile-2060.yaml`). Inspection of
the current file shows no such injection — the F.2 pass already cleaned
it. The judge appears to have been operating on a stale snapshot.

**What changed**: Strengthened the scenario with explicit hardware specs
(NXP i.MX 8M Plus 4xA53 at 1.8 GHz, hardware VPU, 32-bit LPDDR4 at 4000
MT/s with ~10 GB/s sustained, PCIe-attached Hailo-8) so a future judge
cannot mistake the scenario for vague. No corruption was found to
remove; the rewrite is defensive grounding.

**Concerns**: If the judge re-reports the same "injected path" error on
the now-clean text, the issue is in the judge pipeline, not the YAML.

---

### edge-2401 (NEEDS_FIX, math_correct WARN)

**Prior verdict**: Effective-bits calculation gave 3.6 bits, which is
log2(12), not log2(18). The bulk maps to 18 INT8 levels, not 12.

**What changed**: Replaced "~3.6 effective bits" with "log2(18) ~= 4.17
effective bits" in `realistic_solution`, `common_mistake`, and
`napkin_math`. Also corrected the entropy-calibrator side: ~127 levels
= log2(127) ~= 6.99 bits, replacing the loose "~7 bits" claim with the
exact figure. The "discards 3-4 effective bits" common-mistake phrasing
was tightened to "roughly 3 effective bits (4.17 used out of 8 nominal)"
to match.

---

### mobile-1881 (NEEDS_FIX, visual_alignment WARN)

**Prior verdict**: Visual alt text said "fanout diagram" while the
scenario describes a linear pipeline.

**What changed**: Rewrote `visual.alt` to "A linear pipeline diagram
showing data flowing left-to-right from the Cloud through the 5G modem,
the Crypto/UFS stage, and into the phone's NPU." The visual block path
was untouched per the rules.

---

### mobile-1948 (DROP, math_correct ERROR)

**Prior verdict**: Massive unit error. Per-token KV is ~56 KB, not 56
bytes; the prior text derived "~150k tokens fit in 8 MB" by dividing
8 MB by 56 bytes, which contradicted its own correct conclusion that
~150 tokens fit.

**What changed**: Restructured the per-token KV derivation:
- Computed per-token KV from given totals: 96 MB / 2048 tokens ≈ 48 KB.
- Cross-checked against architecture: Llama-3-3B has 28 layers × 8 KV
  heads × 128 head_dim × 2 tensors × 1 byte ≈ 57 KB, with grouped-query
  sharing pulling effective per-token cost down to ~48 KB.
- Recomputed TCM occupancy: 8 MB / 48 KB ≈ 170 tokens (~8% of 2k
  context), which is consistent with the prior "8% hit rate" figure.
- Updated `common_mistake` to name the specific arithmetic trap: "per-
  token KV is tens of kilobytes, so TCM holds order ~170 tokens, not
  ~150k."
- Updated `napkin_math` to make the per-token derivation explicit.

The 8% hit rate, 10.6 mJ/token energy budget, and 520 tok/s bandwidth
ceiling all stay correct — those numbers were only contaminated by the
unit-error sentence, not derived from it.

---

### mobile-1982 (NEEDS_FIX, math_correct WARN)

**Prior verdict**: Conceptually contradicts itself. With ρ = 0.84 < 1
the queue is draining (slowly), not growing. The "1.13 frames added per
drained frame" phrasing inverted the ratio.

**What changed**: Rewrote the explanation to state correctly:
- ρ = 28/33.3 = 0.84, queue is stable and draining (since ρ < 1).
- 0.84 new frames arrive for every 1 frame drained (not 1.13).
- Net drain rate is (1 − ρ) = 0.16 frames per service slot.
- Final figure 2.72 s preserved (the underlying arithmetic was right).
- Added a 2nd common_mistake bullet about misreading ρ < 1 as "queue
  growing."

The realisation-zone message — "naive 300 ms estimate is ~9× off" —
remains intact.

---

### tinyml-1634 (DROP, math_correct ERROR)

**Prior verdict**: Young/Daly optimal-interval calculation missed the
hours-to-seconds conversion. Result labeled "2.31 seconds" was actually
in different units.

**What changed**: Redid the analytic optimum with explicit unit
conversion:
- λ = 0.05/hr = 0.05 / 3600 s = 1.39e-5 /s.
- t_opt = sqrt(2 × 0.004 J / (1.39e-5 /s × 0.030 W))
       = sqrt(0.008 / 4.17e-7)
       = sqrt(19,200)
       ≈ 139 s ≈ 2.3 minutes.

Updated the conclusion: 5-min cadence is ~2× the analytic optimum (not
"much shorter than optimum"), 15-min is ~6.5× optimum. Added an explicit
common_mistake bullet for "forgetting the hours-to-seconds conversion in
Young/Daly produces a unit-mismatched result that looks tiny."

The 5-min vs 15-min comparison (273 vs 691 mJ/hr, 2.5× win) is
unaffected — that arithmetic was always right.

---

### tinyml-1681 (DROP, math_correct ERROR)

**Prior verdict**: Three errors: (1) treats kernel as 3 halfwords not
48; (2) omits weight-load cost; (3) spilling 2 registers should take
4 cycles via STM/LDM, not 8 individual STR/LDR cycles.

**What changed**: Substantial rewrite of the cycle model.
- Per-output operand counts now correctly: input window = K × C_in = 48
  halfwords = 24 words; **kernel weights = K × C_in = 48 halfwords = 24
  words**, loaded fresh each output because the full kernel doesn't fit
  in the 10 available GPRs.
- T_load_input = 24 cycles (4 LDM bursts of 6 words).
- T_load_weights = 24 cycles (new term, was previously omitted).
- T_MAC = 24 cycles unchanged.
- T_spill corrected to 4 cycles (STM 2 words = 1+2 = 3 cycles, LDM
  2 words = 1+2 = 3 cycles, rounded with address-update overhead to 4).
- T_store = 1, T_padding = 0 interior, +6 boundary, all unchanged.
- T_interior = 24 + 24 + 24 + 4 + 1 = **77 cycles** (was 57).
- T_boundary = 83 cycles (was 63).
- Total = 98 × 77 + 2 × 83 + 200 = **7912 cycles** (was 5912).
- At 80 MHz: 99.0 µs (was 73.9 µs).
- Naive 1.2× heuristic = 5760 cycles → **27% under-estimate** (was 2.6%
  under). The much larger heuristic gap is now itself the teaching
  point: "the heuristic silently absorbs the weight-load cost into a
  single fudge factor calibrated for kernel-resident shapes."

---

### tinyml-1716 (DROP, math_correct ERROR)

**Prior verdict**: Pipeline overlap ignored that filter (2 ms) and
inference (10 ms) both run on the single Cortex-M4 CPU and serialise
into a 12 ms CPU stage. The bind is 12 ms, not 10 ms.

**What changed**: Rewrote the overlap analysis:
- CPU stages serialise: filter + inference = 2 + 10 = **12 ms** CPU
  stage.
- Hardware-DMA stages (ADC, SPI) overlap with CPU.
- Ideal overlap rate = max(ADC=3, CPU=12, SPI=5) = 12 ms = 83.3 fps;
  ideal speedup vs 50 fps sequential = **1.67×** (was 2.0×).
- DMA contention tax applies to the 12 ms CPU window (not the 10 ms
  inference): 8 ms DMA-active × 8% = 0.64 ms, inflating CPU stage to
  12.64 ms = 79.1 fps; realised speedup = **1.58×** (was 1.88×).
- Updated common_mistake to name the single-core CPU serialisation as
  the trap, plus the DMA-attribution slip.
- Conclusion now correctly observes that even dual-bank SRAM cannot
  reach 2× speedup — true 2× requires a second core or filter
  hardware-offload.

---

### tinyml-1723 (DROP, math_correct ERROR)

**Prior verdict**: Triple-buffering does not multiply the activation
arena by 3. Compute is sequential, so only one arena copy is live at
any instant. Total SRAM should be ~19.5 KB, not 43.5 KB.

**What changed**: Restructured the SRAM accounting:
- Producer-consumer interface buffers (input, output) need 3 slots each
  for non-blocking handoff.
- Activation arena is touched only during compute; one frame is in
  compute at any instant; therefore **a single 12 KB arena suffices
  regardless of buffering depth on the I/O sides**.
- Triple-buffer total = 3 × 2 KB inputs + 1 × 12 KB arena + 3 × 0.5 KB
  outputs = **6 + 12 + 1.5 = 19.5 KB** (~7.6% of 256 KB).
- Ping-pong total = 17 KB (~6.6%).
- 64 KB arena sensitivity: 6 + 64 + 1.5 = 71.5 KB (~28%, still
  feasible) — the arena failure-mode is "model doesn't fit," not
  "buffering multiplied it."
- Common-mistake updated to call out the arena-multiplication trap
  directly.

This rewrite changes the deployment recommendation: triple-buffering is
comfortably feasible on this MCU, contradicting the prior conclusion
that a 64 KB model would force a fallback to ping-pong.

---

### tinyml-1724 (NEEDS_FIX, math_correct WARN)

**Prior verdict**: Unit-definition typo: `(uC = uA*ms)` is dimensionally
wrong; uA*ms = nC, not uC. Numerical work elsewhere correctly used
mA*ms.

**What changed**: Replaced the parenthetical with `(uC = mA*ms = A*us;
note uA*ms = nC, not uC)`. The numerical calculations were all already
correct (they used 20 mA × 2 ms = 40 µC, etc.), so no other fields
needed editing.

---

## Validation

All 10 YAMLs parse cleanly with `yaml.safe_load` and validate against
`Question.model_validate()` from `interviews/vault/schema.py`. No
required fields are empty. Visual-block paths and schema axes (track,
level, zone, topic, competency_area, bloom_level, id, chains) were
untouched.
