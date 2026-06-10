# Float Exposition Worklist — `benchmarking.qmd` (vol1)

Graded against the Float Exposition Standard by type level.
Flag-only pass; no edits applied.

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| Equation | 🔴 strict | 3 | 2 | 1 | 0 |
| Figure | 🟠 high | 13 | 13 | 0 | 0 |
| Table | 🟠 high | 22 | 15 | 7 | 0 |
| Listing | 🟡 medium | 0 | — | — | — |
| **Total** | | **38** | **30** | **8** | **0** |

---

## Findings (⚠️ only)

---

### `eq-throughput-benchmark` (🔴 equation) — def L2185

**Ref sentence (L2184):**
> "Let $N_{\text{samples}}$ be the total number of training samples processed and $T_{\text{train}}$ the training time from @eq-training-time-benchmark. @Eq-throughput-benchmark shows:"

**Problem — missing Interpret move (consequence/regime).**
Both symbols are defined, so the lead-in passes. But no prose states what a *useful* throughput value looks like, how throughput and training time trade off against each other, or what the equation implies for benchmark design (e.g., why samples/sec can be inflated by large batches that don't improve time-to-accuracy). The payoff paragraph is the footnote about etymology — informative but not a consequence/regime statement. The equation is treated as a definition rather than a claim with a systems implication.

**Takeaway currently lives:** caption (none) / footnote etymology — neither counts.

**Missing move:** Lead-out stating the implication.

**Rule-compliant rewrite for the lead-out (insert after eq-throughput-benchmark, before the footnote):**

> Throughput and training time are not interchangeable: a system that processes more samples per second by increasing batch size may not reduce wall-clock time to the accuracy target, because larger batches often require more total gradient steps to converge. Benchmarks that report only throughput therefore overstate efficiency gains unless time-to-accuracy is verified in parallel.

---

### `tbl-benchmarking-vendor-claims` (🟠 table) — def L536

**Ref sentence (L525):**
> "@Tbl-benchmarking-vendor-claims translates common marketing phrases into the technical caveats behind each."

**Problem — bare-pointer cite, no Interpret move.**
The sentence describes what the table is (a translation) but states no takeaway — which caveat pattern is most common, what the reader should actually do with this table, or what the structural reason is that vendor claims systematically overstate real-world performance. The payoff (L538) pivots immediately to a new topic (hardware infrastructure taxonomy) without closing the loop on any table row.

**Takeaway currently lives:** in the table cells / caption only.

**Missing move:** Lead-out naming the dominant pattern and the reader's action.

**Rule-compliant rewrite (replace the bare cite sentence at L525):**

> Of the four patterns in @tbl-benchmarking-vendor-claims, the precision boundary and workload boundary caveats account for most real-world surprise: vendors routinely report INT8 peak numbers for systems that will run FP32 in production, and maximum-batch throughput for workloads that must serve single-stream latency. Treating any vendor number as a lower bound for exploration rather than a confirmed deployment figure is the operationally correct posture.

---

### `tbl-benchmarking-edgetpu-validation` (🟠 table) — def L2929

**Ref sentence (L2920):**
> "@Tbl-benchmarking-edgetpu-validation reports the validation protocol under the SingleStream scenario."

**Problem — bare-pointer cite, no Interpret move.**
The cite names what the table does (reports a protocol) but states no finding. The payoff (L2935) is thin: "illustrates why benchmarking requires matching the MLPerf scenario to the deployment context" — a generic principle rather than the specific table finding. The specific finding (preprocessing overhead narrows the headline ~12× inference speedup to roughly 3× end-to-end; the EdgeTPU consumes more power per inference than the CPU despite its raw speed advantage) lives only in the table cells.

**Takeaway currently lives:** in the table cells / caption.

**Missing move:** Lead-out stating the specific quantitative finding.

**Rule-compliant rewrite (replace the cite sentence at L2920 and strengthen the payoff at L2935):**

Replace cite (L2920):
> @Tbl-benchmarking-edgetpu-validation reports the SingleStream validation. The headline inference speedup of roughly `{python} EdgeTPUSpeedupCalc.inference_speedup_mult_str` is real, but preprocessing overhead pulls the end-to-end speedup down to about `{python} EdgeTPUSpeedupCalc.e2e_speedup_mult_str`, and the EdgeTPU draws `{python} EdgeTPUSpeedupCalc.edgetpu_power_ratio_mult_str` more power than the CPU, making it more efficient per inference only because it finishes faster.

Replace payoff (L2935):
> The gap between inference and end-to-end speedup is the central lesson: a benchmark that measures only model execution conceals the preprocessing cost that dominates latency for simple models on fast accelerators. The SingleStream scenario exposes this because it does not allow batching to amortize setup overhead.

---

### `tbl-benchmarking-energy-memory-tier` (🟠 table) — def L3304

**Ref sentence (L3295):**
> "@Tbl-benchmarking-energy-memory-tier extends the picture to memory access, with energy cost per byte across each tier of the hierarchy:"

**Problem — float-announcer colon, no Interpret move before or after.**
The cite uses the forbidden colon-as-announcer pattern. More critically, no prose states the key takeaway before the table: DRAM costs 16,000× more per byte than a register read, which is why memory-bound inference (typical for large models) spends most of its energy on data movement, not arithmetic. The payoff at L3309 gives one sentence ("reading one byte from DRAM costs over [X] more energy than a register access") but does not state the *implication* for benchmark design (why energy measurements must attribute load vs. compute separately).

**Takeaway currently lives:** in the table cells (16,000× relative cost column); implication not in prose.

**Missing move:** Interpret move naming the design implication.

**Rule-compliant rewrite (replace cite sentence and strengthen payoff):**

Replace cite (L3295):
> The memory hierarchy compounds the compute savings. @Tbl-benchmarking-energy-memory-tier shows that a DRAM access costs roughly 16,000× more energy per byte than a register read, meaning that a model whose weights do not fit in on-chip cache spends the bulk of its inference energy on data movement rather than arithmetic.

Replace/extend payoff after table (L3309):
> Memory access dominates: reading one byte from DRAM costs over `{python} EnergyPerOp.dram_vs_reg_ratio_mult_str` more energy than a register access. For energy benchmarks this means that model size, not FLOP count, is the primary driver of inference energy for large models — a conclusion that computation-only benchmarks systematically miss.

---

### `tbl-benchmarking-mobilenet-int8-energy` (🟠 table) — def L3319

**Ref sentence (L3311):**
> "@Tbl-benchmarking-mobilenet-int8-energy combines the two effects for a MobileNetV2 inference, decomposing per-inference energy into model-load and compute terms at FP32 vs. INT8:"

**Problem — float-announcer colon, no Interpret move.**
The cite describes the table's content but states no finding. The payoff (L3329) pivots to system-level power and cites a different study (Google data-movement 57.3%) without first closing the loop on what this specific table shows: that model-load (DRAM traffic) accounts for the dominant share of inference energy and that INT8 attacks it by 4× memory footprint reduction, while the compute savings are smaller. The "so what" for the benchmark designer (measure energy decomposed by component, not just total joules) is absent.

**Takeaway currently lives:** in the table cells (savings columns); implication not in prose.

**Missing move:** Lead-out naming the dominant term and the design implication.

**Rule-compliant rewrite (replace cite sentence):**

> @Tbl-benchmarking-mobilenet-int8-energy decomposes per-inference energy into load and compute for FP32 vs. INT8. The load term dominates at FP32 because the model must be read from DRAM each inference; INT8 cuts model size by 4×, halving the load energy, while the compute savings are `{python} EnergyBreakdownCalc.compute_savings_factor_mult_str`. Benchmarks that report only total joules will attribute the INT8 gain to arithmetic efficiency when the primary lever is actually reduced memory traffic.

---

### `tbl-llm-benchmark-failure-taxonomy` (🟠 table) — def L4382

**Ref sentence (L4373):**
> "The useful LLM metric taxonomy in @tbl-llm-benchmark-failure-taxonomy is therefore a decision aid, not a leaderboard. Its rows use MMLU…HELM…and perplexity…as examples of scores that answer different deployment questions:"

**Problem — float-announcer colon, no Interpret move.**
The cite names the three metrics and frames the table as a decision aid, but the colon hands off to the table without stating a takeaway. No prose after the table (before the footnotes) states what the table demonstrates: that a high MMLU score does not imply open-ended generation capability, and that no single metric rules out all deployment failure modes — practitioners must choose metrics that match the specific failure they need to prevent. The payoff position is occupied by footnotes, which do not count.

**Takeaway currently lives:** in the table cells ("What the score cannot prove" column) / caption.

**Missing move:** Lead-out stating the cross-cutting conclusion from the table.

**Rule-compliant rewrite (insert lead-out after the table, before the footnotes at L4384):**

> The table's fourth column is the load-bearing one: every metric has a failure mode it cannot rule out, and those blind spots are structural rather than fixable by tuning. MMLU cannot probe generation capability, HELM cannot guarantee prompt stability, and perplexity cannot confirm groundedness outside the training corpus. Selecting a benchmark therefore means selecting which deployment failure to rule out — and explicitly accepting that the remaining failure modes require separate instrumentation.

---

### `tbl-predeployment-checklist` (🟠 table) — def L4785

**Ref sentence (L4775):**
> "The predeployment benchmark checklist in @tbl-predeployment-checklist summarizes the key validation steps."

**Problem — bare-pointer cite, no Interpret move.**
The cite is a pure summary pointer. No prose states which assumption gap is most consequential (bursty traffic vs. clean inputs vs. cold start), why these gaps are systematic rather than edge cases, or what the engineer should prioritize if validating under time pressure. The payoff position is occupied by a checkpoint callout (not body prose), which does not count toward the prose's job per the standard.

**Takeaway currently lives:** in the table cells / caption; callout is not body prose.

**Missing move:** Lead-out naming the dominant gap and the engineer's priority.

**Rule-compliant rewrite (replace cite sentence and add a lead-out after the table, before the callout):**

Replace cite (L4775):
> Before deployment, validate benchmarking conclusions against production-representative conditions. The largest gap in practice is the first row of @tbl-predeployment-checklist: laboratory benchmarks assume uniform request arrival, but production traffic is bursty, and a system that sustains 1,000 QPS under steady load often fails at 5,000 QPS under a flash event.

Add lead-out (after table, before callout at L4787):
> Among the five rows, the traffic pattern and distribution shift rows are the most frequently skipped and the most consequential: clean inputs and warm system state can be approximated in staging, but bursty arrival patterns and covariate drift require production-trace replay or canary exposure to measure accurately.

---

## Dangling refs noted by scanner (informational, not float-exposition findings)

- `@fig-ai-triad` (L4747): referenced in body prose but definition is in another chapter — not an orphan by design, but worth confirming cross-volume resolution.
- `@tbl-dam-tooling`, `@tbl-dam-scorecard` (L4749): referenced in prose as sibling tables in the appendix; definitions not found in this chapter.
