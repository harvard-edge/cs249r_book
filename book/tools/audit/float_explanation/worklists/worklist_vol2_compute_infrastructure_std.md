# Float Exposition Audit — `compute_infrastructure.qmd` (vol2)

> Standard: FLOAT_EXPOSITION_STANDARD.md
> Scope: body prose only (caption, fig-alt, in-figure labels, code comments excluded)
> Grader: Claude Sonnet 4.6 · 2026-06-09

---

## Summary Table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| eq   | 🔴 strict  |  1 |  1 |  0 |  0 |
| fig  | 🟠 high    | 12 |  8 |  4 |  0 |
| tbl  | 🟠 high    | 11 |  7 |  4 |  0 |
| lst  | 🟡 medium  |  0 |  — |  — |  — |
| **Total** | | **24** | **16** | **8** | **0** |

No orphans. No 🛑 failures. All floats cited; 8 ⚠️ findings where the Interpret move is a bare pointer or float-announcer and the takeaway lives in the caption or cells.

---

## Findings (⚠️ only)

---

### 1. `fig-wafer-scale-engine` (fig 🟠) — def L192

**Ref sentence (L188):**
> "As @fig-wafer-scale-engine shows, the challenge of wafer-scale integration is physical: manufacturing yield, power delivery, and thermal expansion."

**Missing move:** Lead-out/Interpret. The citation uses an "as X shows" pivot to name three challenges, but never states what the figure *demonstrates* — the side-by-side comparison of an 8-GPU cluster (microsecond cross-chip latency, slow inter-node links) versus the WSE-3 single wafer (nanosecond latency, 21 PB/s on-chip fabric) is the figure's proof that eliminating the inter-chip boundary collapses the communication latency by three orders of magnitude. That contrast is the figure's argument; the prose only points at the manufacturing difficulties. The takeaway that the cluster-vs-chip comparison *justifies* the manufacturing cost lives only in the figure's alt-text.

**Rule-compliant diff rewrite — replace L188 with:**

```diff
- As @fig-wafer-scale-engine shows, the challenge of wafer-scale integration is physical: manufacturing yield, power delivery, and thermal expansion. A single defect on a standard chip might render it useless, but on a wafer-scale engine, the software must be "defect-aware," routing around local manufacturing flaws in the silicon fabric. Delivering 23 kW of power to a single piece of silicon and cooling it requires specialized manifold-level liquid cooling that is closer to industrial plumbing than traditional computer engineering.
+ The side-by-side @fig-wafer-scale-engine makes the communication argument concrete: an 8-GPU cluster crosses a microsecond boundary every time one chip reaches another over NVLink or InfiniBand, but a WSE-3 keeps all 900,000 cores within nanoseconds of each other because the fabric is silicon, not copper. That three-order-of-magnitude latency gap is what wafer-scale integration is buying. The cost is equally physical: a single defect that would merely discard a diced chip cannot discard the entire wafer, so the software must be "defect-aware," routing computation around local manufacturing flaws. Delivering 23 kW to one piece of silicon and removing that heat requires manifold-level liquid cooling that is closer to industrial plumbing than traditional computer engineering.
```

---

### 2. `fig-accelerator-spectrum` (fig 🟠) — def L200

**Ref sentence (L198):**
> "@Fig-accelerator-spectrum places these architectures on a continuum, revealing the fundamental trade-off between programmability and efficiency that governs every accelerator design choice."

**Missing move:** Interpret. The sentence names the trade-off by label but does not deliver what the figure's die-area pie charts *show* about that trade-off — that moving from CPU to GPU to TPU to custom ASIC progressively reallocates silicon area from control logic (branch prediction, out-of-order execution, cache coherence) to arithmetic units, and that the fraction of die area devoted to arithmetic is the physical mechanism behind throughput-per-watt differences. The figure's visualization teaches via the area split; the prose only repeats the label. The payoff lands on the table (L224), not on the spectrum figure.

**Rule-compliant diff rewrite — replace L198 with:**

```diff
- The key wafer-scale trade-off is manufacturing complexity and defect-aware routing in exchange for eliminating the inter-chip communication bottleneck entirely, keeping all 900,000 cores within nanoseconds of each other on a single silicon fabric. @Fig-accelerator-spectrum places these architectures on a continuum, revealing the fundamental trade-off between programmability and efficiency that governs every accelerator design choice.
+ The key wafer-scale trade-off is manufacturing complexity and defect-aware routing in exchange for eliminating the inter-chip communication bottleneck entirely, keeping all 900,000 cores within nanoseconds of each other on a single silicon fabric. @Fig-accelerator-spectrum makes the underlying mechanism visible through die-area splits: a CPU devotes roughly 90 percent of its silicon to control logic (branch prediction, out-of-order execution, cache coherence) and only 10 percent to arithmetic. A GPU inverts this to about 55 percent arithmetic. A TPU pushes further to 75 percent. A custom ASIC can exceed 90 percent arithmetic, because every feature not required by the target computation has been stripped. The throughput-per-watt advantage of specialized accelerators is not a marketing claim; it is the physical consequence of substituting arithmetic units for control logic on the die.
```

---

### 3. `fig-node-topology-comparison` (fig 🟠) — def L2176

**Ref sentence (L2170):**
> "We must expand to the next physical level: the node, where @fig-node-topology-comparison contrasts the two dominant approaches to wiring multiple accelerators within a single chassis."

**Missing move:** Lead-out/Interpret. The citation sentence is a structural pointer ("contrasts the two dominant approaches"), and the payoff at L2190 explains why the interconnect matters (cheap or ruinous reassembly) but never says what the figure's three-panel comparison demonstrates. The figure shows ring, full-mesh, and NVSwitch crossbar; the figure's argument is that ring latency scales with distance, full-mesh link count grows as N-1 (impractical beyond 8 GPUs), and NVSwitch solves both by providing non-blocking any-GPU-to-any-GPU in a single hop at 900 GB/s aggregate. That is the figure's proof of *why* NVSwitch is the modern choice. The prose payoff discusses why interconnect matters in general but does not interpret the three topologies.

**Rule-compliant diff rewrite — replace the payoff paragraph at L2190 with a dedicated interpret move inserted after L2178 (the figure block), before L2180 (the LEGO cell):**

```diff
+ The three-panel comparison resolves a concrete design question. A ring topology passes data hop-by-hop around the loop, so a GPU on one end of an 8-GPU ring reaches the GPU on the other end only after seven sequential links; latency and bandwidth both scale with distance, making ring-based AllReduce slow under tensor parallelism. A full-mesh eliminates the distance problem by wiring every GPU directly to every other, but requires N minus 1 links per GPU: seven links for 8 GPUs, thirty-one for 32, making the cabling impractical at any node size beyond the smallest. NVSwitch resolves both limitations with a non-blocking crossbar fabric: any GPU reaches any other GPU in one hop at 900 GB/s aggregate, regardless of how many GPUs share the node. The NVSwitch design is why modern DGX nodes can run tensor-parallel AllReduce within the chassis rather than offloading it to the slower inter-node fabric.
```

---

### 4. `fig-infra-bandwidth-hierarchy` (fig 🟠) — def L2293

**Ref sentence (L2297, first citation):**
> "@Fig-infra-bandwidth-hierarchy visualizes this hierarchy as a series of concentric zones, with bandwidth decreasing at each successive tier from on-chip SRAM down to the global network."

**Missing move (first cite):** Interpret at first cite. The sentence is a pure float-announcer ("visualizes…with bandwidth decreasing"). The takeaway — that crossing each zone boundary costs an order-of-magnitude bandwidth and an order-of-magnitude latency, and that this physical law forces TP inside the node and DP across nodes — appears only in the second citation at L2382 ("TP at InfiniBand bandwidth takes 10-20 ms, which would leave accelerators idle"). The first cite passes no takeaway to the reader; anyone who skips to this paragraph gets nothing about why the figure matters. Note: the second citation (L2382) does deliver a strong payoff.

**Rule-compliant diff rewrite — replace L2297 with:**

```diff
- @Fig-infra-bandwidth-hierarchy visualizes this hierarchy as a series of concentric zones, with bandwidth decreasing at each successive tier from on-chip SRAM down to the global network. As @tbl-bandwidth-hierarchy-compute shows, parallelism strategies must respect these boundaries. A simple synchronization calculation makes the bandwidth gaps concrete.
+ @Fig-infra-bandwidth-hierarchy makes the order-of-magnitude jumps between zones visible: each concentric boundary crossed drops bandwidth roughly tenfold and raises latency roughly tenfold, so a synchronization that takes 1 ms over NVLink takes 10 to 20 ms over InfiniBand. Those latency numbers are not an inconvenience; at training step times of tens to hundreds of milliseconds, a 10 to 20 ms AllReduce per layer would idle the accelerators for the majority of each step. As @tbl-bandwidth-hierarchy-compute confirms, this is why tensor parallelism, which requires an AllReduce after every layer, is physically confined to the intra-node domain. A simple synchronization calculation makes the bandwidth gaps concrete.
```

---

### 5. `tbl-gpu-evolution` (tbl 🟠) — def L451

**Ref sentence (L453):**
> "@Tbl-gpu-evolution compresses four hardware generations into a few columns."

**Missing move:** Interpret. The sentence is a bare table-pointer with no takeaway. The key result the table encodes — that each generation targeted the dominant ML workload pattern of its era, that efficiency improvements are substantial but uneven, and that NVLink bandwidth doubled between generations except for the 1.5x A100-to-H100 step — lives entirely in the caption. The prose immediately pivots to the figure ("@Fig-accelerator-efficiency-wall unpacks two of those columns") without stating what the reader should conclude from the generational comparison.

**Rule-compliant diff rewrite — replace L453 with:**

```diff
- @Tbl-gpu-evolution compresses four hardware generations into a few columns. @Fig-accelerator-efficiency-wall unpacks two of those columns, raw throughput and power efficiency, to reveal a divergence that shapes fleet-scale infrastructure decisions.
+ @Tbl-gpu-evolution compresses four hardware generations into a few columns, and the pattern in those columns is not uniform progress: each generation's key innovation tracks the dominant ML workload of its era (V100 introduced Tensor Cores for mixed-precision training; H100 added the Transformer Engine and FP8 for attention-heavy LLMs; B200 used a dual-die chiplet to break the reticle limit). NVLink bandwidth doubled between most generations but grew only 1.5x from A100 to H100, a step that constrained tensor-parallel scaling for that generation. @Fig-accelerator-efficiency-wall unpacks two of those columns, raw throughput and power efficiency, to reveal a divergence that shapes fleet-scale infrastructure decisions.
```

---

### 6. `tbl-hbm-comparison` (tbl 🟠) — def L770

**Ref sentence (L772):**
> "As @tbl-hbm-comparison shows, this bandwidth advantage comes at a price."

**Missing move:** Interpret (structural takeaway missing). The prose draws the cost consequence (price per GB, manufacturing cost per accelerator), which is one row of the table. However, the table's headline argument — that three simultaneous innovations (3D die stacking, TSV interconnects, on-package placement) each contribute to bandwidth, and that the table's "Scaling Factor" column quantifies the combined 17x bandwidth advantage — is left entirely to the caption. The reader who skips the caption does not learn *why* HBM is 17x faster or which of the three innovations contributes what. The prose treats the table as a price-disclosure device rather than interpreting its bandwidth-mechanism argument.

**Rule-compliant diff rewrite — replace L772 with:**

```diff
- As @tbl-hbm-comparison shows, this bandwidth advantage\index{HBM!bandwidth} comes at a price.
+ @Tbl-hbm-comparison quantifies three simultaneous innovations that together produce the 17x bandwidth advantage: 3D die stacking multiplies the bits per package, TSV interconnects cut signal paths from centimeters to micrometers (reducing the energy per bit by 4 to 10x), and on-package placement eliminates the socketed DIMM's long PCB trace entirely. No single innovation achieves the gap; all three operate together. The cost follows directly from the complexity: HBM costs approximately \$10--15 per GB, compared to roughly \$3 per GB for DDR5 server memory, and for an H100 with 80 GB of HBM3, the memory alone represents approximately \$800--1,200 of the accelerator's manufacturing cost. For a B200 with 192 GB of HBM3e, the memory cost rises to \$1,920--2,880 per accelerator, making HBM one of the most expensive components in the system.
```

---

### 7. `tbl-precision-throughput` (tbl 🟠) — def L1959

**Ref sentence (L1935):**
> "The precision landscape for ML has evolved rapidly. @Tbl-precision-throughput summarizes representative precision formats on H100-class accelerators and their common use cases."

**Missing move:** Interpret at cite. The cite sentence is a bare "summarizes" announcer. The table's key result is a 7.4x throughput multiplier from FP32 to FP8 (67 to 1,979 TFLOP/s), which determines fleet sizing. This insight appears in the payoff paragraph at L1961, but the cite sentence itself adds nothing. By the standard, the Interpret move should accompany the citation, not be deferred to a paragraph after the table. The payoff is strong; only the cite sentence fails.

**Rule-compliant diff rewrite — replace L1935 with:**

```diff
- The precision landscape for ML has evolved rapidly. @Tbl-precision-throughput summarizes representative precision formats on H100-class accelerators and their common use cases.
+ The precision landscape for ML has evolved rapidly, and the throughput differences between formats are not incremental. @Tbl-precision-throughput shows that moving from FP32 (67 TFLOP/s dense) to FP8 (roughly 1,979 TFLOP/s) represents a 30x arithmetic throughput gain on the same H100 silicon; BF16 and FP16 sit at roughly 989 TFLOP/s, a 15x gain over FP32. The reduced mantissa bits are the mechanism: fewer bits per number means more numbers fit the same memory bandwidth budget and more operations complete per clock cycle. Whether a given model can use the lower precision formats without accuracy loss is a training-dynamics question, but the infrastructure implication is fixed regardless of model: precision choice is one of the highest-leverage decisions in cluster sizing.
```

---

### 8. `tbl-memory-breakdown` (tbl 🟠) — def L2774

**Ref sentence (L2717, first citation):**
> "As @tbl-memory-breakdown shows, for our `{python}` model, the training memory breaks down as follows:"

**Missing move:** Interpret at first cite. The sentence introduces the table with a colon that leads into a downstream list, functioning as a float-announcer rather than a takeaway. The actual takeaway — that the Adam optimizer state in FP32 dominates the memory budget, not the FP16 weights, and that strategies targeting only weights leave the largest consumers untouched — appears in the second citation at L2792. The first cite is a structural pointer that adds no insight. Anyone who reads the first cite and skips the table learns nothing.

**Rule-compliant diff rewrite — replace the first-cite sentence at L2717 with:**

```diff
- As @tbl-memory-breakdown shows, for our `{python} InfraFrontierNodeBreakdownRecap.frontier_params_b_str` model, the training memory breaks down as follows:
+ The memory budget for training our `{python} InfraFrontierNodeBreakdownRecap.frontier_params_b_str` model has a counterintuitive structure, which @tbl-memory-breakdown makes concrete: the FP16 weights are the smallest component. The Adam optimizer alone contributes first and second moments in FP32, each as large as the FP16 weight copy but in four-byte floats, making the optimizer state the dominant term. Activations add a variable peak that scales with batch size and sequence length. A parallelism strategy that shards only the weights, such as naive model parallelism, therefore leaves the bulk of the memory pressure on every device.
```

---

## Dangling refs (scanner-reported, not findings)

The scanner flagged three forward references with no matching definition in this file:

- L889: `@eq-fleet-arithmetic-intensity` and `@eq-fleet-ridge-point` (defined in `@Sec-appdx-fleet-foundations-roofline`)
- L1683: `@eq-pue` (expected definition not found in this chapter)
- L3716: `@Eq-distributed-training-scaling-efficiency` (forward ref to a later chapter)

These are cross-chapter refs, not float-exposition issues. No action required here.
