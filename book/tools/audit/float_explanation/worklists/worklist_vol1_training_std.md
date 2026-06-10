# Float-exposition worklist — training.qmd (vol1) · Standard pass

Rubric: FLOAT_EXPOSITION_STANDARD.md
Scanner: scan_floats.py --format bundle
Chapter: vol1/training/training.qmd
Total floats: 65 (3 algorithms, 18 equations, 19 figures, 7 listings, 18 tables)

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|---|
| Algorithm | 🔴 Strict | 3 | 3 | 0 | 0 |
| Equation | 🔴 Strict | 18 | 17 | 1 | 0 |
| Figure | 🟠 High | 19 | 18 | 1 | 0 |
| Listing | 🟡 Medium | 7 | 6 | 1 | 0 |
| Table | 🟠 High | 18 | 14 | 4 | 0 |
| **Total** | | **65** | **58** | **7** | **0** |

---

## Findings (⚠️ and 🛑 only)

---

### ⚠️ `eq-buffer-memory` — def L2538 · Equation / 🔴 Strict

**Caption:** (none)

**Verbatim ref (L2537):**
> To keep accelerators fed despite this bandwidth reduction, pipeline architectures maintain multiple data buffers simultaneously---prefetch buffers loading future batches, processing buffers holding data under transformation, and transfer buffers staging data for accelerator consumption. The total host memory required scales with the per-batch memory footprint $M_{\text{batch}}$ according to @eq-buffer-memory:

**Context:** The preceding sentence names all three buffer types. The equation follows immediately. The payoff paragraph (L2540) pivots directly to `eq-pipeline-condition` without ever stating what the buffer-memory equation implies — specifically, what the practical buffer count or memory cost is for real pipelines, or what the design constraint is for choosing $N_{\text{prefetch}}$, $N_{\text{processing}}$, and $N_{\text{transfer}}$.

**Missing move:** The Interpret move is absent. The equation expresses that total buffer memory is additive across the three buffer types scaled by batch size, but the prose never states the consequence: what typical values of these counts look like, or what the so-what is for a practitioner sizing host memory. Without that, the equation is a definition with no actionable implication.

**Rule-compliant diff rewrite:**
```diff
- To keep accelerators fed despite this bandwidth reduction, pipeline architectures maintain
- multiple data buffers simultaneously---prefetch buffers loading future batches, processing
- buffers holding data under transformation, and transfer buffers staging data for accelerator
- consumption. The total host memory required scales with the per-batch memory footprint
- $M_{\text{batch}}$ according to @eq-buffer-memory:
+ To keep accelerators fed despite this bandwidth reduction, pipeline architectures maintain
+ multiple data buffers simultaneously: prefetch buffers loading future batches, processing
+ buffers holding data under transformation, and transfer buffers staging data for accelerator
+ consumption. The total host memory required scales with the per-batch memory footprint
+ $M_{\text{batch}}$ according to @eq-buffer-memory, where a typical three-stage pipeline
+ (one prefetch buffer, one processing buffer, one transfer buffer) multiplies the per-batch
+ footprint by three. For a 256 MB per-batch workload that means roughly 768 MB of host
+ memory committed purely to pipeline buffering, before any model state.
```

---

### ⚠️ `tbl-compare-activations` — def L750 · Table / 🟠 High

**Caption:** **Activation Function Systems Comparison**: While activation functions contribute only a fraction of total training time, their implementation characteristics (computational complexity, hardware utilization, and memory patterns) significantly impact the efficiency of modern learning pipelines.

**Verbatim ref (L741):**
> @Tbl-compare-activations synthesizes these system-level trade-offs, showing *how* mathematical behavior translates into operational constraints.

**Context:** The preceding prose (L737) is rich, quantifying ReLU peak utilization, sigmoid hardware cost, and sparsity benefits. The payoff (L752) says "ReLU is the default choice for large-scale networks due to its efficiency and scalability. Softmax remains indispensable for classification tasks." That is a high-level conclusion without extracting the table's load-bearing contrast.

**Missing move:** The citation is a float-announcer ("synthesizes... showing") with no takeaway. The payoff states the conclusion ("ReLU is default") but not the specific system implication that drives it, which is that Softmax's global normalization creates non-local memory dependencies that prevent element-wise parallelization — making it structurally different from the others, not just slower. The table's key insight (the architectural discontinuity between element-wise functions and Softmax's all-to-all dependency) never reaches the prose.

**Rule-compliant diff rewrite:**
```diff
- @Tbl-compare-activations synthesizes these system-level trade-offs, showing *how*
- mathematical behavior translates into operational constraints.
+ The system-level contrast in @tbl-compare-activations divides along a structural line:
+ Sigmoid, Tanh, ReLU, and GELU all operate element-wise, letting hardware compute each
+ output independently in parallel, while Softmax requires a global reduction across the
+ entire input vector before any output can be produced. This non-local dependency is
+ what makes Softmax memory-intensive at sequence length and is precisely the pattern
+ that FlashAttention restructures through tiling.
```

---

### ⚠️ `fig-data-pipeline` — def L2037 · Figure / 🟠 High

**Caption:** **CPU-to-GPU Data Flow**: Three distinct zones compose the data pipeline: the storage zone houses raw data on disk, the CPU preprocessing zone handles format conversion, processing, and batching, and the GPU training zone distributes preprocessed batches across multiple accelerator workers for parallel computation.

**Verbatim ref (L2035):**
> The data pipeline running on the CPU bridges raw data storage and accelerator computation. @Fig-data-pipeline breaks down this architecture into three distinct zones.

**Context:** The preceding paragraph (L2033) establishes that data pipeline efficiency determines whether accelerators remain engaged or idle. The citation sentence announces structure ("breaks down into three distinct zones") without delivering any takeaway. The payoff (L2287) is approximately 250 lines below the float, at the far end of a large TikZ figure, and says "These zones matter because each can become the slowest stage." Readers encounter the figure with no guidance on what to look for or why.

**Missing move:** The Interpret move is deferred so far that it cannot serve as payoff for the immediate citation. The citation must tell the reader what the figure demonstrates, not merely what it depicts.

**Rule-compliant diff rewrite:**
```diff
- The data pipeline running on the CPU bridges raw data storage and accelerator computation.
- @Fig-data-pipeline breaks down this architecture into three distinct zones.
+ The data pipeline running on the CPU bridges raw data storage and accelerator computation.
+ @Fig-data-pipeline breaks down this architecture into three distinct zones, each of which
+ can become the slowest stage: storage supplies raw examples, CPU preprocessing converts
+ and batches them, and the GPU training zone consumes the result. When any zone runs slower
+ than its neighbors, the accelerator idles while the pipeline catches up.
```

---

### ⚠️ `tbl-optimization-roadmap` — def L3444 · Table / 🟠 High

**Caption:** **Optimization Technique Roadmap**: Each primary bottleneck category has targeted solutions that address specific performance constraints, matching techniques to profiling results for systematic optimization.

**Verbatim ref (L3433):**
> @Tbl-optimization-roadmap extends the D·A·M-based bottleneck classification from @tbl-dam-training-bottlenecks by mapping each bottleneck to the specific optimization technique that addresses it.

**Context:** The preceding paragraph (L3431) establishes the utilization gap. The payoff (L3446) shifts immediately to the conservation-of-complexity principle and a GPT-2 profiling example, without first stating what the table's mapping reveals. The citation gives only a structural description ("extends... by mapping") with no insight about which technique goes with which bottleneck.

**Missing move:** The load-bearing contrast — that different bottleneck types require non-interchangeable techniques, and that reaching for the wrong tool wastes effort — is never stated. The table's practical value as a lookup (profiling reveals data-bound → prefetching, memory-bound → mixed precision/fusion, compute-bound → FlashAttention or hardware) is left entirely to the cells.

**Rule-compliant diff rewrite:**
```diff
- @Tbl-optimization-roadmap extends the D·A·M-based bottleneck classification from
- @tbl-dam-training-bottlenecks by mapping each bottleneck to the specific optimization
- technique that addresses it.
+ @Tbl-optimization-roadmap translates the D·A·M bottleneck classification into a
+ direct practitioner lookup: data-bound systems need prefetching and pipeline overlap,
+ memory-bound systems need operator fusion and reduced precision, and compute-bound
+ systems respond to FlashAttention, mixed precision, and faster hardware. Reaching for
+ the wrong column wastes engineering effort while the actual bottleneck persists.
```

---

### ⚠️ `lst-flash-attention-comparison` — def L4631 · Listing / 🟡 Medium

**Caption:** **Attention Implementation Comparison**: Standard attention materializes the full $S{\times}S$ matrix in HBM, while Flash Attention uses PyTorch's optimized implementation or the dedicated flash-attn library.

**Verbatim ref (first citation, L4629):**
> Flash Attention's performance gains materialize through careful exploitation of GPU memory hierarchy. Modern frameworks integrate these optimizations transparently, automatically selecting the most efficient attention implementation based on hardware capabilities and input characteristics. @Lst-flash-attention-comparison contrasts standard and optimized attention implementations.

**Context:** The second citation (L4747) is adequate — it names the specific call (`F.scaled_dot_product_attention`) and the engineering decision (use the optimized primitive when the bottleneck is HBM traffic). The first citation, however, gives only a structural announcement with no mechanism named and nothing for the reader to look for before opening the code.

**Missing move:** For a listing, the prose must deliver "what the code shows — the mechanism it embodies and what the reader should notice." The first citation tells the reader only that two implementations are contrasted, not what the contrast reveals (for instance, that the standard path materializes the full score matrix in a separate `torch.mm` call before softmax, while the optimized path hides all of that behind a single fused primitive).

**Rule-compliant diff rewrite (first citation only):**
```diff
- Flash Attention's performance gains materialize through careful exploitation of GPU memory
- hierarchy. Modern frameworks integrate these optimizations transparently, automatically
- selecting the most efficient attention implementation based on hardware capabilities and
- input characteristics. @Lst-flash-attention-comparison contrasts standard and optimized
- attention implementations.
+ Flash Attention's performance gains materialize through careful exploitation of GPU memory
+ hierarchy. Modern frameworks integrate these optimizations transparently, automatically
+ selecting the most efficient attention implementation based on hardware capabilities and
+ input characteristics. In @Lst-flash-attention-comparison, the key difference to notice is
+ in the standard path: the explicit `torch.mm(q, k.T)` call materializes the full $S\times S$
+ score matrix in HBM before softmax can run, whereas the optimized path replaces the entire
+ sequence with `F.scaled_dot_product_attention`, which delegates tiling and SRAM scheduling
+ to the framework's fused kernel and never writes that intermediate matrix to HBM.
```

---

### ⚠️ `tbl-flashattention-benchmarks` — def L4731 · Table / 🟠 High

**Caption:** **FlashAttention Benchmark Comparison**: Illustrative per-call timing and peak memory for standard attention vs. FlashAttention on a 40 GB A100-style configuration across sequence lengths. OOM marks configurations where standard attention exceeds the 40 GB memory budget; the 8192-token row also shows that standard attention would exceed 80 GB.

**Verbatim ref (L4671):**
> The benefits of Flash Attention become concrete when measured on real hardware. @dao2022 reports end-to-end GPT-style training speedups and separate attention-kernel benchmarks showing that IO-aware attention reduces memory traffic and improves runtime. @Tbl-flashattention-benchmarks uses an illustrative A100-style scenario to show the same systems pattern; its timings and memory values are representative chapter numbers, not values reported verbatim by Dao et al.

**Context:** The payoff paragraph (L4735) does not interpret the current table. It pivots immediately to FlashAttention-2 and FlashAttention-3 speedup numbers. The table's own data — specifically the OOM boundary (where standard attention runs out of memory but FlashAttention does not) and the per-sequence-length scaling pattern — are never stated in prose.

**Missing move:** For a table, the prose must deliver "the takeaway the table encodes — the load-bearing contrast, the specific row(s) that matter, or the decision the table drives." Neither the citation sentence nor the payoff names the OOM threshold, which is the most important row in the table. The citation acknowledges the table is illustrative but does not tell the reader what the illustration shows.

**Rule-compliant diff rewrite:**
```diff
- The benefits of Flash Attention become concrete when measured on real hardware. @dao2022
- reports end-to-end GPT-style training speedups and separate attention-kernel benchmarks
- showing that IO-aware attention reduces memory traffic and improves runtime.
- @Tbl-flashattention-benchmarks uses an illustrative A100-style scenario to show the same
- systems pattern; its timings and memory values are representative chapter numbers, not
- values reported verbatim by Dao et al.
+ The benefits of Flash Attention become concrete when measured on real hardware. @dao2022
+ reports end-to-end GPT-style training speedups and separate attention-kernel benchmarks
+ showing that IO-aware attention reduces memory traffic and improves runtime.
+ @Tbl-flashattention-benchmarks uses an illustrative A100-style scenario to show the same
+ systems pattern. The key rows are the OOM entries: at sequence lengths above roughly 2048
+ tokens, standard attention exhausts the 40 GB budget entirely, while FlashAttention fits
+ the same computation within budget at every row. That is not a speedup story — it is a
+ feasibility boundary. The table's timings and memory values are representative chapter
+ numbers, not values reported verbatim by Dao et al.
```

---

### ⚠️ `tbl-scaling-decision` — def L6522 · Table / 🟠 High

**Caption:** **Scaling Decision Guidelines**: Model size, dataset scale, and available hardware determine when distributed training complexity is justified. Single-machine optimization provides better cost-efficiency below these thresholds.

**Verbatim ref (L6513):**
> @Tbl-scaling-decision provides quantitative guidance for scaling decisions across different model and data scales.

**Context:** The preceding paragraph (L6508-6511) lists the four single-machine optimizations to exhaust first. The payoff (L6524) explains when distributed training becomes necessary through three hard limits. Neither paragraph unpacks the table's specific thresholds (sub-1B on single GPU, 1-10B on single node, above-10B or above-10 TB for multi-node) or the rationale behind them.

**Missing move:** The citation is a bare pointer with no content. The table's actual guidance — particularly why the 1-10B tier fits on a single multi-GPU node (intra-node NVLink avoids the slower inter-node fabric) — is never stated in prose. The reader must infer the rationale from the cells alone.

**Rule-compliant diff rewrite:**
```diff
- @Tbl-scaling-decision provides quantitative guidance for scaling decisions across different
- model and data scales.
+ @Tbl-scaling-decision translates these limits into a practical lookup: models below one
+ billion parameters fit on a single GPU with the optimizations above, models in the 1-10B
+ range fit on a single multi-GPU node (keeping high-bandwidth intra-node NVLink rather than
+ the slower inter-node fabric), and only models above 10B or datasets above 10 TB push
+ complexity into multi-node territory where communication overhead and fault tolerance
+ become first-order concerns.
```

---

## ✅ floats not expanded (pass)

**Algorithms (3/3 pass):** `alg-adam-update`, `alg-gradient-checkpointing`, `alg-streaming-attention`

**Equations (17/18 pass):** `eq-training-iron-law`, `eq-gd-memory`, `eq-gd-time`, `eq-batch-memory-decomposition`, `eq-activation-memory-per-batch`, `eq-total-training-memory`, `eq-storage-throughput`, `eq-preprocess-throughput`, `eq-training-bottleneck`, `eq-system-throughput`, `eq-gpu-utilization`, `eq-memory-hierarchy-bandwidth`, `eq-iteration-time`, `eq-pipeline-condition`, `eq-attention`, `eq-gradient-accumulation-equivalence`, `eq-gradient-accumulation-overhead`

**Figures (18/19 pass):** `fig-activation-perf`, `fig-training-roofline`, `fig-training-pipeline`, `fig-training-loop`, `fig-galore-llm-memory-breakdown`, `fig-linear-scaling-failure`, `fig-tf-bottleneck-trace`, `fig-optimization-flowchart`, `fig-fetching-naive`, `fig-fetching-optimized`, `fig-mixed-precision`, `fig-grad-accumulation`, `fig-activation-checkpointing`, `fig-communication-tax`, `fig-train-data-parallelism`, `fig-model-parallelism`, `fig-layers-blocks`, `fig-evolution-systems`

**Listings (6/7 pass):** `lst-gelu-approx`, `lst-adam-training`, `lst-param_update`, `lst-dataloader_usage`, `lst-mixed-precision`, `lst-gradient-accumulation-loop`

**Tables (14/18 pass):** `tbl-iron-law-mapping`, `tbl-training-gpt2-lighthouse-specs`, `tbl-optimizer-properties`, `tbl-training-arithmetic-intensity`, `tbl-training-wave-quantization`, `tbl-dam-training-bottlenecks`, `tbl-precision-comparison`, `tbl-precision-decision-tree`, `tbl-hw-precision-strategy`, `tbl-checkpoint-tradeoffs`, `tbl-optimization`, `tbl-training-gpt2-final-profile`, `tbl-gpt2-summary`, `tbl-computing-eras`
