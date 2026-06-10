# Verified findings — training.qmd (vol1)
Prior findings: 3 | Survived: 1 | Refuted: 2

## SURVIVING findings (genuinely under-explained — keep rewrite)

### ⚠️ `tbl-scaling-decision` — def L6522
- Ref: "@Tbl-scaling-decision provides quantitative guidance for scaling decisions across different model and data scales."
- Why it survives: The ref sentence is a pure pointer with no content. The prev paragraph (L6506-6511) lists four single-machine optimizations to exhaust first but names no thresholds. The caption states the organizing principle but does not unpack any specific value. The payoff paragraph (L6524+) explains three hard limits (memory exhaustion, wall-clock time, dataset scale) in narrative terms but never translates the table's specific numeric brackets into prose — the reader is not told that 1-10B fits on a single multi-GPU node and why ("Model parallelism within node avoids network"), nor that sub-1B fits on a single GPU, nor what drives the >10 TB dataset threshold. The table rows carry all of the substantive guidance; nothing in the surrounding text primes the reader on what to look for or what the threshold values mean.
- Suggested rewrite (rule-compliant; no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - @Tbl-scaling-decision provides quantitative guidance for scaling decisions across different model and data scales.
  + @Tbl-scaling-decision translates these limits into a practical lookup: models below one billion parameters fit on a single GPU with the optimizations above, models in the 1-10 billion range fit on a single multi-GPU node (keeping high-speed intra-node interconnect rather than the slower inter-node fabric), and only models above 10 billion parameters or datasets above 10 TB justify multi-node distributed complexity.
  ```

---

## REFUTED findings (explanation found in neighborhood — drop)

- `fig-data-pipeline` — REFUTED: explanation is in payoff paragraphs L2287-2289, only two lines after the figure's closing `:::` at L2285. The first pass incorrectly reported this payoff as "~250 lines below the figure" (it confused the section-level payoff reference at L2287 with a much later passage). The actual payoff reads: "These zones matter because each can become the slowest stage. Storage supplies raw examples from disk... CPU preprocessing then converts formats, applies resizing, normalization, or data augmentation, and batches examples into tensors the accelerator can consume... Format conversion, processing, and batching are therefore not housekeeping steps; they are throughput gates. If any one runs slower than the training loop, expensive accelerator resources idle while the data pipeline catches up." This fully explains what each zone shows and why the architecture matters.

- `tbl-optimization-roadmap` — REFUTED: explanation is distributed across prev paragraph L3435 and payoff paragraph L3446. The prev paragraph immediately before the table explicitly characterizes each bottleneck type that the table rows address: "Data movement latency emerges when training batches cannot flow from storage through preprocessing to compute units fast enough to keep accelerators in use. Computational throughput limitations occur when mathematical operations execute below hardware peak performance due to suboptimal precision choices or kernel inefficiencies. Memory capacity constraints restrict both the model sizes and batch sizes we can process." This directly primes the reader on what each table row means. The payoff paragraph then closes with the GPT-2 concrete example showing which combination of techniques addresses real-world multi-constraint profiles. Taken together, the neighborhood provides the reader with both what the table shows and why the mapping matters; the ref sentence's structural framing is adequately rescued by the surrounding prose.
