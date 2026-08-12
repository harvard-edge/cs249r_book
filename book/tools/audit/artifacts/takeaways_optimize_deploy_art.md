# Takeaways ART: Optimize Deploy

## book/quarto/contents/vol1/data_selection/data_selection.qmd
- Recommendation: Keep
- Evidence reviewed: Purpose; full heading outline from Data Selection Fundamentals through Fallacies and Pitfalls; Summary; existing `callout-takeaways`; nearby Chinchilla diagnostic, diminishing-returns figure, and fallacies/pitfalls calculation cells.
- Issues: None
- Proposed callout: Existing title `Curate, do not accumulate`; no replacement needed.

## book/quarto/contents/vol1/model_compression/model_compression.qmd
- Recommendation: Modify
- Evidence reviewed: Purpose; full heading outline from Optimization Framework through Implementation Tools and Fallacies and Pitfalls; Summary; existing `callout-takeaways`; nearby BERT compression, Amdahl ceiling, and ResNet-50 INT8 measurement cells.
- Issues:
  - Several bullets are topic labels or commands rather than durable claims.
  - Multiple bullets are below the 25-word target and under-explain the engineering consequence.
  - The current block underuses central quantitative anchors: the BERT compression ratio, the 8x-to-1.5x theory/practice gap, and the 20 percent inference-fraction Amdahl ceiling.
- Proposed callout:

```qmd
::: {.callout-takeaways title="From benchmark winner to production model"}

* **Compression spends surplus capacity**: Production models trade unused parameters, precision, and capacity for latency, memory, power, or cost limits the deployment cannot violate. The right target is not minimum size, but the smallest artifact that preserves the behavior the context actually needs.
* **Savings multiply only when aligned**: Structural pruning, distillation, quantization, and architecture changes can compound, as the BERT mobile pipeline's `{python} BertCompression.compression_ratio_mult_str` footprint reduction shows. The gain becomes real only when the resulting operators match the target runtime and accelerator.
* **Precision is a deployment contract**: INT8 post-training quantization offers a strong first move because it can deliver 4$\times$ storage reduction without retraining, while QAT, distillation, or mixed precision buy back accuracy when calibration or layer sensitivity exposes unacceptable error.
* **Hardware sets the exchange rate**: Unstructured sparsity and theoretical FLOP cuts rarely help commodity GPUs unless kernels and memory layouts can exploit them. The chapter's warning that an 8$\times$ paper speedup can collapse to about 1.5$\times$ makes target-hardware profiling mandatory.
* **End-to-end latency caps model wins**: Compression is valuable only on the critical path. When inference is 20 percent of total request latency, Amdahl's Law caps even perfect model acceleration at 1.25$\times$, so preprocessing, dispatch, and data movement may be the true optimization target.

:::
```

## book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd
- Recommendation: Keep
- Evidence reviewed: Purpose; full heading outline from Acceleration Fundamentals through Compiler Support, Runtime Support, and Fallacies and Pitfalls; Summary; existing `callout-takeaways`; nearby summary roofline recap and hardware feasibility/fallacies calculation cells.
- Issues: None
- Proposed callout: Existing title `Moving data costs more than computing it`; no replacement needed.

## book/quarto/contents/vol1/benchmarking/benchmarking.qmd
- Recommendation: Modify
- Evidence reviewed: Purpose; full heading outline from ML Benchmarking Framework through Production Considerations and Fallacies and Pitfalls; Summary; existing `callout-takeaways`; nearby precision-energy summary anchor and fallacies/pitfalls setup cells.
- Issues:
  - Several bullets are compressed into labels rather than consequences.
  - The block repeats framework categories without fully stating what changes in engineering judgment.
  - Quantitative anchors from the chapter arc, including the 2-10x benchmark-production gap and MobileNet INT8 energy reduction, should be better integrated.
- Proposed callout:

```qmd
::: {.callout-takeaways title="Measuring what matters"}

* **Benchmarks validate co-design**: System, model, and data benchmarks expose different failures: hardware underdelivery, compression quality loss, and distribution mismatch. A system that passes only one axis can still fail when Data, Algorithm, and Machine constraints meet under production load.
* **Proxy numbers need boundaries**: MLPerf-style run rules make comparisons honest, but standardized workloads are still proxies. Batch size, thermal state, input distribution, concurrency, and SLA windows decide whether a lab result survives the 2--10$\times$ benchmark-production gap.
* **Granularity trades diagnosis for realism**: Micro-benchmarks isolate kernels, macro-benchmarks expose model-level costs, and end-to-end benchmarks capture user-visible behavior. Effective measurement stacks all three so teams can see both the symptom and the layer that caused it.
* **Tail latency is the benchmark**: Interactive systems fail at p95 and p99 before averages move. Reporting percentile latency under representative load prevents a benchmark from approving a system whose mean passes while its worst-served requests violate the SLO.
* **Amdahl caps every optimization claim**: A faster model cannot outrun the rest of the pipeline; if preprocessing is 50 percent of latency, an infinitely fast model yields only a 2$\times$ system improvement. Benchmark the whole request path before celebrating kernel speedups.
* **Efficiency still needs quality evidence**: INT8 may cut memory 4$\times$ and reduce MobileNet inference energy by about `{python} PrecisionEnergySummary.total_savings_factor_mult_str`, but calibration, subgroup robustness, and edge-case behavior decide whether the compressed model is deployable.

:::
```

## book/quarto/contents/vol1/model_serving/model_serving.qmd
- Recommendation: Modify
- Evidence reviewed: Purpose; full heading outline from Serving Paradigm through Economics and Planning and Fallacies and Pitfalls; Summary; existing `callout-takeaways`; nearby Llama 3 serving economics and serving fallacy/pitfall calculation cells.
- Issues:
  - A few bullets are below the target length and read like section summaries.
  - The current block covers the right topics but can state the latency-budget, queuing, pipeline-tax, and cost consequences more memorably.
  - Runtime and unit-economics implications should remain explicit because they are central to the chapter's deployment arc.
- Proposed callout:

```qmd
::: {.callout-takeaways title="Inverting every training priority"}

* **Serving is latency economics**: Training rewards throughput over long runs, but serving spends a fixed per-request budget across serialization, preprocessing, queuing, inference, postprocessing, and the network. Optimizing only model latency misses the stages users actually wait on.
* **Utilization turns into waiting**: Queuing theory makes capacity planning nonlinear: at 80 percent utilization, average time in system is 5$\times$ service time; at 90 percent, it reaches 10$\times$. Cost-efficient headroom keeps modest traffic surges from becoming SLO failures.
* **Fast models reveal pipeline taxes**: Once inference falls to roughly 5 ms, image decode, tokenization, and other preprocessing can consume 45-70 percent of total latency. The binding optimization becomes the request path, not the neural network kernel.
* **Batching follows traffic, not habit**: Poisson web arrivals can use dynamic batching, synchronized sensors need aligned batches, and single-user mobile workloads often cannot batch at all. The right batching window converts slack into throughput without spending the latency budget.
* **Skew breaks accuracy without errors**: Resize methods, normalization order, calibration data, or feature definitions that differ between training and serving shift live inputs outside the learned distribution. Reusing identical code paths and monitoring production slices prevents silent degradation.
* **LLM serving is memory management**: Decode often reads weights from VRAM per token, so TPOT is bandwidth-bound unless batching changes the constraint. KV-cache layout, PagedAttention, continuous batching, precision, and runtime choice determine both concurrency and cost per token.
* **Runtime choices become infrastructure bills**: Precision, graph compilation, operator fusion, and serving runtime translate directly into replica count and cost per inference. INT8 can deliver about 3$\times$ FP32 throughput, while TensorRT or ONNX Runtime often adds 2--5$\times$ over framework-native serving.

:::
```

## book/quarto/contents/vol1/ml_ops/ml_ops.qmd
- Recommendation: Keep
- Evidence reviewed: Purpose; full heading outline from MLOps Overview through Case Studies and Fallacies and Pitfalls; Summary; existing `callout-takeaways`; nearby Oura recap, single-model ROI, validation-gap, and operational accuracy calculation cells.
- Issues: None
- Proposed callout: Existing title `Perfectly available, perfectly wrong`; no replacement needed.

## book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd
- Recommendation: Modify
- Evidence reviewed: Purpose; full heading outline from Responsibility as Systems Engineering through Data Governance and Compliance and Fallacies and Pitfalls; Summary; existing `callout-takeaways`; nearby fairness pitfall and TCO recap cells.
- Issues:
  - Several bullets are too terse, especially checklist and documentation items.
  - The block can be consolidated from seven bullets to five stronger system-level lessons.
  - The proposed version should keep fairness disparity, TCO, monitoring, and governance as measurable engineering constraints.
- Proposed callout:

```qmd
::: {.callout-takeaways title="Reliable for whom?"}

* **Aggregate correctness hides harm**: A model can report 95 percent accuracy while producing `{python} GenderShadesDisparity.disparity_factor_mult_str` error-rate disparities across demographic groups. Responsible evaluation therefore starts with disaggregated and intersectional slices, not aggregate accuracy alone.
* **Responsibility becomes testable through thresholds**: "Be fair" is not testable, but bounded disparity, documented intended use, and explainability requirements are. Translating values into measurable constraints lets teams place fairness, accuracy, latency, and cost on the same reviewable Pareto frontier.
* **Efficiency is a social constraint**: A 4$\times$ more efficient model uses 4$\times$ less energy, costs 4$\times$ less, and broadens who can deploy it. Because inference dominates TCO by `{python} ResponsibleTcoRecap.inf_train_ratio_str`:1 over training, per-query optimization is responsible engineering.
* **Monitoring must watch outcomes**: Bias and privacy failures can continue with green uptime dashboards because harmful predictions look operationally normal. Production monitoring must track subgroup outcomes, data lineage, feedback loops, and incident paths with the same rigor as latency regressions.
* **Governance has to be built in**: Model cards, datasheets, access controls, erasure workflows, human-review paths, and audit trails are technical infrastructure. Regulations such as GDPR require capabilities that cannot be retrofitted after a pipeline is already serving decisions.

:::
```

## book/quarto/contents/vol1/conclusion/conclusion.qmd
- Recommendation: Modify
- Evidence reviewed: Purpose; full heading outline from Synthesizing ML Systems through Thirteen Quantitative Invariants, Principles in Practice, Future Directions, Journey Forward, Fallacies and Pitfalls, and Summary; existing `callout-takeaways`; nearby roofline, tail-latency, edge-vs-H100, fleet-failure, and Amdahl calculation cells.
- Issues:
  - As a conclusion, the block should default to four synthesis takeaways rather than seven narrower bullets.
  - Some existing bullets are too short or repeat chapter summary language.
  - Nearby quantitative anchors should be used to make the final systems lens more concrete.
- Proposed callout:

```qmd
::: {.callout-takeaways title="Reasoning across boundaries"}

* **Invariants outlast implementations**: The thirteen quantitative invariants turn ML systems from framework-specific craft into a discipline of measurable constraints. Data, algorithms, and machines change, but memory movement, latency budgets, drift, and responsibility keep governing design.
* **Complexity only moves**: Compression, batching, monitoring, and governance do not destroy complexity; they relocate it across data, algorithm, and machine. The conservation of complexity is the common reason local optimizations become another layer's constraint.
* **Boundaries reveal the bottleneck**: A 70-billion-parameter Llama 2 can be about 295$\times$ memory-bound on H100, and p99 latency can sit 40$\times$ above the mean. Systems thinking means measuring where physics, traffic, and users bind.
* **Scale changes the binding term**: Earlier chapters derive invariants where one machine's memory, bandwidth, and power bind; fleet-scale chapters ask what happens when a thousand-GPU pool turns multi-year component MTTF into sub-day cluster MTBF. The physics stays, but the constraint moves to fleets.

:::
```
