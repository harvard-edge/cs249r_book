# Takeaways ART: Foundations

## book/quarto/contents/vol1/introduction/introduction.qmd
- Recommendation: Modify
- Evidence reviewed: Purpose, learning objectives, clean heading outline, Summary section, `efficiency-scale-summary-anchor` and Amdahl summary recap cells, existing `callout-takeaways` block.
- Issues:
  - Seven bullets are allowed, but this is an introductory chapter and the rule recommends a smaller synthesis for introductions.
  - The five-pillar and lifecycle bullets read partly as coverage inventory rather than durable results.
  - The central quantitative anchors are strong and should be retained in a tighter block.
- Proposed callout:

```qmd
::: {.callout-takeaways title="Constraints drive architecture"}

* **D·A·M bottlenecks migrate, not disappear**: Data, Algorithm, and Machine constraints interact, so improving one axis often exposes another. The systems habit is to ask which axis now binds, then choose the intervention that relieves that constraint without creating a larger downstream failure.
* **Learned behavior decays silently**: Traditional software usually fails when code changes; ML systems can degrade while code and infrastructure stay fixed because the world shifts under the training distribution. The degradation equation turns that drift into retraining triggers rather than surprise accuracy loss.
* **The iron law makes latency diagnostic**: Data movement, computation, and overhead all spend from the same time budget. Cutting inference from `{python} AmdahlsPitfallRecap.t_inference_ms_str` to `{python} AmdahlsPitfallRecap.t_inf_new_ms_str` gives only `{python} AmdahlsPitfallRecap.improv_pct_str` improvement when preprocessing (`{python} AmdahlsPitfallRecap.t_pre_ms_str`) and postprocessing (`{python} AmdahlsPitfallRecap.t_post_ms_str`) dominate, so optimize the term that binds end-to-end behavior.
* **Scale wins inside physical limits**: The bitter lesson explains why general methods with more compute displaced hand-crafted systems, but scale only helps when data, architecture, and machine can support it. Efficiency gains of `{python} EfficiencyScaleSummary.algo_efficiency_max_mult_str` coexisted with roughly `{python} EfficiencyScaleSummary.compute_growth_order_str` orders of compute growth.
* **AI engineering is continuous co-design**: Deployment context, lifecycle monitoring, and the five engineering pillars are not later add-ons; they are how stochastic learned behavior is held to deterministic reliability targets from cloud training through TinyML operation.

:::
```

## book/quarto/contents/vol1/ml_systems/ml_systems.qmd
- Recommendation: Keep
- Evidence reviewed: Purpose, learning objectives, clean heading outline, Summary section, deployment paradigm and fallacy quantitative anchors surfaced in the chapter, existing `callout-takeaways` block.
- Issues: None
- Proposed callout: Existing title "Same model, different engineering"; no replacement needed.

## book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd
- Recommendation: Modify
- Evidence reviewed: Purpose, learning objectives, clean heading outline, Summary section, constraint-propagation and iteration examples referenced from summary, existing `callout-takeaways` block.
- Issues:
  - Several bullets are below the 25-word target and read like compressed section labels.
  - The block has seven bullets even though the chapter summary points to three dominant workflow patterns.
  - The iron-law and constraint-propagation ideas should be synthesized around D·A·M through time.
- Proposed callout:

```qmd
::: {.callout-takeaways title="See the whole map first"}

* **The lifecycle is a loop, not a checklist**: Data and model pipelines advance in parallel, but production feedback is what makes them a system. Monitoring, validation, and retraining send lessons from deployment back into collection, labeling, architecture, and infrastructure decisions.
* **Late constraints compound exponentially**: A deployment limit found at stage $N_{\text{stage}}$ costs roughly $2^{N_{\text{stage}}-1}$ more than if caught at problem definition. The stage-5 mismatch's `{python} FallaciesConstraintPropagation.cost_factor_mult_str` multiplier is why requirements must flow backward early.
* **Iteration velocity becomes model quality**: In the worked example, a lightweight model starting 5 percentage points behind reaches the 99 percent ceiling because faster cycles create more chances to improve data, architecture, and hyperparameters. Workflow speed compounds into capability.
* **Interfaces make feedback actionable**: Stage contracts define inputs, outputs, and quality invariants so that data, model, and deployment teams can detect violations before integration. Without explicit contracts, each stage can optimize locally while the system fails globally.
* **Production speaks on different clocks**: Real-time inference monitoring, batch retraining triggers, and quarterly model revisions answer different failure modes. Treating all feedback as one loop either reacts too slowly to drift or churns expensive workflows without signal.
* **Workflow is D·A·M through time**: Problem definition fixes constraints, data collection sets $D$ and $D_{\text{vol}}$, model development sets $O$, deployment spends $L_{\text{lat}}$, and monitoring sends violations back into re-optimization. The lifecycle is the spatial D·A·M coupling unfolding over time.

:::
```

## book/quarto/contents/vol1/data_engineering/data_engineering.qmd
- Recommendation: Keep
- Evidence reviewed: Purpose, learning objectives, clean heading outline, Summary section, `DataEngineeringSummaryRecap` and `StorageLoadingRecap` cells, existing `callout-takeaways` block.
- Issues: None
- Proposed callout: Existing title "Data is the source code"; no replacement needed.

## book/quarto/contents/vol1/nn_computation/nn_computation.qmd
- Recommendation: Keep
- Evidence reviewed: Purpose, learning objectives, clean heading outline, Summary section, `mnist-weights-calc` and `summary-paradigm-cost-recap` cells, existing `callout-takeaways` block.
- Issues: None
- Proposed callout: Existing title "The math behind the model"; no replacement needed.

## book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd
- Recommendation: Keep
- Evidence reviewed: Purpose, learning objectives, clean heading outline, Summary section, architecture-selection and lighthouse-model synthesis, existing `callout-takeaways` block.
- Issues: None
- Proposed callout: Existing title "Architecture is infrastructure"; no replacement needed.

## book/quarto/contents/vol1/frameworks/frameworks.qmd
- Recommendation: Keep
- Evidence reviewed: Purpose, learning objectives, clean heading outline, Summary section, execution/differentiation/abstraction synthesis, existing `callout-takeaways` block.
- Issues: None
- Proposed callout: Existing title "The layer between math and hardware"; no replacement needed.

## book/quarto/contents/vol1/training/training.qmd
- Recommendation: Modify
- Evidence reviewed: Purpose, learning objectives, clean heading outline, `Optimization impact summary`, `gpt2-summary-calc`, `OptimizationSummaryCalc`, `GPT2SummaryScalingRecap`, final Summary section, `GPT2SummaryChapterRecap`, existing `callout-takeaways` block.
- Issues:
  - Two bullets are below the 25-word target and read more like slogans than durable takeaways.
  - The profiling, bottleneck, and optimization-composition ideas can be made more diagnostic.
  - The replacement should retain the Adam, FP16, Flash Attention, checkpointing, and GPT-2 memory anchors.
- Proposed callout:

```qmd
::: {.callout-takeaways title="Why training costs millions"}

* **Training cost is an iron-law budget**: $T_{\text{train}} = \frac{O}{R_{\text{peak}} \times \eta_{\text{hw}}}$ makes every optimization accountable: reduce work, raise effective throughput, or improve utilization. A change that misses the dominant term only moves cost around the training loop.
* **Memory determines whether convergence matters**: Adam can reach good solutions in roughly one-third the iterations of SGD, but its per-parameter state costs 3$\times$ extra memory before activations enter the budget. Optimizer choice and batch-size-dependent activations often decide whether training fits at all.
* **Profiles choose the remedy**: The loop is profile, diagnose, fix, and re-profile. Compute-bound jobs benefit from larger batches and mixed precision; memory-bound jobs need checkpointing or smaller state; data- and communication-bound jobs need pipeline or parallelism changes.
* **Precision and IO-aware kernels shift bottlenecks**: FP16 with FP32 accumulation can deliver about 2$\times$ throughput and memory reduction, while Flash Attention avoids materializing the full $S{\times}S$ matrix in HBM and can yield 2--4$\times$ speedups when attention IO dominates.
* **Checkpointing buys memory with recompute**: Storing fewer activations and recomputing them during backpropagation cuts activation memory 3--4$\times$ in the walkthrough, from `{python} GPT2SummaryChapterRecap.act_fp16_gb_str` to `{python} GPT2SummaryChapterRecap.act_ckpt_gb_str` at batch 4. Use it when memory, not compute, binds.
* **Composed optimizations postpone scale-out**: Mixed precision, checkpointing, and prefetching together turn `{python} GPT2SummaryChapterRecap.b_total_mem_gb_str` into `{python} GPT2SummaryChapterRecap.o_total_mem_gb_str` for the GPT-2 walkthrough. Exhaust these local levers before paying distributed-training communication, reliability, energy, and cost overheads.

:::
```
