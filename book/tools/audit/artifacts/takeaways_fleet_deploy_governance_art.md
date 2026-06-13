# Takeaways ART: Fleet Deploy Governance

## book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd

- Recommendation: Modify
- Evidence reviewed: Purpose; h1-h3 outline; Summary; current `callout-takeaways`; summary recap cells for H100 roofline ridge and FlashAttention state reduction.
- Issues:
  - Current block has seven bullets and reads like a technique inventory.
  - Several bold leads are topic labels that closely mirror learning objectives.
  - The summary's stronger result is bottleneck displacement through a profile-optimize-reprofile loop.
- Proposed callout:

```qmd
::: {.callout-takeaways title="Match the software to the silicon"}

* **Bytes usually bind**: On H100-class accelerators, the chapter's roofline recap places most transformer work below the FP16 ridge point of `{python} RooflineSummaryRecap.h100_fp16_ridge_str`, so useful speedups come first from reducing HBM traffic, keeping intermediates in SRAM, and spending compute only when it moves the active bottleneck.
* **Fusion makes locality real**: Operator fusion, CUDA graphs, and FlashAttention are not just kernel tricks; they remove launch overhead and materialized attention state. In the 8K example, tiled attention cuts stored intermediate state by `{python} FlashAttentionSummaryRecap.savings_mult_str`, turning memory pressure into usable throughput.
* **Precision buys bandwidth with risk**: FP8, INT8, and INT4 increase effective bandwidth and batch capacity only when outliers, scale factors, and quality checks are managed. Quantization is a systems contract between numerical format, kernel implementation, serving memory, and acceptable model behavior.
* **Algorithms can move the roofline**: Speculative decoding and mixture-of-experts change how much useful work each target-model pass performs, but they introduce acceptance-rate, routing, AllToAll, and load-balancing constraints. The win is real only after communication and scheduler costs are measured.
* **Profiling is every step**: The 70B case study shows optimization as bottleneck displacement: INT4 changes batch size, batch size changes arithmetic intensity, and the next limit moves. Fleet performance engineering means measure, optimize the binding term, then measure again before believing the speedup.

:::
```

## book/quarto/contents/vol2/inference/inference.qmd

- Recommendation: Modify
- Evidence reviewed: Purpose; h1-h3 outline; Summary; current `callout-takeaways`; calculation-cell index for serving cost, batching, KV cache, sharding, routing, quantized serving, and ranking cascade anchors.
- Issues:
  - The opening bullet is underdeveloped relative to the serving-cost arc.
  - The current block overemphasizes named mechanisms and underemphasizes tail-latency budgeting across the service.
  - Autoscaling, multi-tenancy, and model-class-specific serving are present in the chapter arc but only weakly synthesized.
- Proposed callout:

```qmd
::: {.callout-takeaways title="Serving inverts every assumption"}

* **Serving cost compounds forever**: Training is paid once, but inference OpEx accrues on every request; for high-traffic production systems it can dominate by 100$\times$ or more over a model's lifetime. Milliseconds, memory fragments, and utilization points are therefore financial levers, not local optimizations.
* **Tail latency governs architecture**: Inference systems optimize under P99 SLOs, so batching, sharding, caching, and autoscaling must raise utilization without spending the user-visible deadline. Aggregate throughput is necessary but insufficient when a slow request can break downstream services.
* **Decode turns memory into capacity**: LLM prefill is compute-bound, while autoregressive decode rereads weights and expands private KV state token by token. Continuous batching, PagedAttention, prefix reuse, and KV compression matter because they convert scarce HBM bandwidth and memory into admitted requests.
* **Model class sets the primitive**: Vision models benefit from predictable static or dynamic batches, recommenders from feature-parallel embedding sharding, and LLMs from iteration-level scheduling. A fleet scheduler must match the workload shape instead of applying one universal batch-size rule.
* **Distribution always sends a bill**: Tensor parallelism, expert routing, global load balancing, multi-tenancy, and failover all buy capacity or isolation by adding communication and coordination. Production serving works only when that tax is explicitly budgeted inside latency, cost, and reliability envelopes.

:::
```

## book/quarto/contents/vol2/edge_intelligence/edge_intelligence.qmd

- Recommendation: Modify
- Evidence reviewed: Purpose; h1-h3 outline; Summary; current `callout-takeaways`; calculation-cell index for edge spectrum, training overhead, adaptation footprint, federated compression, and orchestration anchors.
- Issues:
  - Current bullets contain strong facts but include topic-label leads such as "The three pillars."
  - The chapter's sign-flip argument, where fleet physics are reversed at the edge, should be explicit.
  - Federated learning and heterogeneity overlap across two current bullets.
- Proposed callout:

```qmd
::: {.callout-takeaways title="Physics sets the budget"}

* **The edge reverses the fleet**: Data centers fight to keep many accelerators busy; edge devices fight to learn under batteries, kilobytes, thermal throttling, and intermittent links. The same Compute, Communication, Coordination constraints produce a different discipline when scarcity, not scale, is binding.
* **Bandwidth beats advertised TOPS**: On-device LLMs are limited by mobile memory bandwidth, not NPU peak compute. The 30--50$\times$ gap between mobile RAM and data-center HBM makes quantization and locality survival requirements for interactive decode.
* **Learning multiplies the footprint**: On-device training needs activations, gradients, optimizer state, and bidirectional traffic, raising resources 4--12$\times$ over inference-only deployment. Adaptation strategies must shrink the update, not merely compress the deployed model.
* **Adaptation needs three levers**: Bias-only updates, LoRA, sparse layers, few-shot data reuse, and experience replay each spend different memory, energy, and privacy budgets. Federated coordination adds population learning only when protocols handle non-IID data and client dropout.
* **Heterogeneity is normal operation**: Edge fleets span microcontrollers, phones, gateways, and NPUs with uneven availability and performance. Federated systems must treat stragglers, participation bias, rollback, and privacy-preserving observability as protocol design constraints, not deployment cleanup.

:::
```

## book/quarto/contents/vol2/ops_scale/ops_scale.qmd

- Recommendation: Modify
- Evidence reviewed: Purpose; h1-h3 outline; Summary; current `callout-takeaways`; calculation-cell index for platform ROI, TCO, feature-store scale, cost monitoring, and monitoring-overhead anchors.
- Issues:
  - Current bullets are mostly accurate but several leads are topic labels or categories.
  - The summary's qualitative result is that the model fleet, not an individual model, becomes the unit of management.
  - The current block can better connect ROI, monitoring, TCO, and edge operations as one platform discipline.
- Proposed callout:

```qmd
::: {.callout-takeaways title="Platforms, not pipelines"}

* **One model is not the unit**: At portfolio scale, the operational object becomes the dependency graph of models, features, pipelines, alerts, and owners. Registries, lineage, and ensemble-aware CI/CD prevent a local update from silently breaking downstream consumers.
* **Platforms turn toil marginal**: The summary's 50-model, \$2M platform example shows why shared infrastructure pays back when repeated pipelines, incident response, and manual coordination are removed. The economic gain is not elegance; each added model costs less to deploy and maintain.
* **TCO chooses the target**: The TCO equation separates training, inference, data, and iteration costs, and the dominant term changes over a system's life. Early fleets should buy velocity; mature high-traffic fleets should buy serving efficiency, utilization, and cost attribution.
* **Monitoring must aggregate signal**: Per-model dashboards and alerts scale into noise when 100 models each emit independent failures. Hierarchical telemetry, fleet-wide anomaly detection, and feature-quality gates make common-cause incidents visible before alert fatigue hides them.
* **Operations reaches the edge**: Model fleets do not end at the data center. Weeks-long rollouts, hardware-in-the-loop validation, version skew, and heterogeneous device failures make edge deployment part of the same platform discipline that governs cloud CI/CD.

:::
```

## book/quarto/contents/vol2/security_privacy/security_privacy.qmd

- Recommendation: Modify
- Evidence reviewed: Purpose; h1-h3 outline; Summary; current `callout-takeaways`; calculation-cell index for model theft, differential privacy, privacy-utility curves, and defense architecture anchors.
- Issues:
  - The first bullets are accurate but formatted as distinctions or topics rather than durable claims.
  - The block should foreground the chapter's central shift: the learned model is both asset and attack surface.
  - Hardware trust, provenance, DP, and LLM safeguards can be synthesized as layered fleet defenses.
- Proposed callout:

```qmd
::: {.callout-takeaways title="Defend the model, not just the server"}

* **The model is the attack surface**: ML systems inherit ordinary server threats but add attacks against the learned boundary itself. Poisoned data, model extraction, adversarial inputs, and prompt injection can compromise behavior without exploiting a traditional software vulnerability.
* **Privacy is a spent budget**: Because models can encode shadows of sensitive data, privacy cannot be treated as anonymization at the perimeter. Differential privacy offers bounded $(\epsilon,\delta)$ claims, but only if budgets are tracked across training, release, and repeated query access.
* **Generative systems need semantic defenses**: LLM failures exploit language and context: prompt injection, tool misuse, and PII leakage pass through normal text channels. Output monitoring, content isolation, policy enforcement, and guardrails must sit in the serving path, not only in network controls.
* **Provenance enables recovery**: Signed datasets, locked dependencies, registry permissions, and deployment attestations let operators answer which data, code, principal, and checkpoint produced a suspect model. Without that chain, rollback and forensics become guesswork at fleet speed.
* **Trust starts below software**: TEEs, secure boot, HSMs, and hardware roots of trust protect multi-tenant and confidential workloads from layers software alone cannot isolate. The defense posture is layered because each level catches failures the others cannot see.

:::
```

## book/quarto/contents/vol2/robust_ai/robust_ai.qmd

- Recommendation: Modify
- Evidence reviewed: Purpose; h1-h3 outline; Summary; current `callout-takeaways`; summary recap cell for robustness tax; calculation-cell index for SDC probability, drift confidence, PSI/KL/KS, adversarial training cost, and certified radius anchors.
- Issues:
  - Current block is close but several leads still name topics rather than consequences.
  - The strongest chapter result is that silent failures must be made visible and budgeted.
  - Robustness cost should be framed as an explicit engineering purchase rather than a generic trade-off.
- Proposed callout:

```qmd
::: {.callout-takeaways title="Silent failure is the real threat"}

* **Silence is the failure mode**: Robustness exists because drift, adversarial perturbations, poisoning, and numerical faults often preserve uptime and latency while corrupting predictions. Production systems need monitors that surface competence loss before user complaints or downstream business metrics reveal it.
* **Robustness is bought explicitly**: Strong adversarial training can cost roughly `{python} RobustnessTaxRecap.acc_drop_pp_str` of clean ImageNet accuracy, while certified defenses and uncertainty sampling add compute. The engineering decision is how much resilience the failure consequence justifies.
* **Drift needs distance measures**: MMD, PSI, KS, KL, and related metrics turn environmental shift into thresholds for review, retraining, rollback, or routing. The metric is useful only when connected to a response path and calibrated against false alarms.
* **Threats masquerade as each other**: A software fault can look like concept drift, a poisoned sample can look like a rare outlier, and an adversarial input can hide inside natural variation. Robust systems combine ingress validation, training defenses, uncertainty signals, and output verification.
* **Generative reliability is semantic**: LLM hallucinations are confidently fluent failures rather than simple label mistakes. Robustness therefore includes grounding checks, self-consistency, entropy or uncertainty signals, and human escalation policies that bound what the model is allowed to assert.

:::
```

## book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd

- Recommendation: Modify
- Evidence reviewed: Purpose; h1-h3 outline; Summary; current `callout-takeaways`; calculation-cell index for AI compute growth, energy wall, PUE, carbon accounting, data-center energy, training/inference energy, lifecycle emissions, and carbon-aware scheduling anchors.
- Issues:
  - Current block preserves important numbers but several leads are topic labels.
  - The chapter's central result is stronger than efficiency: power, carbon, water, and materials set existence constraints.
  - The Jevons rebound and lifecycle boundary should be integrated as engineering implications.
- Proposed callout:

```qmd
::: {.callout-takeaways title="Efficiency alone is not enough"}

* **Power is a hard ceiling**: A fleet cannot compute past the megawatts, cooling, water, and grid capacity its site can supply. Sustainability is therefore an existence constraint on model scale, not a public-relations layer around an otherwise finished architecture.
* **Demand can outrun efficiency**: In the 2012--2019 scaling window, AI compute demand grew about 6.2$\times$ per year against a 1.5$\times$ annual hardware-efficiency curve. Without algorithmic and governance limits, the power wall arrives even as each operation gets cheaper.
* **Decode wastes energy structurally**: Autoregressive serving spends long periods bandwidth-bound, leaving accelerators drawing static power while waiting on memory. Sustainable serving needs quantization, sparsity, batching discipline, and memory-optimized hardware because FLOP efficiency alone misses the dominant loss.
* **Carbon starts before boot**: Up to 30 percent of lifecycle emissions can be embodied in hardware manufacturing before the first query runs. Procurement, hardware lifetime, reuse, and e-waste policy are MLOps decisions when lifecycle accounting is the boundary.
* **Location changes the footprint**: Carbon-aware scheduling can cut emissions by the chapter's 8 to 40 times representative regional factors when flexible jobs move across grids. Efficiency must be paired with workload placement and demand governance or Jevons rebound spends the savings.

:::
```

## book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd

- Recommendation: Modify
- Evidence reviewed: Purpose; h1-h3 outline; Summary; current `callout-takeaways`; summary fairness-gap cell; calculation-cell index for fairness tax, metric disagreement, DP-SGD, unlearning, explanation storage, monitoring scale, automation bias, and SHAP overhead anchors.
- Issues:
  - Current bullets are useful but several leads are mechanism labels rather than results.
  - The fairness example's incompatible gaps should be preserved as the chapter's clearest quantitative anchor.
  - The block should emphasize responsibility as deployment-gating infrastructure, not only ethics or compliance.
- Proposed callout:

```qmd
::: {.callout-takeaways title="Ethics is an engineering constraint"}

* **Responsibility gates deployment**: In regulated or enterprise settings, accuracy and latency do not matter if the system cannot document behavior, explain decisions, support audit, or assign ownership. Governance is a Coordination constraint that determines whether Compute is allowed to run.
* **Fairness has no hidden optimum**: The chapter's table shows one loan system can have a `{python} FairnessGapCalc.fpr_gap_pp_str` false-positive gap while still carrying `{python} FairnessGapCalc.approval_gap_pp_str` approval and `{python} FairnessGapCalc.tpr_gap_pp_str` true-positive gaps. Differing base rates turn fairness into an explicit normative choice, not a single metric.
* **Oversight consumes capacity**: Fairness monitoring, audit logging, explanations, and privacy-preserving training add latency, storage, compute, and accuracy costs. SHAP-style explanations and DP-SGD must be budgeted in architecture and SLOs rather than appended when legal review arrives.
* **Generative governance is infrastructure**: System prompts, RLHF policies, tool permissions, model versions, and safety evaluations are operational artifacts. They need ownership, CI/CD, rollback, and monitoring because they steer model behavior as directly as weights do.
* **Evidence needs an owner**: Dashboards, explanations, and appeals protect users only when a team can investigate, remediate, roll back, or escalate failures. Responsible AI becomes engineering practice when every measured obligation has an accountable operating path.

:::
```

## book/quarto/contents/vol2/conclusion/conclusion.qmd

- Recommendation: Modify
- Evidence reviewed: Purpose; h1-h3 outline; Fallacies and Pitfalls; current `callout-takeaways`; scale-facts, fleet-evolution, post-silicon, and Fermi estimate cells.
- Issues:
  - Current first bullet restates the six principles instead of turning them into a durable result.
  - For a conclusion, four synthesis bullets would better match the takeaways rule than a five-bullet list.
  - The replacement should preserve the p99 fan-out, Llama 3 failure cadence, and 100x fleet-efficiency anchors while emphasizing the volume-level systems discipline.
- Proposed callout:

```qmd
::: {.callout-takeaways title="Systems that scale, endure, and serve"}

* **The fleet is the object**: The volume's six principles reduce to one habit: follow the binding constraint across infrastructure, communication, coordination, serving, and governance. A faster component is not a better system if it hides cost in another layer.
* **Scale changes the probability model**: At the `{python} ConclusionScaleFacts.tail_percentile_str`th percentile, touching `{python} ConclusionScaleFacts.tail_servers_str` gives a `{python} ConclusionScaleFacts.tail_hit_pct_str` chance of a slow server, and Llama 3 saw `{python} ConclusionScaleFacts.llama_failures_str` in `{python} ConclusionScaleFacts.llama_days_str`. Fleet behavior is not single-node behavior repeated.
* **Orchestration becomes capability**: The illustrative `{python} FleetEvolution.fe_target_gain_mult_str` efficiency target cannot come from silicon or algorithms alone; after `{python} FleetEvolution.fe_hw_gain_mult_str` hardware and `{python} FleetEvolution.fe_algo_gain_mult_str` algorithm gains, the residual `{python} FleetEvolution.fe_orch_gain_mult_str` must come from routing, reuse, overlap, and verification.
* **Obligations belong in the path**: Security, privacy, fairness, carbon, accessibility, and auditability are production constraints, not external reviews. The discipline is to design them into data pipelines, schedulers, serving paths, and operating procedures before scale makes the trade-off irreversible.

:::
```
