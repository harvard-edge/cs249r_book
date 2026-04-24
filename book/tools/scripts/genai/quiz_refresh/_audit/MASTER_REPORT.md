# Quiz Audit Master Report — 2026-04-24

**Chapters audited**: 30 / 33
**Questions audited**: 1,508
**Auditor model**: gpt-5.4

## Grade distribution

| Grade | Chapters |
|---|:---:|
| A | 0 |
| B | 30 |
| C | 0 |
| D | 0 |
| F | 0 |

**Verdict**: every audited chapter graded **B** (good with minor issues). 0 chapters need full regeneration.

## High-severity issues (need targeted fix)

### vol2/conclusion — `#sec-conclusion-path-forward-caa2` q4
- **Type**: trivia_fill
- **Description**: This is the weakest item in the chapter. Despite numeric context, the blank is effectively asking for the name of a specific technology ('co-packaged optics'), making it a term-recall check rather than a reasoning question about communication energy, system scaling, or substrate choice.
- **Suggested fix**: Replace with an MCQ or SHORT asking why lower energy per bit changes feasible fleet scale, or why optical interconnects become necessary once communication energy dominates million-GPU systems.

### vol2/fault_tolerance — `#sec-ft-fault-injection-tools-frameworks` q0
- **Type**: redefinition_violation
- **Description**: This question's entire point is defining 'fault model' versus 'error model.' That crosses the spec's buildup rule if those concepts have already been established earlier or are being treated here as general methodological vocabulary rather than chapter-specific application.
- **Suggested fix**: Reframe it as application: present an ML robustness experiment and ask which part is the fault model versus the error model, or ask which model choice would better answer a given resilience question.

### vol1/nn_architectures — `#sec-network-architectures-multilayer-perceptrons-dense-pattern-processing-bc11` q1
- **Type**: build_up_violation
- **Description**: This is essentially a definition check for MLP inductive bias, even though MLP terminology is already part of prior vocabulary. At chapter 6, the quiz should test how that bias creates systems consequences or why it fails on structured data, not simply restate the baseline definition.
- **Suggested fix**: Rewrite as an application question comparing MLP choice against CNN or transformer choice for a concrete workload, forcing the reader to use the prior term in this chapter's context rather than define it.

### vol1/nn_architectures — `#sec-network-architectures-attention-mechanisms-dynamic-pattern-processing-22df` q1
- **Type**: build_up_violation
- **Description**: This is close to a definitional 'what is attention' item, which undershoots the chapter's quality bar for reasoning and also weakens build-up. The section's distinctive contribution is not merely that attention computes weighted sums, but why that changes sequence-processing depth and introduces the quadratic wall.
- **Suggested fix**: Replace with a scenario asking why attention succeeds on a long-range dependency that an RNN struggles with, or why QKV-based routing changes the hardware cost structure.

### vol1/nn_computation — `#sec-neural-computation-evolution-ml-paradigms-ec9c` q6
- **Type**: other
- **Description**: The ORDER answer appears wrong relative to the section's own causal story. The chapter presents data availability, algorithmic innovations, and computing infrastructure as a mutually reinforcing cycle, but when it narrates emergence it repeatedly states that data abundance and algorithmic advances together created demand for stronger infrastructure; putting infrastructure last as a linear order is shaky, and the explanation overcommits to a causal sequence the text does not clearly endorse.
- **Suggested fix**: Either convert this to an MCQ/SHORT about the reinforcing cycle, or if keeping ORDER, use a genuinely sequential process from the section rather than cyclical factors.

### vol2/performance_engineering — `#sec-performance-engineering-speculative` q1
- **Type**: build_up_violation
- **Description**: This is a re-definition question for 'speculative decoding,' a prior-vocabulary term first introduced earlier in the book. The item's whole point is basic mechanism identification rather than applying the concept in this chapter's performance framework.
- **Suggested fix**: Replace with an application question comparing when speculative decoding helps or hurts under different batch sizes, acceptance rates, or memory-bandwidth regimes.
