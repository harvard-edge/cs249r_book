# MLSysim Paper — Related Work & Citation Guide

A structured walkthrough of every cited paper, organized by function in the MLSysim narrative.
Use this to verify each citation is justified and to spot gaps.

---

## How to Read This Document

Each entry follows this format:

- **What they did**: The paper's key contribution in 1–2 sentences
- **Why we cite it**: How it supports the MLSysim argument
- **Where**: Line(s) in `paper.tex` where it appears
- **Verdict**: ✅ (justified), ⚠️ (consider strengthening), or ❌ (unused — decide to cite or remove)

---

## 1. Foundational Framing (Introduction)

These papers establish *why* MLSysim needs to exist.

### `sutton2019bitter` — Rich Sutton, "The Bitter Lesson" (2019)
- **What they did**: Argued that general methods leveraging computation consistently outperform hand-engineered approaches — ML success is fundamentally compute-driven.
- **Why we cite it**: Opens the paper with "ML has become infrastructure." If progress is compute-driven, then reasoning about compute infrastructure is essential.
- **Where**: L165
- **Verdict**: ✅ Perfect opener. Sets the tone that ML is a systems problem.

### `dean2012large` — Dean et al., "Large Scale Distributed Deep Networks" (NeurIPS 2012)
- **What they did**: Pioneered distributed DNN training at Google (DistBelief), introducing data and model parallelism at warehouse scale.
- **Why we cite it**: Supports the claim that training frontier models requires orchestrating tens of thousands of accelerators. Historical anchor for distributed ML.
- **Where**: L165
- **Verdict**: ✅ Canonical reference for the "ML at scale" claim.

### `shoeybi2019megatron` — Shoeybi et al., "Megatron-LM" (2019)
- **What they did**: Introduced efficient intra-layer tensor parallelism and pipeline parallelism for training multi-billion parameter models.
- **Why we cite it**: Co-cited with Dean to show the modern scale of distributed training. Also sourced for the AllReduce equation (Wall 14).
- **Where**: L165, L219 (Wall 14 equation), L617 (DistributedSolver)
- **Verdict**: ✅ Dual role: motivation + equation source.

---

## 2. The Fidelity–Speed–Scope Void (Introduction + Related Work)

These papers define the modeling landscape MLSysim positions against.

### `won2023astrasim2` — Won et al., "ASTRA-sim 2.0" (ISPASS 2023)
- **What they did**: Hierarchical network simulator for distributed training. Models collective communication across realistic topologies with high fidelity.
- **Why we cite it**: Exemplifies the high-fidelity end of the void — accurate but hours per configuration, too slow for design-space exploration.
- **Where**: L170 (void argument), L253 (comparison table), L281–285 (Related Work), L851 (speed comparison), L998 (limitations), L1010 (walls not included)
- **Verdict**: ✅ Key positioning reference. Cited 5+ times — appropriately, as the primary "complement not competitor."

### `calculon2023` — Isaev et al., "Calculon" (SC 2023)
- **What they did**: Analytical co-design tool for LLM training performance. Fast execution but narrow scope (transformer training only).
- **Why we cite it**: Closest analog to MLSysim in speed, but demonstrates the scope gap — no inference, data pipelines, reliability, sustainability, or dimensional enforcement.
- **Where**: L170, L262 (comparison table), L293–295 (Related Work)
- **Verdict**: ✅ Primary analytical competitor. Clear differentiation.

### `binkert2011gem5` — Binkert et al., "The gem5 Simulator" (2011)
- **What they did**: Canonical cycle-accurate CPU/GPU architecture simulator used across academia.
- **Why we cite it**: Represents the highest-fidelity baseline. Shows why cycle-level simulation is impractical for ML systems exploration (no ML abstractions, hours per forward pass).
- **Where**: L252 (comparison table), L281 (Related Work)
- **Verdict**: ✅ Important for establishing the simulation spectrum.

### `wang2025simai` — Wang et al., "SimAI" (NSDI 2025)
- **What they did**: Full-stack distributed training simulator integrating NS3 network modeling with kernel traces. 98% accuracy at 1024 A100 nodes.
- **Why we cite it**: Highest-accuracy trace-driven approach, but requires minutes per config and hardware traces. Extends ASTRA-sim's approach.
- **Where**: L254 (comparison table), L283 (Related Work)
- **Verdict**: ✅ Strengthens the fidelity–speed spectrum argument.

---

## 3. Accelerator Design Tools (Related Work)

Tools that operate at the operator/tile level — one abstraction below MLSysim.

### `parashar2019timeloop` — Parashar et al., "Timeloop" (ISPASS 2019)
- **What they did**: Systematic methodology for evaluating DNN accelerator dataflows. Models how data tiles map onto spatial architectures.
- **Why we cite it**: Two roles: (1) in the void argument, tools that lack dimensional enforcement; (2) in Related Work, representing tile-level tools that MLSysim consumes outputs from.
- **Where**: L170, L287–289 (Related Work)
- **Verdict**: ✅ Important for showing MLSysim operates one level higher.

### `wu2019accelergy` — Wu et al., "Accelergy" (ICCAD 2019)
- **What they did**: Architecture-level energy estimation primitives consumed by Timeloop.
- **Why we cite it**: Co-cited with Timeloop as the accelerator design toolkit. Shows the gap: great for energy estimation, no system-level reasoning.
- **Where**: L170, L289 (Related Work)
- **Verdict**: ✅ Properly paired with Timeloop.

### `zhang2024llmcompass` — Zhang et al., "LLMCompass" (ISCA 2024)
- **What they did**: Automated mapper + area-based cost model for LLM inference hardware design. 4% end-to-end error on A100 within minutes.
- **Why we cite it**: Brings Timeloop-style analysis to LLMs but still at operator level — doesn't model fleet, economics, or sustainability.
- **Where**: L258 (comparison table), L289 (Related Work)
- **Verdict**: ✅ Shows the state of the art in accelerator-level LLM tools.

---

## 4. Analytical & Co-Design Tools (Related Work)

The tools most similar to MLSysim in spirit — analytical approaches trading fidelity for speed.

### `qi2017paleo` — Qi et al., "PALEO" (ICLR 2017)
- **What they did**: Pioneered analytical decomposition of DNN training time into computation and communication components across data/model parallelism.
- **Why we cite it**: Historical precedent for the analytical modeling approach. Predates modern concerns (inference serving, sustainability).
- **Where**: L261 (comparison table), L297 (Related Work)
- **Verdict**: ✅ Important historical anchor.

### `jia2019flexflow` — Jia et al., "FlexFlow" (MLSys 2019)
- **What they did**: Simulation-guided search over parallelism strategies for deep neural networks.
- **Why we cite it**: Shows prior work on parallelism optimization. Narrow scope relative to MLSysim.
- **Where**: L297 (Related Work)
- **Verdict**: ✅ Appropriate — one sentence, contextualizes the parallelism search problem.

### `yu2021habitat` — Yu et al., "Habitat" (ATC 2021)
- **What they did**: Cross-hardware performance extrapolation for DNN training using execution-time scaling curves.
- **Why we cite it**: Prior analytical work enabling what-if across hardware; doesn't cover modern scope.
- **Where**: L297 (Related Work)
- **Verdict**: ✅ Brief mention, appropriate scope.

### `agrawal2024vidur` — Agrawal et al., "Vidur" (MLSys 2024)
- **What they did**: Large-scale LLM inference simulation framework using operator-level profiling. <5% error across multiple LLMs and scheduling policies.
- **Why we cite it**: Closest inference analog in the analytical category. But inference-only — no training, economics, sustainability.
- **Where**: L264 (comparison table), L297 (Related Work)
- **Verdict**: ✅ Key inference comparison point.

### `bambhaniya2024genz` — Bambhaniya et al., "GenZ" (2024)
- **What they did**: Analytical framework for LLM inference platform design with multi-dimensional network topology modeling.
- **Why we cite it**: Similar analytical approach for inference platform design. Inference-only scope.
- **Where**: L265 (comparison table), L297 (Related Work)
- **Verdict**: ✅ Complements Vidur in the inference tool landscape.

### `zhong2024distserve` — Zhong et al., "DistServe" (OSDI 2024)
- **What they did**: Disaggregated prefill and decode onto separate hardware for goodput-optimized LLM serving.
- **Why we cite it**: Validates the two-phase inference model (our Eq. serving). Shows prefill-decode separation is a real systems concern.
- **Where**: L297 (Related Work)
- **Verdict**: ✅ Supports the Serving Wall formulation.

### `agrawal2024sarathi` — Agrawal et al., "Sarathi-Serve" (OSDI 2024)
- **What they did**: Chunked-prefill scheduling to tame throughput-latency tradeoff in LLM inference.
- **Why we cite it**: Same: validates that the two-phase inference model requires increasingly sophisticated scheduling.
- **Where**: L297 (Related Work)
- **Verdict**: ✅ Complements DistServe.

### `yuan2024llmviewer` — Yuan et al., "LLM Inference Unveiled" (2024)
- **What they did**: Survey of LLM inference with Roofline model insights for memory and latency estimation.
- **Why we cite it**: Lightweight profiling tool. Doesn't extend to fleet-level reasoning, multi-tenant scheduling, or cross-domain analysis.
- **Where**: L266 (comparison table), L299 (Related Work)
- **Verdict**: ✅ Appropriate positioning.

### `kim2023llmanalysis` — Li, Cheng, "llm-analysis" (GitHub 2023)
- **What they did**: Open-source lightweight latency and memory estimation tool for transformer inference.
- **Why we cite it**: Same category as LLM-Viewer. Useful but narrow scope.
- **Where**: L266 (comparison table), L299 (Related Work)
- **Verdict**: ✅ Appropriate.

### `liang2025lumos` — Liang et al., "Lumos" (MLSys 2025)
- **What they did**: Trace-driven LLM training performance modeling with 3.3% average error up to 512 H100 GPUs.
- **Why we cite it**: Higher single-point accuracy than MLSysim but requires empirical traces from target hardware, limiting what-if exploration.
- **Where**: L263 (comparison table), L295 (Related Work)
- **Verdict**: ✅ Clean differentiation: accuracy vs. generality.

### `deepseek2025v3` — DeepSeek-AI, "DeepSeek-V3" (ISCA 2025)
- **What they did**: FP8 mixed-precision training + MoE sparsity + multi-plane network topology for frontier training at a fraction of conventional cost.
- **Why we cite it**: Exemplifies the kind of multi-wall co-design analysis MLSysim targets — spans Walls 1, 3, 13, 14, and 17 simultaneously.
- **Where**: L295 (Related Work)
- **Verdict**: ✅ Strong motivating example of why multi-wall analysis matters.

---

## 5. Sustainability & Fleet Efficiency (Related Work)

### `faiz2024llmcarbon` — Faiz et al., "LLMCarbon" (ICLR 2024)
- **What they did**: End-to-end carbon footprint modeling for dense and MoE LLMs. Validated within 8% of Google's published figures.
- **Why we cite it**: Sustainability tool covering one domain. MLSysim integrates carbon into full-stack analysis.
- **Where**: L269 (comparison table), L301 (Related Work)
- **Verdict**: ✅ Key sustainability comparison.

### `lottick2019codecarbon` — Lottick et al., "CodeCarbon" (2019)
- **What they did**: Empirical runtime energy and carbon tracking via hardware power monitors.
- **Why we cite it**: Empirical counterpart to analytical sustainability modeling. One-domain tool.
- **Where**: L270 (comparison table), L301 (Related Work)
- **Verdict**: ✅ Represents the empirical measurement approach.

### `wongpanich2025fleet` — Wongpanich et al., "ML Productivity Goodput" (2025)
- **What they did**: Introduced MPG as a fleet-level efficiency metric for warehouse-scale TPU clusters. Shows traditional utilization metrics are insufficient.
- **Why we cite it**: Two roles: (1) fleet efficiency metric gap in Related Work; (2) heterogeneous fleet limitation in Discussion.
- **Where**: L301, L1002
- **Verdict**: ✅ Dual role is well-justified.

---

## 6. Pedagogical Precedents (Related Work)

### `hennessy2024architecture` — Hennessy & Patterson, *Computer Architecture* 7th ed. (2024)
- **What they did**: Gold-standard quantitative approach to computer architecture education. MIPS ISA + SPIM simulator taught architectural reasoning through simplification.
- **Why we cite it**: **Central pedagogical precedent.** MLSysim aspires to be for ML systems what MIPS/SPIM was for computer architecture. Also sourced for "power wall" terminology, demand-supply separation inspiration, and accuracy-speed tradeoff argument.
- **Where**: L170, L332, L359, L490, L972, L1014
- **Verdict**: ✅ Most important framing reference. 6 citations — all justified.

### `patterson2014organization` — Patterson & Hennessy, *Computer Organization and Design* 5th ed. (2014)
- **What they did**: The undergraduate textbook that introduced MIPS/SPIM to generations of students.
- **Why we cite it**: Direct pedagogical precedent in the Pedagogical Precedents subsection.
- **Where**: L307
- **Verdict**: ✅ Distinct from the 7th ed. grad text — this is the undergrad analog.

### `cox2011xv6` — Cox et al., "xv6" (MIT 2011)
- **What they did**: Simple Unix-like teaching OS that strips production complexity to reveal core abstractions.
- **Why we cite it**: Pedagogical precedent: teaching through simplification.
- **Where**: L307
- **Verdict**: ✅ Clean parallel to MLSysim's approach.

### `tanenbaum2006minix` — Tanenbaum & Woodhull, *MINIX / OS Design and Implementation* (2006)
- **What they did**: Microkernel teaching OS, companion to the operating systems textbook.
- **Why we cite it**: Same pedagogical precedent category.
- **Where**: L307
- **Verdict**: ✅ Completes the trio (MIPS/SPIM, xv6, MINIX).

### `mlsysbook2025` — Reddi et al., *Machine Learning Systems* textbook (2025)
- **What they did**: The companion ML Systems textbook this tool is designed for.
- **Why we cite it**: Establishes MLSysim as the analytical companion to a broader educational ecosystem.
- **Where**: L173, L309, L1045
- **Verdict**: ✅ Self-reference to the host project.

---

## 7. Wall Equation Sources (Taxonomy, Section 4)

Each wall cites the foundational paper for its core equation.

### `williams2009roofline` — Williams et al., "Roofline Model" (CACM 2009)
- **What they did**: Visual performance model relating compute ceilings and memory bandwidth ceilings via arithmetic intensity. Introduced the "ridge point."
- **Why we cite it**: **Theoretical backbone of MLSysim.** Source for Walls 1–2 (Compute/Memory), Wall 21 (Sensitivity), and the Roofline analysis throughout.
- **Where**: L200–201, L394, L490, L497, L504, L702, L808, L862
- **Verdict**: ✅ Most-cited paper (~8 appearances). Appropriately central.

### `chowdhery2022palm` — Chowdhery et al., "PaLM" (JMLR 2023)
- **What they did**: Scaled language modeling to 540B parameters on 6K+ TPU v4 chips. Reported MFU metrics that became the field standard.
- **Why we cite it**: Source for Wall 3 (Software/MFU definition). Also validation Anchor 4 (PaLM scaling efficiency degradation).
- **Where**: L202, L521, L526, L817
- **Verdict**: ✅ Dual role: equation source + validation.

### `pope2023llm` — Pope et al., "Efficiently Scaling Transformer Inference" (MLSys 2023)
- **What they did**: Formalized the prefill (compute-bound) vs. decode (memory-bound) phase analysis for transformer inference.
- **Why we cite it**: Source equation for Wall 4 (Serving) — the two-phase inference model.
- **Where**: L203, L528
- **Verdict**: ✅ Clean equation attribution.

### `kwon2023efficient` — Kwon et al., "PagedAttention / vLLM" (SOSP 2023)
- **What they did**: Virtual memory for KV-cache — non-contiguous paged allocation eliminates 40–50% external fragmentation in LLM serving.
- **Why we cite it**: Three roles: (1) Wall 5 (Batching) equation, (2) Wall 22 (Synthesis) reference, (3) validation Anchor 2 (vLLM inference benchmark).
- **Where**: L167, L204, L231, L536, L811
- **Verdict**: ✅ High-impact paper with multiple justified citations.

### `lie2022cerebras` — Lie, "Cerebras Architecture Deep Dive" (Hot Chips 2022)
- **What they did**: Wafer-scale engine architecture: on-wafer SRAM + external MemoryX weight streaming.
- **Why we cite it**: Source for Wall 6 (Streaming) — the weight-streaming bottleneck model. Also the Silicon Zoo entry for CS-3 specs.
- **Where**: L205, L402, L543
- **Verdict**: ✅ Unique architecture that motivates a distinct wall.

### `dean2013tail` — Dean & Barroso, "The Tail at Scale" (CACM 2013)
- **What they did**: Showed that P99 latency at scale is dominated by the slowest component in fan-out architectures.
- **Why we cite it**: Source for Wall 7 (Tail Latency) — Erlang-C M/M/c queueing model.
- **Where**: L206, L550
- **Verdict**: ✅ Canonical tail latency reference.

### `mohan2021analyzing` — Mohan et al., "Analyzing and Mitigating Data Stalls" (VLDB 2021)
- **What they did**: Showed that data pipeline stalls are a significant but overlooked bottleneck in DNN training.
- **Why we cite it**: Source for Wall 8 (Ingestion) — demand/supply ratio for data I/O.
- **Where**: L209, L562
- **Verdict**: ✅ Established the data stall problem.

### `murray2021tf` — Murray et al., "tf.data" (VLDB 2021)
- **What they did**: ML data processing framework that formalized the CPU preprocessing pipeline abstraction.
- **Why we cite it**: Source for Wall 9 (Transformation) — CPU preprocessing throughput model.
- **Where**: L210, L569
- **Verdict**: ✅ Clean attribution.

### `leiserson1985fat` — Leiserson, "Fat-Trees" (IEEE Trans. Computers 1985)
- **What they did**: Introduced fat-tree networks for hardware-efficient supercomputing. Defined bisection bandwidth.
- **Why we cite it**: Source for Wall 10 (Locality) — bisection bandwidth fraction by topology.
- **Where**: L211, L576
- **Verdict**: ✅ Foundational network topology reference.

### `hoffmann2022chinchilla` — Hoffmann et al., "Chinchilla" (NeurIPS 2022)
- **What they did**: Established compute-optimal scaling laws: doubling parameters requires doubling tokens. $C = 6PD$, $D^* \approx 20P$.
- **Why we cite it**: Source for Wall 11 (Complexity) — scaling law equations. Also validation Anchor 5.
- **Where**: L214, L490, L588, L820
- **Verdict**: ✅ Central to the algorithmic scaling domain.

### `snell2025scaling` — Snell et al., "Scaling LLM Test-Time Compute" (ICLR 2025)
- **What they did**: Showed that scaling inference-time compute (chain-of-thought, tree search) can be more effective than scaling model parameters.
- **Why we cite it**: Source for Wall 12 (Reasoning) — $T = K \times T_{step}$. Also motivates the CoT cost multiplier use case.
- **Where**: L215, L597, L602
- **Verdict**: ✅ Timely reference for inference-time scaling.

### `han2016deep` — Han et al., "Deep Compression" (ICLR 2016, Best Paper)
- **What they did**: Pruning + trained quantization + Huffman coding for DNN compression. Achieved 35–49× compression.
- **Why we cite it**: Source for Wall 13 (Fidelity) — compression ratio equations.
- **Where**: L216, L604
- **Verdict**: ✅ Seminal compression paper.

### `gholami2021survey` — Gholami et al., "Survey of Quantization Methods" (2021)
- **What they did**: Comprehensive survey covering PTQ, QAT, mixed-precision, and accuracy impact across architectures.
- **Why we cite it**: Co-source for Wall 13 accuracy degradation curves. Referenced for empirical compression baselines.
- **Where**: L604, L610
- **Verdict**: ✅ Appropriate survey reference.

### `narayanan2021efficient` — Narayanan et al., "Efficient Large-Scale LM Training" (SC 2021)
- **What they did**: Megatron-LM at scale with pipeline parallelism, interleaved scheduling, and 3D parallelism analysis.
- **Why we cite it**: Source for the pipeline bubble equation (Eq. bubble) in Wall 14.
- **Where**: L627
- **Verdict**: ✅ Direct equation attribution.

### `daly2006higher` — Daly, "Higher Order Estimate of Optimal Checkpoint Interval" (2006)
- **What they did**: Extended Young's 1974 checkpoint formula with higher-order correction terms.
- **Why we cite it**: Source for Wall 15 (Fragility) — cluster MTBF formula and Young-Daly optimal checkpoint interval.
- **Where**: L220, L638, L648
- **Verdict**: ✅ Standard checkpoint theory reference.

### `young1974first` — Young, "First Order Approximation to Optimum Checkpoint Interval" (CACM 1974)
- **What they did**: Original formula for optimal checkpoint frequency as a function of MTBF and checkpoint duration.
- **Why we cite it**: Co-source with Daly for the checkpoint interval formula.
- **Where**: L648
- **Verdict**: ✅ Appropriate to cite the original alongside the extension.

### `little1961proof` — Little, "A Proof for L = λW" (Operations Research 1961)
- **What they did**: The foundational queueing theory result relating queue length, arrival rate, and wait time.
- **Why we cite it**: Source for Wall 16 (Multi-tenant) — the queueing delay model.
- **Where**: L221, L655
- **Verdict**: ✅ Classic operations research reference.

### `barroso2018datacenter` — Barroso et al., *The Datacenter as a Computer* 3rd ed. (2018)
- **What they did**: Definitive reference on warehouse-scale computing: TCO decomposition, power, cooling, efficiency.
- **Why we cite it**: Source for Wall 17 (Capital) — TCO = CapEx + OpEx formulation.
- **Where**: L224, L667
- **Verdict**: ✅ Standard datacenter economics reference.

### `barroso2007case` — Barroso & Hölzle, "The Case for Energy-Proportional Computing" (IEEE Computer 2007)
- **What they did**: Argued servers should consume power proportional to utilization. Showed idle servers waste significant energy.
- **Why we cite it**: Source for the energy-proportional power model: idle = 30% TDP, rest scales linearly with MFU.
- **Where**: L681
- **Verdict**: ✅ Clean, specific attribution.

### `patterson2021carbon` — Patterson et al., "Carbon Emissions and Large Neural Network Training" (2021)
- **What they did**: Methodology for computing operational carbon footprint of ML training runs. Reported GPT-3 emissions.
- **Why we cite it**: Source for Wall 18 (Sustainability) equations. Also validation Anchor 6 (GPT-3 carbon footprint).
- **Where**: L225, L396, L490, L674, L822
- **Verdict**: ✅ Dual role: equation source + validation anchor.

### `eisenman2022checknrun` — Eisenman et al., "Check-N-Run" (NSDI 2022)
- **What they did**: Checkpointing system for training deep learning recommendation models. Analyzed I/O burst costs.
- **Why we cite it**: Source for Wall 19 (Checkpoint) — I/O burst penalty equation.
- **Where**: L226, L683
- **Verdict**: ✅ Direct equation attribution.

### `abadi2016deep` — Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016)
- **What they did**: Introduced DP-SGD — differentially private stochastic gradient descent with per-example clipping and Gaussian noise.
- **Why we cite it**: Source for Wall 20 (Safety) — DP-SGD overhead model.
- **Where**: L227, L690
- **Verdict**: ✅ Foundational privacy-in-ML reference.

---

## 8. Serving Wall — Subtechnique Citations

### `dao2022flashattention` — Dao et al., "FlashAttention" (NeurIPS 2022)
- **What they did**: IO-aware fused attention kernel achieving 2.5× speedup by reducing redundant HBM reads.
- **Why we cite it**: Example of raising η from ~0.3 to ~0.75 for attention layers — demonstrates that software optimization (Wall 3) can overcome hardware limitations.
- **Where**: L526
- **Verdict**: ✅ Concrete example of the Software Wall remedy.

### `zheng2024sglang` — Zheng et al., "SGLang" (2024)
- **What they did**: Efficient execution of structured language model programs with prefix caching.
- **Why we cite it**: Prompt caching technique cited in the Serving Wall — reduces TTFT by skipping prefill for cached KV entries.
- **Where**: L534
- **Verdict**: ✅ Modern serving optimization.

### `leviathan2023fast` — Leviathan et al., "Fast Inference via Speculative Decoding" (ICML 2023)
- **What they did**: Draft-verify paradigm where a small model proposes tokens and a large model verifies in parallel.
- **Why we cite it**: Speculative decoding technique in the Serving Wall.
- **Where**: L534
- **Verdict**: ✅ Key inference optimization.

### `patel2024splitwise` — Patel et al., "Splitwise" (ISCA 2024)
- **What they did**: Phase-splitting LLM inference: disaggregates prefill and decode onto different hardware with KV-cache network transfer.
- **Why we cite it**: Disaggregated serving technique in the Serving Wall.
- **Where**: L534
- **Verdict**: ✅ Represents the disaggregation trend.

---

## 9. Compression Wall — Additional Techniques

### `frantar2023gptq` — Frantar et al., "GPTQ" (ICLR 2023)
- **What they did**: Accurate one-shot post-training quantization for GPT-scale models using approximate second-order information.
- **Why we cite it**: Demonstrates that 4-bit quantization preserves most accuracy for LLMs — makes $r_{quant} = 4×$ a practical operating point.
- **Where**: L610
- **Verdict**: ✅ Evidence for the Fidelity Wall's practical range.

### `lin2024awq` — Lin et al., "AWQ" (MLSys 2024)
- **What they did**: Activation-aware weight quantization that protects salient weights based on activation magnitudes.
- **Why we cite it**: Same: 4-bit practical operating point evidence. Complementary approach to GPTQ.
- **Where**: L610
- **Verdict**: ✅ Paired with GPTQ for completeness.

### `nvidia2023h100` — NVIDIA, "H100 Tensor Core GPU Datasheet" (2023)
- **What they did**: Official hardware specifications for the H100 GPU.
- **Why we cite it**: Three roles: (1) "No Magic Numbers" invariant — every constant must trace to a source; (2) 2:4 structured sparsity via Sparse Tensor Cores; (3) sustained vs. peak bandwidth gap.
- **Where**: L334, L610, L811, L981
- **Verdict**: ✅ Essential hardware provenance.

---

## 10. Validation Anchors (Section 5)

### `mlperf2020` — Mattson et al., "MLPerf" (IEEE Micro 2020)
- **What they did**: Industry-standard benchmark suite for ML performance across training, inference, and other tasks.
- **Why we cite it**: Anchor 1 ground truth — ResNet-50 throughput on DGX A100 (38,200 samples/s).
- **Where**: L808
- **Verdict**: ✅ Gold-standard benchmark reference.

### `llama3team2024` — Llama Team @ Meta, "The Llama 3 Herd of Models" (2024)
- **What they did**: 405B-parameter model trained on 16K H100 GPUs with 38–43% MFU. Detailed 4D parallelism strategy.
- **Why we cite it**: Anchor 3 ground truth — distributed training MFU at production scale. Also η default range source.
- **Where**: L526, L814
- **Verdict**: ✅ Most important recent validation anchor.

---

## 11. Discussion & Future Work (Section 6)

### `box1976science` — Box, "Science and Statistics" (JASA 1976)
- **What they did**: The famous statistical philosophy paper: "All models are wrong, but some are useful."
- **Why we cite it**: Opens the Discussion section — frames MLSysim's limitations as inherent to all modeling, not a deficiency.
- **Where**: L992
- **Verdict**: ✅ Perfect framing for limitations section.

### `stephenson1999mco` — Stephenson et al., "Mars Climate Orbiter Mishap Investigation Board Phase I Report" (NASA 1999)
- **What they did**: Investigation of the most infamous unit-conversion failure: pound-force seconds vs. newton seconds destroyed a $327M spacecraft.
- **Why we cite it**: Footnote motivating dimensional strictness. Memorable, high-stakes example of why units matter.
- **Where**: L349 (footnote)
- **Verdict**: ✅ Compelling motivating example.

### `tinytorch2025` — Reddi et al., "TinyTorch" (2025)
- **What they did**: Progressive educational ML framework — students build a framework from scratch (tensors → autograd → optimizers).
- **Why we cite it**: Complementary tool in Future Work and Conclusion. TinyTorch = bottom-up (internals), MLSysim = top-down (systems analysis).
- **Where**: L1030, L1045
- **Verdict**: ✅ Self-reference to companion project.

### `shazeer2017outrageously` — Shazeer et al., "Outrageously Large Neural Networks: MoE" (ICLR 2017)
- **What they did**: Introduced the Sparsely-Gated Mixture-of-Experts layer — active params ≠ total params.
- **Why we cite it**: Justifies why `SparseTransformerWorkload.lower()` uses active params for FLOPs but total params for memory.
- **Where**: L425
- **Verdict**: ✅ Explains a key architectural decision in the type system.

---

## 12. UNCITED ENTRIES ⚠️

These 6 papers exist in `references.bib` but are **never referenced** in `paper.tex`.

### `kaplan2020scaling` — Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- **What they did**: First systematic scaling laws (predating Chinchilla). Different compute-optimal exponents.
- **Action needed**: Either cite as the precursor to Chinchilla in Wall 11 ("Building on the scaling laws of \citealt{kaplan2020scaling}, Chinchilla established…") or **remove from bib**.
- **Verdict**: ⚠️ Recommend citing — it's the foundational scaling paper Chinchilla refined.

### `rajbhandari2020zero` — Rajbhandari et al., "ZeRO" (SC 2020)
- **What they did**: Memory optimization partitioning optimizer states, gradients, and parameters across data-parallel ranks.
- **Action needed**: **Must cite at L617** where the prose says "ZeRO/FSDP" without any citation. This is a gap.
- **Verdict**: ⚠️ **Fix required** — uncited in-text mention.

### `rasley2020deepspeed` — Rasley et al., "DeepSpeed" (KDD 2020)
- **What they did**: The DeepSpeed library enabling ZeRO and other large-model training optimizations.
- **Action needed**: Could co-cite with ZeRO at L617, or remove if ZeRO alone suffices.
- **Verdict**: ⚠️ Optional — ZeRO citation is more critical.

### `gupta2022act` — Gupta et al., "ACT" (ISCA 2022)
- **What they did**: Architectural Carbon Modeling Tool — models embodied + operational carbon for computing systems.
- **Action needed**: Could cite alongside LLMCarbon in Related Work for embodied carbon perspective. Or remove.
- **Verdict**: ⚠️ Good paper but not currently used. Remove unless adding an embodied carbon discussion.

### `amodei2018ai` — Amodei & Hernandez, "AI and Compute" (OpenAI 2018)
- **What they did**: Documented the exponential growth of compute used in AI training (doubling every 3.4 months).
- **Action needed**: Could cite alongside Sutton's Bitter Lesson in the Introduction. Or remove.
- **Verdict**: ⚠️ Would strengthen the "ML as infrastructure" opening if cited.

### `jouppi2017datacenter` — Jouppi et al., "TPU v1" (ISCA 2017)
- **What they did**: Landmark paper on the first Tensor Processing Unit deployed at Google-scale.
- **Action needed**: Could cite when discussing TPU entries in the Silicon Zoo (L402) or in the validation section for TPU v4 specs. Or remove.
- **Verdict**: ⚠️ Iconic paper but not currently needed. Remove unless adding TPU history.

---

## Summary Statistics

| Category | Count | Notes |
|----------|-------|-------|
| Actively cited | 64 keys | All justified |
| Uncited in bib | 6 keys | 1 fix required (ZeRO), 5 optional |
| Most-cited paper | `williams2009roofline` | ~8 appearances — the theoretical backbone |
| Second most-cited | `hennessy2024architecture` | 6 appearances — the pedagogical backbone |
| Papers with dual roles | 7 | Equation source + validation anchor (PaLM, Chinchilla, etc.) |
| Unique venues represented | 25+ | ISCA, SOSP, NeurIPS, ICLR, MLSys, SC, OSDI, NSDI, etc. |

---

## Recommended Actions

1. **Fix**: Add `\citep{rajbhandari2020zero}` at L617 where "ZeRO/FSDP" appears uncited
2. **Consider**: Cite `kaplan2020scaling` as the precursor to Chinchilla in Wall 11
3. **Consider**: Cite `amodei2018ai` alongside Sutton in the Introduction
4. **Clean**: Remove `rasley2020deepspeed`, `gupta2022act`, `jouppi2017datacenter` if not adding citations
5. **Verify**: All 64 active citations have been cross-checked against real papers (completed in prior session)
