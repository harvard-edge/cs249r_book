# The 22 Walls of ML Systems: Interview Cheatsheet 🧱

In an AI Systems interview, you are often asked to discuss trade-offs. The "22 Walls" provides the definitive taxonomy for identifying exactly which physical or logical constraint is binding your system.

Use this cheatsheet to anchor your whiteboard designs in silicon reality.

---

## Domain 1: Node (Single-Accelerator Resources)
*What a single GPU/TPU can achieve in isolation.*

| Wall | Name | Equation | Interview Context |
| :--- | :--- | :--- | :--- |
| **Wall 1** | **Compute** | $T = Ops / (Peak \cdot \eta)$ | Use when discussing model throughput and TFLOPS ceilings. |
| **Wall 2** | **Memory** | $T = \lvert W vert / BW_{HBM}$ | Use for memory-bound decoding or capacity OOM issues. |
| **Wall 4** | **Serving** | Prefill vs. Decode | Discussing the duality of LLM inference phases. |
| **Wall 5** | **Batching** | $KV_{paged}$ | Solving KV-cache fragmentation (vLLM/PagedAttention). |
| **Wall 7** | **Tail Latency** | Erlang-C / P99 | Discussing why latency explodes at high utilization (>70%). |

## Domain 2: Data (Movement & Pipelines)
*How data moves to and through the accelerator.*

| Wall | Name | Equation | Interview Context |
| :--- | :--- | :--- | :--- |
| **Wall 8** | **Ingestion** | $ho = BW_{demand} / BW_{supply}$ | Discussing storage I/O bottlenecks (NVMe vs. Network). |
| **Wall 9** | **Transformation**| $T = B \cdot S / CPU_{tp}$ | The "CPU Starvation" problem during JPEG decoding. |
| **Wall 10**| **Locality** | $BW_{eff} = BW_{link} / oversub$ | Network topology impact on data loading. |

## Domain 3: Algorithm (Scaling & Compression)
*The mathematical demand of the model.*

| Wall | Name | Equation | Interview Context |
| :--- | :--- | :--- | :--- |
| **Wall 11**| **Complexity** | $C = 6PD$ | Chinchilla scaling laws for compute-optimal training. |
| **Wall 13**| **Fidelity** | $r = 32/b$ | Trading accuracy for efficiency via Quantization (INT8/4). |

## Domain 4: Fleet (Multi-Node Coordination)
*The penalty of distributed scale.*

| Wall | Name | Equation | Interview Context |
| :--- | :--- | :--- | :--- |
| **Wall 14**| **Communication** | $T = 	ext{Comm} / 	ext{Comp}$ | Bisection bandwidth and AllReduce overhead (Amdahl's Law). |
| **Wall 15**| **Fragility** | $MTBF_{cluster} = M/N$ | Why 10,000 GPUs require optimal checkpointing logic. |

---

## How to use the Walls in an Interview:
1.  **Identify the Binding Wall:** "Our profiling shows low GPU utilization; we are likely hitting **Wall 9 (Transformation)** due to CPU-bound JPEG decoding."
2.  **Propose a Wall-Buster:** "To overcome the **Memory Wall (Wall 2)** during decoding, we should implement 4-bit quantization to reduce the bytes moved per token."
3.  **Cite the Invariant:** "As we scale to 1,000 nodes, **Wall 14 (Communication)** will dominate unless we use a Fat-Tree topology to maximize bisection bandwidth."

---
> *For deep derivations, see the [MLSysBook Textbook](https://mlsysbook.ai/book/).*
