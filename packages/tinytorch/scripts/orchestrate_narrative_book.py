#!/usr/bin/env python3
"""
TinyTorch Narrative Book Orchestrator
====================================
Orchestrates the generation and refinement of the narrative textbook:
"TinyTorch: The xv6 of Machine Learning Systems"

Enforces strict progressive disclosure, first-principles systems intuition,
Chicago Manual of Style (CMOS) prose rules, and seamless chapter-to-chapter flow.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
NARRATIVE_BOOK_DIR = REPO_ROOT / "packages" / "tinytorch" / "narrative_book"
SRC_DIR = REPO_ROOT / "packages" / "tinytorch" / "tinytorch"

MASTER_SYSTEM_PROMPT = """
You are the Lead Systems Architect and Author of "TinyTorch: The xv6 of Machine Learning Systems".
Your goal is to write a deeply intuitive, elegant, and engaging systems book for machine learning engineers.

CORE PEDAGOGICAL PHILOSOPHY:
- This is NOT a dry university textbook filled with mathematical boilerplate and detached proofs.
- This is NOT a homework lab manual: NEVER include "Purpose" sections, NEVER include "Learning Objectives" bullet lists, and NEVER include "TODO: fill in this function" instructions.
- This IS the "xv6 of Machine Learning Systems": a clear, first-principles engineering journey where a curious learner builds an entire modern deep learning framework from raw Python lists to a complete GPT-2 transformer with hardware acceleration.

CMOS (CHICAGO MANUAL OF STYLE) PROSE RULES:
1. SPELLING & PUNCTUATION: American English throughout (e.g. "optimize", "backward", "modeling", "labeled"). Use serial (Oxford) commas. Use em-dashes without surrounding spaces (---).
2. NUMBERS: Spell out whole numbers from zero through one hundred in general prose ("four tiers", "two matrices", "twenty-one chapters"). Use numerals for percentages (4.2x speedup, 90%), memory byte sizes (64 bytes, 4 MB), dimensions, coordinates, and equations.
3. HEADINGS: Title Case for all chapter and section headings.
4. TONE & REGISTER: Authoritative, inviting, narrative, and deeply grounded in systems reality. Speak directly to the engineer ("When we execute...", "Notice what happens on the hardware...").

FIGURE PLACEHOLDERS & DIAGRAMS:
- Embed existing vector diagrams with clear, descriptive captions: `![Caption text](assets/images/diagrams/xx_name.svg){#fig-label}`.
- If an additional conceptual diagram is needed that does not exist yet, leave a structured placeholder:
  `::: {#fig-concept-name .figure-placeholder}`
  `*Figure X.Y: Detailed descriptive caption explaining the physical/visual layout to be illustrated.*`
  `:::`

CHAPTER STRUCTURE (4 MANDATORY MOVEMENTS):
1. THE CRISIS / HOOK: Open immediately with the engineering tension. Why does standard software fail? (Memory fragmentation, cache line misses, linear collapse walls, O(N) GPU memory bandwidth stalls).
2. THE MENTAL MODEL & GEOMETRY: The physical insight before touching code.
3. THE PURE CONSTRUCTION: Clean, minimal, working TinyTorch Python implementations with syntax-highlighted code blocks.
4. THE PRODUCTION BRIDGE: Connecting our implementation directly to how production engines (PyTorch c10::TensorImpl, CUDA kernels, cuDNN, Triton JIT) solve the same invariant at scale.
5. BUILDING THE SYSTEM: HOW IT ALL CONNECTS: Every chapter MUST conclude with an integration section that recaps what was built, how it locks into the previous chapters, and how the global machine learning engine is coming together, finishing with a forward cliffhanger to the next chapter.

STRICT PROGRESSIVE DISCLOSURE INVARIANT:
- Never assume knowledge of tools or layers not yet built.
- In Chapter 1, the reader ONLY has raw memory and Python lists. Do NOT mention autograd graphs, loss functions, or transformers.
- In Chapter 2, we introduce non-linearities because linear layers collapse.
- In Chapter 6, we introduce autograd because analytical manual backprop becomes impossible.
- In Chapter 12, we introduce attention because recurrence cannot parallelize over time.
- In Chapter 18, we introduce KV caching because autoregressive decode recomputation wastes DRAM bandwidth.
"""

CHAPTER_REGISTRY = {
    "preface": {
        "title": "Preface: The xv6 Philosophy for Machine Learning Systems",
        "file": "preface.qmd",
        "crisis": "Why do modern ML developers feel alienated by 500,000-line C++ framework codebases?",
        "mental_model": "The pedagogical power of minimal, complete systems (MIT xv6 analogy).",
        "knowledge_prev": "Basic Python and linear algebra.",
        "knowledge_new": "The 4-tier architecture of TinyTorch (4,558 pure lines of code) and how to read this book as an active builder.",
        "forbidden": "Do not give implementation details of specific layers yet.",
        "diagram": "00_journey-diag-1.svg",
        "next_bridge": "Stepping into the engine room: Why we must start from scratch."
    },
    "00_welcome": {
        "title": "Welcome: The Black Box Crisis",
        "file": "00_welcome.qmd",
        "crisis": "Machine learning has become 'import torch; model.fit()' alchemy without understanding the machine.",
        "mental_model": "De-mystifying the abstraction layers from DRAM memory bytes to token generation.",
        "knowledge_prev": "Preface.",
        "knowledge_new": "The contract of TinyTorch: Zero external ML dependencies, complete transparency, test-driven validation.",
        "forbidden": "Do not dive into tensor stride math yet.",
        "diagram": "00_journey-diag-1.svg",
        "next_bridge": "Chapter 1: The foundation of all ML computing—building the multidimensional Tensor."
    },
    "01_tensors": {
        "title": "Chapter 1: Tensors & Strides — The Flat Memory Illusion",
        "file": "01_tensors.qmd",
        "crisis": "Nested Python lists fragment pointers across the OS heap, destroying CPU cache lines and preventing vectorized SIMD math.",
        "mental_model": "A tensor is a flat 1D memory buffer viewed through an algebraic coordinate translator (shape, strides, storage_offset).",
        "knowledge_prev": "Raw Python and flat 1D arrays.",
        "knowledge_new": "Multi-dimensional indexing formula: Offset = Offset_0 + sum(i_k * s_k). Zero-copy views (transpose, reshape, slice) and broadcast stride-0 trick.",
        "forbidden": "Do not mention backward passes, autograd, or neural layers.",
        "diagram": "01_tensor-diag-1.svg",
        "src_file": "core/tensor.py",
        "next_bridge": "Now that we have multi-dimensional matrix operations, what happens when we stack linear layers? The linear collapse wall."
    },
    "02_activations": {
        "title": "Chapter 2: Activations — Breaking the Linear Collapse Wall",
        "file": "02_activations.qmd",
        "crisis": "Stacking 100 linear layers W_2(W_1 x) mathematically collapses into a single matrix W_comb = W_2 W_1. Deep networks become shallow linear regressions.",
        "mental_model": "Non-linear activations act as mathematical gates that warp decision boundaries and prevent linear collapse.",
        "knowledge_prev": "Tensor class with contiguous memory and matmul.",
        "knowledge_new": "ReLU (piecewise linear), Sigmoid, Tanh, and modern GELU (Gaussian Error Linear Unit) used in GPT-2.",
        "forbidden": "Do not introduce autograd backward derivatives yet; focus on forward activation mechanics.",
        "diagram": "02_activations-diag-1.svg",
        "src_file": "core/activations.py",
        "next_bridge": "We have tensors and non-linearities. How do we package them into reusable, modular neural network building blocks?"
    },
    "03_layers": {
        "title": "Chapter 3: Layers & Parameters — Packaging the Affine Transform",
        "file": "03_layers.qmd",
        "crisis": "Hardcoding matrix multiplications and bias additions manually across networks causes parameter tracking leaks and shape mismatches.",
        "mental_model": "The Layer container: encapsulation of state (learnable weights and biases) and computation (forward transformation) with initialization variance control.",
        "knowledge_prev": "Tensor, matmul, activations.",
        "knowledge_new": "Linear layer Y = XW^T + b, Kaiming/He initialization bounds (Var(W) = 2/fan_in), Dropout, and Sequential container.",
        "forbidden": "Do not introduce backward passes yet.",
        "diagram": "03_layers-diag-1.svg",
        "src_file": "core/layers.py",
        "next_bridge": "Our model produces raw continuous numbers (logits). How do we measure how wrong our predictions are without causing floating-point overflow?"
    },
    "04_losses": {
        "title": "Chapter 4: Loss Functions & Log-Sum-Exp — Numerical Hazards in Probability Space",
        "file": "04_losses.qmd",
        "crisis": "Computing exp(z) on logits like z=1000 causes IEEE 754 float overflow (inf), while z=-1000 causes underflow (0.0), crashing training with NaN gradients.",
        "mental_model": "Log-Sum-Exp subtraction invariant: shifting logits by max(z) guarantees mathematical equivalence while clamping float values into safe exponent ranges.",
        "knowledge_prev": "Tensors, Layers, forward logits.",
        "knowledge_new": "MSE Loss, Cross-Entropy Loss, Softmax probabilities, and the Log-Sum-Exp numerical stabilization trick.",
        "forbidden": "Do not introduce backward autograd engine yet.",
        "diagram": "04_losses-diag-1.svg",
        "src_file": "core/losses.py",
        "next_bridge": "We have a model and a loss. But where does the data come from? How do we feed millions of samples without stalling the CPU?"
    },
    "05_dataloader": {
        "title": "Chapter 5: The DataLoader — Asynchronous Feeding of the Compute Engine",
        "file": "05_dataloader.qmd",
        "crisis": "Loading single samples from disk synchronously during training starves the compute engine, causing 90% GPU/CPU idle stall time.",
        "mental_model": "The Producer-Consumer Pipeline: Decoupling disk I/O, dataset indexing, shuffling, and batch collation from the compute execution loop.",
        "knowledge_prev": "Dataset structures, Tensors.",
        "knowledge_new": "Dataset protocol, random sampling without replacement, batch collation, and async memory pinning.",
        "forbidden": "Do not introduce backprop yet.",
        "diagram": "05_dataloader-diag-1.svg",
        "src_file": "core/dataloader.py",
        "next_bridge": "We have batches of data flowing and loss computing. Now comes the central miracle of deep learning: How do we compute gradients for millions of parameters automatically?"
    },
    "06_autograd": {
        "title": "Chapter 6: Automatic Differentiation — The Dynamic Tape DAG",
        "file": "06_autograd.qmd",
        "crisis": "Symbolic differentiation expressions explode exponentially in size, while numerical finite differences require O(N) forward passes for N parameters.",
        "mental_model": "Reverse-mode automatic differentiation on a dynamic tape: recording elementary vector-Jacobian operations during forward pass, then executing a reverse topological sort.",
        "knowledge_prev": "Forward passes across Tensors, Layers, Losses.",
        "knowledge_new": "Computational graph (DAG), Node, Op registration, reverse-mode chain rule, in-place gradient accumulation (param.grad += grad), and zero_grad().",
        "forbidden": "Do not introduce momentum or AdamW yet; focus purely on exact gradient calculation.",
        "diagram": "06_autograd-diag-1.svg",
        "src_file": "core/autograd.py",
        "next_bridge": "We now have exact gradient vectors pointing uphill on the loss surface. How do we use them to update parameters without getting stuck in saddle points?"
    },
    "07_optimizers": {
        "title": "Chapter 7: Optimizers — Momentum, AdamW, and Decoupled Weight Decay",
        "file": "07_optimizers.qmd",
        "crisis": "Vanilla SGD oscillates wildly in high-curvature ravines and stalls in flat plateaus. Adding L2 regularization to Adam distorts gradient variance estimates.",
        "mental_model": "Heavy-ball physics (Momentum velocity buffer) and decoupled adaptive moment estimation (AdamW: separate step update from weight decay).",
        "knowledge_prev": "Tensors, Layers, autograd backward pass.",
        "knowledge_new": "SGD with Momentum, Adam, AdamW (Loshchilov & Hutter), Cosine Annealing learning rate schedules.",
        "forbidden": "Do not build full training loops yet.",
        "diagram": "07_optimizers-diag-1.svg",
        "src_file": "core/optimizers.py",
        "next_bridge": "We have all individual components: Model, DataLoader, Loss, Autograd, and Optimizer. How do we orchestrate them into a bulletproof 5-step training engine?"
    },
    "08_training": {
        "title": "Chapter 8: The Training Engine — The Rigid Five-Step Loop Contract",
        "file": "08_training.qmd",
        "crisis": "Misordering loop steps (e.g. stepping optimizer before backward, or forgetting zero_grad) causes silent mathematical corruption and exploding gradients.",
        "mental_model": "The rigid 5-step state transition contract: zero_grad -> forward -> loss -> backward -> step, paired with gradient norm clipping and atomic serialization.",
        "knowledge_prev": "DataLoader, Autograd, Optimizers, Losses.",
        "knowledge_new": "The 5-step loop invariant, eval mode vs train mode, gradient norm clipping (clip_grad_norm), metric tracking, and checkpoint save/load.",
        "forbidden": "Do not introduce spatial vision or transformers yet.",
        "diagram": "08_training-diag-1.svg",
        "src_file": "core/training.py",
        "next_bridge": "Milestone I: Recreating historical breakthroughs from Rosenblatt's 1958 Perceptron to solving Minsky's XOR crisis."
    },
    "milestone_01": {
        "title": "Milestone I: The Historic Leap — From Perceptrons to Rumelhart's MLP",
        "file": "milestone_01.qmd",
        "crisis": "Minsky & Papert (1969) proved single-layer perceptrons cannot solve XOR, triggering the First AI Winter.",
        "mental_model": "Solving XOR with hidden representations and validating our complete Tier 1 & 2 engine against historical milestones.",
        "knowledge_prev": "Full Core Engine (Modules 01-08).",
        "knowledge_new": "Historical validation on Rosenblatt Perceptron, XOR problem, and 1986 Rumelhart MLP on TinyDigits.",
        "forbidden": "Do not introduce 2D convolutions or transformers yet.",
        "diagram": "08_training-diag-1.svg",
        "next_bridge": "Part II: Moving beyond 1D vectors into multidimensional spatial vision and human language."
    },
    "09_convolutions": {
        "title": "Chapter 9: Spatial Convolutions — Preserving 2D Topology via im2col GEMMs",
        "file": "09_convolutions.qmd",
        "crisis": "Flattening 2D images into 1D vectors destroys spatial locality and causes parameter explosion (a 1000x1000 image requires 1,000,000 weights per hidden neuron).",
        "mental_model": "Translational equivariance via shared local receptive fields, and the im2col memory unrolling trick to convert sliding windows into dense systolic GEMMs.",
        "knowledge_prev": "Linear layers, autograd, 2D tensors.",
        "knowledge_new": "Conv2d forward, padding and stride algebra, MaxPool2d, Flatten, and im2col matrix layout transformation.",
        "forbidden": "Do not introduce language models or tokenizers yet.",
        "diagram": "09_convolutions-diag-1.svg",
        "src_file": "core/spatial.py",
        "next_bridge": "We have conquered continuous spatial grids (pixels). How do we handle discrete, variable-length symbolic data: human language?"
    },
    "10_tokenization": {
        "title": "Chapter 10: Byte-Pair Encoding — Compressing Language into Subword Tokens",
        "file": "10_tokenization.qmd",
        "crisis": "Character-level tokens make sequences too long (exploding memory), while word-level tokens cause out-of-vocabulary crashes on unseen words.",
        "mental_model": "Information entropy compression via iterative frequency merging: starting with 256 UTF-8 bytes and iteratively merging frequent pairs (BPE).",
        "knowledge_prev": "Raw strings and byte streams.",
        "knowledge_new": "Byte-Pair Encoding (BPE), vocabulary induction, merge rank tables, encode/decode pipelines with zero out-of-vocabulary errors.",
        "forbidden": "Do not introduce embedding lookup tables or attention yet.",
        "diagram": "10_tokenization-diag-1.svg",
        "src_file": "core/tokenization.py",
        "next_bridge": "Tokens are discrete integer IDs (e.g. 15496). How do we project discrete indices into continuous semantic vector spaces?"
    },
    "11_embeddings": {
        "title": "Chapter 11: Embeddings & Position — Projecting Meaning and Temporal Waveforms",
        "file": "11_embeddings.qmd",
        "crisis": "One-hot vectors have zero semantic distance (every word is orthogonal), and permutation-invariant attention has zero knowledge of word order.",
        "mental_model": "Zero-compute pointer lookup into continuous dense embedding matrices, plus geometric sinusoidal frequency waves for temporal position.",
        "knowledge_prev": "Token IDs, linear layers.",
        "knowledge_new": "Embedding lookup table, learned positional embeddings, Vaswani sinusoidal geometric encodings (sin/cos frequency scales).",
        "forbidden": "Do not introduce attention QKV projections yet.",
        "diagram": "11_embeddings-diag-1.svg",
        "src_file": "core/embeddings.py",
        "next_bridge": "We have token vectors with position. How do tokens dynamically communicate with every other token in the sequence? Attention."
    },
    "12_attention": {
        "title": "Chapter 12: Attention Mechanisms — Scaled Dot-Product & Causal Masking",
        "file": "12_attention.qmd",
        "crisis": "Recurrent neural networks (RNNs) cannot parallelize over time, and raw dot products explode in variance with dimension d_k, saturating softmax gradients to zero.",
        "mental_model": "The database lookup analogy: Queries match Keys to produce routing weights for Values, scaled by 1/sqrt(d_k) to maintain unit variance.",
        "knowledge_prev": "Embeddings, positional vectors, softmax.",
        "knowledge_new": "Scaled dot-product attention, multi-head projection split/merge, causal lower-triangular masking (-inf), and batch attention parallelization.",
        "forbidden": "Do not build full transformer blocks with LayerNorm yet.",
        "diagram": "12_attention-diag-1.svg",
        "src_file": "core/attention.py",
        "next_bridge": "We have multi-head attention. How do we stabilize 100-layer stacks and provide feed-forward non-linear transformations? The Transformer Block."
    },
    "13_transformers": {
        "title": "Chapter 13: The Transformer — Assembling GPT-2 with Pre-LN Residual Highways",
        "file": "13_transformers.qmd",
        "crisis": "Deep attention stacks suffer from vanishing/exploding gradient highways unless activations are strictly normalized before transformations.",
        "mental_model": "Pre-LayerNorm residual streams: clean gradient highways through skip connections, interleaved with Multi-Head Attention and 4x MLP expansions.",
        "knowledge_prev": "Attention, Embeddings, Linear layers.",
        "knowledge_new": "LayerNorm (mean/variance normalization), Pre-LN Transformer Block, MLP expansion (4x hidden dim with GELU), and complete TinyGPT language model.",
        "forbidden": "Do not introduce hardware quantization or KV cache yet.",
        "diagram": "13_transformers-diag-1.svg",
        "src_file": "core/transformers.py",
        "next_bridge": "Milestone II: Training our TinyGPT model to generate coherent language from scratch."
    },
    "milestone_02": {
        "title": "Milestone II: Generative Intelligence — Training an Autoregressive Model",
        "file": "milestone_02.qmd",
        "crisis": "Moving from static classification to autoregressive sequence generation: temperature scaling, top-k sampling, and context length limits.",
        "mental_model": "The autoregressive generation loop: predicting the next token distribution and feeding outputs back as inputs.",
        "knowledge_prev": "Full Tier 1 & Tier 2 & Tier 3 architecture.",
        "knowledge_new": "Autoregressive generation loop, temperature sampling, top-k filtering, and training on language corpora.",
        "forbidden": "Do not introduce hardware profiling or quantization yet.",
        "diagram": "13_transformers-diag-1.svg",
        "next_bridge": "Part III: Our framework is mathematically complete. But why is it slow? Entering the realm of systems performance engineering."
    },
    "14_profiling": {
        "title": "Chapter 14: Profiling & Bottlenecks — The Roofline Model and Memory Stalls",
        "file": "14_profiling.qmd",
        "crisis": "Optimizing compute instructions (FLOPs) on memory-bound operators delivers 0% speedup because the hardware is stalled waiting on DRAM bandwidth.",
        "mental_model": "The Roofline Model: Arithmetic Intensity I = FLOPs / Memory_Bytes determines whether an operation is memory-bound or compute-bound.",
        "knowledge_prev": "Complete TinyTorch execution stack.",
        "knowledge_new": "High-resolution latency timing, memory footprint profiling, arithmetic intensity calculation, and identifying memory-bound bottlenecks.",
        "forbidden": "Do not introduce quantization math yet.",
        "diagram": "14_profiling-diag-1.svg",
        "src_file": "perf/profiling.py",
        "next_bridge": "We discovered DRAM memory traffic is our bottleneck. How do we cut memory traffic by 4x without retraining? INT8 Quantization."
    },
    "15_quantization": {
        "title": "Chapter 15: INT8 Quantization — Compressing Floats into Symmetric Integers",
        "file": "15_quantization.qmd",
        "crisis": "FP32 weights take 4 bytes per parameter; transferring them across memory buses dominates 80% of inference energy and latency.",
        "mental_model": "Symmetric affine quantization: mapping continuous 32-bit float ranges into 8-bit signed integers [-128, 127] via scale factor S = max(|X|) / 127.",
        "knowledge_prev": "Linear layers, memory profiling.",
        "knowledge_new": "Quantizer, QuantizedLinear, symmetric uniform scaling, calibration on validation batches, and 4x memory footprint reduction.",
        "forbidden": "Do not introduce pruning or distillation yet.",
        "diagram": "15_quantization-diag-1.svg",
        "src_file": "perf/quantization.py",
        "next_bridge": "Quantization shrinks precision. Can we also eliminate parameters entirely? Model Compression and Pruning."
    },
    "16_compression": {
        "title": "Chapter 16: Model Compression — Pruning, Low-Rank SVD, and Distillation",
        "file": "16_compression.qmd",
        "crisis": "Unstructured sparse pruning zeroes out individual weights but leaves matrix dimensions unchanged, resulting in zero speedup on dense hardware GEMM cores.",
        "mental_model": "Structured pruning (removing entire channels/heads) and Low-Rank SVD factorizations that physically shrink matrix dimensions for dense hardware acceleration.",
        "knowledge_prev": "Linear layers, Quantization.",
        "knowledge_new": "Magnitude pruning vs Structured channel pruning, Low-Rank SVD decomposition W = U V, and Knowledge Distillation loss.",
        "forbidden": "Do not introduce kernel fusion yet.",
        "diagram": "16_compression-diag-1.svg",
        "src_file": "perf/compression.py",
        "next_bridge": "Our model is compressed. But why do consecutive elementwise operations launch separate GPU kernels? Kernel Fusion."
    },
    "17_acceleration": {
        "title": "Chapter 17: Hardware Acceleration — Kernel Fusion and SRAM Register Residency",
        "file": "17_acceleration.qmd",
        "crisis": "Executing Add followed by GELU requires 3 full roundtrips to slow off-chip DRAM, wasting 70% of execution time on memory bus traffic.",
        "mental_model": "Kernel Fusion: keeping intermediate results resident in ultra-fast on-chip SRAM registers, doing multiple mathematical operations in a single memory pass.",
        "knowledge_prev": "Vector operations, Roofline intensity.",
        "knowledge_new": "Vectorized SIMD operations, cache-tiled matrix multiplication, and Fused Add-GELU operator engines.",
        "forbidden": "Do not introduce KV cache yet.",
        "diagram": "17_acceleration-diag-1.svg",
        "src_file": "perf/acceleration.py",
        "next_bridge": "During transformer text generation, why do we recompute attention keys and values for all past tokens? The KV Cache."
    },
    "18_memoization": {
        "title": "Chapter 18: The KV Cache — Converting O(N) Generation Bandwidth into O(1)",
        "file": "18_memoization.qmd",
        "crisis": "Autoregressive generation re-calculates attention Keys and Values for all previous N-1 tokens at every step, causing O(N^2) total compute and severe memory bandwidth thrashing.",
        "mental_model": "KV Cache Memoization: appending only the newest token's (K_new, V_new) into static pre-allocated SRAM/DRAM buffers, turning token decode into an O(1) matrix-vector operation.",
        "knowledge_prev": "Attention mechanism, Transformer blocks.",
        "knowledge_new": "KVCache buffer implementation, cache hit/update lifecycle, and 4.2x autoregressive generation throughput speedup.",
        "forbidden": "Do not introduce MLPerf benchmark harness yet.",
        "diagram": "18_memoization-diag-1.svg",
        "src_file": "perf/memoization.py",
        "next_bridge": "We have built multiple performance optimizations. How do we measure their real-world impact with scientific and statistical rigor?"
    },
    "19_benchmarking": {
        "title": "Chapter 19: Rigorous Benchmarking — GPU Synchronization and Tail Latencies",
        "file": "19_benchmarking.qmd",
        "crisis": "Measuring execution time without warmup passes captures cold JIT/cache misses, and timing asynchronous GPU calls without synchronization measures queue latency rather than compute time.",
        "mental_model": "Statistical benchmarking rigor: Warmup stabilization, hardware synchronization barriers, throughput (samples/sec), and tail latency percentiles (P50, P95, P99).",
        "knowledge_prev": "Full optimization tier.",
        "knowledge_new": "Benchmark suite, warmup passes, device synchronization, P50/P95/P99 percentile calculation, and MLPerf compliance.",
        "forbidden": "Do not assemble the final capstone stack yet.",
        "diagram": "19_benchmarking-diag-1.svg",
        "src_file": "perf/benchmarking.py",
        "next_bridge": "Now, we bring everything together into a unified system: The 16x Capstone Performance Stack."
    },
    "20_capstone": {
        "title": "Chapter 20: The Capstone — Building the 16× Cumulative Acceleration Stack",
        "file": "20_capstone.qmd",
        "crisis": "Individual optimizations only yield 2x-3x speedups in isolation; how do we stack quantization, operator fusion, and KV caching to achieve a 16x multiplicative leap?",
        "mental_model": "Amdahl's Multiplier Stack: attacking memory footprint, kernel launches, and decode complexity across every tier of the systems architecture.",
        "knowledge_prev": "All 19 previous modules.",
        "knowledge_new": "Full TinyTorch Capstone architecture, cumulative speedup multiplication, and complete framework verification.",
        "forbidden": "None (all modules integrated).",
        "diagram": "20_capstone-diag-1.svg",
        "src_file": "olympics.py",
        "next_bridge": "Milestone III: The Torch Olympics — Evaluating TinyTorch on production MLPerf workloads."
    },
    "milestone_03": {
        "title": "Milestone III: The Torch Olympics — Real-World MLPerf Performance Showdown",
        "file": "milestone_03.qmd",
        "crisis": "Does our from-scratch Python framework hold up against rigorous industry benchmarks?",
        "mental_model": "The Torch Olympics: automated scoring across correctness, training throughput, and inference speedup.",
        "knowledge_prev": "Full TinyTorch Framework.",
        "knowledge_new": "Torch Olympics leaderboard, MLPerf test verification, and automated evaluation metrics.",
        "forbidden": "None.",
        "diagram": "20_capstone-diag-1.svg",
        "next_bridge": "Part IV: Where does modern ML systems engineering go from here? Beyond TinyTorch."
    },
    "21_extensions": {
        "title": "Chapter 21: The Modern Stack — Triton JIT, Custom Silicon, and Physical Systems",
        "file": "21_extensions.qmd",
        "crisis": "Python interpreters have overhead; how does the industry bridge from high-level Python code to custom GPU silicon?",
        "mental_model": "The modern compiler stack: PyTorch torch.compile, TorchInductor, OpenAI Triton JIT, and domain-specific accelerators (TPUs, NPUs).",
        "knowledge_prev": "Complete TinyTorch book.",
        "knowledge_new": "OpenAI Triton block programming model, torch.compile graph capture, custom silicon memory hierarchies, and edge robotics.",
        "forbidden": "None.",
        "diagram": "00_journey-diag-1.svg",
        "next_bridge": "Epilogue: You didn't just import torch. You built it."
    }
}

def build_prompt_for_chapter(chapter_key: str) -> str:
    """Constructs a comprehensive, CMOS-compliant prompt for generating a book chapter."""
    info = CHAPTER_REGISTRY.get(chapter_key)
    if not info:
        raise ValueError(f"Unknown chapter key: {chapter_key}")

    src_code_snippet = ""
    if "src_file" in info:
        src_path = SRC_DIR / info["src_file"]
        if src_path.exists():
            with open(src_path, "r", encoding="utf-8") as f:
                src_code_snippet = f.read()

    prompt = f"""{MASTER_SYSTEM_PROMPT}

You are writing:
CHAPTER TITLE: {info['title']}
TARGET FILE: packages/tinytorch/narrative_book/{info['file']}

CHAPTER CONTEXT & INVARIANTS:
- The Engineering Crisis: {info['crisis']}
- The Mental Model: {info['mental_model']}
- Reader's Prior Knowledge: {info['knowledge_prev']}
- Concepts Introduced in this Chapter: {info['knowledge_new']}
- STRICT BOUNDARY (Do NOT introduce): {info['forbidden']}
- Primary Vector Diagram: assets/images/diagrams/{info.get('diagram', '')}
- Chapter Exit Cliffhanger: {info['next_bridge']}

TINYTORCH SOURCE CODE REFERENCE:
```python
{src_code_snippet[:4500]}
```

WRITING INSTRUCTIONS:
1. Write the complete, publication-grade Quarto Markdown (.qmd) file for this chapter adhering strictly to CMOS rules.
2. DO NOT include "Purpose" or "Learning Objectives" sections.
3. Open immediately with Section X.1: The Crisis (the engineering problem/tension).
4. Embed the primary vector diagram and include clear figure placeholders for any additional visual architectures needed.
5. Provide clean, tested TinyTorch Python implementations with syntax-highlighted code blocks.
6. Add Systems Perspective callouts explaining hardware realities (cache lines, DRAM bandwidth, C10/CUDA equivalents).
7. End with "Section X.N: Building the System: How It All Connects" which recaps what was built, how it locks into the previous chapters, and how the global machine learning engine is coming together, finishing with the specified cliffhanger bridge.
8. Write the complete file directly to `packages/tinytorch/narrative_book/{info['file']}`.
"""
    return prompt


def orchestrate_chapter(chapter_key: str, dry_run: bool = False):
    """Invokes agy with the tailored prompt for a given chapter."""
    prompt = build_prompt_for_chapter(chapter_key)
    target_file = NARRATIVE_BOOK_DIR / CHAPTER_REGISTRY[chapter_key]["file"]

    print(f"\n================================================================================")
    print(f"📖 ORCHESTRATING CHAPTER: {CHAPTER_REGISTRY[chapter_key]['title']}")
    print(f"📁 TARGET FILE: {target_file}")
    print(f"================================================================================\n")

    if dry_run:
        print("[DRY RUN] Generated CMOS Prompt:\n")
        print(prompt[:1200] + "\n... [TRUNCATED] ...\n")
        return

    cmd = [
        "/Users/VJ/.local/bin/agy",
        "-p", prompt,
        "--dangerously-skip-permissions"
    ]
    print(f"🚀 Invoking Antigravity (agy)...")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    print(f"✓ Chapter {chapter_key} successfully authored!")


def main():
    parser = argparse.ArgumentParser(description="TinyTorch Narrative Book Orchestrator")
    parser.add_argument("--chapter", type=str, help="Specific chapter key to orchestrate (e.g., '01_tensors', 'preface')")
    parser.add_argument("--all", action="store_true", help="Orchestrate all chapters sequentially")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling agy")
    parser.add_argument("--render", action="store_true", help="Render Quarto book to PDF after generation")

    args = parser.parse_args()

    if args.chapter:
        orchestrate_chapter(args.chapter, dry_run=args.dry_run)
    elif args.all:
        for ch in CHAPTER_REGISTRY:
            orchestrate_chapter(ch, dry_run=args.dry_run)
    else:
        print("Please specify --chapter <name> or --all. Available chapters:")
        for k, v in CHAPTER_REGISTRY.items():
            print(f"  - {k:15s}: {v['title']}")
        return

    if args.render and not args.dry_run:
        print("\n🔨 Rendering TinyTorch Narrative Book to PDF...")
        subprocess.run(["quarto", "render", "--to", "pdf"], cwd=str(NARRATIVE_BOOK_DIR), check=True)
        print("✓ PDF compilation complete!")


if __name__ == "__main__":
    main()
