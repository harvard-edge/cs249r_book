#!/usr/bin/env python3
"""
Applies curated marginal navigation signposts (jump back / jump ahead) across all chapters.
Reinforces progressive disclosure by providing clickable cross-links in the margin
with ↩ (backward) and ↪ (forward) indicators.
"""

from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent

# Chapter-specific curated marginal jump signposts
# Format: (target_file, search_pattern, position, signpost_content)
# position: 'after' or 'before'
JUMP_SIGNPOSTS = [
    # Chapter 01: Tensors
    (
        "01_tensors.qmd",
        r"### Broadcasting is a zero stride\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-layers  
*Broadcast strides allow parameter bias vectors to add across mini-batches without memory copies.*
:::

"""
    ),
    # Chapter 02: Activations
    (
        "02_activations.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-tensors  
*Elementwise activation kernels operate directly over contiguous 1D memory buffers.*
:::

"""
    ),
    (
        "02_activations.qmd",
        r"### Derivative formulas and the backward pass\n",
        "before",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-acceleration  
*Operator fusion combines activation arithmetic with linear layers, slashing DRAM round-trips.*
:::

"""
    ),
    # Chapter 03: Layers
    (
        "03_layers.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-activations  
*Weight initialization variance scaling accounts for non-linear activation compression (e.g., ReLU).*
:::

"""
    ),
    (
        "03_layers.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-quantization  
*Compacts 32-bit floating-point layer weight matrices into 8-bit integers (INT8).*
:::

"""
    ),
    # Chapter 04: Losses
    (
        "04_losses.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-activations  
*Softmax normalization and the max-subtraction shift prevent catastrophic floating-point overflow.*
:::

"""
    ),
    (
        "04_losses.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-training  
*How scalar loss reductions drive backward gradient propagation and parameter updates.*
:::

"""
    ),
    # Chapter 05: DataLoader
    (
        "05_dataloader.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-tensors  
*Batch collation packs scattered dataset samples into contiguous row-major tensor batches.*
:::

"""
    ),
    (
        "05_dataloader.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-training  
*The DataLoader mini-batch stream powers the unified five-step training execution loop.*
:::

"""
    ),
    # Chapter 06: Autograd
    (
        "06_autograd.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-tensors  
*Reverse topological traversal writes gradient updates directly into leaf tensor `.grad` buffers.*
:::

"""
    ),
    (
        "06_autograd.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-optimizers  
*How the optimizer transforms computed gradient vectors into stable parameter update trajectories.*
:::

"""
    ),
    # Chapter 07: Optimizers
    (
        "07_optimizers.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-autograd  
*Why `zero_grad()` must clear accumulated `.grad` buffers before the next backward pass.*
:::

"""
    ),
    (
        "07_optimizers.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-training  
*Coordinates in-place optimizer parameter mutation within the unified training lifecycle.*
:::

"""
    ),
    # Chapter 08: Training
    (
        "08_training.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-optimizers  
*In-place optimizer updates and gradient clipping bound parameter trajectories during training.*
:::

"""
    ),
    (
        "08_training.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-milestone-1  
*Deploying the five-step training engine to train a multi-layer perceptron from scratch.*
:::

"""
    ),
    # Milestone 01
    (
        "milestone_01.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-layers  
*Composing Linear layers and ReLU activations to bend representation space.*
:::

"""
    ),
    (
        "milestone_01.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-transformers  
*Scaling from 2-layer MLPs to deep, autoregressive Transformer foundation models.*
:::

"""
    ),
    # Chapter 09: Convolutions
    (
        "09_convolutions.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-layers  
*The `im2col` lowering transformation converts spatial sliding windows into standard GEMM operations.*
:::

"""
    ),
    (
        "09_convolutions.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-acceleration  
*How hardware accelerators optimize convolution memory layouts and high-throughput GEMM tiling.*
:::

"""
    ),
    # Chapter 10: Tokenization
    (
        "10_tokenization.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-dataloader  
*Tokenization acts as the front-end CPU ingestion stage that feeds numerical tensor batches.*
:::

"""
    ),
    (
        "10_tokenization.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-embeddings  
*How discrete token IDs map into continuous, high-dimensional vector representations.*
:::

"""
    ),
    # Chapter 11: Embeddings
    (
        "11_embeddings.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-tokenization  
*Discrete token ID integers serve as row indices into the embedding weight table.*
:::

"""
    ),
    (
        "11_embeddings.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-attention  
*Continuous sequence representation tensors $(B, S, D)$ feed scaled dot-product attention.*
:::

"""
    ),
    # Chapter 12: Attention (already has forward link, add backward link)
    (
        "12_attention.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-embeddings  
*Input sequence vectors project into query, key, and value subspaces.*
:::

"""
    ),
    # Chapter 13: Transformers
    (
        "13_transformers.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-attention  
*Causal multi-head self-attention forms the core token-mixing sub-layer of the Transformer block.*
:::

"""
    ),
    (
        "13_transformers.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-milestone-2  
*Autoregressive text generation and next-token probability distribution sampling with TinyGPT.*
:::

"""
    ),
    # Milestone 02
    (
        "milestone_02.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-transformers  
*Full-prefix forward passes through TinyGPT's stacked Pre-LN Transformer decoder blocks.*
:::

"""
    ),
    (
        "milestone_02.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-memoization  
*Eliminating the redundant $O(S^2)$ key-value recomputation during autoregressive generation.*
:::

"""
    ),
    # Chapter 14: Profiling
    (
        "14_profiling.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-transformers  
*Accounting for TinyGPT's parameter budget across multi-head attention and feed-forward sub-layers.*
:::

"""
    ),
    (
        "14_profiling.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-quantization  
*Using INT8 quantization to break memory-bandwidth bottlenecks identified by the Roofline model.*
:::

"""
    ),
    # Chapter 15: Quantization
    (
        "15_quantization.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-profiling  
*Applying operational intensity to determine when weight memory traffic dominates FP32 inference.*
:::

"""
    ),
    (
        "15_quantization.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-compression  
*Combining weight quantization with knowledge distillation for compound model compression.*
:::

"""
    ),
    # Chapter 16: Compression
    (
        "16_compression.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-losses  
*Temperature-scaled Kullback-Leibler (KL) divergence transfers dark knowledge from teacher to student.*
:::

"""
    ),
    (
        "16_compression.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-capstone  
*Integrating distilled student models into the final production optimization pipeline.*
:::

"""
    ),
    # Chapter 17: Acceleration
    (
        "17_acceleration.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-activations  
*Fusing elementwise GELU arithmetic into linear layer buffers eliminates intermediate DRAM traffic.*
:::

"""
    ),
    (
        "17_acceleration.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-extensions  
*How modern ML compilers (OpenAI Triton and PyTorch Inductor) automate kernel fusion at scale.*
:::

"""
    ),
    # Chapter 18: Memoization (already has backward link, add forward link)
    (
        "18_memoization.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-benchmarking  
*Measuring KV cache latency and memory footprint under strict monotonic timing controls.*
:::

"""
    ),
    # Chapter 19: Benchmarking
    (
        "19_benchmarking.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-memoization  
*Benchmarking cached versus uncached autoregressive generation under strict warmup controls.*
:::

"""
    ),
    (
        "19_benchmarking.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-capstone  
*The multi-axis scorecard: evaluating accuracy retention, compression ratio, and runtime speedup.*
:::

"""
    ),
    # Chapter 20: Capstone
    (
        "20_capstone.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-quantization  
*Synthesizing INT8 quantization, distillation, and KV caching into a coherent serving pipeline.*
:::

"""
    ),
    (
        "20_capstone.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-milestone-3  
*The Torch Olympics: deploying optimized candidate models against strict competitive scoring gates.*
:::

"""
    ),
    # Milestone 03
    (
        "milestone_03.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-capstone  
*Evaluating multi-objective trade-offs across accuracy retention, memory compression, and speedup.*
:::

"""
    ),
    (
        "milestone_03.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↪ Jump ahead:** @sec-extensions  
*Scaling from single-node edge deployment to distributed cluster training and custom accelerators.*
:::

"""
    ),
    # Chapter 21: Extensions
    (
        "21_extensions.qmd",
        r"## The Problem\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-transformers  
*Mapping TinyGPT's attention projections and feed-forward networks onto custom systolic array silicon.*
:::

"""
    ),
    (
        "21_extensions.qmd",
        r"## The Idea\n",
        "after",
        """::: {.column-margin}
**↩ Jump back:** @sec-autograd  
*Tracing dynamic Python autograd tapes into static intermediate representation (IR) computation graphs.*
:::

"""
    ),
]


def apply_signposts():
    applied_count = 0
    for filename, pattern, position, signpost in JUMP_SIGNPOSTS:
        filepath = BASE_DIR / filename
        if not filepath.exists():
            print(f"Skipping missing file: {filename}")
            continue

        content = filepath.read_text(encoding="utf-8")

        # Skip if signpost target is already present in this file to avoid duplicates
        target_marker = signpost.strip().split("\n")[1]  # e.g., "**↩ Jump back:** @sec-tensors"
        if target_marker in content:
            print(f"Already present in {filename}: {target_marker}")
            continue

        m = re.search(pattern, content)
        if not m:
            print(f"Pattern '{pattern.strip()}' not found in {filename}")
            continue

        if position == "after":
            idx = m.end()
            new_content = content[:idx] + "\n" + signpost + content[idx:]
        else:
            idx = m.start()
            new_content = content[:idx] + signpost + "\n" + content[idx:]

        filepath.write_text(new_content, encoding="utf-8")
        print(f"Applied to {filename}: {target_marker}")
        applied_count += 1

    print(f"\nTotal signposts applied: {applied_count}")


if __name__ == "__main__":
    apply_signposts()
