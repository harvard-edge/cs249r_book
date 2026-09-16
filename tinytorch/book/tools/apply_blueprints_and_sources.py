#!/usr/bin/env python3
"""
Apply Framework Blueprint "You Are Here" Navigators and Terminal Source Code Mapping Cards
across all remaining chapters (02-21, M01-M03).
"""
import re
from pathlib import Path

CHAPTER_DATA = {
    "02_activations.qmd": {
        "title": "Module 02 Activations",
        "bp_name": "02_framework-you-are-here",
        "src_name": "02_activation-margin-source",
        "bp_caption": "Module 02 introduces non-linear activation functions that bend representation space between linear transformations.",
        "src_caption": "Module 02 extracts elementwise activations (ReLU, Sigmoid, Tanh, GELU) and Softmax normalization directly from the working repository.",
    },
    "03_layers.qmd": {
        "title": "Module 03 Layers",
        "bp_name": "03_framework-you-are-here",
        "src_name": "03_layers-margin-source",
        "bp_caption": "Module 03 encapsulates stateful weights and modular execution into composable layer blocks.",
        "src_caption": "Module 03 extracts Linear affine transformations, Dropout regularization, and Sequential containers directly from the working repository.",
    },
    "04_losses.qmd": {
        "title": "Module 04 Losses",
        "bp_name": "04_framework-you-are-here",
        "src_name": "04_losses-margin-source",
        "bp_caption": "Module 04 evaluates network predictions against targets while guarding against numerical overflow.",
        "src_caption": "Module 04 extracts MSELoss, numerically stabilized CrossEntropyLoss, and BinaryCrossEntropyLoss directly from the working repository.",
    },
    "05_dataloader.qmd": {
        "title": "Module 05 DataLoader",
        "bp_name": "05_framework-you-are-here",
        "src_name": "05_dataloader-margin-source",
        "bp_caption": "Module 05 feeds the compute engine by batching, shuffling, and collating samples into contiguous memory.",
        "src_caption": "Module 05 extracts Dataset sample indexing, DataLoader shuffling, and batch collation directly from the working repository.",
    },
    "06_autograd.qmd": {
        "title": "Module 06 Autograd",
        "bp_name": "06_framework-you-are-here",
        "src_name": "06_autograd-margin-source",
        "bp_caption": "Module 06 builds the computation DAG and executes reverse-mode automatic differentiation via topological sort.",
        "src_caption": "Module 06 extracts automatic differentiation, the backward execution tape, and gradient accumulation directly from the working repository.",
    },
    "07_optimizers.qmd": {
        "title": "Module 07 Optimizers",
        "bp_name": "07_framework-you-are-here",
        "src_name": "07_optimizers-margin-source",
        "bp_caption": "Module 07 translates raw backward gradients into parameter trajectory updates through momentum and adaptive moments.",
        "src_caption": "Module 07 extracts SGD with momentum, Adam moment trackers, and AdamW decoupled weight decay directly from the working repository.",
    },
    "08_training.qmd": {
        "title": "Module 08 Training",
        "bp_name": "08_framework-you-are-here",
        "src_name": "08_training-margin-source",
        "bp_caption": "Module 08 enforces the canonical training step cycle and coordinates convergence across epochs.",
        "src_caption": "Module 08 extracts the canonical training loop, cosine learning rate schedules, and gradient clipping directly from the working repository.",
    },
    "milestone_01.qmd": {
        "title": "Milestone I MLP Synthesis",
        "bp_name": "m01_framework-you-are-here",
        "src_name": "m01_xor-margin-source",
        "bp_caption": "Milestone I synthesizes all Part I mechanisms into a complete neural network to solve the non-linear XOR boundary.",
        "src_caption": "Milestone I combines Modules 01–08 to build and train a multi-layer perceptron from scratch.",
    },
    "09_convolutions.qmd": {
        "title": "Module 09 Convolutions",
        "bp_name": "09_framework-you-are-here",
        "src_name": "09_conv-margin-source",
        "bp_caption": "Module 09 exploits 2D spatial locality by lowering sliding receptive field convolutions to optimized GEMM matrix multiplication.",
        "src_caption": "Module 09 extracts Conv2d spatial convolution, im2col buffer unrolling, and MaxPool2d downsampling directly from the working repository.",
    },
    "10_tokenization.qmd": {
        "title": "Module 10 Tokenization",
        "bp_name": "10_framework-you-are-here",
        "src_name": "10_token-margin-source",
        "bp_caption": "Module 10 bridges unstructured text and tensor mathematics through Byte-Pair Encoding subword tokenization.",
        "src_caption": "Module 10 extracts Byte-Pair Encoding (BPE) vocabulary construction, token encoding, and decoding directly from the working repository.",
    },
    "11_embeddings.qmd": {
        "title": "Module 11 Embeddings",
        "bp_name": "11_framework-you-are-here",
        "src_name": "11_embed-margin-source",
        "bp_caption": "Module 11 projects discrete token integers into dense continuous geometry and encodes sequential ordering.",
        "src_caption": "Module 11 extracts embedding table lookup and sinusoidal positional encodings directly from the working repository.",
    },
    "12_attention.qmd": {
        "title": "Module 12 Attention",
        "bp_name": "12_framework-you-are-here",
        "src_name": "12_attn-margin-source",
        "bp_caption": "Module 12 implements scaled dot-product attention and multi-head projection with lower-triangular causal masking.",
        "src_caption": "Module 12 extracts scaled dot-product attention, causal mask application, and MultiHeadAttention directly from the working repository.",
    },
    "13_transformers.qmd": {
        "title": "Module 13 Transformers",
        "bp_name": "13_framework-you-are-here",
        "src_name": "13_trans-margin-source",
        "bp_caption": "Module 13 synthesizes causal self-attention, feedforward MLPs, and Pre-LN residual highways into the complete TinyGPT model.",
        "src_caption": "Module 13 extracts LayerNorm, MLP expansion, TransformerBlock, and the flagship TinyGPT model directly from the working repository.",
    },
    "milestone_02.qmd": {
        "title": "Milestone II Language Model Generation",
        "bp_name": "m02_framework-you-are-here",
        "src_name": "m02_gen-margin-source",
        "bp_caption": "Milestone II closes the generative loop: prompting TinyGPT, shaping probability distributions, and streaming autoregressive tokens.",
        "src_caption": "Milestone II extracts the autoregressive token generation loop, temperature scaling, and top-k sampling directly from the working repository.",
    },
    "14_profiling.qmd": {
        "title": "Module 14 Profiling",
        "bp_name": "14_framework-you-are-here",
        "src_name": "14_prof-margin-source",
        "bp_caption": "Module 14 grounds execution in physics, computing operational intensity and plotting operations on the Roofline model.",
        "src_caption": "Module 14 extracts analytical FLOP accounting, parameter memory quantification, and roofline profiling directly from the working repository.",
    },
    "15_quantization.qmd": {
        "title": "Module 15 Quantization",
        "bp_name": "15_framework-you-are-here",
        "src_name": "15_quant-margin-source",
        "bp_caption": "Module 15 compresses 32-bit floats into 8-bit integers, quadrupling cache line density while maintaining numerical precision.",
        "src_caption": "Module 15 extracts affine INT8 quantization, dynamic range calibration, and integer matrix multiplication directly from the working repository.",
    },
    "16_compression.qmd": {
        "title": "Module 16 Compression",
        "bp_name": "16_framework-you-are-here",
        "src_name": "16_comp-margin-source",
        "bp_caption": "Module 16 transfers teacher dark knowledge to a compact student network via softened probability distributions.",
        "src_caption": "Module 16 extracts knowledge distillation loss, magnitude pruning, and low-rank approximation directly from the working repository.",
    },
    "17_acceleration.qmd": {
        "title": "Module 17 Acceleration",
        "bp_name": "17_framework-you-are-here",
        "src_name": "17_accel-margin-source",
        "bp_caption": "Module 17 eliminates round-trip memory traffic to DRAM by fusing adjacent operators into single-pass kernel loops.",
        "src_caption": "Module 17 extracts operator fusion, memory traffic accounting, and tiled matrix multiplication directly from the working repository.",
    },
    "18_memoization.qmd": {
        "title": "Module 18 Memoization",
        "bp_name": "18_framework-you-are-here",
        "src_name": "18_memo-margin-source",
        "bp_caption": "Module 18 eliminates O(S²) autoregressive recomputation through static Key-Value cache buffer management.",
        "src_caption": "Module 18 extracts pre-allocated Key-Value caches, cursor write indexing, and cached attention directly from the working repository.",
    },
    "19_benchmarking.qmd": {
        "title": "Module 19 Benchmarking",
        "bp_name": "19_framework-you-are-here",
        "src_name": "19_bench-margin-source",
        "bp_caption": "Module 19 establishes experimental rigour: discarding cold-start caches, measuring percentiles, and auditing throughput.",
        "src_caption": "Module 19 extracts benchmark timing harnesses, latency percentile metrics, and MLPerf compliance directly from the working repository.",
    },
    "20_capstone.qmd": {
        "title": "Module 20 Capstone",
        "bp_name": "20_framework-you-are-here",
        "src_name": "20_caps-margin-source",
        "bp_caption": "Module 20 synthesizes training, profiling, quantization, and caching into a verified production optimization report.",
        "src_caption": "Module 20 extracts end-to-end benchmark reporting, optimization comparison, and competition submission generation directly from the working repository.",
    },
    "milestone_03.qmd": {
        "title": "Milestone III Torch Olympics",
        "bp_name": "m03_framework-you-are-here",
        "src_name": "m03_edge-margin-source",
        "bp_caption": "Milestone III deploys optimized TinyGPT to edge hardware constraints, charting the multi-objective Pareto frontier.",
        "src_caption": "Milestone III benchmarks the combined impact of INT8 quantization, KV caching, and operator fusion under strict hardware ceilings.",
    },
    "21_extensions.qmd": {
        "title": "Chapter 21 Extensions",
        "bp_name": "21_framework-you-are-here",
        "src_name": "21_ext-margin-source",
        "bp_caption": "Chapter 21 reaches beyond the educational interpreter to examine graph compilers, custom GPU kernels, and TPU systolic arrays.",
        "src_caption": "Chapter 21 presents architectural models of computational graph capture, Triton GPU execution, and 2D systolic array hardware.",
    },
}


def main():
    for filename, info in CHAPTER_DATA.items():
        p = Path(filename)
        if not p.exists():
            print(f"Skipping {filename}: not found")
            continue

        content = p.read_text(encoding="utf-8")

        # 1. Blueprint block to insert at top right under H1
        bp_block = f"""::: {{.column-margin}}
![](assets/images/diagrams/{info['bp_name']}.svg){{width="100%" fig-alt="TinyTorch execution datapath blueprint highlighting {info['title']} as the active subsystem."}}

*Framework Blueprint: {info['bp_caption']}*
:::
"""

        # Insert after line 1 (# Heading {#sec-...}\n\n)
        h1_match = re.search(r'^(# [^\n]+\n+)', content, re.MULTILINE)
        if h1_match:
            h1_end = h1_match.end()
            # Check if blueprint already inserted
            if info['bp_name'] not in content[:h1_end + 300]:
                content = content[:h1_end] + bp_block + "\n" + content[h1_end:]
                print(f"[{filename}] Inserted Blueprint Navigator at line 2")
            else:
                print(f"[{filename}] Blueprint Navigator already present")

        # 2. Source Code Mapping block to replace under ## The Code
        src_block = f"""::: {{.column-margin}}
![](assets/images/diagrams/{info['src_name']}.svg){{width="100%" fig-alt="Source code mapping card for {info['title']} showing file path, exported package, and tested primitives."}}

*Source Code Mapping: {info['src_caption']}*
:::"""

        # Replace the existing Source Code Mapping margin block
        src_pattern = r':::\s*\{\.column-margin\}\s*\n(\*{0,2}Source Code Mapping.*?\n):::'
        if re.search(src_pattern, content, re.DOTALL):
            content = re.sub(src_pattern, src_block, content, count=1, flags=re.DOTALL)
            print(f"[{filename}] Replaced Source Code Mapping card")
        else:
            print(f"[{filename}] WARNING: Source Code Mapping block not found for replacement!")

        p.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
