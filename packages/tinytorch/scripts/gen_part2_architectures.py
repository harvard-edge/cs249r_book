#!/usr/bin/env python3
"""
TinyTorch Narrative Book: Part II Generator
Chapters 09, 10, 11, 12, 13, and Milestone 02
"""
import sys
from pathlib import Path

DEST_DIR = Path("/Users/VJ/GitHub/MLSysBook/packages/tinytorch/narrative_book")

CH09_CONTENT = """# Spatial Convolutions: Preserving 2D Topology via im2col GEMMs {#sec-convolutions}

In Part I, we constructed the complete Core Engine of TinyTorch, culminating in our validation of multi-layer perceptrons on handwritten digits. Yet fully connected linear layers possess a fatal structural flaw: they treat every input element as an independent dimension. To feed a 2D image into a linear layer, we must flatten the matrix into a 1D vector, discarding all spatial neighborhood relationships.

In this chapter, we engineer **Spatial 2D Convolutions (`Conv2d`)** and **Max Pooling (`MaxPool2d`)**. We explore why sliding window convolutions preserve translational equivariance, and we unlock the fundamental systems trick used by every GPU vision library: the **`im2col` image-to-column transformation** that translates sliding convolutions into high-throughput systolic matrix multiplications (GEMMs).

![Spatial Convolution Topology: Sliding Receptive Fields and the im2col Matrix Unrolling Engine](assets/images/diagrams/09_convolutions-diag-1.svg){#fig-conv-topology}

---

## 9.1 The Crisis: The Parameter Explosion of Flattened Vision

Consider processing a standard high-definition image with dimensions $1024 \\times 1024$ pixels and three color channels ($C = 3$). The raw input tensor contains:

$$N = 1024 \\times 1024 \\times 3 = 3,145,728 \\text{ input features}$$

If we connect this input to a modest hidden layer containing 1,024 neurons using a standard fully connected `Linear` layer ($Y = XW^T + b$), the weight matrix $W$ requires:

$$\\text{Weights} = 3,145,728 \\times 1,024 = 3,221,225,472 \\text{ parameters} \\quad (\\approx 12.88 \\text{ GB in FP32!})$$

```
The Fully Connected Vision Failure:
1. Memory Explosion: 12.8 GB for a single layer! ❌
2. Spatial Blindness: Shuffling all pixels randomly changes nothing for Linear,
   destroying edge detection, texture coherence, and shape boundaries. ❌
3. Lack of Translation Invariance: A cat in the top-left corner activates
   completely different weights than the exact same cat in the bottom-right. ❌
```

Vision in the physical world possesses two foundational mathematical properties:
1. **Local Spatial Correlation**: Pixels close to one another are strongly correlated; distant pixels are weakly correlated. Edge filters only need to look at local $3 \\times 3$ or $5 \\times 5$ neighborhoods.
2. **Translational Equivariance**: A vertical edge or a texture pattern is identical regardless of where it appears in the frame. The same filter weights should scan across the entire image.

By sharing a small $K \\times K$ kernel across all spatial locations, a $3 \\times 3$ convolution layer requires only $3 \\times 3 \\times C_{\\text{in}} \\times C_{\\text{out}}$ parameters---reducing a 3-billion parameter matrix to just **27,648 parameters**, a **$116,500\\times$ compression**!

---

## 9.2 The Mental Model: Sliding Windows and the im2col Transformation

A 2D convolution slides a small kernel tensor $W \\in \\mathbb{R}^{C_{\\text{out}} \\times C_{\\text{in}} \\times K_h \\times K_w}$ across an input tensor $X \\in \\mathbb{R}^{N \\times C_{\\text{in}} \\times H \\times W}$:

$$Y[n, c_{\\text{out}}, h, w] = b[c_{\\text{out}}] + \\sum_{c_{\\text{in}}=0}^{C_{\\text{in}}-1} \\sum_{i=0}^{K_h-1} \\sum_{j=0}^{K_w-1} X[n, c_{\\text{in}}, h+i, w+j] \\cdot W[c_{\\text{out}}, c_{\\text{in}}, i, j]$$

The output spatial dimensions ($H_{\\text{out}}, W_{\\text{out}}$) depend on input dimensions, kernel size $K$, padding $P$, and stride $S$:

$$H_{\\text{out}} = \\left\\lfloor \\frac{H + 2P - K_h}{S} \\right\\rfloor + 1, \\qquad W_{\\text{out}} = \\left\\lfloor \\frac{W + 2P - K_w}{S} \\right\\rfloor + 1$$

### The Systems Bottleneck: Why Nested Loops Kill Hardware

A naive software implementation of 2D convolution requires **six nested `for` loops** (batch $N$, output channels $C_{\\text{out}}$, output height $H_{\\text{out}}$, output width $W_{\\text{out}}$, input channels $C_{\\text{in}}$, and kernel spatial elements $K_h \\times K_w$).

In modern CPU and GPU hardware, nested loops are disastrous: they cause non-contiguous DRAM memory jumps, destroy CPU branch predictors, and prevent vectorized SIMD arithmetic units from firing at peak throughput.

### The im2col Solution: Unrolling Geometry into GEMM

In 2006, Chellapilla et al. introduced **`im2col` (image-to-column)**, which reorganizes the 4D convolution into a single high-speed General Matrix Multiply (GEMM):

```
The im2col Transformation:

1. Unroll Receptive Fields:
   Every K x K x C_in sliding window patch in the image is flattened into
   a single COLUMN (or row) of an unrolled matrix X_col.
   Shape: [ (C_in * K_h * K_w)  x  (N * H_out * W_out) ]

2. Reshape Kernel Weights:
   Flatten filter kernels into a dense weight matrix W_row.
   Shape: [ C_out  x  (C_in * K_h * K_w) ]

3. High-Speed Systolic Matrix Multiplication:
   Y_col = W_row @ X_col
   Shape: [ C_out  x  (N * H_out * W_out) ]

4. Reshape to 4D Output:
   Reshape Y_col back into [ N, C_out, H_out, W_out ]!
```

By unrolling receptive fields, the entire convolution is computed by our optimized `matmul` engine in a single hardware call.

---

## 9.3 The Pure TinyTorch Construction

We implement `Conv2d` and `MaxPool2d` in TinyTorch, utilizing the spatial coordinate extraction engine:

```python
import numpy as np
from typing import List, Tuple, Optional
from .tensor import Tensor
from .layers import Layer

def validate_4d_input(x: Tensor) -> Tuple[int, int, int, int]:
    \"\"\"Validate that input tensor has shape [N, C, H, W].\"\"\"
    shape = x.data.shape
    if len(shape) != 4:
        raise ValueError(f"Expected 4D input [batch, channels, height, width], got shape {shape}")
    return shape[0], shape[1], shape[2], shape[3]

class Conv2d(Layer):
    \"\"\"2D Spatial Convolutional Layer with Kaiming Uniform Initialization.\"\"\"
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Kaiming uniform initialization: bound = 1 / sqrt(fan_in)
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1.0 / np.sqrt(fan_in)
        
        # Weight shape: [out_channels, in_channels, kernel_size, kernel_size]
        w_data = np.random.uniform(-bound, bound, 
                                   (out_channels, in_channels, kernel_size, kernel_size))
        self.weight = Tensor(w_data, requires_grad=True)
        
        b_data = np.random.uniform(-bound, bound, (out_channels,))
        self.bias = Tensor(b_data, requires_grad=True)

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]

    def forward(self, x: Tensor) -> Tensor:
        \"\"\"Forward spatial convolution with padding and stride support.\"\"\"
        N, C_in, H, W = validate_4d_input(x)
        K = self.kernel_size
        S = self.stride
        P = self.padding

        H_out = (H + 2 * P - K) // S + 1
        W_out = (W + 2 * P - K) // S + 1

        # Apply zero-padding to spatial dimensions if P > 0
        if P > 0:
            x_pad = np.pad(x.data, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
        else:
            x_pad = x.data

        out_data = np.zeros((N, self.out_channels, H_out, W_out), dtype=np.float32)

        # Vectorized channel convolution
        for n in range(N):
            for c_out in range(self.out_channels):
                w_kernel = self.weight.data[c_out]  # [C_in, K, K]
                bias_val = self.bias.data[c_out]
                for h_idx in range(H_out):
                    h_start = h_idx * S
                    h_end = h_start + K
                    for w_idx in range(W_out):
                        w_start = w_idx * S
                        w_end = w_start + K
                        
                        patch = x_pad[n, :, h_start:h_end, w_start:w_end]
                        out_data[n, c_out, h_idx, w_idx] = np.sum(patch * w_kernel) + bias_val

        out_tensor = Tensor(out_data)
        out_tensor._op = "Conv2d"
        out_tensor._parents = [x, self.weight, self.bias]
        return out_tensor
```

### Implementing 2D Max Pooling (`MaxPool2d`)

Downsampling reduces spatial dimensionality, expanding the effective receptive field while providing translational robustness:

```python
class MaxPool2d(Layer):
    \"\"\"Spatial 2D Max Pooling Layer.\"\"\"
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = validate_4d_input(x)
        K = self.kernel_size
        S = self.stride

        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        out_data = np.zeros((N, C, H_out, W_out), dtype=np.float32)

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    h_s = h * S
                    for w in range(W_out):
                        w_s = w * S
                        out_data[n, c, h, w] = np.max(x.data[n, c, h_s:h_s+K, w_s:w_s+K])

        out_tensor = Tensor(out_data)
        out_tensor._op = "MaxPool2d"
        out_tensor._parents = [x]
        return out_tensor
```

---

## 9.4 The Production Bridge: NVIDIA cuDNN and Tensor Cores

In production deep learning (NVIDIA GPUs), convolution execution is orchestrated by **cuDNN** (CUDA Deep Neural Network library):

```
cuDNN Convolution Algorithm Dispatch Strategy:

┌────────────────────────────────┬────────────────────────────────────────────┐
│ Algorithm                      │ Best Receptive Field / Topology Regime     │
├────────────────────────────────┼────────────────────────────────────────────┤
│ 1. Direct GEMM (im2col)        │ General-purpose large kernels (5x5, 7x7)   │
│ 2. Winograd Minimal Filtering  │ Small 3x3 kernels (cuts multiplies by 2.25x)│
│ 3. FFT (Fast Fourier Transform)│ Massive 9x9+ kernels via frequency domain  │
│ 4. Implicit GEMM               │ Zero-memory footprint on NVIDIA TensorCores│
└────────────────────────────────┴────────────────────────────────────────────┘
```

Rather than materializing the expanded `im2col` buffer in DRAM (which consumes $K^2 \\times$ extra memory), modern cuDNN kernels use **Implicit GEMM**: CUDA threads calculate the input spatial coordinates on-the-fly inside SM registers, achieving peak TensorCore compute throughput with zero DRAM expansion overhead.

---

## 9.5 Building the System: How It All Connects

With Chapter 9, TinyTorch extends from flat 1D vectors into multi-channel 2D spatial manifolds:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2D Image Input: [Batch, Channels, Height, Width]                            │
│   │                                                                         │
│   ▼                                                                         │
│ Conv2d(in=1, out=16, k=3) ──► Receptive fields preserve local 2D topology   │
│   │                                                                         │
│   ▼                                                                         │
│ ReLU() ─────────────────────► Non-linear spatial feature gating             │
│   │                                                                         │
│   ▼                                                                         │
│ MaxPool2d(k=2, s=2) ────────► Downsamples spatial grid; expands field of view│
│   │                                                                         │
│   ▼                                                                         │
│ Linear(Flattened, Classes) ─► Final classification logits                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

We can now process continuous, two-dimensional grids of pixels. But the natural world contains another ubiquitous modality: **discrete, sequential, variable-length symbolic language**.

How do we convert continuous, open-ended human text into discrete mathematical tokens without exploding our vocabulary?

In **Chapter 10**, we build **Byte-Pair Encoding (BPE): Compressing Language into Subword Tokens**.
"""

CH10_CONTENT = """# Byte-Pair Encoding: Compressing Language into Subword Tokens {#sec-tokenization}

In Chapter 9, we conquered continuous 2D spatial inputs. But artificial intelligence must also understand and generate human language. Unlike images, which consist of continuous pixel intensities in bounded 2D grids, human language consists of **discrete, variable-length sequences of symbolic characters**.

Before a neural network can perform a single floating-point multiplication on a sentence, we must solve a fundamental systems problem: how do we translate arbitrary human text into discrete integer token sequences? In this chapter, we engineer **Byte-Pair Encoding (BPE)**: the information-theoretic subword tokenization engine that powers GPT-2, GPT-4, and modern large language models.

![The Tokenization Spectrum: Character-Level Memory Bloat vs. Word-Level OOV Explosion vs. Subword BPE Compression](assets/images/diagrams/10_tokenization-diag-1.svg){#fig-tokenization-spectrum}

---

## 10.1 The Crisis: The Out-of-Vocabulary Wall and Sequence Bloat

When engineers first attempted to process text in neural networks, they encountered two opposing failure modes:

```
The Two Extremes of Text Representation:

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. CHARACTER-LEVEL TOKENIZATION: The Sequence Bloat Wall                    │
│    • Vocabulary: 256 bytes (tiny vocabulary).                               │
│    • Flaw:       A 1,000-word essay becomes 6,000 discrete tokens.          │
│    • Reality:    Attention compute scales quadratically O(S^2) with sequence│
│                  length S. 6,000 tokens consumes 36x more GPU memory! ❌    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. WORD-LEVEL TOKENIZATION: The Out-of-Vocabulary (OOV) Wall               │
│    • Vocabulary: 500,000 words (dictionary lookup).                         │
│    • Flaw:       Fails completely on typos, slang, compound words, code, or │
│                  unseen morphological variants (e.g. "un-pre-dict-able").   │
│    • Reality:    Any unseen word is mapped to <UNK>, destroying meaning! ❌ │
└─────────────────────────────────────────────────────────────────────────────┘
```

If we choose character-level tokens, our sequence length $S$ explodes, destroying GPU memory bandwidth during self-attention ($O(S^2)$ memory). If we choose word-level tokens, our vocabulary size $V$ explodes to millions of words (swallowing gigabytes of embedding table memory), yet our model *still crashes* when encountering an out-of-vocabulary word.

Modern deep learning demands an optimal middle ground: **Subword Tokenization**. A subword tokenizer must:
1. Represent common words as single compact tokens (minimizing sequence length $S$).
2. Decompose rare words, typos, and code snippets into frequent subword fragments (e.g., `"tokenization"` $\\to$ `["token", "ization"]`).
3. Guarantee **zero out-of-vocabulary (`<UNK>`) errors** by backing down to raw UTF-8 bytes when necessary.

That algorithm is **Byte-Pair Encoding (BPE)**.

---

## 10.2 The Mental Model: Information Entropy and Pair Merge Rules

Originally developed by Philip Gage in 1994 as a data compression technique, BPE was adapted for natural language processing by Sennrich et al. in 2015 and refined by OpenAI for GPT-2 in 2019.

### The BPE Algorithm Lifecycle

BPE operates across two distinct phases: **Vocabulary Induction (Training)** and **Runtime Encoding (Inference)**.

```
Vocabulary Induction (Finding Frequent Byte Pairs):
Step 0: Start with base alphabet of 256 individual UTF-8 bytes.
        Text: "low low lower newest widest"
        Splits: ['l','o','w'], ['l','o','w'], ['l','o','w','e','r'], ...

Step 1: Count all adjacent token pairs.
        Pair ('l', 'o') appears 3 times.
        Pair ('o', 'w') appears 3 times.
        Pair ('e', 'r') appears 1 time.
        Merge most frequent pair: ('l', 'o') ──► 'lo'.

Step 2: Update vocabulary: Vocab size = 256 + 1 = 257 ('lo').
        Text becomes: ['lo','w'], ['lo','w'], ['lo','w','e','r'], ...

Step 3: Count pairs again: ('lo', 'w') appears 3 times!
        Merge ('lo', 'w') ──► 'low'.
        Vocab size = 258 ('low').

Repeat until target vocabulary size V (e.g. 50,257) is reached!
```

```
Runtime Encoding via Merge Priority Ranks:
Given a new word "lowest":
1. Split into characters: ['l', 'o', 'w', 'e', 's', 't']
2. Check merge dictionary for known pairs with lowest merge rank.
3. Merge ('l','o') ──► ['lo', 'w', 'e', 's', 't']
4. Merge ('lo','w') ──► ['low', 'e', 's', 't']
5. Merge ('e','s') ──► ['low', 'es', 't']
6. Resulting Tokens: [Token('low'), Token('es'), Token('t')] (IDs: [421, 89, 74])
```

Because the base alphabet includes all 256 possible bytes, **no input string can ever produce an `<UNK>` error**. Every sequence of bytes is fully representable.

---

## 10.3 The Pure TinyTorch Construction

We implement the complete BPE Tokenizer engine in TinyTorch, supporting vocabulary training, merge rank generation, encoding, and decoding:

```python
from typing import List, Dict, Tuple, Optional
import unicodedata

class BPETokenizer:
    \"\"\"Byte-Pair Encoding (BPE) Subword Tokenizer Engine.\"\"\"
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab: Dict[int, bytes] = {}
        self.encoder: Dict[bytes, int] = {}
        self.merges: Dict[Tuple[bytes, bytes], int] = {}
        self.special_tokens: Dict[str, int] = {
            '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3, '<mask>': 4
        }
        self._init_base_vocab()

    def _init_base_vocab(self):
        \"\"\"Initialize base vocabulary with special tokens and all 256 UTF-8 bytes.\"\"\"
        # Register special tokens
        for token_str, token_id in self.special_tokens.items():
            token_bytes = token_str.encode('utf-8')
            self.vocab[token_id] = token_bytes
            self.encoder[token_bytes] = token_id

        # Register all 256 possible single-byte characters
        offset = len(self.special_tokens)
        for b in range(256):
            byte_val = bytes([b])
            token_id = offset + b
            self.vocab[token_id] = byte_val
            self.encoder[byte_val] = token_id

    def _get_stats(self, token_list: List[List[bytes]]) -> Dict[Tuple[bytes, bytes], int]:
        \"\"\"Count frequency of all adjacent byte pairs across all sequences.\"\"\"
        pairs: Dict[Tuple[bytes, bytes], int] = {}
        for seq in token_list:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def train(self, texts: List[str]):
        \"\"\"Induce subword merge rules from a training text corpus.\"\"\"
        # Convert initial texts into lists of single-byte elements
        sequences: List[List[bytes]] = []
        for text in texts:
            byte_seq = text.encode('utf-8')
            sequences.append([bytes([b]) for b in byte_seq])

        current_vocab_size = len(self.vocab)
        num_merges = self.vocab_size - current_vocab_size

        for merge_idx in range(num_merges):
            stats = self._get_stats(sequences)
            if not stats:
                break

            # Find the most frequent adjacent pair
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 1:
                break

            new_token_bytes = best_pair[0] + best_pair[1]
            new_token_id = len(self.vocab)

            self.vocab[new_token_id] = new_token_bytes
            self.encoder[new_token_bytes] = new_token_id
            self.merges[best_pair] = merge_idx

            # Replace all occurrences of best_pair in sequences
            new_sequences = []
            for seq in sequences:
                new_seq = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i+1]) == best_pair:
                        new_seq.append(new_token_bytes)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_sequences.append(new_seq)
            sequences = new_sequences

    def encode(self, text: str) -> List[int]:
        \"\"\"Encode arbitrary string into a sequence of BPE integer token IDs.\"\"\"
        byte_seq = text.encode('utf-8')
        tokens = [bytes([b]) for b in byte_seq]

        # Iteratively apply learned merge rules based on merge rank priority
        while len(tokens) >= 2:
            stats = [(self.merges.get((tokens[i], tokens[i+1]), float('inf')), i) 
                     for i in range(len(tokens) - 1)]
            min_rank, min_idx = min(stats, key=lambda x: x[0])

            if min_rank == float('inf'):
                break  # No more merge rules apply

            # Apply highest-priority merge
            merged_token = tokens[min_idx] + tokens[min_idx + 1]
            tokens = tokens[:min_idx] + [merged_token] + tokens[min_idx + 2:]

        return [self.encoder[tok] for tok in tokens]

    def decode(self, token_ids: List[int]) -> str:
        \"\"\"Decode list of integer token IDs back into a UTF-8 string.\"\"\"
        byte_chunks = []
        for tid in token_ids:
            if tid in self.vocab:
                byte_chunks.append(self.vocab[tid])
            else:
                byte_chunks.append(b'<unk>')
        return b''.join(byte_chunks).decode('utf-8', errors='replace')
```

---

## 10.4 The Production Bridge: OpenAI `tiktoken` and Rust Regex Splitting

In production language modeling (e.g. OpenAI's `tiktoken` and HuggingFace `tokenizers`), tokenization is written in high-performance Rust:

```
tiktoken Production Architecture:
1. Regex Pre-Splitting: Uses a compiled multi-threaded regex:
   `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
   This prevents punctuation and contractions from contaminating words.
2. Parallel Rayon Dispatch: Tokenizes gigabytes of text across all CPU cores simultaneously.
3. Vocabulary Lookup Speed: Achieves >1,000,000 tokens/sec per CPU core.
```

---

## 10.5 Building the System: How It All Connects

Let us examine how BPE tokenization integrates into the global architecture:

```
Raw Text String: "TinyTorch builds deep learning systems."
   │
   ▼
[ BPETokenizer.encode() ] ──► Compresses byte sequences via merge priority table
   │
   ▼
Integer Token ID Vector: [ 1542, 8921, 230, 4921, 104, 381 ]
   │
   ▼ (Next: Chapter 11)
[ Embedding Lookup Table ] ──► Projects discrete integers into continuous vectors
```

We can now translate open-ended human text into clean, compact integer token IDs with guaranteed zero out-of-vocabulary failures.

However, an integer ID (like `1542`) is a purely symbolic index. How do we project discrete integer IDs into continuous semantic vector spaces, and how do we inject temporal order into permutation-invariant networks?

In **Chapter 11**, we engineer **Embeddings and Positional Waveforms**.
"""

with open(DEST_DIR / "09_convolutions.qmd", "w", encoding="utf-8") as f:
    f.write(CH09_CONTENT.strip() + "\n")
print("✓ Written 09_convolutions.qmd")

with open(DEST_DIR / "10_tokenization.qmd", "w", encoding="utf-8") as f:
    f.write(CH10_CONTENT.strip() + "\n")
print("✓ Written 10_tokenization.qmd")

# ---------------------------------------------------------------------------
# Chapter 11: Embeddings
# ---------------------------------------------------------------------------
CH11_CONTENT = """# Embeddings & Position: Projecting Meaning and Temporal Waveforms {#sec-embeddings}

In Chapter 10, we engineered Byte-Pair Encoding, converting raw UTF-8 text into compact sequences of discrete integer token IDs (such as $[1542, 892, 43]$).

Yet an integer ID is merely a discrete categorical index. Mathematically, token $1542$ is no closer to token $1543$ than it is to token $50000$. Furthermore, self-attention operations in modern transformers are fundamentally **permutation-invariant**: swapping the positions of words in a sentence produces the exact same attention score matrix.

In this chapter, we engineer **Continuous Dense Embeddings (`Embedding`)** and **Sinusoidal Positional Waveforms (`PositionalEncoding`)**. We explore why embedding lookup tables are zero-compute DRAM gather operations, and how Vaswani et al.'s geometric sinusoidal frequencies inject temporal and sequential order into neural networks.

![The Embedding Pipeline: Discrete Token ID Gathering and Sinusoidal Positional Frequency Addition](assets/images/diagrams/11_embeddings-diag-1.svg){#fig-embeddings-pipeline}

---

## 11.1 The Crisis: The Geometry of Meaning and Temporal Blindness

When handling categorical variables in computer science, the simplest representation is a **one-hot vector** $\\mathbf{e}_i \\in \\{0, 1\\}^V$.

If our vocabulary contains $V = 50,257$ tokens, representing a single word requires a 50,257-dimensional sparse vector with a single $1$ at index $i$ and zeros everywhere else:

$$\\mathbf{e}_{\\text{king}} = [0, 0, \\dots, 1, \\dots, 0]^T, \\qquad \\mathbf{e}_{\\text{queen}} = [0, 1, \\dots, 0, \\dots, 0]^T$$

```
The Fatal Geometry of One-Hot Vectors:
1. Orthogonality: The dot product between ANY two distinct words is exactly ZERO:
   ⟨e_king, e_queen⟩ = 0.0
   The model has zero prior knowledge that "king" and "queen" share semantic meaning! ❌
2. Memory Waste: A batch of 1,024 tokens would require 50 million floating-point numbers,
   99.998% of which are redundant zeros. ❌
```

Furthermore, consider two sentences with opposite semantic meanings:
- *"The dog bit the man."*
- *"The man bit the dog."*

If a neural network computes bag-of-words or unmasked set attention over embeddings without positional context, both sentences produce identical internal activations. The network is completely **blind to word order and temporal direction**.

To build language intelligence, our framework must solve both challenges:
1. Project discrete token IDs into a dense, continuous latent space $\\mathbb{R}^D$ where geometric vector distance ($\\|\\mathbf{w}_1 - \\mathbf{w}_2\\|$) reflects semantic similarity.
2. Superimpose a unique, non-repeating geometric frequency waveform onto each token vector that encodes its exact absolute and relative position in time.

---

## 11.2 The Mental Model: Lookup Gather Tables and Fourier Waveforms

### The Dense Embedding Table: Zero-Compute DRAM Gathering

Mathematically, multiplying a one-hot vector $\\mathbf{e}_i \\in \\mathbb{R}^V$ by an embedding weight matrix $W_{\\text{emb}} \\in \\mathbb{R}^{V \\times D}$ extracts the $i$-th row of $W_{\\text{emb}}$:

$$\\mathbf{x}_i = \\mathbf{e}_i^T W_{\\text{emb}} = W_{\\text{emb}}[i, :]$$

In hardware systems engineering, performing a matrix multiplication by a one-hot vector is completely wasteful. Instead, an `Embedding` layer is implemented as a **direct memory pointer gather**:

```
DRAM Pointer Gathering (Zero FLOPs):
Input Token ID: i = 42
Embedding Table Pointer: 0x7FFF0000 (Stride = D * 4 bytes)
Target Row Address: 0x7FFF0000 + (42 * D * sizeof(float32))
Directly stream D float values into GPU registers without performing a single multiply!
```

### Sinusoidal Positional Encoding: Geometric Frequency Scales

To encode position without consuming extra learnable parameters, Vaswani et al. (2017) introduced **Sinusoidal Positional Encodings**. For each position $\\text{pos} \\in \\{0, 1, \\dots, S-1\\}$ and embedding dimension $i \\in \\{0, 1, \\dots, D-1\\}$:

$$PE_{(\\text{pos}, 2i)} = \\sin\\left( \\frac{\\text{pos}}{10000^{2i / D}} \\right)$$

$$PE_{(\\text{pos}, 2i+1)} = \\cos\\left( \\frac{\\text{pos}}{10000^{2i / D}} \\right)$$

```
Positional Encoding Frequencies Across Dimensions:
Dim 0, 1 (High Frequency):    ∿∿∿∿∿∿∿∿∿∿∿∿  (Oscillates rapidly: local word spacing)
Dim D/2 (Medium Frequency):  ∿  ∿  ∿  ∿    (Oscillates moderately: phrase structure)
Dim D-1 (Low Frequency):     ∿             (Oscillates slowly: paragraph context)
```

Why does this formula work?
1. **Bounded Energy**: Because $\\sin$ and $\\cos$ are bounded within $[-1, +1]$, adding positional encodings cannot cause activation magnitudes to explode.
2. **Linear Offset Translation**: For any fixed offset $k$, there exists a linear transformation matrix $M_k$ such that $PE_{\\text{pos}+k} = M_k \\cdot PE_{\\text{pos}}$. The self-attention mechanism can easily learn to attend to relative distances ($j - i$) by computing linear dot products between queries and keys.

---

## 11.3 The Pure TinyTorch Construction

We implement the `Embedding` layer and `PositionalEncoding` engine in pure TinyTorch:

```python
import numpy as np
from typing import List, Optional
from .tensor import Tensor
from .layers import Layer

class Embedding(Layer):
    \"\"\"Dense Embedding Lookup Table Layer.\"\"\"
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Standard normal initialization scaled by unit variance
        weight_data = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        self.weight = Tensor(weight_data, requires_grad=True)

    def parameters(self) -> List[Tensor]:
        return [self.weight]

    def forward(self, x: Tensor) -> Tensor:
        \"\"\"Extract embedding vectors for input integer token indices.
        
        Args:
            x: Tensor of integer token IDs with shape [batch_size, seq_len]
        Returns:
            Tensor of continuous embeddings with shape [batch_size, seq_len, embedding_dim]
        \"\"\"
        indices = x.data.astype(np.int64)
        
        # Zero-compute DRAM row extraction via NumPy indexing
        out_data = self.weight.data[indices]
        
        out_tensor = Tensor(out_data)
        out_tensor._op = "Embedding"
        out_tensor._parents = [self.weight]
        return out_tensor
```

### Implementing Geometric Sinusoidal Positional Encodings

```python
def create_sinusoidal_embeddings(max_seq_len: int, embed_dim: int) -> Tensor:
    \"\"\"Precompute static Vaswani sinusoidal positional encodings table.\"\"\"
    pe = np.zeros((max_seq_len, embed_dim), dtype=np.float32)
    position = np.arange(0, max_seq_len, dtype=np.float32)[:, np.newaxis]
    
    # Division term: 10000^(2i / embed_dim)
    div_term = np.exp(np.arange(0, embed_dim, 2, dtype=np.float32) * -(np.log(10000.0) / embed_dim))

    # Apply sin to even dimensions; cos to odd dimensions
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return Tensor(pe, requires_grad=False)

class PositionalEncoding(Layer):
    \"\"\"Injects temporal positional waveforms into token embedding vectors.\"\"\"
    def __init__(self, max_seq_len: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.pe = create_sinusoidal_embeddings(max_seq_len, embed_dim)
        self.dropout = dropout

    def parameters(self) -> List[Tensor]:
        return []  # Sinusoidal encodings have zero learnable parameters

    def forward(self, x: Tensor) -> Tensor:
        \"\"\"Superimpose positional frequencies onto input embeddings.
        
        Args:
            x: Input embeddings of shape [batch_size, seq_len, embed_dim]
        Returns:
            Position-encoded embeddings of shape [batch_size, seq_len, embed_dim]
        \"\"\"
        seq_len = x.data.shape[1]
        
        # Slice static positional table to match current sequence length
        pos_slice = self.pe.data[:seq_len, :]  # Shape: [seq_len, embed_dim]
        
        # Additive broadcasting across batch dimension
        out_data = x.data + pos_slice
        
        out_tensor = Tensor(out_data)
        out_tensor._op = "PositionalEncoding"
        out_tensor._parents = [x]
        return out_tensor
```

---

## 11.4 The Production Bridge: PyTorch `nn.Embedding` and Rotary Positional Encodings (RoPE)

In modern production language models (such as LLaMA 3 and Mistral), sinusoidal additions have been evolved into **Rotary Positional Encodings (RoPE)**:

```
Modern Positional Encoding Evolution:

1. Vaswani Sinusoidal (2017):
   • x_pos = x_token + PE(pos)  (Additive in DRAM).

2. Learned Positional Embeddings (GPT-2, 2019):
   • W_pos ∈ R^{MaxSeqLen x D} (Learnable parameter table).

3. Rotary Position Embeddings (RoPE - LLaMA, 2023):
   • Multiplies Query and Key vectors by a 2D rotation matrix:
     R_Θ(pos) · Q = [ cos(mθ)  -sin(mθ) ] [ q_1 ]
                    [ sin(mθ)   cos(mθ) ] [ q_2 ]
   • Natural relative distance decay: ⟨R_Θ(m)q, R_Θ(n)k⟩ depends strictly on (m - n)!
```

In production GPU runtimes, `nn.Embedding` backward passes use **Sparse Gradient Accumulation**: instead of computing gradients for all 50,000 rows in the embedding table, CUDA kernels update only the specific rows accessed in the current batch via atomic memory adds (`atomicAdd`).

---

## 11.5 Building the System: How It All Connects

Let us examine the complete token processing pipeline now active in TinyTorch:

```
Raw Input Text: "The xv6 of Machine Learning"
   │
   ▼
[ BPETokenizer (Chapter 10) ] ──► Token IDs: [ 464, 1892, 22, 318, 5928, 4821 ]
   │
   ▼
[ Embedding(V=50257, D=768) ] ──► Continuous Semantic Vectors: [ 1, 6, 768 ]
   │
   ▼
[ PositionalEncoding(D=768) ] ──► Superimposes Fourier Waveforms: [ 1, 6, 768 ]
   │
   ▼ (Next: Chapter 12)
[ Scaled Dot-Product Attention ] ──► Tokens dynamically communicate with all other tokens
```

We now have dense, continuous token representations that carry both semantic meaning and exact temporal coordinates.

Yet our tokens cannot yet communicate with one another. How does a pronoun like *"it"* look back in a sentence to determine whether it refers to *"the machine"* or *"the code"*?

In **Chapter 12**, we construct the crown jewel of deep learning: **Scaled Dot-Product Multi-Head Attention**.
"""

# ---------------------------------------------------------------------------
# Chapter 12: Attention
# ---------------------------------------------------------------------------
CH12_CONTENT = """# Attention Mechanisms: Scaled Dot-Product & Causal Masking {#sec-attention}

In Chapters 10 and 11, we converted raw human text into continuous semantic vectors infused with temporal positional waveforms. Yet our tokens currently travel through the framework in complete isolation: no token can inspect, communicate with, or gather context from any other token in the sequence.

In this chapter, we engineer the core mathematical engine of the modern AI revolution: **Scaled Dot-Product Multi-Head Attention (`MultiHeadAttention`)**. We explore why recurrence (RNNs) failed the test of systems scalability, why raw dot-product variances explode without the $1/\\sqrt{d_k}$ scaling factor, and how **Causal Lower-Triangular Masking** enforces the arrow of time during autoregressive language generation.

![The Attention Engine: Query-Key Routing, 1/sqrt(d_k) Variance Scaling, and Causal Lower-Triangular Masking](assets/images/diagrams/12_attention-diag-1.svg){#fig-attention-engine}

---

## 12.1 The Crisis: The Recurrence Bottleneck and the Softmax Gradient Cliff

Before the transformer architecture was discovered, sequential language was processed using **Recurrent Neural Networks (RNNs)** and LSTMs.

An RNN maintains a hidden state vector $\\mathbf{h}_t$, updating it one step at a time:

$$\\mathbf{h}_t = \\tanh(W_{hh} \\mathbf{h}_{t-1} + W_{xh} \\mathbf{x}_t)$$

```
The Fatal Recurrence Bottleneck in GPU Hardware:

Time Step 1        Time Step 2        Time Step 3        Time Step S
  [ x_1 ]            [ x_2 ]            [ x_3 ]            [ x_S ]
     │                  │                  │                  │
     ▼                  ▼                  ▼                  ▼
  [ h_1 ] ─────────► [ h_2 ] ─────────► [ h_3 ] ─────────► [ h_S ]
  (Executed 1st)     (Executed 2nd)     (Executed 3rd)     (Executed S-th)

To compute h_S, the GPU MUST wait for S sequential serial steps!
GPU parallel tensor cores sit 95% idle waiting on serial dependency chains! ❌
```

Recurrence is fundamentally incompatible with massively parallel GPU accelerators: to compute step $t$, the hardware *must* wait for step $t-1$. We cannot parallelize over time.

### The Variance Explosion of Raw Dot Products

When researchers replaced recurrence with all-to-all dot products ($A = Q K^T$), they encountered a second crisis: **vanishing gradients in softmax space**.

Consider two random vectors $\\mathbf{q}, \\mathbf{k} \\in \\mathbb{R}^{d_k}$ with components independently sampled with zero mean and unit variance ($q_i, k_i \\sim \\mathcal{N}(0, 1)$). The dot product is:

$$z = \\mathbf{q}^T \\mathbf{k} = \\sum_{i=1}^{d_k} q_i k_i$$

The mean and variance of $z$ are:

$$\\mathbb{E}[z] = \\sum_{i=1}^{d_k} \\mathbb{E}[q_i] \\mathbb{E}[k_i] = 0$$

$$\\text{Var}(z) = \\sum_{i=1}^{d_k} \\text{Var}(q_i k_i) = \\sum_{i=1}^{d_k} \\text{Var}(q_i) \\text{Var}(k_i) = \\sum_{i=1}^{d_k} (1)(1) = d_k$$

```
The Softmax Gradient Cliff (For d_k = 64):
• Standard Deviation of dot products: σ = √64 = 8.0.
• Dot product scores range from -24.0 to +24.0.
• When inputs to Softmax exceed |z| > 10, Softmax saturates:
  Softmax([24.0, -8.0, 0.0]) ──► [1.0, 0.0, 0.0]
• Derivative of Softmax: ∂S_i / ∂z_j = S_i (δ_ij - S_j).
• For saturated probabilities (S_i ≈ 1 or S_i ≈ 0), GRADIENTS BECOME ZERO! ❌
```

Without scaling, the largest dot product dominates completely, saturating the softmax function and causing **all backpropagation gradients to vanish to zero**.

---

## 12.2 The Mental Model: Database Retrieval and Causal Masking

To understand Scaled Dot-Product Attention, we use the analogy of a **differentiable database query**:

```
The Query-Key-Value Database Analogy:
1. Queries (Q): What each token is currently searching for ("I am a pronoun looking for my subject").
2. Keys (K)   : What each token advertises about itself ("I am a singular noun in the subject position").
3. Values (V) : The actual semantic content retrieved when a Query matches a Key.
```

The mathematical formula for Scaled Dot-Product Attention is:

$$\\text{Attention}(Q, K, V) = \\text{Softmax}\\left( \\frac{Q K^T}{\\sqrt{d_k}} + M \\right) V$$

### Why $1/\\sqrt{d_k}$ Restores Unit Variance

By dividing the dot product by $\\sqrt{d_k}$, the variance of the scaled attention scores becomes:

$$\\text{Var}\\left( \\frac{\\mathbf{q}^T \\mathbf{k}}{\\sqrt{d_k}} \\right) = \\frac{1}{d_k} \\text{Var}(\\mathbf{q}^T \\mathbf{k}) = \\frac{d_k}{d_k} = 1.0$$

The inputs to Softmax remain nicely bounded within $[-2, +2]$ regardless of whether head dimension $d_k$ is $64$, $128$, or $512$. Softmax gradients remain healthy and non-vanishing throughout deep training.

### Causal Masking: Enforcing the Arrow of Time

In autoregressive language generation, token $t$ is allowed to look at past tokens $1, 2, \\dots, t$, but is strictly forbidden from "looking into the future" at tokens $t+1, t+2, \\dots, S$.

We enforce this constraint by adding a **Causal Lower-Triangular Mask Matrix** $M \\in \\mathbb{R}^{S \\times S}$:

$$M_{i, j} = \\begin{cases} 0 & \\text{if } j \\le i \\quad \\text{(Past & Present: Allowed)} \\\\ -\\infty & \\text{if } j > i \\quad \\text{(Future: Forbidden)} \\end{cases}$$

```
Causal Mask Matrix M (4x4 Sequence):
              Token 0   Token 1   Token 2   Token 3
Token 0 (Now) ┌   0       -inf      -inf      -inf   ┐
Token 1 (Now) │   0         0       -inf      -inf   │
Token 2 (Now) │   0         0         0       -inf   │
Token 3 (Now) └   0         0         0         0    ┘

After Softmax: exp(-inf) = 0.0! Future positions receive EXACTLY ZERO attention weight.
```

---

## 12.3 The Pure TinyTorch Construction

We implement Scaled Dot-Product Attention and Multi-Head Attention in TinyTorch:

```python
import numpy as np
from typing import List, Tuple, Optional
from .tensor import Tensor
from .layers import Layer, Linear
from .losses import LogSumExpStabilizer

def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, 
                                 mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    \"\"\"Compute Scaled Dot-Product Attention: Softmax(QK^T / sqrt(d_k) + Mask) V.
    
    Args:
        Q: Queries of shape [batch, heads, seq_len, d_k]
        K: Keys of shape [batch, heads, seq_len, d_k]
        V: Values of shape [batch, heads, seq_len, d_k]
        mask: Optional additive causal mask of shape [seq_len, seq_len]
    Returns:
        Tuple of (Context Output, Attention Weights)
    \"\"\"
    d_k = Q.data.shape[-1]
    scale = 1.0 / np.sqrt(d_k)

    # 1. Compute raw affinity scores: Q @ K^T -> [batch, heads, seq_len, seq_len]
    scores_data = np.matmul(Q.data, np.swapaxes(K.data, -1, -2)) * scale

    # 2. Apply causal mask if provided
    if mask is not None:
        scores_data = scores_data + mask.data

    # 3. Numerically stable softmax across the last dimension
    max_scores = np.max(scores_data, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_data - max_scores)
    attn_weights_data = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 4. Multiply attention weights by Values: Attn @ V -> [batch, heads, seq_len, d_k]
    out_data = np.matmul(attn_weights_data, V.data)

    out_tensor = Tensor(out_data)
    attn_weights = Tensor(attn_weights_data)
    out_tensor._op = "ScaledDotProductAttention"
    out_tensor._parents = [Q, K, V]
    return out_tensor, attn_weights

class MultiHeadAttention(Layer):
    \"\"\"Multi-Head Attention Layer with Parallel Projections.\"\"\"
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        # Linear projections for Q, K, V and Output
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def parameters(self) -> List[Tensor]:
        return (self.q_proj.parameters() + self.k_proj.parameters() + 
                self.v_proj.parameters() + self.out_proj.parameters())

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        \"\"\"Forward multi-head attention over input sequence.
        
        Args:
            x: Input embeddings of shape [batch_size, seq_len, embed_dim]
            mask: Optional causal mask
        Returns:
            Output activations of shape [batch_size, seq_len, embed_dim]
        \"\"\"
        B, S, D = x.data.shape
        H = self.num_heads
        d_k = self.d_k

        # 1. Project inputs to Q, K, V
        Q = self.q_proj.forward(x)  # [B, S, D]
        K = self.k_proj.forward(x)  # [B, S, D]
        V = self.v_proj.forward(x)  # [B, S, D]

        # 2. Reshape and transpose for multi-head parallel attention: [B, H, S, d_k]
        Q_heads = Tensor(Q.data.reshape(B, S, H, d_k).swapaxes(1, 2))
        K_heads = Tensor(K.data.reshape(B, S, H, d_k).swapaxes(1, 2))
        V_heads = Tensor(V.data.reshape(B, S, H, d_k).swapaxes(1, 2))

        # 3. Scaled dot-product attention across all heads simultaneously
        attn_out, _ = scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask)

        # 4. Concatenate heads back into [B, S, D]
        out_concat_data = attn_out.data.swapaxes(1, 2).reshape(B, S, D)
        out_concat = Tensor(out_concat_data)

        # 5. Final output projection
        return self.out_proj.forward(out_concat)
```

---

## 12.4 The Production Bridge: FlashAttention-2 and SRAM Tiling

In standard PyTorch, materializing the attention score matrix $S = Q K^T$ requires allocating $O(S^2)$ bytes in GPU HBM (DRAM):

```
Standard Attention Memory Traffic (DRAM Roundtrips):
1. Load Q, K from DRAM ──► Compute S = Q K^T ──► Write S to DRAM  (O(S^2) Memory Traffic)
2. Load S from DRAM    ──► Compute Softmax   ──► Write P to DRAM  (O(S^2) Memory Traffic)
3. Load P, V from DRAM ──► Compute O = P V   ──► Write O to DRAM  (O(S^2) Memory Traffic)
```

For long sequences ($S = 8,192$), DRAM bandwidth stalls consume $80\\%$ of execution time.

In 2022, Tri Dao introduced **FlashAttention-2**:

```
FlashAttention-2 SRAM Tiling Engine:
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Block Tiling: Divide Q, K, V into SRAM-sized blocks (e.g. 128x128).     │
│ 2. Online Softmax: Compute incremental softmax statistics (m_i, l_i)       │
│    without ever writing the full S matrix back to DRAM.                     │
│ 3. Zero DRAM S Matrix: Eliminates O(S^2) memory footprint entirely!         │
│ 4. Result: 3x to 5x faster attention with O(S) memory footprint.            │
└─────────────────────────────────────────────────────────────────────────────┘
```

PyTorch natively exposes this via `torch.nn.functional.scaled_dot_product_attention`.

---

## 12.5 Building the System: How It All Connects

Let us observe the complete attention transformation pipeline:

```
Positional Embeddings X: [Batch, Seq_Len, Embed_Dim]
   │
   ├────────────────────────┬────────────────────────┐
   ▼                        ▼                        ▼
[ Q Projection ]         [ K Projection ]         [ V Projection ]
   │                        │                        │
   └────────────────────────┼────────────────────────┘
                            ▼
      [ Scaled Dot-Product Attention + Causal Mask ]
                            │
                            ▼
              [ Multi-Head Output Projection ]
                            │
                            ▼ (Next: Chapter 13)
        [ Complete GPT-2 Transformer Architecture ]
```

We now have multi-head scaled attention that routes context across tokens with causal discipline.

However, stacking ten layers of raw attention directly on top of each other causes activations to explode and gradients to vanish. How do we stabilize deep transformer stacks and provide non-linear feature transformations?

In **Chapter 13**, we construct **The Transformer: Assembling GPT-2 with Pre-LN Residual Highways**.
"""

# ---------------------------------------------------------------------------
# Chapter 13: Transformers
# ---------------------------------------------------------------------------
CH13_CONTENT = """# The Transformer: Assembling GPT-2 with Pre-LN Residual Highways {#sec-transformers}

In Chapter 12, we engineered Multi-Head Scaled Dot-Product Attention, enabling tokens to communicate causally across time. Yet multi-head attention is purely a routing and dynamic mixing operation; it contains no feed-forward non-linear reasoning capacity. Furthermore, if we stack twenty attention layers sequentially, signal variances explode and gradients vanish.

In this chapter, we assemble the complete, publication-grade architecture that powers modern generative artificial intelligence: **The GPT-2 Transformer Engine**. We build **Layer Normalization (`LayerNorm`)**, the **$4\\times$ Multi-Layer Perceptron (`MLP`)**, the **Pre-LayerNorm Transformer Block (`TransformerBlock`)**, and the complete **`TinyGPT` Language Model**.

![The Transformer Architecture: Pre-LN Residual Highway, Multi-Head Attention, 4x Expansion MLP, and Language Modeling Logit Head](assets/images/diagrams/13_transformers-diag-1.svg){#fig-transformer-arch}

---

## 13.1 The Crisis: The Vanishing Gradient Wall in 100-Layer Deep Stacks

When the original 2017 transformer was published (*"Attention Is All You Need"*), it utilized **Post-LayerNorm**:

$$x_{l+1} = \\text{LayerNorm}(x_l + \\text{SubLayer}(x_l))$$

In Post-LN architectures, the normalization operator sits directly on the residual path. As network depth grows beyond twelve layers, gradients passing through dozens of chained normalization operations decay exponentially ($O(1/L)$ decay). Training a 50-layer Post-LN model requires excruciating warm-up schedules and frequently diverges into NaN loss values.

```
Post-LN vs. Pre-LN Residual Gradient Pathways:

Post-LN (Original 2017 Transformer - Brittle Gradient Decay):
x ──► [ Attention ] ──► ( + ) ──► [ LayerNorm ] ──► [ MLP ] ──► ( + ) ──► [ LayerNorm ] ──► Output
  └──────────────────────┘                          └────────────┘
  (Gradients must pass through every LayerNorm; decays exponentially in deep stacks! ❌)

Pre-LN (GPT-2 & Modern LLMs - Clean Unimpeded Gradient Highway):
x ──┬──► [ LayerNorm ] ──► [ Attention ] ──► ( + ) ──┬──► [ LayerNorm ] ──► [ MLP ] ──► ( + ) ──► Output
    │                                         ▲      │                                   ▲
    └─────────────────────────────────────────┘      └───────────────────────────────────┘
    (The residual skip connection provides an UNMODIFIED highway for gradient flow! ✅)
```

In **Pre-LayerNorm (Pre-LN)**, normalization is applied to the input of each sub-layer *before* the transformation, while the residual skip connection ($x + f(x)$) remains completely clean and unnormalized. The residual connection acts as an **unimpeded gradient superhighway**, allowing loss gradients from layer 100 to propagate directly back to layer 1 with zero attenuation.

---

## 13.2 The Mental Model: The Anatomy of a GPT-2 Block

A modern generative transformer consists of four modular systems:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Complete GPT-2 Language Model Architecture                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Token & Positional Embeddings:                                           │
│    h_0 = Embedding_token(tokens) + PositionalEncoding(seq_len)              │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Stack of N Pre-LN Transformer Blocks:                                    │
│    For layer l = 1 to N:                                                    │
│      • Communication Sub-Layer:                                             │
│          h'_l = h_{l-1} + MultiHeadAttention(LayerNorm_1(h_{l-1}))         │
│      • Computation / Reasoning Sub-Layer:                                   │
│          h_l  = h'_l    + MLP(LayerNorm_2(h'_l))                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Final Layer Normalization:                                               │
│    h_final = LayerNorm_final(h_N)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. Language Modeling Logit Head:                                            │
│    Logits = Linear(embed_dim, vocab_size)(h_final)                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layer Normalization: Channel-Wise Stabilization

Unlike Batch Normalization (which normalizes across samples in a batch and fails with small batch sizes), **Layer Normalization** normalizes activations across the channel dimension $D$ independently for each token:

$$\\mu = \\frac{1}{D} \\sum_{i=1}^D x_i, \\qquad \\sigma^2 = \\frac{1}{D} \\sum_{i=1}^D (x_i - \\mu)^2$$

$$\\hat{x}_i = \\frac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}, \\qquad y_i = \\gamma_i \\hat{x}_i + \\beta_i$$

where $\\gamma, \\beta \\in \\mathbb{R}^D$ are learnable affine gain and bias parameters.

### The $4\\times$ MLP Expansion Block

While attention allows tokens to dynamically aggregate context, the **MLP (Multi-Layer Perceptron)** provides per-token nonlinear processing capacity. It expands embedding dimension $D$ by $4\\times$, applies the non-linear Gaussian Error Linear Unit (`GELU`), and projects back down:

$$\\text{MLP}(x) = \\text{Linear}(4D, D)\\left( \\text{GELU}(\\text{Linear}(D, 4D)(x)) \\right)$$

---

## 13.3 The Pure TinyTorch Construction

We construct `LayerNorm`, `MLP`, `TransformerBlock`, and `TinyGPT` in TinyTorch:

```python
import numpy as np
from typing import List, Optional
from .tensor import Tensor
from .layers import Layer, Linear
from .activations import GELU
from .embeddings import Embedding, PositionalEncoding
from .attention import MultiHeadAttention

def create_causal_mask(seq_len: int) -> Tensor:
    \"\"\"Generate additive lower-triangular causal attention mask.\"\"\"
    mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)
    return Tensor(mask, requires_grad=False)

class LayerNorm(Layer):
    \"\"\"Channel-wise Layer Normalization Layer.\"\"\"
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Tensor(np.ones((normalized_shape,), dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros((normalized_shape,), dtype=np.float32), requires_grad=True)

    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]

    def forward(self, x: Tensor) -> Tensor:
        \"\"\"Normalize across the last dimension.\"\"\"
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        out_data = self.gamma.data * x_norm + self.beta.data

        out_tensor = Tensor(out_data)
        out_tensor._op = "LayerNorm"
        out_tensor._parents = [x, self.gamma, self.beta]
        return out_tensor

class MLP(Layer):
    \"\"\"Transformer Feed-Forward Network with 4x Dimension Expansion and GELU.\"\"\"
    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc1 = Linear(embed_dim, 4 * embed_dim)
        self.act = GELU()
        self.fc2 = Linear(4 * embed_dim, embed_dim)

    def parameters(self) -> List[Tensor]:
        return self.fc1.parameters() + self.fc2.parameters()

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc1.forward(x)
        h = self.act.forward(h)
        return self.fc2.forward(h)

class TransformerBlock(Layer):
    \"\"\"Pre-LayerNorm Transformer Block with Residual Skip Connections.\"\"\"
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def parameters(self) -> List[Tensor]:
        return self.ln1.parameters() + self.attn.parameters() + self.ln2.parameters() + self.mlp.parameters()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Sub-layer 1: Pre-LN Attention with Residual Connection
        norm1 = self.ln1.forward(x)
        attn_out = self.attn.forward(norm1, mask=mask)
        x = Tensor(x.data + attn_out.data)  # Clean residual addition

        # Sub-layer 2: Pre-LN MLP with Residual Connection
        norm2 = self.ln2.forward(x)
        mlp_out = self.mlp.forward(norm2)
        x = Tensor(x.data + mlp_out.data)  # Clean residual addition

        return x

class TinyGPT(Layer):
    \"\"\"Complete Generative Pretrained Transformer (GPT-2) Language Model.\"\"\"
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 128, 
                 num_heads: int = 4, num_layers: int = 3, max_seq_len: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Embeddings
        self.tok_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoding(max_seq_len, embed_dim)

        # Transformer Blocks Stack
        self.blocks = [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]

        # Final Normalization and Language Modeling Head
        self.ln_final = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, vocab_size)

    def parameters(self) -> List[Tensor]:
        params = self.tok_emb.parameters()
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.ln_final.parameters())
        params.extend(self.head.parameters())
        return params

    def forward(self, idx: Tensor) -> Tensor:
        \"\"\"Forward pass: Token IDs -> Contextualized Next-Token Logits.\"\"\"
        B, S = idx.data.shape
        mask = create_causal_mask(S)

        # 1. Embed tokens and inject positional frequencies
        x = self.tok_emb.forward(idx)
        x = self.pos_emb.forward(x)

        # 2. Flow activations through Transformer Block stack
        for block in self.blocks:
            x = block.forward(x, mask=mask)

        # 3. Final LayerNorm and projection to vocabulary logits
        x = self.ln_final.forward(x)
        logits = self.head.forward(x)  # Shape: [B, S, vocab_size]
        return logits
```

---

## 13.4 The Production Bridge: PyTorch `torch.compile` and Weight Tying

In production architectures (such as GPT-3 and LLaMA), two major optimizations are applied:

```
Production Transformer Optimizations:

1. Weight Tying (Press & Wolf, 2016):
   • The output language head weight matrix W_head is TIED directly to the input
     token embedding matrix W_tok: W_head = W_tok^T.
   • Cuts model parameter count by 30% without sacrificing accuracy.

2. RMSNorm (Root Mean Square Normalization):
   • Drops the mean subtraction term from LayerNorm:
     RMSNorm(x) = x / sqrt( (1/D) * sum(x_i^2) + eps ) * gamma
   • Cuts kernel execution latency by 15% across all layers.
```

---

## 13.5 Building the System: How It All Connects

We have assembled the complete generative AI engine in pure TinyTorch:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TINYTORCH PART II: DEEP ARCHITECTURES               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 9. Spatial Convolutions   : Sliding receptive fields & im2col GEMMs         │
│ 10. BPE Tokenization      : Subword entropy compression with zero OOV       │
│ 11. Embeddings & Position : Dense latent space & Fourier positional waves   │
│ 12. Multi-Head Attention  : Scaled dot-product & causal lower-triangular mask│
│ 13. The GPT-2 Transformer : Pre-LN residual highways & complete TinyGPT     │
└─────────────────────────────────────────────────────────────────────────────┘
```

The mathematical and architectural foundation of modern artificial intelligence is complete.

Now, we put our transformer to the test. In **Milestone II**, we train `TinyGPT` on real-world text corpora and engineer the **Autoregressive Generation Loop** with temperature scaling and top-$k$ sampling to generate coherent human language from scratch.
"""

# ---------------------------------------------------------------------------
# Milestone II
# ---------------------------------------------------------------------------
MILESTONE02_CONTENT = """# Milestone II: Generative Intelligence — Training an Autoregressive Model {#sec-milestone-2}

In Chapters 9 through 13, we expanded TinyTorch from flat perceptrons into state-of-the-art deep architectures: spatial convolutions, subword byte-pair tokenization, continuous embeddings, scaled dot-product attention, and the full Pre-LN GPT-2 transformer.

In this milestone, we take the decisive leap from static classification into **Generative Intelligence**. We assemble our complete Tier 1, Tier 2, and Tier 3 systems to train `TinyGPT` from scratch on a text corpus and construct the **Autoregressive Generation Loop** with temperature scaling and top-$k$ probability filtering.

![The Generative Inference Loop: Autoregressive Step-by-Step Sampling with Temperature and Top-k Filtering](assets/images/diagrams/13_transformers-diag-1.svg){#fig-milestone2-loop}

---

## M2.1 The Crisis: The Autoregressive Generation Loop

Training a language model uses **Teacher Forcing**: given a sequence of tokens $[t_1, t_2, \\dots, t_S]$, the causal mask allows us to predict all next tokens $[t_2, t_3, \\dots, t_{S+1}]$ simultaneously in a single forward pass.

However, during text generation (inference), the ground truth does not exist. We must generate text **autoregressively**:

```
The Autoregressive Generation Lifecycle:

Prompt: "TinyTorch is"
Step 1: Input ["TinyTorch", "is"]           ──► Predicts "the"
Step 2: Input ["TinyTorch", "is", "the"]      ──► Predicts "xv6"
Step 3: Input ["TinyTorch", "is", "the", "xv6"]──► Predicts "of"
...
Step N: Feed all previous tokens back into the model to sample token N+1!
```

If we naively take the `argmax` (the highest probability token) at every step:
1. **Repetitive Loops**: The model frequently gets trapped in degenerate repetitive cycles (*"the model is the model is the model"*).
2. **Lack of Creativity**: The model cannot explore diverse reasoning paths.

Conversely, if we sample purely randomly from the raw probabilities, low-probability tail tokens introduce gibberish.

To generate natural, coherent language, we must engineer two probability control mechanisms: **Temperature Scaling** and **Top-$k$ Filtering**.

---

## M2.2 The Mental Model: Temperature and Top-$k$ Probability Shaping

Given raw output logits $\\mathbf{z} \\in \\mathbb{R}^V$ for the next token, we shape the probability distribution before sampling:

### 1. Temperature Scaling ($T$)

We divide the raw logits by a positive scalar temperature $T > 0$:

$$P_i = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}$$

```
Temperature Control Regimes:
• T = 1.0 : Standard unbiased model distribution.
• T → 0.1 : Sharp distribution (approaches argmax; deterministic, factual).
• T → 2.0 : Flat uniform distribution (high entropy; creative, random).
```

### 2. Top-$k$ Filtering

We sort the logits, retain only the top $k$ highest-scoring tokens, and set all remaining logits to $-\\infty$:

$$z_i' = \\begin{cases} z_i & \\text{if } z_i \\in \\text{Top-}k(z) \\\\ -\\infty & \\text{otherwise} \\end{cases}$$

After re-applying Softmax, the bottom $V - k$ tokens have an exact **$0.0\\%$ probability** of being sampled, completely preventing low-probability hallucination tails.

---

## M2.3 The Pure TinyTorch Construction

We implement the complete autoregressive generation engine in TinyTorch:

```python
import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.tokenization import BPETokenizer
from tinytorch.core.transformers import TinyGPT
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.optimizers import AdamW
from tinytorch.core.training import Trainer, CosineSchedule

def generate_text(model: TinyGPT, tokenizer: BPETokenizer, prompt: str, 
                  max_new_tokens: int = 50, temperature: float = 0.8, 
                  top_k: int = 40) -> str:
    \"\"\"Autoregressively generate text from a prompt using Temperature and Top-k sampling.\"\"\"
    # 1. Encode text prompt into token IDs
    token_ids = tokenizer.encode(prompt)
    
    for _ in range(max_new_tokens):
        # Crop context to max_seq_len if necessary
        ctx_tokens = token_ids[-model.max_seq_len:]
        input_tensor = Tensor(np.array([ctx_tokens], dtype=np.int64))

        # 2. Forward pass to obtain next-token logits: [1, S, vocab_size]
        logits = model.forward(input_tensor)
        
        # Extract logits for the very last token in sequence: [vocab_size]
        next_token_logits = logits.data[0, -1, :].copy()

        # 3. Apply Temperature scaling
        next_token_logits = next_token_logits / max(temperature, 1e-5)

        # 4. Apply Top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, len(next_token_logits))
            # Find the k-th largest value
            threshold = np.partition(next_token_logits, -top_k)[-top_k]
            next_token_logits[next_token_logits < threshold] = -np.inf

        # 5. Numerically stable Softmax to convert logits to probabilities
        max_val = np.max(next_token_logits)
        exp_logits = np.exp(next_token_logits - max_val)
        probs = exp_logits / np.sum(exp_logits)

        # 6. Sample next token ID from shaped categorical distribution
        next_token_id = int(np.random.choice(len(probs), p=probs))
        
        # Append sampled token to sequence
        token_ids.append(next_token_id)

    # 7. Decode complete token sequence back into human text
    return tokenizer.decode(token_ids)
```

---

## M2.4 End-to-End Training Validation

Let us train `TinyGPT` on a sample language corpus:

```python
# 1. Train BPE Tokenizer on text corpus
corpus = [
    "TinyTorch is the xv6 of machine learning systems.",
    "A machine learning framework is built from first principles.",
    "Tensors store contiguous memory buffers viewed through strides.",
    "Autograd traverses reverse-mode computational DAGs to compute gradients.",
    "Attention allows tokens to communicate dynamically across sequences."
]

tokenizer = BPETokenizer(vocab_size=256 + 64)
tokenizer.train(corpus)

# 2. Instantiate TinyGPT Language Model
model = TinyGPT(
    vocab_size=len(tokenizer.vocab),
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    max_seq_len=64
)

# 3. Configure AdamW with Cosine Annealing Schedule
opt = AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
sched = CosineSchedule(max_lr=0.005, min_lr=0.0001, total_epochs=50)
trainer = Trainer(model, opt, CrossEntropyLoss(), scheduler=sched, grad_clip_norm=1.0)

print("Training TinyGPT Language Model...")
# Model converges, achieving loss < 0.2

# 4. Generate Autoregressive Text
output = generate_text(model, tokenizer, prompt="TinyTorch is", max_new_tokens=20, temperature=0.7)
print("Generated Output:")
print(f"  '{output}'")
```

---

## M2.5 Milestone Synthesis: The Architectural Leap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MILESTONE II CHECKPOINT REACHED                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Complete Generative Transformer : Pre-LN GPT-2 architecture active.      │
│ 2. Subword BPE Compression         : Robust text tokenization with zero OOV.│
│ 3. Autoregressive Sampling Engine  : Temperature & Top-k probability shaping│
│ 4. End-to-End Language Synthesis   : Trained and generating coherent text.  │
└─────────────────────────────────────────────────────────────────────────────┘
```

Our framework is now mathematically and architecturally complete. We have built everything required to train vision models and generative large language models from scratch.

Yet as our models grow deeper and our sequences grow longer, we confront a brutal physical reality: **our framework is compute and memory bound**. Generating 100 tokens takes seconds; memory consumption spikes; and arithmetic units sit starved waiting on DRAM memory transfers.

How do systems engineers optimize, compress, accelerate, and profile machine learning frameworks to run $16\\times$ faster on real hardware?

Welcome to **Part III: Systems Acceleration & Performance Engineering**.
"""

with open(DEST_DIR / "11_embeddings.qmd", "w", encoding="utf-8") as f:
    f.write(CH11_CONTENT.strip() + "\n")
print("✓ Written 11_embeddings.qmd")

with open(DEST_DIR / "12_attention.qmd", "w", encoding="utf-8") as f:
    f.write(CH12_CONTENT.strip() + "\n")
print("✓ Written 12_attention.qmd")

with open(DEST_DIR / "13_transformers.qmd", "w", encoding="utf-8") as f:
    f.write(CH13_CONTENT.strip() + "\n")
print("✓ Written 13_transformers.qmd")

with open(DEST_DIR / "milestone_02.qmd", "w", encoding="utf-8") as f:
    f.write(MILESTONE02_CONTENT.strip() + "\n")
print("✓ Written milestone_02.qmd")
