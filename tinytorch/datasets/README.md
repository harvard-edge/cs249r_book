# TinyTorch Datasets

This directory contains datasets for TinyTorch milestone examples.

## Directory Structure

```
datasets/
├── tinydigits/         ← 8×8 handwritten digits (ships with repo, ~310KB)
├── tinypy/             ← curated, verified Python code for code completion (~67KB)
├── tinyshakespeare/    ← bundled sample of Shakespeare plays (~21KB)
├── tinytalks/          ← conversational Q&A text for transformers (~40KB)
├── DATASHEET.md        ← Comprehensive TinyVerse suite datasheet (Gebru et al.)
└── README.md           ← This file
```

## Shipped Datasets (No Download Required)

### TinyDigits
- **Used by:** Milestones 03 & 04 (MLP and CNN examples)
- **Contents:** 1,000 training + 200 test samples (balanced 0 to 9)
- **Format:** 8x8 grayscale images, pickled
- **Size:** ~310 KB
- **Documentation:** README.md and DATASHEET.md
- **Purpose:** Fast iteration on real image classification

### TinyPy
- **Used by:** Milestone 05 Part 3 (Code completion and syntax verification)
- **Contents:** 102 curated Python snippets across math, sorting, data structures, and deep learning primitives
- **Format:** Clean, 100% AST-valid Python source code
- **Size:** ~73 KB
- **Documentation:** README.md and DATASHEET.md
- **Purpose:** Fast, offline code completion, syntax learning, and AST compiler gates

### TinyTalks
- **Used by:** Milestone 05 Part 4 (Conversational modeling and Overfitting Detective)
- **Contents:** 350 general Q&A pairs plus 81 specialized TinyTorch Concepts Q&A pairs with train/test splits
- **Format:** Plain text (Q: / A: lines), character-level friendly
- **Size:** ~53 KB combined
- **Documentation:** README.md and DATASHEET.md
- **Purpose:** Conversational text, concept retrieval, and measuring memorization versus generalization

### TinyShakespeare (Sample)
- **Used by:** Milestone 05 Part 1 (TinyGPT character-level language modeling)
- **Contents:** Curated excerpt from William Shakespeare plays
- **Format:** Plain text, character-level friendly
- **Size:** ~21 KB
- **Documentation:** README.md and DATASHEET.md
- **Purpose:** Fast, offline next-token prediction and autoregressive sampling without downloading the full corpus

## Downloaded Datasets (On-Demand)

The milestones automatically download larger datasets when needed:

### MNIST
- **Used by:** Optional scaling benchmark via `DatasetManager().get_mnist()`
- **Downloads to:** `milestones/datasets/mnist/`
- **Contents:** 60K training + 10K test samples
- **Format:** 28×28 grayscale images
- **Size:** ~10 MB compressed
- **Auto-downloaded by:** `milestones/data_manager.py`

### CIFAR-10
- **Used by:** `milestones/04_1998_cnn/02_lecun_cifar10.py`
- **Downloads to:** `milestones/datasets/cifar-10/`
- **Contents:** 50K training + 10K test samples
- **Format:** 32×32 RGB images
- **Size:** ~170 MB compressed
- **Auto-downloaded by:** `milestones/data_manager.py`

## Design Philosophy

**Shipped datasets** follow Karpathy's "~1K samples" philosophy:
- Small enough to ship with repo
- Large enough for meaningful learning
- Fast training (seconds to minutes)
- Instant gratification for students

**Downloaded datasets** are full benchmarks:
- Standard ML benchmarks (MNIST, CIFAR-10)
- Larger, slower, more realistic
- Auto-downloaded only when needed
- Used for scaling demonstrations

## Total Repository Size

- **Shipped data:** ~440 KB (TinyDigits, TinyPy, TinyTalks, TinyShakespeare combined)
- **USB-friendly:** Entire repo fits on any device
- **Offline-capable:** Core milestones work without internet
- **Git-friendly:** No large binary files in version control
