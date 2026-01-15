# TinyTorch Datasets

This directory contains datasets for TinyTorch milestone examples.

## Directory Structure

```
datasets/
├── tinydigits/     ← 8×8 handwritten digits (ships with repo, ~310KB)
├── tinytalks/      ← Q&A dataset for transformers (ships with repo, ~40KB)
└── README.md       ← This file
```

## Shipped Datasets (No Download Required)

### TinyDigits
- **Used by:** Milestones 03 & 04 (MLP and CNN examples)
- **Contents:** 1,000 training + 200 test samples
- **Format:** 8×8 grayscale images, pickled
- **Size:** ~310 KB
- **Purpose:** Fast iteration on real image classification

### TinyTalks
- **Used by:** Milestone 05 (Transformer/GPT examples)
- **Contents:** 350 Q&A pairs across 5 difficulty levels
- **Format:** Plain text (Q: ... A: ... format)
- **Size:** ~40 KB
- **Purpose:** Character-level conversational AI training

## Downloaded Datasets (On-Demand)

The milestones automatically download larger datasets when needed:

### MNIST
- **Used by:** `milestones/03_1986_mlp/02_rumelhart_mnist.py`
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

- **Shipped data:** ~350 KB (tinydigits + tinytalks)
- **USB-friendly:** Entire repo fits on any device
- **Offline-capable:** Core milestones work without internet
- **Git-friendly:** No large binary files in version control
