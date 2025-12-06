# TinyTorch Learning Journey
**From Zero to Transformer: A 20-Module Adventure**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    🎯 YOUR LEARNING DESTINATION                      │
│                                                                       │
│  Start: "What's a tensor?"                                           │
│    ↓                                                                  │
│  Finish: "I built a transformer from scratch using only NumPy!"      │
│                                                                       │
│  🏆 North Star Achievement: Train CNNs on CIFAR-10 to 75%+ accuracy │
└─────────────────────────────────────────────────────────────────────┘
```

## Overview: 4 Phases, 20 Modules, 6 Milestones

**Total Time**: 100-130 hours (5-7 weeks at 20 hrs/week)
**Prerequisites**: Python, NumPy basics, basic linear algebra
**Tools**: Just Python + NumPy + Jupyter notebooks

---

## Phase 1: FOUNDATION (Modules 01-04)
**Goal**: Build the fundamental data structures and operations
**Time**: 14-19 hours | **Difficulty**: ⭐-⭐⭐ Beginner-friendly

```
┌──────────┐      ┌──────────────┐      ┌─────────┐      ┌─────────┐
│    01    │─────▶│      02      │─────▶│   03    │─────▶│   04    │
│  Tensor  │      │ Activations  │      │ Layers  │      │ Losses  │
│          │      │              │      │         │      │         │
│ • Shape  │      │ • ReLU       │      │ • Linear│      │ • MSE   │
│ • Data   │      │ • Sigmoid    │      │ • Module│      │ • Cross │
│ • Ops    │      │ • Softmax    │      │ • Params│      │   Entropy│
└──────────┘      └──────────────┘      └─────────┘      └─────────┘
  4-6 hrs           3-4 hrs              4-5 hrs          3-4 hrs
    ⭐                ⭐⭐                  ⭐⭐              ⭐⭐
```

### Module Details

**Module 01: Tensor** (4-6 hours, ⭐)
- Build the foundation: n-dimensional arrays with operations
- Implement: shape, reshape, indexing, broadcasting
- Operations: add, multiply, matmul, transpose
- Why it matters: Everything in ML is tensor operations

**Module 02: Activations** (3-4 hours, ⭐⭐)
- Add non-linearity: ReLU, Sigmoid, Softmax
- Understand: Why neural networks need activations
- Implement: Forward passes for each activation
- Why it matters: Without activations, networks are just linear algebra

**Module 03: Layers** (4-5 hours, ⭐⭐)
- Build neural network components: Linear layers
- Implement: nn.Module system, Parameter class
- Create: Weight initialization, layer composition
- Why it matters: Foundation for all network architectures

**Module 04: Losses** (3-4 hours, ⭐⭐)
- Measure performance: MSE and CrossEntropy
- Understand: How to quantify model errors
- Implement: Loss calculation and aggregation
- Why it matters: Without loss, we can't train networks

### Milestone Checkpoint 1: 1957 Perceptron
**Unlock After**: Module 04
```
🏆 CHECKPOINT: Train Rosenblatt's Original Perceptron
├─ Dataset: Linearly separable binary classification
├─ Architecture: Single layer, no hidden units
├─ Achievement: First trainable neural network in history!
└─ Test: Can your implementation learn AND/OR logic?
```

---

## Phase 2: TRAINING SYSTEMS (Modules 05-08)
**Goal**: Make your networks learn from data
**Time**: 24-31 hours | **Difficulty**: ⭐⭐⭐-⭐⭐⭐⭐ Core ML concepts

```
┌──────────┐      ┌────────────┐      ┌──────────┐      ┌────────────┐
│    05    │─────▶│     06     │─────▶│    07    │─────▶│     08     │
│ Autograd │      │ Optimizers │      │ Training │      │ DataLoader │
│          │      │            │      │          │      │            │
│ • Graph  │      │ • SGD      │      │ • Loops  │      │ • Batching │
│ • Forward│      │ • Momentum │      │ • Epochs │      │ • Shuffling│
│ • Backward│     │ • Adam     │      │ • Eval   │      │ • Pipeline │
└──────────┘      └────────────┘      └──────────┘      └────────────┘
 8-10 hrs          6-8 hrs             6-8 hrs           4-5 hrs
 ⭐⭐⭐⭐          ⭐⭐⭐⭐             ⭐⭐⭐⭐           ⭐⭐⭐
     │                 │                  │                  │
     └─────────────────┴──────────────────┴──────────────────┘
                    ALL BUILD ON TENSOR (Module 01)
```

### Module Details

**Module 05: Autograd** (8-10 hours, ⭐⭐⭐⭐) **CRITICAL MODULE**
- Implement automatic differentiation: The magic of modern ML
- Build: Computational graph, gradient tracking
- Implement: backward() for all operations
- Why it matters: This IS machine learning - without gradients, no training

**Module 06: Optimizers** (6-8 hours, ⭐⭐⭐⭐)
- Update weights intelligently: SGD, Momentum, Adam
- Understand: Learning rates, momentum, adaptive methods
- Implement: Parameter updates, state management
- Why it matters: How networks actually improve over time

**Module 07: Training** (6-8 hours, ⭐⭐⭐⭐) **CRITICAL MODULE**
- Complete training loops: The full ML pipeline
- Implement: Epochs, batches, forward/backward passes
- Add: Metrics tracking, model evaluation
- Why it matters: This is where everything comes together

**Module 08: DataLoader** (4-5 hours, ⭐⭐⭐)
- Efficient data handling: Batching, shuffling, pipelines
- Implement: Batch creation, data iteration
- Optimize: Memory efficiency, preprocessing
- Why it matters: Real ML needs to handle millions of examples

### Milestone Checkpoint 2: 1969 XOR Crisis & Solution
**Unlock After**: Module 07
```
🏆 CHECKPOINT: Solve the Problem That Nearly Killed AI
├─ Dataset: XOR (the "impossible" problem for single-layer networks)
├─ Architecture: Multi-layer perceptron with hidden units
├─ Achievement: Prove Minsky wrong - MLPs can learn XOR!
└─ Test: 100% accuracy on XOR with your backpropagation
```

### Milestone Checkpoint 3: 1986 MLP Revival
**Unlock After**: Module 08
```
🏆 CHECKPOINT: Recognize Handwritten Digits (MNIST)
├─ Dataset: MNIST (60,000 handwritten digits)
├─ Architecture: 2-3 layer MLP with ReLU activations
├─ Achievement: 95%+ accuracy on real computer vision!
└─ Test: Your network recognizes digits you draw yourself
```

---

## Phase 3: ADVANCED ARCHITECTURES (Modules 09-13)
**Goal**: Build modern CV and NLP architectures
**Time**: 26-33 hours | **Difficulty**: ⭐⭐⭐-⭐⭐⭐⭐ Advanced concepts

```
┌──────────┐      ┌───────────────┐      ┌─────────────┐
│    09    │─────▶│      10       │─────▶│     11      │
│ Spatial  │      │ Tokenization  │      │ Embeddings  │
│          │      │               │      │             │
│ • Conv2d │      │ • BPE         │      │ • Token Emb │
│ • Pool2d │      │ • Vocab       │      │ • Position  │
│ • CNNs   │      │ • Encoding    │      │ • Learned   │
└──────────┘      └───────────────┘      └─────────────┘
  6-8 hrs          4-5 hrs                4-5 hrs
  ⭐⭐⭐            ⭐⭐                    ⭐⭐
     │                  │                      │
     │                  └──────────┬───────────┘
     │                             ▼
     │            ┌──────────┐      ┌──────────────┐
     │            │    12    │─────▶│      13      │
     │            │Attention │      │Transformers  │
     │            │          │      │              │
     │            │ • Q,K,V  │      │ • Encoder    │
     │            │ • Multi  │      │ • Decoder    │
     │            │   -Head  │      │ • Complete   │
     │            └──────────┘      └──────────────┘
     │              5-6 hrs           6-8 hrs
     │              ⭐⭐⭐             ⭐⭐⭐⭐
     │                  │                  │
     └──────────────────┴──────────────────┘
              ALL USE AUTOGRAD (Module 05)
```

### Module Details

**Module 09: Spatial Operations** (6-8 hours, ⭐⭐⭐) **CRITICAL MODULE**
- Convolutional Neural Networks: Modern computer vision
- Implement: Conv2d (with 6 nested loops!), MaxPool2d
- Understand: Why CNNs revolutionized image processing
- Why it matters: The foundation of modern computer vision

**Module 10: Tokenization** (4-5 hours, ⭐⭐)
- Text preprocessing: From strings to numbers
- Implement: Byte-Pair Encoding (BPE), vocabulary building
- Understand: How transformers see language
- Why it matters: Can't process text without tokenization

**Module 11: Embeddings** (4-5 hours, ⭐⭐)
- Convert tokens to vectors: Token and positional embeddings
- Implement: Embedding lookup, sinusoidal position encoding
- Understand: How models represent meaning
- Why it matters: Foundation for all language models

**Module 12: Attention** (5-6 hours, ⭐⭐⭐) **CRITICAL MODULE**
- The transformer revolution: Multi-head self-attention
- Implement: Q, K, V projections, scaled dot-product attention
- Understand: Why attention changed everything
- Why it matters: The core of GPT, BERT, and all modern LLMs

**Module 13: Transformers** (6-8 hours, ⭐⭐⭐⭐) **CRITICAL MODULE**
- Complete transformer architecture: GPT-style models
- Implement: Encoder/decoder blocks, layer norm, residuals
- Build: Full transformer from components
- Why it matters: You're building GPT from scratch!

### Milestone Checkpoint 4: 1998 CNN Revolution
**Unlock After**: Module 09
```
🏆 CHECKPOINT: CIFAR-10 Image Classification (North Star!)
├─ Dataset: CIFAR-10 (50,000 color images, 10 classes)
├─ Architecture: LeNet-inspired CNN with Conv2d + MaxPool
├─ Achievement: 75%+ accuracy on real-world images!
├─ Test: Classify airplanes, cars, birds, cats, etc.
└─ Impact: This is where your framework becomes REAL
```

### Milestone Checkpoint 5: 2017 Transformer Era
**Unlock After**: Module 13
```
🏆 CHECKPOINT: Build a Language Model
├─ Dataset: Text corpus (Shakespeare, WikiText, etc.)
├─ Architecture: GPT-style decoder with multi-head attention
├─ Achievement: Generate coherent text character-by-character
├─ Test: Your model completes sentences meaningfully
└─ Impact: You've built the architecture behind ChatGPT!
```

---

## Phase 4: PRODUCTION SYSTEMS (Modules 14-20)
**Goal**: Optimize and deploy ML systems at scale
**Time**: 36-47 hours | **Difficulty**: ⭐⭐⭐-⭐⭐⭐⭐ Systems engineering

```
┌──────────┐      ┌──────────────┐      ┌──────────────┐
│    14    │─────▶│      15      │─────▶│      16      │
│Profiling │      │ Quantization │      │ Compression  │
│          │      │              │      │              │
│ • Time   │      │ • INT8       │      │ • Pruning    │
│ • Memory │      │ • Calibrate  │      │ • Distill    │
│ • FLOPs  │      │ • Compress   │      │ • Sparse     │
└──────────┘      └──────────────┘      └──────────────┘
  5-6 hrs          5-6 hrs                5-6 hrs
  ⭐⭐⭐            ⭐⭐⭐                  ⭐⭐⭐

       ▼                 ▼                     ▼

┌──────────┐      ┌──────────────┐      ┌──────────┐      ┌──────────┐
│    17    │─────▶│      18      │─────▶│    19    │─────▶│    20    │
│Memoization│    │Acceleration  │      │Benchmark │      │ Capstone │
│          │      │              │      │          │      │          │
│ • KV-Cache│     │ • Vectorize  │      │ • Compare│      │ • Full   │
│ • Reuse  │      │ • Hardware   │      │ • Report │      │   System │
│ • Speedup│      │ • Parallel   │      │ • Analyze│      │ • Deploy │
└──────────┘      └──────────────┘      └──────────┘      └──────────┘
  4-5 hrs          6-8 hrs               5-6 hrs          5-8 hrs
  ⭐⭐⭐            ⭐⭐⭐                 ⭐⭐⭐            ⭐⭐⭐⭐
```

### Module Details

**Module 14: Profiling** (5-6 hours, ⭐⭐⭐)
- Measure everything: Time, memory, FLOPs
- Implement: Profiling decorators, bottleneck analysis
- Understand: Where computation actually happens
- Why it matters: Can't optimize what you don't measure

**Module 15: Quantization** (5-6 hours, ⭐⭐⭐)
- Compress models: Float32 → INT8
- Implement: Quantization, calibration, dequantization
- Achieve: 4× smaller models, faster inference
- Why it matters: Deploy models on edge devices

**Module 16: Compression** (5-6 hours, ⭐⭐⭐)
- Shrink models: Pruning and distillation
- Implement: Weight pruning, knowledge distillation
- Achieve: 10× smaller models with minimal accuracy loss
- Why it matters: Mobile ML and resource-constrained deployment

**Module 17: Memoization** (4-5 hours, ⭐⭐⭐)
- Cache computations: KV-cache for transformers
- Implement: Memoization decorators, cache management
- Optimize: 10-100× speedup for inference
- Why it matters: How production LLMs run efficiently

**Module 18: Acceleration** (6-8 hours, ⭐⭐⭐)
- Hardware optimization: Vectorization, parallelization
- Implement: NumPy tricks, batch processing
- Achieve: 10-100× speedups
- Why it matters: Production systems need speed

**Module 19: Benchmarking** (5-6 hours, ⭐⭐⭐)
- Compare implementations: Rigorous performance testing
- Implement: Benchmark suite, statistical analysis
- Report: Scientific measurements
- Why it matters: Engineering decisions need data

**Module 20: Capstone** (5-8 hours, ⭐⭐⭐⭐) **FINAL PROJECT**
- Build complete system: End-to-end ML pipeline
- Integrate: All 19 modules into production-ready system
- Deploy: Real application with optimization
- Why it matters: This is your portfolio piece!

### Milestone Checkpoint 6: 2024 Systems Age
**Unlock After**: Module 20
```
🏆 FINAL CHECKPOINT: Production-Optimized ML System
├─ Challenge: Take any milestone and make it production-ready
├─ Requirements:
│   ├─ 10× faster inference (profiling + acceleration)
│   ├─ 4× smaller model (quantization + compression)
│   ├─ <100ms latency (memoization + optimization)
│   └─ Rigorous benchmarks (statistical significance)
├─ Achievement: You're now an ML systems engineer!
└─ Test: Deploy your system, measure everything, compare to PyTorch
```

---

## Dependency Map: How Modules Connect

```
CORE FOUNDATION
├─ Module 01 (Tensor)
│   ├─▶ Module 02 (Activations)
│   ├─▶ Module 03 (Layers)
│   ├─▶ Module 04 (Losses)
│   └─▶ Module 08 (DataLoader)
│
TRAINING ENGINE
├─ Module 05 (Autograd) ← Enhances Module 01
│   ├─▶ Module 06 (Optimizers)
│   └─▶ Module 07 (Training)
│
COMPUTER VISION BRANCH
├─ Module 09 (Spatial) ← Uses 01,02,03,05
│   └─▶ Module 20 (Capstone)
│
NLP BRANCH
├─ Module 10 (Tokenization) ← Uses 01
│   ├─▶ Module 11 (Embeddings)
│   └─▶ Module 12 (Attention) ← Uses 01,03,05,11
│       └─▶ Module 13 (Transformers) ← Uses 02,11,12
│
OPTIMIZATION BRANCH
├─ Module 14 (Profiling) ← Measures any module
│   ├─▶ Module 15 (Quantization) ← Compresses any module
│   ├─▶ Module 16 (Compression) ← Shrinks any module
│   ├─▶ Module 17 (Memoization) ← Optimizes 12,13
│   ├─▶ Module 18 (Acceleration) ← Speeds up any module
│   └─▶ Module 19 (Benchmarking) ← Measures optimizations
│       └─▶ Module 20 (Capstone)
```

---

## Time Estimates by Experience Level

```
┌──────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Experience Level │ Phase 1  │ Phase 2  │ Phase 3  │ Phase 4  │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Beginner         │ 17-23h   │ 29-37h   │ 31-40h   │ 43-56h   │
│ (New to ML)      │          │          │          │          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Intermediate     │ 14-19h   │ 24-31h   │ 26-33h   │ 36-47h   │
│ (Used PyTorch)   │          │          │          │          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Advanced         │ 11-15h   │ 19-25h   │ 21-26h   │ 29-38h   │
│ (Built models)   │          │          │          │          │
└──────────────────┴──────────┴──────────┴──────────┴──────────┘

Total Time: 100-130 hours (Intermediate) | 5-7 weeks at 20 hrs/week
```

---

## Difficulty Ratings Explained

```
⭐⭐         │ Beginner-friendly
            │ - Follow clear instructions
            │ - Build intuition for concepts
            │ - ~2 hours per module
            │
⭐⭐⭐       │ Core ML concepts
            │ - Implement fundamental algorithms
            │ - Connect multiple concepts
            │ - ~3 hours per module
            │
⭐⭐⭐⭐     │ Advanced implementation
            │ - Complex algorithms
            │ - Systems thinking required
            │ - ~4 hours per module
            │
⭐⭐⭐⭐⭐   │ Expert-level systems
            │ - Multi-layered complexity
            │ - Production considerations
            │ - ~5-6 hours per module
```

---

## Suggested Learning Paths

### Fast Track (Core ML Only) - 64 hours
Focus on the essentials to build and train networks:
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09
(Tensor through Spatial for CNNs)

Milestones: Perceptron → XOR → MNIST → CIFAR-10
```

### NLP Focus - 85 hours
Core + Language models:
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
          ↓
10 → 11 → 12 → 13
(Add Tokenization through Transformers)

Milestones: All ML history + Transformer Era
```

### Systems Engineering Path - Full 100-130 hours
Everything + optimization:
```
Complete all 20 modules
(Tensor → Transformers → Optimization → Capstone)

Milestones: All 6 checkpoints + Production Systems
```

---

## Success Metrics: What "Done" Looks Like

```
✅ Module Complete When:
├─ All unit tests pass (test_unit_* functions)
├─ Module integration test passes (test_module())
├─ You can explain the algorithm to someone else
└─ Code matches PyTorch API (but implemented from scratch)

✅ Phase Complete When:
├─ All modules in phase pass tests
├─ Milestone checkpoint achieved
└─ You understand connections between modules

✅ Course Complete When:
├─ All 20 modules implemented
├─ All 6 milestones achieved
├─ Capstone project deployed
└─ You can confidently say: "I built a transformer from scratch!"
```

---

## Common Questions

**Q: Do I need to complete modules in order?**
A: YES! Each module builds on previous ones. Module 05 (Autograd) enhances Module 01 (Tensor), Module 12 (Attention) uses Modules 01, 03, 05, and 11. The dependency chain is strict.

**Q: Can I skip modules?**
A: Modules 01-08 are REQUIRED. Modules 09-13 split into CV (09) and NLP (10-13) tracks - you can choose one. Modules 14-20 are optimization - recommended but optional for core understanding.

**Q: How do I know if I'm ready for the next module?**
A: Run `test_module()` - if all tests pass, you're ready! Each module has comprehensive integration tests.

**Q: What if I get stuck?**
A: Each module has reference solutions, detailed scaffolding, and clear error messages. Plus milestone checkpoints validate your progress.

**Q: How is this different from online courses?**
A: You BUILD everything from scratch. No black boxes. No "just import PyTorch." You implement every line of a production ML framework.

---

## Your Journey Starts Now

```
┌─────────────────────────────────────────────┐
│  📍 YOU ARE HERE                             │
│                                              │
│  Next Step: cd modules/01_tensor/    │
│             jupyter notebook tensor_dev.py   │
│                                              │
│  First Goal: Understand what a tensor is    │
│  First Win: Implement your first matmul     │
│  First Checkpoint: Train a perceptron       │
│                                              │
│  🎯 Final Destination (60-80 hours ahead):  │
│     "I built a transformer from scratch!"   │
└─────────────────────────────────────────────┘
```

**Remember**: Every expert was once a beginner. Every line of PyTorch was written by someone who understood these fundamentals. Now it's your turn.

**Ready to start building?**

```bash
cd modules/01_tensor
jupyter notebook tensor_dev.py
```

Let's build something amazing! 🚀
