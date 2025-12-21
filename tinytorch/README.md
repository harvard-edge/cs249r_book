<div align="center">

# Tiny🔥Torch

### The Hands-On Lab Environment for Machine Learning Systems

[![ML Systems Book](https://img.shields.io/badge/textbook-mlsysbook.ai-blue?logo=bookstack)](https://mlsysbook.ai)
[![Docs](https://img.shields.io/badge/labs-tinytorch-orange?logo=jupyter)](https://mlsysbook.ai/tinytorch)
[![Python](https://img.shields.io/badge/python-3.8+-3776ab?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Harvard](https://img.shields.io/badge/Harvard-CS249r-A51C30)](https://mlsysbook.ai)

</div>

---

**[Machine Learning Systems](https://mlsysbook.ai)** is the textbook.
**TinyTorch** is where you build what you read.

---

## The ML Systems Learning Ecosystem

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    MACHINE LEARNING SYSTEMS                 │
                    │                        mlsysbook.ai                         │
                    │                                                             │
                    │   The comprehensive textbook for ML systems engineering     │
                    └─────────────────────────────┬───────────────────────────────┘
                                                  │
                                                  │ Read the theory
                                                  ▼
        ┌─────────────────────────────────────────────────────────────────────────────────────┐
        │                                                                                     │
        │   ┌───────────────────────┐    ┌───────────────────────┐    ┌───────────────────┐   │
        │   │                       │    │                       │    │                   │   │
        │   │    Tiny🔥Torch        │    │    Hardware Labs      │    │  Software Co-Labs │   │
        │   │                       │    │                       │    │                   │   │
        │   │  Build the framework  │    │  Deploy to devices    │    │  Scale with cloud │   │
        │   │  from scratch         │    │  Arduino, Raspberry   │    │  (coming soon)    │   │
        │   │                       │    │  Pi, Edge devices     │    │                   │   │
        │   │  ► Tensors → Autograd │    │                       │    │                   │   │
        │   │  ► CNNs → Transformers│    │  ► Image recognition  │    │  ► Distributed    │   │
        │   │  ► Quantization → KV$ │    │  ► Keyword spotting   │    │  ► Production     │   │
        │   │                       │    │  ► Motion detection   │    │  ► MLOps          │   │
        │   │                       │    │                       │    │                   │   │
        │   └───────────┬───────────┘    └───────────┬───────────┘    └─────────┬─────────┘   │
        │               │                            │                          │             │
        │               └────────────────────────────┼──────────────────────────┘             │
        │                                            │                                        │
        │                                            ▼                                        │
        │                              ┌─────────────────────────┐                            │
        │                              │                         │                            │
        │                              │     Torch Olympics      │                            │
        │                              │                         │                            │
        │                              │   Compete. Benchmark.   │                            │
        │                              │   Prove what you built. │                            │
        │                              │                         │                            │
        │                              └─────────────────────────┘                            │
        │                                                                                     │
        └─────────────────────────────────────────────────────────────────────────────────────┘
```

**The Learning Path:**

1. **[Read the Textbook](https://mlsysbook.ai)** — Understand ML systems concepts
2. **[Build in TinyTorch](https://mlsysbook.ai/tinytorch)** — Implement what you learned from scratch
3. **[Deploy to Hardware](https://mlsysbook.ai/contents/labs/labs.html)** — Run models on real embedded devices
4. **Compete in Olympics** — Prove mastery through benchmarked competition

Just as Patterson & Hennessy pairs with RISC-V for computer architecture, ML Systems pairs with TinyTorch for machine learning.

---

## What You Build

A **complete ML framework from scratch** that runs the same code as PyTorch:

```python
# After completing TinyTorch, this is YOUR code:
from tinytorch.nn import Linear, Conv2d, Transformer
from tinytorch.optim import Adam
from tinytorch import Tensor

model = Transformer(vocab=1000, d_model=64, n_heads=4)
optimizer = Adam(model.parameters())

for batch in dataloader:
    loss = model(batch.x, batch.y)
    loss.backward()   # You built this
    optimizer.step()  # You built this too
```

**No PyTorch. No TensorFlow. Everything is code you wrote.**

---

## 20 Modules Across Three Tiers

| Tier | Modules | Textbook Chapters | What You Build |
|------|---------|-------------------|----------------|
| **Foundation** | 01-08 | DL Primer, Training, Frameworks | Tensors, autograd, optimizers, training loops |
| **Architecture** | 09-13 | DNN Architectures | CNNs, tokenization, attention, transformers |
| **Optimization** | 14-19 | Efficient AI, Optimizations, Benchmarking | Profiling, quantization, compression, KV-cache |
| **Capstone** | 20 | — | Torch Olympics competition |

Each module follows the textbook:
> **Read** the chapter → **Build** in TinyTorch → **Deploy** to hardware (optional)

---

## Historical Milestones

Validate your implementation by recreating landmark ML achievements:

| Year | Milestone | What You Prove |
|------|-----------|----------------|
| 1958 | Rosenblatt's Perceptron | Your tensors and layers work |
| 1969 | XOR Solution | Your multi-layer networks learn non-linear functions |
| 1986 | Backpropagation | Your autograd computes gradients correctly |
| 1998 | LeNet CNN | Your convolutions classify images |
| 2017 | Transformer | Your attention mechanism generates text |
| 2018+ | MLPerf | Your optimizations achieve competitive benchmarks |

These aren't toy demos—they're historically significant achievements rebuilt with YOUR framework.

---

## Torch Olympics

*Coming Soon*

The capstone competition where you optimize the framework you built:

| Track | Challenge |
|-------|-----------|
| **Vision** | Highest CIFAR accuracy with your Conv2d |
| **Language** | Best text generation with your transformer |
| **Speed** | Fastest inference with your optimizations |
| **Compression** | Smallest model that still works |

Compete using code you wrote, not someone else's framework.

---

## Current Status

> **Preview Release** — TinyTorch is functional but evolving. We're sharing early to gather community feedback.

| Ready | In Progress | Coming Soon |
|-------|-------------|-------------|
| All 20 modules | Documentation polish | NBGrader integration |
| 600+ tests | Instructor resources | Torch Olympics leaderboard |
| `tito` CLI | Textbook cross-references | Software Co-Labs |
| Historical milestones | | Binder/Colab support |

**Classroom Target**: Fall 2026

---

## Getting Started

### For Students

1. **Read the textbook chapter** at [mlsysbook.ai](https://mlsysbook.ai)
2. **Complete the TinyTorch module** for that chapter
3. **Validate with milestones** to prove your implementation works

```bash
git clone https://github.com/harvard-edge/tinytorch
cd tinytorch
pip install -e .
```

See the [Getting Started Guide](https://mlsysbook.ai/tinytorch/getting-started.html) for detailed setup.

### For Instructors

TinyTorch supports multiple integration models:

| Model | Modules | Use Case |
|-------|---------|----------|
| **Full Course** | All 20 | Standalone ML systems course |
| **Half Semester** | 01-09 | Foundation + CNNs |
| **Optimization Focus** | 14-19 | Add-on to existing ML course |
| **Self-Paced** | Any | Professional development |

See the [Instructor Guide](INSTRUCTOR.md) for curriculum integration.

---

## Repository Structure

```
tinytorch/
├── src/                    # Source modules (01-20)
├── modules/                # Generated Jupyter notebooks
├── tinytorch/              # Your built framework (import from here)
├── milestones/             # Historical validation scripts
├── tests/                  # 600+ tests
├── site/                   # Course website
└── tito/                   # CLI tool
```

---

## Related Educational Frameworks

We acknowledge other excellent educational frameworks:

- **[MiniTorch](https://minitorch.github.io/)** — Cornell's autodiff-focused framework
- **[micrograd](https://github.com/karpathy/micrograd)** — Karpathy's tiny autograd engine
- **[tinygrad](https://github.com/tinygrad/tinygrad)** — Hotz's minimalist deep learning

**TinyTorch's distinction**: It's the lab environment for the [Machine Learning Systems](https://mlsysbook.ai) textbook, with 20 progressive modules covering the full stack from tensors to deployment optimization.

---

## Contributing

TinyTorch is part of the open [ML Systems Book](https://mlsysbook.ai) project. We welcome contributions:

- Report issues or suggest improvements
- Contribute modules, tests, or documentation
- Share how you're using TinyTorch in your course

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Acknowledgments

Created by [Prof. Vijay Janapa Reddi](https://vijay.seas.harvard.edu) at Harvard University as the hands-on companion to [Machine Learning Systems](https://mlsysbook.ai).

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**[Textbook](https://mlsysbook.ai)** · **[TinyTorch Labs](https://mlsysbook.ai/tinytorch)** · **[Hardware Labs](https://mlsysbook.ai/contents/labs/labs.html)** · **[Discussions](https://github.com/harvard-edge/cs249r_book/discussions)**

*Read the book. Build the framework. Deploy to hardware.*

</div>
