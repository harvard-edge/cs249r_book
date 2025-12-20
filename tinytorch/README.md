<div align="center">

# Tiny🔥Torch

### Build Your Own ML Framework From Scratch

[![Status](https://img.shields.io/badge/status-preview-orange?logo=github)](https://github.com/harvard-edge/cs249r_book/discussions/1076)
[![Docs](https://img.shields.io/badge/docs-mlsysbook.ai-blue?logo=readthedocs)](https://mlsysbook.ai/tinytorch)
[![Python](https://img.shields.io/badge/python-3.8+-3776ab?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Harvard](https://img.shields.io/badge/Harvard-CS249r-A51C30?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsMTAgNSAxMC01TTIgMTJsMTAgNSAxMC01Ii8+PC9zdmc+)](https://mlsysbook.ai)

**Most ML courses teach you to *use* frameworks. TinyTorch teaches you to *build* them.**

[The Vision](#why-tinytorch) · [20 Modules](#-20-progressive-modules) · [Share Feedback](https://github.com/harvard-edge/cs249r_book/discussions/1076)

</div>

---

> 🚧 **Preview Release** — TinyTorch is functional but evolving. We're sharing early to shape the direction with community input rather than building in isolation.
>
> 📅 **Classroom Ready**: Summer/Fall 2026 · **Right Now**: [We want your feedback](#-help-shape-tinytorch)

---

## Why TinyTorch?

Everyone wants to be an astronaut 🧑‍🚀. Very few want to be the rocket scientist 🚀.

In machine learning, we see the same pattern. Everyone wants to train models, run inference, deploy AI. Very few want to understand how the frameworks actually work. Even fewer want to build one.

**The world is full of users. We do not have enough builders.**

### The Solution: AI Bricks 🧱

TinyTorch teaches you the **AI bricks**—the stable engineering foundations you can use to build any AI system.

- **Small enough to learn from**: bite-sized code that runs even on a Raspberry Pi
- **Big enough to matter**: showing the real architecture of how frameworks are built

A Harvard University course that transforms you from framework user to systems engineer, giving you the deep understanding needed to optimize, debug, and innovate at the foundation of AI.

---

## What You'll Build

A **complete ML framework** capable of:

🎯 **North Star Achievement**: Train CNNs for image classification
- Real computer vision on standard benchmark datasets
- Built entirely from scratch using only NumPy
- Competitive performance with modern frameworks

**Additional Capabilities**:
- GPT-style language models with attention mechanisms
- Modern optimizers (Adam, SGD) with learning rate scheduling
- Performance profiling, optimization, and competitive benchmarking

**No dependencies on PyTorch or TensorFlow - everything is YOUR code!**

---

## 🛠 Help Shape TinyTorch

We're sharing TinyTorch early because we'd rather shape the direction with community input than build in isolation. Before diving into code, we want to hear from you:

**If you're a student:**
→ What hands-on labs or projects would help you learn ML systems?

**If you teach:**
→ What would make TinyTorch easy to bring into a course?

**If you're a practitioner:**
→ What real-world systems tasks should we simulate?

**For everyone:**
→ What natural extensions belong in this "AI bricks" model?

📣 **[Share your thoughts in the discussion →](https://github.com/harvard-edge/cs249r_book/discussions/1076)**

---

## Current Status

| Ready | In Progress | Coming Soon |
|-------|-------------|-------------|
| ✅ All 20 modules implemented | 🔧 Documentation polish | 📅 NBGrader integration |
| ✅ Complete test suite (600+ tests) | 🔧 Edge case handling | 📅 Community leaderboard |
| ✅ `tito` CLI for workflows | 🔧 Instructor resources | 📅 Binder/Colab support |
| ✅ Historical milestone scripts | | |

**Want to explore the code?** [Browse the repository structure](#repository-structure) to see how modules are organized.

**Adventurous early adopter?** Local installation works, but expect rough edges. See the [setup guide](site/getting-started.md).

---

## 20 Progressive Modules

Build your framework through four progressive parts:

| Part | Modules | What You Build |
|------|---------|----------------|
| **I. Foundations** | 01-08 | Tensors, activations, layers, losses, dataloader, autograd, optimizers, training |
| **II. Vision** | 09 | Conv2d, CNNs for image classification |
| **III. Language** | 10-13 | Tokenization, embeddings, attention, transformers |
| **IV. Optimization** | 14-20 | Profiling, quantization, compression, acceleration, benchmarking, capstone |

Each module asks: **"Can I build this capability from scratch?"**

📖 **[Full curriculum and module details →](https://mlsysbook.ai/tinytorch)**

---

## Historical Milestones

As you progress, unlock recreations of landmark ML achievements:

| Year | Milestone | Your Achievement |
|------|-----------|------------------|
| 1957 | Perceptron | Binary classification with gradient descent |
| 1969 | XOR Crisis | Multi-layer networks solve non-linear problems |
| 1986 | Backpropagation | Multi-layer network training |
| 1998 | CNN Revolution | **Image classification with convolutions** |
| 2017 | Transformer Era | Language generation with self-attention |
| 2018+ | MLPerf | Production-ready optimization |

**These aren't toy demos** - they're historically significant ML achievements rebuilt with YOUR framework!

---

## Learning Philosophy

```python
# Traditional Course:
import torch
model.fit(X, y)  # Magic happens

# TinyTorch:
# You implement every component
# You measure memory usage
# You optimize performance
# You understand the systems
```

**Why Build Your Own Framework?**
- **Deep Understanding** - Know exactly what `loss.backward()` does
- **Systems Thinking** - Understand memory, compute, and scaling
- **Debugging Skills** - Fix problems at any level of the stack
- **Production Ready** - Learn patterns used in real ML systems

---

## Documentation

| Audience | Resources |
|----------|-----------|
| **Students** | [Course Website](https://mlsysbook.ai/tinytorch) ・ [Getting Started](site/getting-started.md) |
| **Instructors** | [Instructor Guide](INSTRUCTOR.md) |
| **Contributors** | [Contributing Guide](CONTRIBUTING.md) |

---

## Repository Structure

```
TinyTorch/
├── src/                        # 💻 Python source files (developers/contributors edit here)
│   ├── 01_tensor/              # Module 01: Tensor operations from scratch
│   │   ├── 01_tensor.py        # Python source (version controlled)
│   │   └── ABOUT.md            # Conceptual overview & learning objectives
│   ├── 02_activations/         # Module 02: ReLU, Softmax activations
│   ├── 03_layers/              # Module 03: Linear layers, Module system
│   ├── 04_losses/              # Module 04: MSE, CrossEntropy losses
│   ├── 05_dataloader/          # Module 05: Efficient data pipelines
│   ├── 06_autograd/            # Module 06: Automatic differentiation
│   ├── 07_optimizers/          # Module 07: SGD, Adam optimizers
│   ├── 08_training/            # Module 08: Complete training loops
│   ├── 09_convolutions/        # Module 09: Conv2d, MaxPool2d, CNNs
│   ├── 10_tokenization/        # Module 10: Text processing
│   ├── 11_embeddings/          # Module 11: Token & positional embeddings
│   ├── 12_attention/           # Module 12: Multi-head attention
│   ├── 13_transformers/        # Module 13: Complete transformer blocks
│   ├── 14_profiling/           # Module 14: Performance analysis
│   ├── 15_quantization/        # Module 15: Model compression (precision reduction)
│   ├── 16_compression/         # Module 16: Pruning & distillation
│   ├── 17_acceleration/        # Module 17: Hardware optimization
│   ├── 18_memoization/         # Module 18: KV-cache/memoization
│   ├── 19_benchmarking/        # Module 19: Performance measurement
│   └── 20_capstone/            # Module 20: Complete ML systems
│
├── modules/                    # 📓 Generated notebooks (learners work here)
│   ├── 01_tensor/              # Auto-generated from src/
│   │   ├── 01_tensor.ipynb     # Jupyter notebook for learning
│   │   ├── README.md           # Practical implementation guide
│   │   └── tensor.py           # Your implementation
│   └── ...                     # (20 module directories)
│
├── site/                       # 🌐 Course website & documentation (Jupyter Book)
│   ├── intro.md                # Landing page
│   ├── _toc.yml                # Site navigation (links to modules)
│   ├── _config.yml             # HTML website configuration
│   ├── chapters/               # Course content chapters
│   └── modules/                # Module documentation
│
├── milestones/                 # 🏆 Historical ML evolution - prove what you built!
│   ├── 01_1957_perceptron/     # Rosenblatt's first trainable network
│   ├── 02_1969_xor/            # Minsky's challenge & multi-layer solution
│   ├── 03_1986_mlp/            # Backpropagation & MNIST digits
│   ├── 04_1998_cnn/            # LeCun's CNNs & CIFAR-10
│   ├── 05_2017_transformer/    # Attention mechanisms & language
│   └── 06_2018_mlperf/         # Modern optimization & profiling
│
├── tito/                       # 🎛️ CLI tool for streamlined workflows
│   ├── main.py                 # Entry point
│   ├── commands/               # 23 command modules
│   └── core/                   # Core utilities
│
├── tinytorch/                  # 📦 Generated package (import from here)
│   ├── core/                   # Core ML components
│   └── ...                     # Your built framework!
│
└── tests/                      # ✅ Comprehensive test suite (600+ tests)
```

**Key workflow**: `src/*.py` → `modules/*.ipynb` → `tinytorch/*.py`

---

## Join the Community

TinyTorch is part of the [ML Systems Book](https://mlsysbook.ai) ecosystem. We're building an open community of learners and educators passionate about ML systems.

**Ways to get involved:**
- ⭐ Star this repo to show support
- 💬 Join [Discussions](https://github.com/harvard-edge/cs249r_book/discussions) to ask questions
- 🐛 Report issues or suggest improvements
- 🤝 Contribute modules, fixes, or documentation

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Related Projects

"TinyTorch" is a popular name for educational ML frameworks. We acknowledge excellent projects with similar names:

- [tinygrad](https://github.com/tinygrad/tinygrad) - George Hotz's minimalist framework
- [micrograd](https://github.com/karpathy/micrograd) - Andrej Karpathy's tiny autograd
- [MiniTorch](https://minitorch.github.io/) - Cornell's educational framework

**Our TinyTorch** distinguishes itself through its 20-module curriculum, NBGrader integration, ML systems focus, and connection to the [ML Systems Book](https://mlsysbook.ai) ecosystem.

---

## Acknowledgments

Created by [Prof. Vijay Janapa Reddi](https://vijay.seas.harvard.edu) at Harvard University.

Special thanks to students and contributors who helped build this framework.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**[📖 Full Documentation](https://mlsysbook.ai/tinytorch)** ・ **[💬 Discussions](https://github.com/harvard-edge/cs249r_book/discussions)** ・ **[🌐 ML Systems Book](https://mlsysbook.ai)**

**Start Small. Go Deep. Build ML Systems.**

</div>
