# TinyTorch

**Build ML Systems From First Principles**

[![CI](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/tinytorch-ci.yml?branch=dev&label=CI&logo=githubactions)](https://github.com/harvard-edge/cs249r_book/actions/workflows/tinytorch-ci.yml)
[![Docs](https://img.shields.io/badge/Docs-mlsysbook.ai%2Ftinytorch-blue)](https://mlsysbook.ai/tinytorch)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> 🧪 **Early Access** - TinyTorch is available for early adopters! All 20 modules are implemented with complete solutions and the `tito` CLI for streamlined workflows.
>
> 🎯 **Spring 2025**: Full public release with community features and classroom deployment resources.

---

## Why TinyTorch?

**"Most ML education teaches you to _use_ frameworks. TinyTorch teaches you to _build_ them."**

The difference between ML users and ML engineers determines who drives innovation versus who merely consumes it. TinyTorch bridges this gap by teaching you to build every component of modern AI systems from scratch.

A Harvard University course that transforms you from framework user to systems engineer, giving you the deep understanding needed to optimize, debug, and innovate at the foundation of AI.

---

## What You'll Build

A **complete ML framework** capable of:

🎯 **North Star Achievement**: Train CNNs on CIFAR-10 to **75%+ accuracy**
- Real computer vision with 50,000 training images
- Built entirely from scratch using only NumPy
- Competitive performance with modern frameworks

**Additional Capabilities**:
- GPT-style language models with attention mechanisms
- Modern optimizers (Adam, SGD) with learning rate scheduling
- Performance profiling, optimization, and competitive benchmarking

**No dependencies on PyTorch or TensorFlow - everything is YOUR code!**

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/harvard-edge/cs249r_book.git
cd cs249r_book/tinytorch

# Install in editable mode
pip install -e .

# Verify installation
tito doctor

# Start building!
tito module start 01
```

> **Note**: TinyTorch is part of the [ML Systems Book](https://mlsysbook.ai) project. Installing from the book repository keeps everything together and lets you easily update with `git pull`.

---

## 20 Progressive Modules

Build your framework through four progressive parts:

| Part | Modules | What You Build |
|------|---------|----------------|
| **I. Foundations** | 01-07 | Tensors, activations, layers, losses, autograd, optimizers, training |
| **II. Vision** | 08-09 | DataLoaders, Conv2d, CNNs → CIFAR-10 @ 75%+ |
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
| 1986 | Backpropagation | MNIST digit recognition at 95%+ |
| 1998 | CNN Revolution | **CIFAR-10 @ 75%+ accuracy** |
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
| **Students** | [Course Website](https://mlsysbook.ai/tinytorch) ・ [Quick Start](site/STUDENT_QUICKSTART.md) ・ [FAQ](site/faq.md) |
| **Instructors** | [Instructor Guide](INSTRUCTOR.md) ・ [NBGrader Setup](site/nbgrader/) ・ [TA Guide](TA_GUIDE.md) |
| **Contributors** | [Contributing Guide](CONTRIBUTING.md) ・ [Module Development](site/development/module-rules.md) |

---

## Repository Structure

```
tinytorch/
├── src/           # Source files (contributors edit here)
├── modules/       # Generated notebooks (learners work here)
├── tinytorch/     # Generated package (import from here)
├── milestones/    # Historical ML achievements
├── tests/         # Comprehensive test suite
├── site/          # Documentation website
└── tito/          # CLI tool
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
