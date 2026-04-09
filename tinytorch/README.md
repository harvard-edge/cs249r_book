<div align="center">

# Tiny🔥Torch

### Build Your Own ML Framework From Scratch

[![Version](https://img.shields.io/github/v/tag/harvard-edge/cs249r_book?filter=tinytorch-v*&label=version&color=D4740C&logo=fireship&logoColor=white)](https://github.com/harvard-edge/cs249r_book/releases?q=tinytorch)
[![Status](https://img.shields.io/badge/status-preview-orange?logo=github)](https://github.com/harvard-edge/cs249r_book/discussions/1076)
[![Docs](https://img.shields.io/badge/docs-mlsysbook.ai-blue?logo=readthedocs)](https://mlsysbook.ai/tinytorch)
[![Python](https://img.shields.io/badge/python-3.10+-3776ab?logo=python&logoColor=white)](https://python.org)
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

<table>
  <thead>
    <tr>
      <th width="33%">Ready</th>
      <th width="33%">In Progress</th>
      <th width="34%">Coming Soon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>✅ All 20 modules implemented</td>
      <td>🔧 Documentation polish</td>
      <td>📅 NBGrader integration</td>
    </tr>
    <tr>
      <td>✅ Complete test suite (600+ tests)</td>
      <td>🔧 Edge case handling</td>
      <td>📅 Community leaderboard</td>
    </tr>
    <tr>
      <td>✅ <code>tito</code> CLI for workflows</td>
      <td>🔧 Instructor resources</td>
      <td>📅 Binder/Colab support</td>
    </tr>
    <tr>
      <td>✅ Historical milestone scripts</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

**Want to explore the code?** [Browse the repository structure](#repository-structure) to see how modules are organized.

**Adventurous early adopter?** Local installation works, but expect rough edges. See the [setup guide](site/getting-started.md).

---

## 🏗️ 20 Progressive Modules

Build your framework through four progressive parts:

<table>
  <thead>
    <tr>
      <th width="20%">Part</th>
      <th width="15%">Modules</th>
      <th width="65%">What You Build</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>I. Foundations</b></td>
      <td align="center">01-08</td>
      <td>Tensors, activations, layers, losses, dataloader, autograd, optimizers, training</td>
    </tr>
    <tr>
      <td align="center"><b>II. Vision</b></td>
      <td align="center">09</td>
      <td>Conv2d, CNNs for image classification</td>
    </tr>
    <tr>
      <td align="center"><b>III. Language</b></td>
      <td align="center">10-13</td>
      <td>Tokenization, embeddings, attention, transformers</td>
    </tr>
    <tr>
      <td align="center"><b>IV. Optimization</b></td>
      <td align="center">14-20</td>
      <td>Profiling, quantization, compression, acceleration, benchmarking, capstone</td>
    </tr>
  </tbody>
</table>

Each module asks: **"Can I build this capability from scratch?"**

📖 **[Full curriculum and module details →](https://mlsysbook.ai/tinytorch)**

---

## 🏆 Historical Milestones

As you progress, unlock recreations of landmark ML achievements:

<table>
  <thead>
    <tr>
      <th width="15%">Year</th>
      <th width="35%">Milestone</th>
      <th width="50%">Your Achievement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>1958</b></td>
      <td>Perceptron</td>
      <td>Binary classification with gradient descent</td>
    </tr>
    <tr>
      <td align="center"><b>1969</b></td>
      <td>XOR Crisis</td>
      <td>Multi-layer networks solve non-linear problems</td>
    </tr>
    <tr>
      <td align="center"><b>1986</b></td>
      <td>Backpropagation</td>
      <td>Multi-layer network training</td>
    </tr>
    <tr>
      <td align="center"><b>1998</b></td>
      <td>CNN Revolution</td>
      <td><b>Image classification with convolutions</b></td>
    </tr>
    <tr>
      <td align="center"><b>2017</b></td>
      <td>Transformer Era</td>
      <td>Language generation with self-attention</td>
    </tr>
    <tr>
      <td align="center"><b>2018+</b></td>
      <td>MLPerf</td>
      <td>Production-ready optimization</td>
    </tr>
  </tbody>
</table>

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

<table>
  <thead>
    <tr>
      <th width="25%">Audience</th>
      <th width="75%">Resources</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Students</b></td>
      <td><a href="https://mlsysbook.ai/tinytorch">Course Website</a> ・ <a href="site/getting-started.md">Getting Started</a></td>
    </tr>
    <tr>
      <td><b>Instructors</b></td>
      <td><a href="INSTRUCTOR.md">Instructor Guide</a></td>
    </tr>
    <tr>
      <td><b>Contributors</b></td>
      <td><a href="CONTRIBUTING.md">Contributing Guide</a></td>
    </tr>
  </tbody>
</table>

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
│   │   ├── tensor.ipynb         # Jupyter notebook for learning
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
│   ├── 01_1958_perceptron/     # Rosenblatt's first trainable network
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

## Contributors

Thanks to these wonderful people who helped improve TinyTorch!

**Legend:** 🪲 Bug Hunter · ⚡ Code Warrior · 📚 Documentation Hero · 🎨 Design Artist · 🧠 Idea Generator · 🔎 Code Reviewer · 🧪 Test Engineer · 🛠️ Tool Builder

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=80" width="80px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧠 🔎 🧪 🛠️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kai4avaya"><img src="https://avatars.githubusercontent.com/kai4avaya?v=4?s=80" width="80px;" alt="kai"/><br /><sub><b>kai</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧪</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/minhdang26403"><img src="https://avatars.githubusercontent.com/minhdang26403?v=4?s=80" width="80px;" alt="Dang Truong"/><br /><sub><b>Dang Truong</b></sub></a><br />🪲 🧑‍💻 ✍️ 🧪</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/didier-durand"><img src="https://avatars.githubusercontent.com/didier-durand?v=4?s=80" width="80px;" alt="Didier Durand"/><br /><sub><b>Didier Durand</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rnjema"><img src="https://avatars.githubusercontent.com/rnjema?v=4?s=80" width="80px;" alt="rnjema"/><br /><sub><b>rnjema</b></sub></a><br />🧑‍💻 ✍️ 🛠️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Pratham-ja"><img src="https://avatars.githubusercontent.com/u/114498234?v=4?v=4?s=80" width="80px;" alt="Pratham Chaudhary"/><br /><sub><b>Pratham Chaudhary</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/karthikdani"><img src="https://avatars.githubusercontent.com/karthikdani?v=4?s=80" width="80px;" alt="Karthik Dani"/><br /><sub><b>Karthik Dani</b></sub></a><br />🪲 🧑‍💻</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/avikde"><img src="https://avatars.githubusercontent.com/avikde?v=4?s=80" width="80px;" alt="Avik De"/><br /><sub><b>Avik De</b></sub></a><br />🪲 🧪</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Takosaga"><img src="https://avatars.githubusercontent.com/Takosaga?v=4?s=80" width="80px;" alt="Takosaga"/><br /><sub><b>Takosaga</b></sub></a><br />🪲 ✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/joeswagson"><img src="https://avatars.githubusercontent.com/joeswagson?v=4?s=80" width="80px;" alt="joeswagson"/><br /><sub><b>joeswagson</b></sub></a><br />🧑‍💻 🛠️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AndreaMattiaGaravagno"><img src="https://avatars.githubusercontent.com/u/22458187?v=4?v=4?s=80" width="80px;" alt="AndreaMattiaGaravagno"/><br /><sub><b>AndreaMattiaGaravagno</b></sub></a><br />🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Roldao-Neto"><img src="https://avatars.githubusercontent.com/u/148023227?v=4?v=4?s=80" width="80px;" alt="Rolds"/><br /><sub><b>Rolds</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/asgalon"><img src="https://avatars.githubusercontent.com/u/45242704?v=4?v=4?s=80" width="80px;" alt="asgalon"/><br /><sub><b>asgalon</b></sub></a><br />🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AmirAlasady"><img src="https://avatars.githubusercontent.com/AmirAlasady?v=4?s=80" width="80px;" alt="Amir Alasady"/><br /><sub><b>Amir Alasady</b></sub></a><br />🪲</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jettythek"><img src="https://avatars.githubusercontent.com/jettythek?v=4?s=80" width="80px;" alt="jettythek"/><br /><sub><b>jettythek</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/wz1114841863"><img src="https://avatars.githubusercontent.com/wz1114841863?v=4?s=80" width="80px;" alt="wzz"/><br /><sub><b>wzz</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ngbolin"><img src="https://avatars.githubusercontent.com/u/9389997?v=4?v=4?s=80" width="80px;" alt="Ng Bo Lin"/><br /><sub><b>Ng Bo Lin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/keo-dara"><img src="https://avatars.githubusercontent.com/u/175544368?v=4?v=4?s=80" width="80px;" alt="keo-dara"/><br /><sub><b>keo-dara</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Kobra299"><img src="https://avatars.githubusercontent.com/u/4283156?v=4?v=4?s=80" width="80px;" alt="Wayne Norman"/><br /><sub><b>Wayne Norman</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lalalostcode"><img src="https://avatars.githubusercontent.com/u/149884766?v=4?v=4?s=80" width="80px;" alt="Ilham Rafiqin"/><br /><sub><b>Ilham Rafiqin</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/oscarf189"><img src="https://avatars.githubusercontent.com/u/28113740?v=4?v=4?s=80" width="80px;" alt="Oscar Flores"/><br /><sub><b>Oscar Flores</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harishb00a"><img src="https://avatars.githubusercontent.com/harishb00a?v=4?s=80" width="80px;" alt="harishb00a"/><br /><sub><b>harishb00a</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sotoblanco"><img src="https://avatars.githubusercontent.com/u/46135649?v=4?v=4?s=80" width="80px;" alt="Pastor Soto"/><br /><sub><b>Pastor Soto</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/salmanmkc"><img src="https://avatars.githubusercontent.com/u/32169182?v=4?v=4?s=80" width="80px;" alt="Salman Chishti"/><br /><sub><b>Salman Chishti</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/adityamulik"><img src="https://avatars.githubusercontent.com/u/10626835?v=4?v=4?s=80" width="80px;" alt="Aditya Mulik"/><br /><sub><b>Aditya Mulik</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AdemolaAri"><img src="https://avatars.githubusercontent.com/u/49918815?v=4?v=4?s=80" width="80px;" alt="Ademola Arigbabuwo"/><br /><sub><b>Ademola Arigbabuwo</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yarikoptic"><img src="https://avatars.githubusercontent.com/u/39889?v=4?v=4?s=80" width="80px;" alt="Yaroslav Halchenko"/><br /><sub><b>Yaroslav Halchenko</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harishb00"><img src="https://avatars.githubusercontent.com/u/43300971?v=4?v=4?s=80" width="80px;" alt="Harish"/><br /><sub><b>Harish</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/farhan523"><img src="https://avatars.githubusercontent.com/u/62025759?v=4?v=4?s=80" width="80px;" alt="Farhan Asghar"/><br /><sub><b>Farhan Asghar</b></sub></a><br />✍️</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

**Recognize a contributor:** Comment on any issue or PR:
```
@all-contributors please add @username for bug, code, doc, or ideas
```

---

## Acknowledgments

Created by [Prof. Vijay Janapa Reddi](https://vijay.seas.harvard.edu) at Harvard University.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

<b><a href="https://mlsysbook.ai/tinytorch">Full Documentation</a></b> ・ <b><a href="https://github.com/harvard-edge/cs249r_book/discussions">Discussions</a></b> ・ <b><a href="https://mlsysbook.ai">ML Systems Book</a></b>

<b>Start Small. Go Deep. Build ML Systems.</b>

</div>
