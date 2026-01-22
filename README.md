# Machine Learning Systems
*Principles and Practices of Engineering Artificially Intelligent Systems*

<p align="center">
  <a href="README.md">English</a> •
  <a href="README/README_zh.md">中文</a> •
  <a href="README/README_ja.md">日本語</a> •
  <a href="README/README_ko.md">한국어</a>
</p>

<div align="center">

<p align="center">

  [![Book](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/book-validate-dev.yml?branch=dev&label=Book&logo=githubactions&cacheSeconds=300)](https://github.com/harvard-edge/cs249r_book/actions/workflows/book-validate-dev.yml)
  [![TinyTorch](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/tinytorch-ci.yml?branch=dev&label=TinyTorch&logo=python&cacheSeconds=300)](https://github.com/harvard-edge/cs249r_book/actions/workflows/tinytorch-ci.yml)
  ![Updated](https://img.shields.io/github/last-commit/harvard-edge/cs249r_book/dev?label=Updated&logo=git&cacheSeconds=300)
  [![License](https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-blue.svg)](https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE.md)
  [![Cite](https://img.shields.io/badge/Cite-IEEE%202024-blue?logo=ieee)](#-citation--license)
  [![Fund Us](https://img.shields.io/badge/Fund%20Us-Open%20Collective-blue.svg?logo=open-collective)](https://opencollective.com/mlsysbook)

</p>

<p align="center">

  <!-- Reader Navigation -->
  **[📖 Read Online](https://mlsysbook.ai)** •
  **[Tiny🔥Torch](https://mlsysbook.ai/tinytorch)** •
  **[📄 Download PDF](https://mlsysbook.ai/assets/downloads/Machine-Learning-Systems.pdf)** •
  **[📓 Download EPUB](https://mlsysbook.ai/epub)** •
  **[🌐 Explore Ecosystem](https://mlsysbook.org)**

</p>

📚 **Hardcopy edition coming 2026 with MIT Press.**

</div>

---

## Mission

**The world is rushing to build AI systems. It is not engineering them.**

That gap is what we mean by AI engineering.

**AI engineering is the discipline of building efficient, reliable, safe, and robust intelligent systems that operate in the real world, not just models in isolation.**

**Our mission:** Establish AI engineering as a foundational discipline, alongside software engineering and computer engineering, by teaching how to design, build, and evaluate end to end intelligent systems. The long term impact of AI will be shaped by engineers who can turn ideas into working, dependable systems.

---

## What’s in this repo

This repository is the open learning stack for AI systems engineering.

It includes the textbook source, TinyTorch, hardware kits, and upcoming co-labs that connect principles to runnable code and real devices.

---

## Start Here

Choose a path based on your goal.

**READ** Start with the [textbook](https://mlsysbook.ai). Try [Chapter 1](https://www.mlsysbook.ai/contents/core/introduction/introduction.html) and the [Benchmarking chapter](https://mlsysbook.ai/contents/core/benchmarking/benchmarking.html).

**BUILD** Start TinyTorch with the [getting started guide](https://mlsysbook.ai/tinytorch/getting-started.html). Begin with Module 01 and work up from CNNs to transformers and the MLPerf benchmarks.

**DEPLOY** Pick a [hardware kit](https://mlsysbook.ai/kits) and run the labs on Arduino, Raspberry Pi, and other edge devices.

**CONNECT** Say hello in [Discussions](https://github.com/harvard-edge/cs249r_book/discussions). We will do our best to reply.

---

## The Learning Stack

The learning stack below shows how the textbook connects to hands on work and deployment. Read the textbook, then pick your path:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│                           MACHINE LEARNING SYSTEMS                            │
│                              Read the Textbook                                │
│                                                                               │
│                    Theory • Concepts • Best Practices                         │
│                                                                               │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │
                          ┌─────────────┼─────────────┐
                          │             │             │
                          ▼             ▼             ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                            HANDS-ON ACTIVITIES                                │
│                           (pick one or all)                                   │
│                                                                               │
│     ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐     │
│     │                 │      │                 │      │                 │     │
│     │    SOFTWARE     │      │    TINYTORCH    │      │    HARDWARE     │     │
│     │    CO-LABS      │      │    FRAMEWORK    │      │      LABS       │     │
│     │                 │      │                 │      │                 │     │
│     │ EXPLORE         │      │ BUILD           │      │ DEPLOY          │     │
│     │                 │      │                 │      │                 │     │
│     │ Run controlled  │      │ Understand      │      │ Engineer under  │     │
│     │ experiments on  │      │ frameworks by   │      │ real constraints│     │
│     │ latency, memory,│      │ implementing    │      │ memory, power,  │     │
│     │ energy, cost    │      │ them            │      │ timing, safety  │     │
│     │                 │      │                 │      │                 │     │
│     │ (coming 2026)   │      │                 │      │ Arduino, Pi     │     │
│     └─────────────────┘      └─────────────────┘      └─────────────────┘     │
│                                                                               │
│           EXPLORE                  BUILD                   DEPLOY             │
│                                                                               │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│                                  AI OLYMPICS                                  │
│                                 Prove Mastery                                 │
│                                                                               │
│       Compete across all tracks • University teams • Public leaderboards      │
│                                                                               │
│                                (coming 2026)                                  │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

| | Component | What You Do | Link |
|--|-----------|-------------|------|
| **READ** | [📖 Textbook](https://mlsysbook.ai) | Understand ML systems concepts | [book/](book/README.md) |
| **EXPLORE** | 🔮 Software Co-Labs | Run controlled experiments on latency, memory, energy, cost | *Coming 2026* |
| **BUILD** | [🔥 TinyTorch](https://mlsysbook.ai/tinytorch) | Understand frameworks by implementing them | [tinytorch/](tinytorch/README.md) |
| **DEPLOY** | [🔧 Hardware Kits](https://mlsysbook.ai/kits) | Engineer under real constraints: memory, power, timing, safety | [kits/](kits/README.md) |
| **PROVE** | 🏆 AI Olympics | Compete and benchmark across all tracks | *Coming 2026* |

**What each path teaches:**
- **EXPLORE** teaches *why* — Understand tradeoffs. Change batch sizes, precision, model architectures and see how latency, memory, and accuracy shift.
- **BUILD** teaches *how* — Understand internals. Implement autograd, optimizers, and attention from scratch to see how TensorFlow and PyTorch actually work.
- **DEPLOY** teaches *where* — Understand constraints. Face real memory limits, power budgets, and latency requirements on actual hardware.

---

## What You Will Learn

This textbook teaches you to think at the intersection of machine learning and systems engineering. Each chapter bridges algorithmic concepts with the infrastructure that makes them work in practice.

### The ML ↔ Systems Bridge

| ML Concept | Systems Concept | What You Learn |
|------------|-----------------|----------------|
| Model parameters | Memory constraints | How to fit large models on resource-limited devices |
| Inference latency | Hardware acceleration | How GPUs, TPUs, and accelerators execute neural networks |
| Training convergence | Compute efficiency | How mixed-precision and optimization techniques reduce cost |
| Model accuracy | Quantization and pruning | How to compress models while preserving performance |
| Data requirements | Pipeline infrastructure | How to build efficient data loading and preprocessing |
| Model deployment | MLOps practices | How to monitor, version, and update models in production |
| Privacy constraints | On-device learning | How to train and adapt models without sending data to the cloud |

### Book Structure

| Part | Focus | Chapters |
|------|-------|----------|
| **I. Foundations** | Core concepts | Introduction, ML Systems, DL Primer, Architectures |
| **II. Design** | Building blocks | Workflow, Data Engineering, Frameworks, Training |
| **III. Performance** | Making it fast | Efficient AI, Optimizations, HW Acceleration, Benchmarking |
| **IV. Deployment** | Making it work | MLOps, On-device Learning, Privacy, Robustness |
| **V. Trust** | Making it right | Responsible AI, Sustainable AI, AI for Good |
| **VI. Frontiers** | What's next | Emerging trends and future directions |

---

## What Makes This Different

This is a living textbook. We keep it updated as the field grows, with community input along the way.

AI may feel like it is moving at lightning speed, but the engineering building blocks that make it work do not change as quickly as the headlines. This project is built around those stable foundations.

Think of it like LEGO. New sets arrive all the time, but the bricks themselves stay the same. Once you learn how the bricks fit together, you can build anything. Here, those "AI bricks" are the solid systems principles that make AI work.

Whether you are reading a chapter, running a lab, or sharing feedback, you are helping make these ideas more accessible to the next learner.

### Research to Teaching Loop

We use the same loop for research and teaching: define the system problem, build a reference implementation, benchmark it, then turn it into curriculum and tooling so others can reproduce and extend it.

| Loop Step | Research Artifacts | Teaching Artifacts |
|-----------|-------------------|-------------------|
| **Measure** | Benchmarks, suites, metrics | Benchmarking chapter, assignments |
| **Build** | Reference systems, compilers, runtimes | TinyTorch modules, co-labs |
| **Deploy** | Hardware targets, constraints, reliability | Hardware labs, kits |

---

## Support This Work

We are working toward **1 million learners by 2030** so that AI engineering becomes a shared, teachable discipline, not a collection of isolated practices. Every star, share, and contribution helps move this effort forward.

### Why GitHub Stars Matter

<div align="center">

*What gets measured gets improved.*

Each star is a learner, educator, or supporter who believes AI systems should be engineered with rigor and real world constraints in mind.

[![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)

[![Star History Chart](https://api.star-history.com/svg?repos=harvard-edge/cs249r_book&type=Date)](https://star-history.com/#harvard-edge/cs249r_book&Date)

1 learner → 10 learners → 100 learners → 1,000 learners → **10,000 learners** → 100,000 learners → **1M learners**

</div>

Stars are not the goal. They are a signal.

A visible, growing community makes it easier for universities, foundations, and industry partners to adopt this material, donate hardware, and fund workshops. That momentum lowers the barrier for the next institution, the next classroom, and the next cohort of learners.

Support raised through this signal flows into [Open Collective](https://opencollective.com/mlsysbook) and funds concrete outcomes such as TinyML4D workshops, hardware kits for underserved classrooms, and the infrastructure required to keep this resource free and open.

One click can unlock the next classroom, the next contributor, and the next generation of AI engineers.

### Fund the Mission

<div align="center">

All contributions go to [Open Collective](https://opencollective.com/mlsysbook), a transparent fund that supports educational outreach.

[![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

</div>

---

## Community and Resources

| Resource | Description |
|---|---|
| [📖 **Textbook**](https://mlsysbook.ai) | Interactive online textbook |
| [🔥 **TinyTorch**](https://mlsysbook.ai/tinytorch) | Build ML frameworks from scratch |
| [🔧 **Hardware Kits**](https://mlsysbook.ai/kits) | Deploy to Arduino, Raspberry Pi, edge devices |
| [🌐 **Ecosystem**](https://mlsysbook.org) | Resources, workshops, and community |
| [💬 **Discussions**](https://github.com/harvard-edge/cs249r_book/discussions) | Questions and ideas |

---

## Contributing

We welcome contributions to the book, TinyTorch, and hardware kits!

| I want to... | Go here |
|--------------|---------|
| Fix a typo or improve a chapter | [book/docs/CONTRIBUTING.md](book/docs/CONTRIBUTING.md) |
| Add a TinyTorch module or fix a bug | [tinytorch/CONTRIBUTING.md](tinytorch/CONTRIBUTING.md) |
| Improve hardware labs | [kits/README.md](kits/README.md) |
| Report an issue | [GitHub Issues](https://github.com/harvard-edge/cs249r_book/issues) |
| Ask a question | [GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions) |

---

## Citation & License

### Citation
```bibtex
@inproceedings{reddi2024mlsysbook,
  title        = {MLSysBook.AI: Principles and Practices of Machine Learning Systems Engineering},
  author       = {Reddi, Vijay Janapa},
  booktitle    = {2024 International Conference on Hardware/Software Codesign and System Synthesis (CODES+ ISSS)},
  pages        = {41--42},
  year         = {2024},
  organization = {IEEE},
  url          = {https://mlsysbook.org}
}
```

### License

This project uses a dual-license structure:

| Component | License | What It Means |
|-----------|---------|---------------|
| **Book content** | [CC BY-NC-ND 4.0](LICENSE.md) | Share freely with attribution; no commercial use; no derivatives |
| **TinyTorch code** | [Apache 2.0](tinytorch/LICENSE) | Use, modify, and distribute freely; includes patent protection |

The textbook content (chapters, figures, explanations) is educational material that should circulate with attribution and without commercial exploitation. The software framework is a tool designed to be easy for anyone to use, modify, or integrate into their own projects.

---

## Contributors

Thanks goes to these wonderful people who have contributed to making this resource better for everyone!

**Legend:** 🪲 Bug Hunter · 🧑‍💻 Code Contributor · ✍️ Word Wizard · 🎨 Design Artist · 🧠 Idea Generator · 🔎 Code Reviewer · 🧪 Test Engineer · 🛠️ Tool Builder · 🏗️ Infrastructure Builder · 🔩 Maintenance Master · 🔬 Researcher · 📖 Tutorial Author

### 📖 Textbook Contributors

<!-- BOOK-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧠 🔎 🧪 🛠️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/Mjrovai"><img src="https://avatars.githubusercontent.com/Mjrovai?v=4?s=50" width="50px;" alt="Marcelo Rovai"/><br /><sub><b>Marcelo Rovai</b></sub></a><br />🧑‍💻 🎨 🧪</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/GabrielAmazonas"><img src="https://avatars.githubusercontent.com/GabrielAmazonas?v=4?s=50" width="50px;" alt="Gabriel Amazonas"/><br /><sub><b>Gabriel Amazonas</b></sub></a><br />🪲 ✍️ 🧠</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/kai4avaya"><img src="https://avatars.githubusercontent.com/kai4avaya?v=4?s=50" width="50px;" alt="Kai Kleinbard"/><br /><sub><b>Kai Kleinbard</b></sub></a><br />🧑‍💻 🛠️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/didier-durand"><img src="https://avatars.githubusercontent.com/didier-durand?v=4?s=50" width="50px;" alt="Didier Durand"/><br /><sub><b>Didier Durand</b></sub></a><br />✍️ 🪲</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/hzeljko"><img src="https://avatars.githubusercontent.com/hzeljko?v=4?s=50" width="50px;" alt="Zeljko Hrcek"/><br /><sub><b>Zeljko Hrcek</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/jasonjabbour"><img src="https://avatars.githubusercontent.com/jasonjabbour?v=4?s=50" width="50px;" alt="Jason Jabbour"/><br /><sub><b>Jason Jabbour</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/uchendui"><img src="https://avatars.githubusercontent.com/uchendui?v=4?s=50" width="50px;" alt="Ikechukwu Uchendu"/><br /><sub><b>Ikechukwu Uchendu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/Naeemkh"><img src="https://avatars.githubusercontent.com/Naeemkh?v=4?s=50" width="50px;" alt="Naeem Khoshnevis"/><br /><sub><b>Naeem Khoshnevis</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/Sara-Khosravi"><img src="https://avatars.githubusercontent.com/Sara-Khosravi?v=4?s=50" width="50px;" alt="Sara Khosravi"/><br /><sub><b>Sara Khosravi</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/V0XNIHILI"><img src="https://avatars.githubusercontent.com/V0XNIHILI?v=4?s=50" width="50px;" alt="Douwe den Blanken"/><br /><sub><b>Douwe den Blanken</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/18jeffreyma"><img src="https://avatars.githubusercontent.com/18jeffreyma?v=4?s=50" width="50px;" alt="Jeffrey Ma"/><br /><sub><b>Jeffrey Ma</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/shanzehbatool"><img src="https://avatars.githubusercontent.com/shanzehbatool?v=4?s=50" width="50px;" alt="shanzehbatool"/><br /><sub><b>shanzehbatool</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/eliasab16"><img src="https://avatars.githubusercontent.com/eliasab16?v=4?s=50" width="50px;" alt="Elias"/><br /><sub><b>Elias</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/JaredP94"><img src="https://avatars.githubusercontent.com/JaredP94?v=4?s=50" width="50px;" alt="Jared Ping"/><br /><sub><b>Jared Ping</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/ishapira1"><img src="https://avatars.githubusercontent.com/ishapira1?v=4?s=50" width="50px;" alt="Itai Shapira"/><br /><sub><b>Itai Shapira</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/8863743b4f26c1a20e730fcf7ebc3bc0?d=identicon&s=100?v=4?s=50" width="50px;" alt="Maximilian Lam"/><br /><sub><b>Maximilian Lam</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/jaysonzlin"><img src="https://avatars.githubusercontent.com/jaysonzlin?v=4?s=50" width="50px;" alt="Jayson Lin"/><br /><sub><b>Jayson Lin</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/sophiacho1"><img src="https://avatars.githubusercontent.com/sophiacho1?v=4?s=50" width="50px;" alt="Sophia Cho"/><br /><sub><b>Sophia Cho</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/andreamurillomtz"><img src="https://avatars.githubusercontent.com/andreamurillomtz?v=4?s=50" width="50px;" alt="Andrea"/><br /><sub><b>Andrea</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/alxrod"><img src="https://avatars.githubusercontent.com/alxrod?v=4?s=50" width="50px;" alt="Alex Rodriguez"/><br /><sub><b>Alex Rodriguez</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/korneelf1"><img src="https://avatars.githubusercontent.com/korneelf1?v=4?s=50" width="50px;" alt="Korneel Van den Berghe"/><br /><sub><b>Korneel Van den Berghe</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/foundingnimo"><img src="https://avatars.githubusercontent.com/foundingnimo?v=4?s=50" width="50px;" alt="Nimo"/><br /><sub><b>Nimo</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/colbybanbury"><img src="https://avatars.githubusercontent.com/colbybanbury?v=4?s=50" width="50px;" alt="Colby Banbury"/><br /><sub><b>Colby Banbury</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/zishenwan"><img src="https://avatars.githubusercontent.com/zishenwan?v=4?s=50" width="50px;" alt="Zishen Wan"/><br /><sub><b>Zishen Wan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/mmaz"><img src="https://avatars.githubusercontent.com/mmaz?v=4?s=50" width="50px;" alt="Mark Mazumder"/><br /><sub><b>Mark Mazumder</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/ma3mool"><img src="https://avatars.githubusercontent.com/ma3mool?v=4?s=50" width="50px;" alt="Abdulrahman Mahmoud"/><br /><sub><b>Abdulrahman Mahmoud</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/DivyaAmirtharaj"><img src="https://avatars.githubusercontent.com/DivyaAmirtharaj?v=4?s=50" width="50px;" alt="Divya Amirtharaj"/><br /><sub><b>Divya Amirtharaj</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/srivatsankrishnan"><img src="https://avatars.githubusercontent.com/srivatsankrishnan?v=4?s=50" width="50px;" alt="Srivatsan Krishnan"/><br /><sub><b>Srivatsan Krishnan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/arnaumarin"><img src="https://avatars.githubusercontent.com/arnaumarin?v=4?s=50" width="50px;" alt="marin-llobet"/><br /><sub><b>marin-llobet</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/aptl26"><img src="https://avatars.githubusercontent.com/aptl26?v=4?s=50" width="50px;" alt="Aghyad Deeb"/><br /><sub><b>Aghyad Deeb</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/James-QiuHaoran"><img src="https://avatars.githubusercontent.com/James-QiuHaoran?v=4?s=50" width="50px;" alt="Haoran Qiu"/><br /><sub><b>Haoran Qiu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/Ekhao"><img src="https://avatars.githubusercontent.com/Ekhao?v=4?s=50" width="50px;" alt="Emil Njor"/><br /><sub><b>Emil Njor</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/ELSuitorHarvard"><img src="https://avatars.githubusercontent.com/ELSuitorHarvard?v=4?s=50" width="50px;" alt="ELSuitorHarvard"/><br /><sub><b>ELSuitorHarvard</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/kaiM0ves"><img src="https://avatars.githubusercontent.com/kaiM0ves?v=4?s=50" width="50px;" alt="kaiM0ves"/><br /><sub><b>kaiM0ves</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/oishib"><img src="https://avatars.githubusercontent.com/oishib?v=4?s=50" width="50px;" alt="oishib"/><br /><sub><b>oishib</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/jared-ni"><img src="https://avatars.githubusercontent.com/jared-ni?v=4?s=50" width="50px;" alt="Jared Ni"/><br /><sub><b>Jared Ni</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/AditiR-42"><img src="https://avatars.githubusercontent.com/AditiR-42?v=4?s=50" width="50px;" alt="Aditi Raju"/><br /><sub><b>Aditi Raju</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/MichaelSchnebly"><img src="https://avatars.githubusercontent.com/MichaelSchnebly?v=4?s=50" width="50px;" alt="Michael Schnebly"/><br /><sub><b>Michael Schnebly</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/VThuong99"><img src="https://avatars.githubusercontent.com/VThuong99?v=4?s=50" width="50px;" alt="Thuong Duong"/><br /><sub><b>Thuong Duong</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/leo47007"><img src="https://avatars.githubusercontent.com/leo47007?v=4?s=50" width="50px;" alt="Yu-Shun Hsiao"/><br /><sub><b>Yu-Shun Hsiao</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/BaeHenryS"><img src="https://avatars.githubusercontent.com/BaeHenryS?v=4?s=50" width="50px;" alt="Henry Bae"/><br /><sub><b>Henry Bae</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/eimlav"><img src="https://avatars.githubusercontent.com/eimlav?v=4?s=50" width="50px;" alt="Eimhin Laverty"/><br /><sub><b>Eimhin Laverty</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/jaywonchung"><img src="https://avatars.githubusercontent.com/jaywonchung?v=4?s=50" width="50px;" alt="Jae-Won Chung"/><br /><sub><b>Jae-Won Chung</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/ShvetankPrakash"><img src="https://avatars.githubusercontent.com/ShvetankPrakash?v=4?s=50" width="50px;" alt="Shvetank Prakash"/><br /><sub><b>Shvetank Prakash</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/marcozennaro"><img src="https://avatars.githubusercontent.com/marcozennaro?v=4?s=50" width="50px;" alt="Marco Zennaro"/><br /><sub><b>Marco Zennaro</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/aryatschand"><img src="https://avatars.githubusercontent.com/aryatschand?v=4?s=50" width="50px;" alt="Arya Tschand"/><br /><sub><b>Arya Tschand</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/arbass22"><img src="https://avatars.githubusercontent.com/arbass22?v=4?s=50" width="50px;" alt="Andrew Bass"/><br /><sub><b>Andrew Bass</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/pongtr"><img src="https://avatars.githubusercontent.com/pongtr?v=4?s=50" width="50px;" alt="Pong Trairatvorakul"/><br /><sub><b>Pong Trairatvorakul</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/euranofshin"><img src="https://avatars.githubusercontent.com/euranofshin?v=4?s=50" width="50px;" alt="Eura Nofshin"/><br /><sub><b>Eura Nofshin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/0c931fcfd03cd548d44c90602dd773ba?d=identicon&s=100?v=4?s=50" width="50px;" alt="Matthew Stewart"/><br /><sub><b>Matthew Stewart</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/af39c27c6090c50a1921a9b6366e81cc?d=identicon&s=100?v=4?s=50" width="50px;" alt="Emeka Ezike"/><br /><sub><b>Emeka Ezike</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/jianqingdu"><img src="https://avatars.githubusercontent.com/jianqingdu?v=4?s=50" width="50px;" alt="jianqingdu"/><br /><sub><b>jianqingdu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/jzhou1318"><img src="https://avatars.githubusercontent.com/jzhou1318?v=4?s=50" width="50px;" alt="Jennifer Zhou"/><br /><sub><b>Jennifer Zhou</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/vitasam"><img src="https://avatars.githubusercontent.com/vitasam?v=4?s=50" width="50px;" alt="The Random DIY"/><br /><sub><b>The Random DIY</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/468ef35acc69f3266efd700992daa369?d=identicon&s=100?v=4?s=50" width="50px;" alt="Fatima Shah"/><br /><sub><b>Fatima Shah</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/BrunoScaglione"><img src="https://avatars.githubusercontent.com/BrunoScaglione?v=4?s=50" width="50px;" alt="Bruno Scaglione"/><br /><sub><b>Bruno Scaglione</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/Allen-Kuang"><img src="https://avatars.githubusercontent.com/Allen-Kuang?v=4?s=50" width="50px;" alt="Allen-Kuang"/><br /><sub><b>Allen-Kuang</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/4ad8cdf19eb3b666ace97d3eedb19278?d=identicon&s=100?v=4?s=50" width="50px;" alt="Tess314"/><br /><sub><b>Tess314</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/taunoe"><img src="https://avatars.githubusercontent.com/taunoe?v=4?s=50" width="50px;" alt="Tauno Erik"/><br /><sub><b>Tauno Erik</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/gnodipac886"><img src="https://avatars.githubusercontent.com/gnodipac886?v=4?s=50" width="50px;" alt="gnodipac886"/><br /><sub><b>gnodipac886</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/serco425"><img src="https://avatars.githubusercontent.com/serco425?v=4?s=50" width="50px;" alt="Sercan Aygün"/><br /><sub><b>Sercan Aygün</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/TheHiddenLayer"><img src="https://avatars.githubusercontent.com/TheHiddenLayer?v=4?s=50" width="50px;" alt="TheHiddenLayer"/><br /><sub><b>TheHiddenLayer</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/Gjain234"><img src="https://avatars.githubusercontent.com/Gjain234?v=4?s=50" width="50px;" alt="Gauri Jain"/><br /><sub><b>Gauri Jain</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/FinAminToastCrunch"><img src="https://avatars.githubusercontent.com/FinAminToastCrunch?v=4?s=50" width="50px;" alt="Fin Amin"/><br /><sub><b>Fin Amin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/alex-oesterling"><img src="https://avatars.githubusercontent.com/alex-oesterling?v=4?s=50" width="50px;" alt="Alex Oesterling"/><br /><sub><b>Alex Oesterling</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/AbenezerKb"><img src="https://avatars.githubusercontent.com/AbenezerKb?v=4?s=50" width="50px;" alt="Abenezer Angamo"/><br /><sub><b>Abenezer Angamo</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/BravoBaldo"><img src="https://avatars.githubusercontent.com/BravoBaldo?v=4?s=50" width="50px;" alt="Baldassarre Cesarano"/><br /><sub><b>Baldassarre Cesarano</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/Jahnic-kb"><img src="https://avatars.githubusercontent.com/Jahnic-kb?v=4?s=50" width="50px;" alt="Jahnic Beck"/><br /><sub><b>Jahnic Beck</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/aethernavshulkraven-allain"><img src="https://avatars.githubusercontent.com/aethernavshulkraven-allain?v=4?s=50" width="50px;" alt="अरनव शुक्ला | Arnav Shukla"/><br /><sub><b>अरनव शुक्ला | Arnav Shukla</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/RinZ27"><img src="https://avatars.githubusercontent.com/RinZ27?v=4?s=50" width="50px;" alt="Rin"/><br /><sub><b>Rin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/bilgeacun"><img src="https://avatars.githubusercontent.com/bilgeacun?v=4?s=50" width="50px;" alt="Bilge Acun"/><br /><sub><b>Bilge Acun</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/atcheng2"><img src="https://avatars.githubusercontent.com/atcheng2?v=4?s=50" width="50px;" alt="Andy Cheng"/><br /><sub><b>Andy Cheng</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/arighosh05"><img src="https://avatars.githubusercontent.com/arighosh05?v=4?s=50" width="50px;" alt="Aritra Ghosh"/><br /><sub><b>Aritra Ghosh</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/abigailswallow"><img src="https://avatars.githubusercontent.com/abigailswallow?v=4?s=50" width="50px;" alt="abigailswallow"/><br /><sub><b>abigailswallow</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/YangZhou1997"><img src="https://avatars.githubusercontent.com/YangZhou1997?v=4?s=50" width="50px;" alt="Yang Zhou"/><br /><sub><b>Yang Zhou</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/XaicuL"><img src="https://avatars.githubusercontent.com/XaicuL?v=4?s=50" width="50px;" alt="JEON HYUNJUN(Luciano)"/><br /><sub><b>JEON HYUNJUN(Luciano)</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/emmanuel2406"><img src="https://avatars.githubusercontent.com/emmanuel2406?v=4?s=50" width="50px;" alt="Emmanuel Rassou"/><br /><sub><b>Emmanuel Rassou</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/jasonlyik"><img src="https://avatars.githubusercontent.com/jasonlyik?v=4?s=50" width="50px;" alt="Jason Yik"/><br /><sub><b>Jason Yik</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/jessicaquaye"><img src="https://avatars.githubusercontent.com/jessicaquaye?v=4?s=50" width="50px;" alt="Jessica Quaye"/><br /><sub><b>Jessica Quaye</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/cursoragent"><img src="https://avatars.githubusercontent.com/cursoragent?v=4?s=50" width="50px;" alt="Cursor Agent"/><br /><sub><b>Cursor Agent</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/happyappledog"><img src="https://avatars.githubusercontent.com/happyappledog?v=4?s=50" width="50px;" alt="happyappledog"/><br /><sub><b>happyappledog</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/snuggs"><img src="https://avatars.githubusercontent.com/snuggs?v=4?s=50" width="50px;" alt="Snuggs"/><br /><sub><b>Snuggs</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/swilcock0"><img src="https://avatars.githubusercontent.com/swilcock0?v=4?s=50" width="50px;" alt="Sam Wilcock"/><br /><sub><b>Sam Wilcock</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/sjohri20"><img src="https://avatars.githubusercontent.com/sjohri20?v=4?s=50" width="50px;" alt="Shreya Johri"/><br /><sub><b>Shreya Johri</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/skmur"><img src="https://avatars.githubusercontent.com/skmur?v=4?s=50" width="50px;" alt="Sonia Murthy"/><br /><sub><b>Sonia Murthy</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/fc4f3460cdfb9365ab59bdeafb06413e?d=identicon&s=100?v=4?s=50" width="50px;" alt="Costin-Andrei Oncescu"/><br /><sub><b>Costin-Andrei Oncescu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/0d6b8616427d8b19d425c9808692e347?d=identicon&s=100?v=4?s=50" width="50px;" alt="formlsysbookissue"/><br /><sub><b>formlsysbookissue</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/7cd8d5dfd83071f23979019d97655dc5?d=identicon&s=100?v=4?s=50" width="50px;" alt="Annie Laurie Cook"/><br /><sub><b>Annie Laurie Cook</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/5aa037840c0ca11ee42784ed4843c655?d=identicon&s=100?v=4?s=50" width="50px;" alt="Parampreet Singh"/><br /><sub><b>Parampreet Singh</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/b15b6e0e9adf58099905c1a0fd474cb9?d=identicon&s=100?v=4?s=50" width="50px;" alt="Vijay Edupuganti"/><br /><sub><b>Vijay Edupuganti</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/f88052cca4f401d9b0f43aed0a53434a?d=identicon&s=100?v=4?s=50" width="50px;" alt="Jothi Ramaswamy"/><br /><sub><b>Jothi Ramaswamy</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/35a8d9ffd03f05e79a2c6ce6206a56f2?d=identicon&s=100?v=4?s=50" width="50px;" alt="Batur Arslan"/><br /><sub><b>Batur Arslan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/bd53d146aa888548c8db4da02bf81e7a?d=identicon&s=100?v=4?s=50" width="50px;" alt="Curren Iyer"/><br /><sub><b>Curren Iyer</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/8d8410338458e08bd5e4b96f58e1c217?d=identicon&s=100?v=4?s=50" width="50px;" alt="Edward Jin"/><br /><sub><b>Edward Jin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/28c6123d2c9f75578d3ccdedb0df3d11?d=identicon&s=100?v=4?s=50" width="50px;" alt="Tess Watt"/><br /><sub><b>Tess Watt</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/ef139181fe00190f21730f6912532e9e?d=identicon&s=100?v=4?s=50" width="50px;" alt="bluebaer7"/><br /><sub><b>bluebaer7</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/f5d58ba6aa9b00189d4c018d370e8f43?d=identicon&s=100?v=4?s=50" width="50px;" alt="yanjingl"/><br /><sub><b>yanjingl</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/a5a47df988ab1720dd706062e523ca32?d=identicon&s=100?v=4?s=50" width="50px;" alt="a-saraf"/><br /><sub><b>a-saraf</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/c2dc311aa8122d5f5f061e1db14682b1?d=identicon&s=100?v=4?s=50" width="50px;" alt="songhan"/><br /><sub><b>songhan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/4814aad67982ab07a69006a1ce9d2a72?d=identicon&s=100?v=4?s=50" width="50px;" alt="jvijay"/><br /><sub><b>jvijay</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/43b1feff77c8a95fd581774fb8ec891f?d=identicon&s=100?v=4?s=50" width="50px;" alt="Zishen"/><br /><sub><b>Zishen</b></sub></a><br />✍️</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- BOOK-CONTRIBUTORS-END -->

---

### 🔥 TinyTorch Contributors

<!-- TINYTORCH-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧠 🔎 🧪 🛠️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/kai4avaya"><img src="https://avatars.githubusercontent.com/kai4avaya?v=4?s=50" width="50px;" alt="kai"/><br /><sub><b>kai</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧪</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/minhdang26403"><img src="https://avatars.githubusercontent.com/minhdang26403?v=4?s=50" width="50px;" alt="Dang Truong"/><br /><sub><b>Dang Truong</b></sub></a><br />🪲 🧑‍💻 ✍️ 🧪</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/didier-durand"><img src="https://avatars.githubusercontent.com/didier-durand?v=4?s=50" width="50px;" alt="Didier Durand"/><br /><sub><b>Didier Durand</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/karthikdani"><img src="https://avatars.githubusercontent.com/karthikdani?v=4?s=50" width="50px;" alt="Karthik Dani"/><br /><sub><b>Karthik Dani</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/avikde"><img src="https://avatars.githubusercontent.com/avikde?v=4?s=50" width="50px;" alt="Avik De"/><br /><sub><b>Avik De</b></sub></a><br />🪲 🧪</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/AmirAlasady"><img src="https://avatars.githubusercontent.com/AmirAlasady?v=4?s=50" width="50px;" alt="Amir Alasady"/><br /><sub><b>Amir Alasady</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/jettythek"><img src="https://avatars.githubusercontent.com/jettythek?v=4?s=50" width="50px;" alt="jettythek"/><br /><sub><b>jettythek</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/Takosaga"><img src="https://avatars.githubusercontent.com/Takosaga?v=4?s=50" width="50px;" alt="Takosaga"/><br /><sub><b>Takosaga</b></sub></a><br />🪲</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- TINYTORCH-CONTRIBUTORS-END -->

---

### 🛠️ Hardware Kits Contributors

<!-- KITS-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧪 🛠️</td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/Mjrovai"><img src="https://avatars.githubusercontent.com/Mjrovai?v=4?s=50" width="50px;" alt="Marcelo Rovai"/><br /><sub><b>Marcelo Rovai</b></sub></a><br />✍️ 🧑‍💻 🎨 📖</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- KITS-CONTRIBUTORS-END -->

---

### 🧪 Labs Contributors

<!-- LABS-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧑‍💻 🎨 ✍️</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- LABS-CONTRIBUTORS-END -->

---

<div align="center">

**[⭐ Star us on GitHub](https://github.com/harvard-edge/cs249r_book#support-this-work) • [✉️ Subscribe](https://buttondown.email/mlsysbook) • [💬 Join discussions](https://github.com/harvard-edge/cs249r_book/discussions) • [🌐 Visit mlsysbook.ai](https://mlsysbook.ai)**

**Made with ❤️ for AI learners worldwide**

</div>
