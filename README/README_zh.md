# 机器学习系统
*人工智能系统工程的原理与实践*

<p align="center">
  <a href="../README.md">English</a> •
  <a href="README_zh.md">中文</a> •
  <a href="README_ja.md">日本語</a> •
  <a href="README_ko.md">한국어</a>
</p>

<div align="center">

<p align="center">

  [![Book](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/book-validate-dev.yml?branch=dev&label=Book&logo=githubactions&cacheSeconds=300)](https://github.com/harvard-edge/cs249r_book/actions/workflows/book-validate-dev.yml)
  [![TinyTorch](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/tinytorch-validate-dev.yml?branch=dev&label=TinyTorch&logo=python&cacheSeconds=300)](https://github.com/harvard-edge/cs249r_book/actions/workflows/tinytorch-validate-dev.yml)
  ![Updated](https://img.shields.io/github/last-commit/harvard-edge/cs249r_book/dev?label=Updated&logo=git&cacheSeconds=300)
  [![License](https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-blue.svg)](https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE.md)
  [![Cite](https://img.shields.io/badge/Cite-IEEE%202024-blue?logo=ieee)](#-citation--license)
  [![Fund Us](https://img.shields.io/badge/Fund%20Us-Open%20Collective-blue.svg?logo=open-collective)](https://opencollective.com/mlsysbook)

</p>

<p align="center">

  <!-- Reader Navigation -->
  **[📖 在线阅读](https://mlsysbook.ai)** •
  **[Tiny🔥Torch](https://mlsysbook.ai/tinytorch)** •
  **[📄 下载 PDF](https://mlsysbook.ai/assets/downloads/Machine-Learning-Systems.pdf)** •
  **[📓 下载 EPUB](https://mlsysbook.ai/epub)** •
  **[🌐 探索生态系统](https://mlsysbook.org)**

</p>

📚 **2026 年 MIT Press 将出版纸质版**

</div>

---

## 使命

**世界正急速构建 AI 系统，却缺乏系统性的工程方法。**

这正是我们所说的 AI 工程。

**AI 工程是一门学科，致力于在真实世界中构建高效、可靠、安全且稳健的智能系统，而不仅仅是孤立的模型。**

**我们的使命：** 将 AI 工程确立为基础学科，与软件工程和计算机工程并列，通过教学让人们掌握端到端智能系统的设计、构建与评估方法。AI 的长期影响将由能够将想法转化为可运行、可信赖系统的工程师塑造。

---

## 本仓库包含的内容

本仓库是 AI 系统工程的开放学习栈。

它包括教材源码、TinyTorch、硬件套件以及即将推出的将原理与可运行代码、真实设备相连接的协作实验（co‑labs）。

---

## 入门指南

根据你的目标选择路径。

**READ** 从[教材](https://mlsysbook.ai)开始。先阅读[第 1 章](https://www.mlsysbook.ai/contents/core/introduction/introduction.html)和[Benchmarking 章节](https://mlsysbook.ai/contents/core/benchmarking/benchmarking.html)。

**BUILD** 按照[入门指南](https://mlsysbook.ai/tinytorch/getting-started.html)启动 TinyTorch。从 Module 01 开始，逐步学习 CNN、Transformer 以及 MLPerf 基准。

**DEPLOY** 选择[硬件套件](https://mlsysbook.ai/kits)，在 Arduino、Raspberry Pi 等边缘设备上进行实验。

**CONNECT** 在[Discussions](https://github.com/harvard-edge/cs249r_book/discussions)中打声招呼，我们会尽快回复。

---

## 学习栈

下面的示意图展示了教材如何与动手实践和部署相连接。阅读教材后，挑选你感兴趣的路径：

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
│                            HANDS‑ON ACTIVITIES                                │
│                           (pick one or all)                                   │
│                                                                               │
│     ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐     │
│     │                 │      │                 │      │                 │     │
│     │    SOFTWARE     │      │    TINYTORCH    │      │    HARDWARE     │     │
│     │    CO‑LABS      │      │    FRAMEWORK    │      │      LABS       │     │
│     │                 │      │                 │      │                 │     │
│     │ EXPLORE         │      │ BUILD           │      │ DEPLOY          │     │
│     │                 │      │                 │      │                 │     │
│     │ Run controlled  │      │ Understand      │      │ Engineer under  │     │
│     │ experiments on  │      │ frameworks by   │      │ real constraints│     │
│     │ latency, memory,│      │ implementing    │      │ memory, power,  │     │
│     │ energy, cost    │      │ them            │      │ timing, safety  │     │
│     │                 │      │                 │      │                 │
│     │ (coming 2026)   │      │                 │      │ Arduino, Pi     │
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

|   | Component | What You Do | Link |
|---|-----------|-------------|------|
| **READ** | [📖 教材](https://mlsysbook.ai) | 理解机器学习系统概念 | [book/](book/README.md) |
| **EXPLORE** | 🔮 Software Co‑Labs | 进行延迟、内存、能耗、成本实验 | *Coming 2026* |
| **BUILD** | [🔥 TinyTorch](https://mlsysbook.ai/tinytorch) | 亲手实现框架 | [tinytorch/](tinytorch/README.md) |
| **DEPLOY** | [🔧 Hardware Kits](https://mlsysbook.ai/kits) | 在受内存、功耗、时延、安全约束的硬件上工程实现 | [kits/](kits/README.md) |
| **PROVE** | 🏆 AI Olympics | 参与所有赛道的竞技与基准测试 | *Coming 2026* |

**每条路径的学习内容：**
- **EXPLORE** 解释 *为什么* —— 了解权衡。改变 batch size、精度、模型结构，观察延迟、内存和准确率的变化。
- **BUILD** 解释 *怎么做* —— 理解内部实现。自行实现 autograd、优化器、注意力机制，感受 TensorFlow 与 PyTorch 的工作原理。
- **DEPLOY** 解释 *在哪里* —— 了解约束。真实硬件的内存上限、功耗预算和时延要求下进行实验。

---

## 你将学到的内容

本教材教会你在机器学习与系统工程的交叉点思考。每一章都将算法概念与支撑其运行的基础设施相连接。

### ML ↔ Systems Bridge

| ML Concept | Systems Concept | What You Learn |
|------------|-----------------|----------------|
| Model parameters | Memory constraints | 如何在资源受限的设备上容纳大型模型 |
| Inference latency | Hardware acceleration | GPU、TPU、加速器如何执行神经网络 |
| Training convergence | Compute efficiency | 混合精度与优化技术如何降低计算成本 |
| Model accuracy | Quantization and pruning | 在保持性能的前提下压缩模型的方法 |
| Data requirements | Pipeline infrastructure | 如何构建高效的数据加载与预处理流水线 |
| Model deployment | MLOps practices | 在生产环境中监控、版本管理与更新模型的方式 |
| Privacy constraints | On‑device learning | 如何在不将数据上传云端的情况下进行学习与适应 |

### 书的结构

| Part | Focus | Chapters |
|------|-------|----------|
| **I. Foundations** | 基础概念 | Introduction, ML Systems, DL Primer, Architectures |
| **II. Design** | 构建模块 | Workflow, Data Engineering, Frameworks, Training |
| **III. Performance** | 加速性能 | Efficient AI, Optimizations, HW Acceleration, Benchmarking |
| **IV. Deployment** | 实际部署 | MLOps, On‑device Learning, Privacy, Robustness |
| **V. Trust** | 可信可靠 | Responsible AI, Sustainable AI, AI for Good |
| **VI. Frontiers** | 前沿展望 | Emerging trends and future directions |

---

## 与众不同之处

本书是一本活的教材。随着领域的发展，我们会持续更新，并吸收社区的反馈。

AI 的发展速度如闪电般，但支撑其运行的工程模块并不会像头条新闻那样快速更迭。本项目围绕这些稳固的基础构建。

把它想象成乐高。新套装层出不穷，但积木本身保持不变。只要学会如何拼接积木，就能构建任何东西。这里的 "AI 积木" 就是让 AI 正常工作的坚实系统原理。

无论是阅读章节、动手实验还是提供反馈，你都在帮助让这些理念对下一代学习者更加易得。

### Research to Teaching Loop

我们使用相同的循环进行研究与教学：定义系统问题 → 构建参考实现 → 基准测试 → 将其转化为课程与工具 → 让他人能够复现与扩展。

| Loop Step | Research Artifacts | Teaching Artifacts |
|-----------|-------------------|-------------------|
| **Measure** | Benchmarks, suites, metrics | Benchmarking chapter, assignments |
| **Build** | Reference systems, compilers, runtimes | TinyTorch modules, co‑labs |
| **Deploy** | Hardware targets, constraints, reliability | Hardware labs, kits |

---

## 支持我们的工作

我们目标是在 **2030 年之前培养 100 万学习者**，让 AI 工程成为共享的、可教学的学科，而不是零散的实践集合。每一次星标、分享与贡献都在推动这一目标前进。

### 为什么 GitHub Stars 很重要？

<div align="center">

*有度量才会改进。*

每一个星标代表一位相信 AI 系统应在严格且真实约束下进行工程化的学习者、教育者或支持者。

[![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)

[![Star History Chart](https://api.star-history.com/svg?repos=harvard-edge/cs249r_book&type=Date)](https://star-history.com/#harvard-edge/cs249r_book&Date)

1 位学习者 → 10 位 → 100 位 → 1,000 位 → **10,000 位** → 100,000 位 → **1M 位**

</div>

星标不是终点，而是信号。

一个可见的社区让大学、基金会和行业合作伙伴更容易采用本资源、捐赠硬件、资助研讨会。此举降低了下一所学校、下一间教室以及下一批学习者的门槛。

捐助将流向 [Open Collective](https://opencollective.com/mlsysbook)，用于 TinyML4D 研讨会、为资源匮乏的课堂提供硬件套件以及维持免费、开放资源的基础设施。

一次点击即可打开下一间教室、下一位贡献者以及下一代 AI 工程师。

### 为使命捐款

<div align="center">

All contributions go to [Open Collective](https://opencollective.com/mlsysbook), a transparent fund that supports educational outreach.

[![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

</div>

---

## 社区与资源

| Resource | Description |
|---|---|
| [📖 **教材**](https://mlsysbook.ai) | 交互式在线教材 |
| [🔥 **TinyTorch**](https://mlsysbook.ai/tinytorch) | 从零实现机器学习框架 |
| [🔧 **Hardware Kits**](https://mlsysbook.ai/kits) | 部署至 Arduino、Raspberry Pi、边缘设备 |
| [🌐 **Ecosystem**](https://mlsysbook.org) | 资源、研讨会、社区 |
| [💬 **Discussions**](https://github.com/harvard-edge/cs249r_book/discussions) | 提问与想法 |

---

## 贡献指南

我们欢迎对教材、TinyTorch 与硬件套件的贡献！

| 我想… | 前往 |
|--------------|---------|
| 修正错别字或改进章节 | [book/docs/CONTRIBUTING.md](book/docs/CONTRIBUTING.md) |
| 添加 TinyTorch 模块或修复 bug | [tinytorch/CONTRIBUTING.md](tinytorch/CONTRIBUTING.md) |
| 改进硬件实验 | [kits/README.md](kits/README.md) |
| 报告问题 | [GitHub Issues](https://github.com/harvard-edge/cs249r_book/issues) |
| 提问 | [GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions) |

---

## 引用与许可证

### 引用
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

### 许可证

本项目采用双许可证结构：

| Component | License | What It Means |
|-----------|---------|---------------|
| **Book content** | [CC BY‑NC‑ND 4.0](LICENSE.md) | 在署名、非商业、禁止演绎的前提下自由分发 |
| **TinyTorch code** | [Apache 2.0](tinytorch/LICENSE) | 自由使用、修改、分发并附带专利保护 |

教材内容（章节、图表、解释）属于教育资料，应在署名且非商业使用的前提下自由共享。软件框架则是供任何人使用、修改、集成的工具。

---

## 贡献者

以下优秀的贡献者让本资源更加完善：

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- ... (contributors omitted for brevity) -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

<div align="center">

**[⭐ 在 GitHub 上给我们加星](https://github.com/harvard-edge/cs249r_book#support-this-work) • [✉️ 订阅更新](https://buttondown.email/mlsysbook) • [💬 参与讨论](https://github.com/harvard-edge/cs249r_book/discussions) • [🌐 访问 mlsysbook.ai](https://mlsysbook.ai)**

*本教材由 MLSysBook 社区倾情打造。*

</div>
