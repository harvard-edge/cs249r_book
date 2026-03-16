# Machine Learning Systems
*Principles and Practices of Engineering Artificially Intelligent Systems*

<p align="center">
  <a href="README.md">English</a> •
  <a href="README/README_zh.md">中文</a> •
  <a href="README/README_ja.md">日本語</a> •
  <a href="README/README_ko.md">한국어</a>
</p>

<div align="center">

<!-- Build Status -->
<p align="center">
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/book-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/book-validate-dev.yml?branch=dev&label=Book&logo=githubactions&cacheSeconds=300" alt="Book"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/tinytorch-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/tinytorch-validate-dev.yml?branch=dev&label=TinyTorch&logo=python&cacheSeconds=300" alt="TinyTorch"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/kits-preview-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/kits-preview-dev.yml?branch=dev&label=Kits&logo=arduino&cacheSeconds=300" alt="Kits"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/labs-preview-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/labs-preview-dev.yml?branch=dev&label=Labs&logo=jupyter&cacheSeconds=300" alt="Labs"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/mlsysim-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/mlsysim-validate-dev.yml?branch=dev&label=MLSys%C2%B7im&logo=python&cacheSeconds=300" alt="MLSys·im"></a>
  <img src="https://img.shields.io/github/last-commit/harvard-edge/cs249r_book/dev?label=Updated&logo=git&cacheSeconds=300" alt="Updated">
</p>

<!-- Meta -->
<p align="center">
  <a href="https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE.md"><img src="https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-blue.svg" alt="License"></a>
  <a href="#-citation--license"><img src="https://img.shields.io/badge/Cite-IEEE%202024-blue?logo=ieee" alt="Cite"></a>
  <a href="https://opencollective.com/mlsysbook"><img src="https://img.shields.io/badge/Fund%20Us-Open%20Collective-blue.svg?logo=open-collective" alt="Fund Us"></a>
</p>

<p align="center">
  **[📘 Volume I](https://mlsysbook.ai/vol1/)** •
  **[📙 Volume II *(Summer 2026)*](https://mlsysbook.ai/vol2/)** •
  **[🔥 TinyTorch](https://mlsysbook.ai/tinytorch/)** •
  **[🔮 MLSys·im](https://mlsysbook.ai/mlsysim/)** •
  **[🌐 Ecosystem](https://mlsysbook.org)**
</p>

📚 **Hardcopy edition coming 2026 with MIT Press.**

</div>

---

## 🚀 Quick Start for Architects

Are you an AI engineer preparing for a systems design interview or building a production cluster? Start here.

<table>
  <thead>
    <tr>
      <th width="25%">Asset</th>
      <th width="50%">What It Is</th>
      <th width="25%">Action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>🚀 Local Audit</b></td>
      <td>Profile <b>your hardware</b> against the Iron Law instantly.</td>
      <td align="center"><kbd>pip install mlsysim</kbd></td>
    </tr>
    <tr>
      <td align="center"><b>💼 Interview Hub</b></td>
      <td>The <b>Napkin Math</b> for FAANG AI Systems Design.</td>
      <td align="center"><a href="interviews/README.md"><b>View Guide</b></a></td>
    </tr>
    <tr>
      <td align="center"><b>🗺️ Blueprint</b></td>
      <td>The <b>AI Engineering Blueprint</b>: Turnkey curricula for educators.</td>
      <td align="center"><a href="instructors/index.qmd"><b>View Hub</b></a></td>
    </tr>
    <tr>
      <td align="center"><b>👩‍🏫 Foundations</b></td>
      <td>A 12-week path for <b>single-machine</b> AI systems.</td>
      <td align="center"><a href="instructors/foundations-syllabus.qmd"><b>View Syllabus</b></a></td>
    </tr>
    <tr>
      <td align="center"><b>🚀 Scale</b></td>
      <td>A 12-week path for <b>distributed fleets</b> and frontier models.</td>
      <td align="center"><a href="instructors/scale-syllabus.qmd"><b>View Syllabus</b></a></td>
    </tr>
  </tbody>
</table>

---

## Mission

<div align="center">
  <blockquote>
    <b>The world is rushing to build AI systems. It is not engineering them.</b>
  </blockquote>
</div>

That gap is what I mean by AI engineering.

**AI engineering is the discipline of building efficient, reliable, safe, and robust intelligent systems that operate in the real world, not just models in isolation.**

**Our goal:** Help **100,000 learners** master ML Systems this year, and reach **1 million by 2030**. We believe AI engineering is a foundational discipline alongside software and computer engineering.

---

## Branch Guide

> [!NOTE]
> **You are on the `dev` branch.** Active development happens here. For the last stable release, see the [`main` branch](https://github.com/harvard-edge/cs249r_book/tree/main).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          BRANCH STRUCTURE                               │
├─────────────────────────────────────────────────────────────────────────┤
│   main (last stable release)                                            │
│   └── Single-volume textbook (published and available)                  │
│         │                                                               │
│   dev (default branch, you are here)                                    │
│   ├── 📘 Volume I: Introduction to Machine Learning Systems             │
│   │      Status: Content complete, undergoing editorial polish          │
│   └── 📙 Volume II: Machine Learning Systems at Scale                   │
│          Status: Active development, chapters being written             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Start Here

This repository is the open learning stack for AI systems engineering: textbook source, TinyTorch, hardware kits, and interactive co-labs.

<table>
  <tbody>
    <tr>
      <td width="15%" align="center"><b>📖 READ</b></td>
      <td>Start with the <a href="https://mlsysbook.ai/book/">textbook</a>:
        <ul>
          <li><a href="https://mlsysbook.ai/vol1/">Volume I: Foundations</a> covers ML basics, development, and optimization.</li>
          <li><a href="https://mlsysbook.ai/vol2/">Volume II: At Scale</a> covers distributed training and production fleets.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td width="15%" align="center"><b>🎯 PRACTICE</b></td>
      <td>Master the "Silicon Realism" required for frontier AI jobs:
        <ul>
          <li><a href="interviews/README.md">Interview Hub</a>: Interactive flashcards and whiteboard math.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td width="15%" align="center"><b>👩‍🏫 INSTRUCT</b></td>
      <td>Adopt the curriculum with the <a href="instructors/index.qmd">AI Engineering Blueprint</a>.</td>
    </tr>
    <tr>
      <td width="15%" align="center"><b>🔥 BUILD</b></td>
      <td>Start TinyTorch with the <a href="https://mlsysbook.ai/tinytorch/getting-started.html">getting started guide</a>. Implement autograd and transformers from scratch.</td>
    </tr>
    <tr>
      <td width="15%" align="center"><b>📦 DEPLOY</b></td>
      <td>Pick a <a href="https://mlsysbook.ai/kits/">hardware kit</a> and run the labs on Arduino, Raspberry Pi, or Jetson.</td>
    </tr>
    <tr>
      <td width="15%" align="center"><b>🔮 SIMULATE</b></td>
      <td>Explore the <a href="mlsysim/README.md">MLSys·im Engine</a> to calculate the physics of ML infrastructure.</td>
    </tr>
  </tbody>
</table>

---

## Support This Work

We are working toward **1 million learners by 2030**. Every star, share, and contribution helps move this effort forward.

### Why GitHub Stars Matter

<div align="center">

*What gets measured gets improved.*

Each star is a learner or supporter who believes AI systems should be engineered with rigor.

[![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)

[![Star History Chart](https://api.star-history.com/svg?repos=harvard-edge/cs249r_book&type=Date)](https://star-history.com/#harvard-edge/cs249r_book&Date)

100 → 1,000 → **10,000** → 100,000 → **1M learners**

</div>

Stars are a signal that universities, foundations, and industry partners use to fund workshops and hardware kits for underserved classrooms.

### Fund the Mission

<div align="center">

All contributions go to [Open Collective](https://opencollective.com/mlsysbook), a transparent fund that supports educational outreach.

[![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

</div>

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

Dual-licensed: [CC BY-NC-ND 4.0](LICENSE.md) (Textbook) and [Apache 2.0](tinytorch/LICENSE) (TinyTorch).

---

<div align="center">

**[✉️ Subscribe](https://buttondown.email/mlsysbook) • [💬 Join discussions](https://github.com/harvard-edge/cs249r_book/discussions) • [🌐 Visit mlsysbook.ai](https://mlsysbook.ai/)**

**Made with ❤️ for AI engineers**<br>
*in the making, around the world* 🌎
</div>
