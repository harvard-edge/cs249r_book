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
  <a href="#citation--license"><img src="https://img.shields.io/badge/Cite-IEEE%202024-blue?logo=ieee" alt="Cite"></a>
  <a href="https://opencollective.com/mlsysbook"><img src="https://img.shields.io/badge/Fund%20Us-Open%20Collective-blue.svg?logo=open-collective" alt="Fund Us"></a>
</p>

<p align="center">
  <b><a href="https://mlsysbook.ai">📘 Volume I</a></b> •
  <b>📙 Volume II <i>(Summer 2026)</i></b> •
  <b><a href="https://mlsysbook.ai/tinytorch/">🔥 TinyTorch</a></b> •
  <b><a href="mlsysim/README.md">🔮 MLSys·im <i>(dev)</i></a></b> •
  <b><a href="interviews/README.md">💼 Interview Playbook <i>(dev)</i></a></b> •
  <b><a href="https://mlsysbook.org">🌐 Ecosystem</a></b>
</p>

<p align="center">📚 <b>Hardcopy edition coming 2026 with MIT Press.</b></p>

</div>

---

## Mission

<div align="center">
  <blockquote>
    <b>The world is rushing to build AI systems. It is not engineering them.</b>
  </blockquote>
</div>

That gap is what we mean by AI engineering.

**AI engineering is the discipline of building efficient, reliable, safe, and robust intelligent systems that operate in the real world, not just models in isolation.** Our mission is to establish AI engineering as a foundational discipline alongside software engineering and computer engineering, by teaching how to design, build, and evaluate end-to-end intelligent systems.

**Our goal:** Help **100,000 learners** master ML Systems this year, and reach **1 million by 2030**.

---

## Why One Repository

I designed this as a single integrated curriculum, not a collection of independent projects. The textbook teaches the theory. TinyTorch makes you build the internals. The hardware kits force you to confront real constraints. The simulator lets you reason about infrastructure you can't afford to rent. Each piece exists because I found that students who only read don't internalize, and students who only code don't generalize.

The repository is the curriculum.

A growing community of contributors helps improve every part of it: fixing errors, sharpening explanations, testing on new hardware. Their work makes this better for everyone, and I'm grateful for every pull request.

---

## The Curriculum

Every component connects. The textbook gives you the mental models. The labs let you explore trade-offs interactively, powered by MLSys·im, the modeling engine for infrastructure you can't physically access. TinyTorch makes you build the machinery yourself. The hardware kits put you face-to-face with real constraints. The interview playbook tests whether you actually understand it. And the instructor hub, slides, and newsletter give educators everything they need to bring this into a classroom.

<p align="center">
  <img src="README/curriculum-map.svg?v=2" alt="Curriculum map showing how the textbook, labs, Tiny Torch, hardware kits, MLSys im, and interview playbook connect" width="760">
</p>

### For Students

<table>
  <thead>
    <tr>
      <th width="5%"></th>
      <th width="15%">Component</th>
      <th width="50%">Role in the Curriculum</th>
      <th width="30%">Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">📖</td>
      <td><b>Textbook</b></td>
      <td>Two-volume MIT Press textbook. The theory, the mental models, and the quantitative reasoning that everything else builds on.</td>
      <td><a href="https://mlsysbook.ai">Volume I</a> · Volume II <i>(Summer 2026)</i></td>
    </tr>
    <tr>
      <td align="center">🔬</td>
      <td><b>Labs</b></td>
      <td>Interactive Marimo notebooks where you explore trade-offs from the textbook: change a parameter, see what breaks, build intuition. Powered by MLSys·im under the hood.</td>
      <td><a href="labs/README.md">Read more</a> <i>(dev)</i></td>
    </tr>
    <tr>
      <td align="center">🔥</td>
      <td><b>Tiny🔥Torch</b></td>
      <td>Build your own ML framework from scratch across 20 progressive modules. You don't understand a system until you've built one.</td>
      <td><a href="https://mlsysbook.ai/tinytorch/">Get started</a></td>
    </tr>
    <tr>
      <td align="center">🛠️</td>
      <td><b>Hardware Kits</b></td>
      <td>Deploy ML to Arduino, Raspberry Pi, and Jetson. Real memory limits, real power budgets, real latency.</td>
      <td><a href="kits/README.md">Browse labs</a> <i>(dev)</i></td>
    </tr>
    <tr>
      <td align="center">🔮</td>
      <td><b>MLSys·im</b></td>
      <td>Calculate memory bottlenecks, network saturation, and scheduling limits at infrastructure scales you can't physically access.</td>
      <td><a href="mlsysim/README.md">Read more</a> <i>(dev)</i></td>
    </tr>
    <tr>
      <td align="center">💼</td>
      <td><b>Interview Playbook</b></td>
      <td>40+ systems design questions for AI infrastructure roles. Silicon physics, distributed infra, production serving, and ML operations.</td>
      <td><a href="interviews/README.md">Start drilling</a> <i>(dev)</i></td>
    </tr>
  </tbody>
</table>

### For Educators

<table>
  <thead>
    <tr>
      <th width="5%"></th>
      <th width="15%">Component</th>
      <th width="50%">What It Provides</th>
      <th width="30%">Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">🎓</td>
      <td><b>Instructor Hub</b></td>
      <td>The AI Engineering Blueprint: two 12-week syllabi, pedagogy guide, assessment rubrics, and a TA handbook.</td>
      <td><a href="instructors/index.qmd">View hub</a></td>
    </tr>
    <tr>
      <td align="center">🎬</td>
      <td><b>Lecture Slides</b></td>
      <td>Beamer slide decks for every chapter, with four theme variants. Drop into your course and teach.</td>
      <td><a href="slides/README.md">Browse decks</a> <i>(dev)</i></td>
    </tr>
    <tr>
      <td align="center">📬</td>
      <td><b>Newsletter</b></td>
      <td>Updates on the curriculum, new chapters, and what the community is building.</td>
      <td><a href="https://buttondown.email/mlsysbook">Subscribe</a></td>
    </tr>
  </tbody>
</table>

---

## What You Will Learn

This textbook teaches you to think at the intersection of machine learning and systems engineering. Each chapter bridges algorithmic concepts with the infrastructure that makes them work in practice.

<table>
  <thead>
    <tr>
      <th width="50%">You know...</th>
      <th width="50%">You will learn...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>How to train a model</td>
      <td>How training scales across GPU clusters</td>
    </tr>
    <tr>
      <td>That quantization shrinks models</td>
      <td>How INT8 math maps to silicon</td>
    </tr>
    <tr>
      <td>What a transformer is</td>
      <td>Why KV-cache dominates memory at inference</td>
    </tr>
    <tr>
      <td>Models run on GPUs</td>
      <td>How schedulers balance latency vs throughput</td>
    </tr>
    <tr>
      <td>Edge devices have limits</td>
      <td>How to co-design models and hardware</td>
    </tr>
  </tbody>
</table>

### Book Structure

The textbook follows the Hennessy & Patterson pedagogical model across two volumes:

<table>
  <thead>
    <tr>
      <th width="20%">Volume</th>
      <th width="30%">Theme</th>
      <th width="50%">Scope</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Volume I</b></td>
      <td>Build, Optimize, Deploy</td>
      <td>Single-machine ML systems (1-8 GPUs)</td>
    </tr>
    <tr>
      <td><b>Volume II</b></td>
      <td>Scale, Distribute, Govern</td>
      <td>Distributed systems at production scale</td>
    </tr>
  </tbody>
</table>

---

## Quick Start

<table>
  <tbody>
    <tr>
      <td width="10%" align="center"><b>1</b></td>
      <td><b>Read the textbook.</b> Start with <a href="https://mlsysbook.ai">Volume I</a>. It's the foundation for everything else.</td>
    </tr>
    <tr>
      <td width="10%" align="center"><b>2</b></td>
      <td><b>Pick a hands-on path.</b> <a href="https://mlsysbook.ai/tinytorch/">Build a framework</a> (TinyTorch), <a href="labs/README.md">explore trade-offs</a> (Labs), or <a href="kits/README.md">deploy to real hardware</a> (Kits).</td>
    </tr>
    <tr>
      <td width="10%" align="center"><b>3</b></td>
      <td><b>Test yourself.</b> Drill the <a href="interviews/README.md">interview playbook</a>: 40+ systems design questions across cloud, edge, mobile, and TinyML.</td>
    </tr>
    <tr>
      <td width="10%" align="center"><b>4</b></td>
      <td><b>Teach it.</b> Adopt the curriculum with the <a href="instructors/index.qmd">AI Engineering Blueprint</a> and <a href="slides/README.md">lecture slides</a>.</td>
    </tr>
  </tbody>
</table>

---

## Branch Guide

> [!NOTE]
> **You are on the `dev` branch.** Active development happens here. For the last stable release, see the [`main` branch](https://github.com/harvard-edge/cs249r_book/tree/main).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          BRANCH STRUCTURE                               │
├─────────────────────────────────────────────────────────────────────────┤
│   main (live at mlsysbook.ai)                                           │
│   └── Single-volume textbook (what readers see today)                   │
│                                                                         │
│   dev (default branch, you are here)                                    │
│   ├── Volume I: Introduction to Machine Learning Systems                │
│   │      Status: Content complete, undergoing editorial polish          │
│   ├── Volume II: Machine Learning Systems at Scale                      │
│   │      Status: Active development, chapters being written             │
│   ├── TinyTorch, Hardware Kits, MLSys·im, Labs, Interview Playbook     │
│   │      Status: In development, not yet on the live site               │
│   └── Two-volume split replaces the single-volume edition at launch     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Support This Work

We are working toward **1 million learners by 2030**. Every star, share, and contribution helps move this effort forward.

### Why GitHub Stars Matter

<div align="center">

<i>What gets measured gets improved.</i>

Each star is a learner or supporter who believes AI systems should be engineered with rigor.

<a href="https://github.com/harvard-edge/cs249r_book/stargazers"><img src="https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold" alt="Stars"></a>

<a href="https://star-history.com/#harvard-edge/cs249r_book&Date"><img src="https://api.star-history.com/svg?repos=harvard-edge/cs249r_book&type=Date" alt="Star History Chart"></a>

100 → 1,000 → <b>10,000</b> → 100,000 → <b>1M learners</b>

</div>

Stars are a signal that universities, foundations, and industry partners use to fund workshops and hardware kits for underserved classrooms.

### Fund the Mission

<div align="center">

All contributions go to <a href="https://opencollective.com/mlsysbook">Open Collective</a>, a transparent fund that supports educational outreach.

<a href="https://opencollective.com/mlsysbook"><img src="https://img.shields.io/badge/Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge&logo=open-collective" alt="Open Collective"></a>

</div>

---

## Contributing

<table>
  <thead>
    <tr>
      <th width="40%">I want to...</th>
      <th width="60%">Go here</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Fix a typo or improve a chapter</td>
      <td><a href="book/docs/CONTRIBUTING.md">book/docs/CONTRIBUTING.md</a></td>
    </tr>
    <tr>
      <td>Add a TinyTorch module or fix a bug</td>
      <td><a href="tinytorch/CONTRIBUTING.md">tinytorch/CONTRIBUTING.md</a></td>
    </tr>
    <tr>
      <td>Improve hardware labs</td>
      <td><a href="kits/README.md">kits/README.md</a></td>
    </tr>
    <tr>
      <td>Report an issue</td>
      <td><a href="https://github.com/harvard-edge/cs249r_book/issues">GitHub Issues</a></td>
    </tr>
    <tr>
      <td>Ask a question</td>
      <td><a href="https://github.com/harvard-edge/cs249r_book/discussions">GitHub Discussions</a></td>
    </tr>
  </tbody>
</table>

---

## Contributors

Thanks goes to these wonderful people who have contributed to making this resource better for everyone!

### Textbook Contributors

<!-- BOOK-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/Mjrovai"><img src="https://avatars.githubusercontent.com/Mjrovai?v=4?s=50" width="50px;" alt="Marcelo Rovai"/><br /><sub><b>Marcelo Rovai</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/GabrielAmazonas"><img src="https://avatars.githubusercontent.com/GabrielAmazonas?v=4?s=50" width="50px;" alt="Gabriel Amazonas"/><br /><sub><b>Gabriel Amazonas</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/kai4avaya"><img src="https://avatars.githubusercontent.com/kai4avaya?v=4?s=50" width="50px;" alt="Kai Kleinbard"/><br /><sub><b>Kai Kleinbard</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/didier-durand"><img src="https://avatars.githubusercontent.com/didier-durand?v=4?s=50" width="50px;" alt="Didier Durand"/><br /><sub><b>Didier Durand</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/hzeljko"><img src="https://avatars.githubusercontent.com/hzeljko?v=4?s=50" width="50px;" alt="Zeljko Hrcek"/><br /><sub><b>Zeljko Hrcek</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/jasonjabbour"><img src="https://avatars.githubusercontent.com/jasonjabbour?v=4?s=50" width="50px;" alt="Jason Jabbour"/><br /><sub><b>Jason Jabbour</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/uchendui"><img src="https://avatars.githubusercontent.com/uchendui?v=4?s=50" width="50px;" alt="Ikechukwu Uchendu"/><br /><sub><b>Ikechukwu Uchendu</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/Naeemkh"><img src="https://avatars.githubusercontent.com/Naeemkh?v=4?s=50" width="50px;" alt="Naeem Khoshnevis"/><br /><sub><b>Naeem Khoshnevis</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/Sara-Khosravi"><img src="https://avatars.githubusercontent.com/Sara-Khosravi?v=4?s=50" width="50px;" alt="Sara Khosravi"/><br /><sub><b>Sara Khosravi</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/V0XNIHILI"><img src="https://avatars.githubusercontent.com/V0XNIHILI?v=4?s=50" width="50px;" alt="Douwe den Blanken"/><br /><sub><b>Douwe den Blanken</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/18jeffreyma"><img src="https://avatars.githubusercontent.com/18jeffreyma?v=4?s=50" width="50px;" alt="Jeffrey Ma"/><br /><sub><b>Jeffrey Ma</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/shanzehbatool"><img src="https://avatars.githubusercontent.com/shanzehbatool?v=4?s=50" width="50px;" alt="shanzehbatool"/><br /><sub><b>shanzehbatool</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/eliasab16"><img src="https://avatars.githubusercontent.com/eliasab16?v=4?s=50" width="50px;" alt="Elias"/><br /><sub><b>Elias</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/JaredP94"><img src="https://avatars.githubusercontent.com/JaredP94?v=4?s=50" width="50px;" alt="Jared Ping"/><br /><sub><b>Jared Ping</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/ishapira1"><img src="https://avatars.githubusercontent.com/ishapira1?v=4?s=50" width="50px;" alt="Itai Shapira"/><br /><sub><b>Itai Shapira</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/8863743b4f26c1a20e730fcf7ebc3bc0?d=identicon&s=100?v=4?s=50" width="50px;" alt="Maximilian Lam"/><br /><sub><b>Maximilian Lam</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/jaysonzlin"><img src="https://avatars.githubusercontent.com/jaysonzlin?v=4?s=50" width="50px;" alt="Jayson Lin"/><br /><sub><b>Jayson Lin</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/sophiacho1"><img src="https://avatars.githubusercontent.com/sophiacho1?v=4?s=50" width="50px;" alt="Sophia Cho"/><br /><sub><b>Sophia Cho</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/andreamurillomtz"><img src="https://avatars.githubusercontent.com/andreamurillomtz?v=4?s=50" width="50px;" alt="Andrea"/><br /><sub><b>Andrea</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/alxrod"><img src="https://avatars.githubusercontent.com/alxrod?v=4?s=50" width="50px;" alt="Alex Rodriguez"/><br /><sub><b>Alex Rodriguez</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/korneelf1"><img src="https://avatars.githubusercontent.com/korneelf1?v=4?s=50" width="50px;" alt="Korneel Van den Berghe"/><br /><sub><b>Korneel Van den Berghe</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/foundingnimo"><img src="https://avatars.githubusercontent.com/foundingnimo?v=4?s=50" width="50px;" alt="Nimo"/><br /><sub><b>Nimo</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/colbybanbury"><img src="https://avatars.githubusercontent.com/colbybanbury?v=4?s=50" width="50px;" alt="Colby Banbury"/><br /><sub><b>Colby Banbury</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/zishenwan"><img src="https://avatars.githubusercontent.com/zishenwan?v=4?s=50" width="50px;" alt="Zishen Wan"/><br /><sub><b>Zishen Wan</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/mmaz"><img src="https://avatars.githubusercontent.com/mmaz?v=4?s=50" width="50px;" alt="Mark Mazumder"/><br /><sub><b>Mark Mazumder</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/ma3mool"><img src="https://avatars.githubusercontent.com/ma3mool?v=4?s=50" width="50px;" alt="Abdulrahman Mahmoud"/><br /><sub><b>Abdulrahman Mahmoud</b></sub></a></td>
    </tr>
  </tbody>
</table>

<p align="center"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors">View all contributors</a></p>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- BOOK-CONTRIBUTORS-END -->

### TinyTorch Contributors

<!-- TINYTORCH-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/kai4avaya"><img src="https://avatars.githubusercontent.com/kai4avaya?v=4?s=50" width="50px;" alt="kai"/><br /><sub><b>kai</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/minhdang26403"><img src="https://avatars.githubusercontent.com/minhdang26403?v=4?s=50" width="50px;" alt="Dang Truong"/><br /><sub><b>Dang Truong</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/didier-durand"><img src="https://avatars.githubusercontent.com/didier-durand?v=4?s=50" width="50px;" alt="Didier Durand"/><br /><sub><b>Didier Durand</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/karthikdani"><img src="https://avatars.githubusercontent.com/karthikdani?v=4?s=50" width="50px;" alt="Karthik Dani"/><br /><sub><b>Karthik Dani</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/avikde"><img src="https://avatars.githubusercontent.com/avikde?v=4?s=50" width="50px;" alt="Avik De"/><br /><sub><b>Avik De</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/Takosaga"><img src="https://avatars.githubusercontent.com/Takosaga?v=4?s=50" width="50px;" alt="Takosaga"/><br /><sub><b>Takosaga</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/rnjema"><img src="https://avatars.githubusercontent.com/rnjema?v=4?s=50" width="50px;" alt="rnjema"/><br /><sub><b>rnjema</b></sub></a></td>
      <td align="center" valign="top" width="11.11%"><a href="https://github.com/joeswagson"><img src="https://avatars.githubusercontent.com/joeswagson?v=4?s=50" width="50px;" alt="joeswagson"/><br /><sub><b>joeswagson</b></sub></a></td>
    </tr>
  </tbody>
</table>

<p align="center"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors">View all contributors</a></p>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- TINYTORCH-CONTRIBUTORS-END -->

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

<b><a href="https://buttondown.email/mlsysbook">✉️ Subscribe</a> • <a href="https://github.com/harvard-edge/cs249r_book/discussions">💬 Join discussions</a> • <a href="https://mlsysbook.ai/">🌐 Visit mlsysbook.ai</a></b>

<b>Made with ❤️ for AI engineers</b><br>
<i>in the making, around the world</i> 🌎
</div>
