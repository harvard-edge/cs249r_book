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
  <b><a href="https://mlsysbook.ai/vol1/">📘 Volume I</a></b> •
  <b><a href="https://mlsysbook.ai/vol2/">📙 Volume II <i>(Summer 2026)</i></a></b> •
  <b><a href="https://mlsysbook.ai/tinytorch/">🔥 TinyTorch</a></b> •
  <b><a href="https://mlsysbook.ai/mlsysim/">🔮 MLSys·im</a></b> •
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

## The Learning Stack

This repository is the open learning stack for AI systems engineering. Read the textbook, then pick your path: build a framework from scratch, deploy to real hardware, or simulate infrastructure at scale.

```
                              ┌───────────────────────────┐
                              │      TEXTBOOK              │
                              │  Theory + Quantitative     │
                              │  Reasoning + Case Studies  │
                              └─────────────┬─────────────┘
                                            │
                  ┌─────────────────────────┼─────────────────────────┐
                  │                         │                         │
                  ▼                         ▼                         ▼
   ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
   │    TINYTORCH          │  │    HARDWARE KITS      │  │    MLSYS-IM           │
   │                       │  │                       │  │                       │
   │  Build an ML framework│  │  Deploy models to     │  │  Simulate the physics │
   │  from scratch using   │  │  Arduino, Raspberry   │  │  of ML infrastructure │
   │  only NumPy           │  │  Pi, and edge devices  │  │  from chip to fleet   │
   │                       │  │                       │  │                       │
   │  20 progressive       │  │  Face real memory,    │  │  Iron Law, roofline,  │
   │  modules              │  │  power, and latency   │  │  TCO, and carbon      │
   │                       │  │  constraints          │  │  analysis             │
   └──────────────────────┘  └──────────────────────┘  └──────────────────────┘
         BUILD                      DEPLOY                    SIMULATE
```

<table>
  <thead>
    <tr>
      <th width="20%">Component</th>
      <th width="50%">What It Is</th>
      <th width="30%">Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Textbook</b></td>
      <td>Two-volume MIT Press textbook on ML systems</td>
      <td><a href="https://mlsysbook.ai/vol1/">Volume I</a> · <a href="https://mlsysbook.ai/vol2/">Volume II</a></td>
    </tr>
    <tr>
      <td><b>TinyTorch</b></td>
      <td>Build your own ML framework from scratch (20 modules)</td>
      <td><a href="https://mlsysbook.ai/tinytorch/">Get started</a></td>
    </tr>
    <tr>
      <td><b>Hardware Kits</b></td>
      <td>Deploy ML to Arduino, Raspberry Pi, and edge devices</td>
      <td><a href="https://mlsysbook.ai/kits/">Browse labs</a></td>
    </tr>
    <tr>
      <td><b>MLSys·im</b></td>
      <td>Physics-grounded simulator for ML infrastructure</td>
      <td><a href="https://mlsysbook.ai/mlsysim/">Try it</a> · <kbd>pip install mlsysim</kbd></td>
    </tr>
    <tr>
      <td><b>Labs</b></td>
      <td>Interactive notebooks for algorithm-system exploration</td>
      <td><i>Coming Summer 2026</i></td>
    </tr>
    <tr>
      <td><b>Instructor Hub</b></td>
      <td>Turnkey curricula with two 12-week syllabi</td>
      <td><a href="instructors/index.qmd">View hub</a></td>
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

## Start Here

<table>
  <tbody>
    <tr>
      <td width="15%" align="center"><b>READ</b></td>
      <td>Start with the <a href="https://mlsysbook.ai/book/">textbook</a>:
        <ul>
          <li><a href="https://mlsysbook.ai/vol1/">Volume I: Foundations</a> covers ML basics, development, and optimization.</li>
          <li><a href="https://mlsysbook.ai/vol2/">Volume II: At Scale</a> covers distributed training and production fleets.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td width="15%" align="center"><b>BUILD</b></td>
      <td>Start TinyTorch with the <a href="https://mlsysbook.ai/tinytorch/getting-started.html">getting started guide</a>. Implement autograd and transformers from scratch.</td>
    </tr>
    <tr>
      <td width="15%" align="center"><b>DEPLOY</b></td>
      <td>Pick a <a href="https://mlsysbook.ai/kits/">hardware kit</a> and run the labs on Arduino, Raspberry Pi, or Jetson.</td>
    </tr>
    <tr>
      <td width="15%" align="center"><b>SIMULATE</b></td>
      <td>Explore the <a href="mlsysim/README.md">MLSys·im Engine</a> to calculate the physics of ML infrastructure.</td>
    </tr>
    <tr>
      <td width="15%" align="center"><b>PRACTICE</b></td>
      <td>Prepare for AI systems design interviews with the <a href="interviews/README.md">Interview Hub</a>.</td>
    </tr>
    <tr>
      <td width="15%" align="center"><b>INSTRUCT</b></td>
      <td>Adopt the curriculum with the <a href="instructors/index.qmd">AI Engineering Blueprint</a> (two 12-week syllabi).</td>
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
│   main (last stable release)                                            │
│   └── Single-volume textbook (published and available)                  │
│         │                                                               │
│   dev (default branch, you are here)                                    │
│   ├── Volume I: Introduction to Machine Learning Systems                │
│   │      Status: Content complete, undergoing editorial polish          │
│   └── Volume II: Machine Learning Systems at Scale                      │
│          Status: Active development, chapters being written             │
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
