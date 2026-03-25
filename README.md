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
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/labs-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/labs-validate-dev.yml?branch=dev&label=Labs&logo=jupyter&cacheSeconds=300" alt="Labs"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/kits-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/kits-validate-dev.yml?branch=dev&label=Kits&logo=arduino&cacheSeconds=300" alt="Kits"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/mlsysim-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/mlsysim-validate-dev.yml?branch=dev&label=MLSys%C2%B7im&logo=python&cacheSeconds=300" alt="MLSys·im"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/staffml-preview-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/staffml-preview-dev.yml?branch=dev&label=StaffML&logo=next.js&cacheSeconds=300" alt="StaffML"></a></br>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/slides-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/slides-validate-dev.yml?branch=dev&label=Slides&logo=googleslides&cacheSeconds=300" alt="Slides"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/instructors-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/instructors-validate-dev.yml?branch=dev&label=Instructors&logo=googleclassroom&cacheSeconds=300" alt="Instructors"></a>
  <img src="https://img.shields.io/github/last-commit/harvard-edge/cs249r_book/dev?label=Updated&logo=git&cacheSeconds=300" alt="Updated">
</p>

<!-- Meta -->
<p align="center">
  <a href="https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE.md"><img src="https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-blue.svg" alt="License"></a>
  <a href="#citation--license"><img src="https://img.shields.io/badge/Cite-IEEE%202024-blue?logo=ieee" alt="Cite"></a>
  <a href="https://opencollective.com/mlsysbook"><img src="https://img.shields.io/badge/Fund%20Us-Open%20Collective-blue.svg?logo=open-collective" alt="Fund Us"></a>
</p>

<p align="center">
  <b><a href="https://mlsysbook.ai">📘 Textbook (current edition)</a></b> •
  <b>📙 Vol I + Vol II <i>(Summer 2026)</i></b> •
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

I designed this as a single integrated curriculum, not a collection of independent projects. The textbook teaches the theory. TinyTorch makes you *build* the internals. The hardware kits force you to confront *real* constraints. The simulator lets you reason about infrastructure you can't afford to rent. Each piece exists because I found that students who only read don't internalize, and students who only code don't generalize.

<div align="center">
  <blockquote>
    <b>The repository is the curriculum.</b>
  </blockquote>
</div>

A growing community of contributors helps improve every part of it: fixing errors, sharpening explanations, testing on new hardware. Their work makes this better for everyone, and I'm grateful for every pull request.

---

## The Curriculum

Every component connects. The textbook gives you the mental models. The labs let you reason through trade-offs interactively, powered by MLSys·im — a modeling engine for infrastructure you can't physically access, and a standalone tool in its own right. TinyTorch makes you build the machinery yourself. The hardware kits put you face-to-face with real deployment constraints. The interview playbook tests whether you actually understand it. And the instructor hub, slides, and newsletter give educators everything they need to bring this into a classroom.

<p align="center">
  <img src="README/curriculum-map.svg?v=3" alt="Curriculum map showing how the textbook, labs, Tiny Torch, hardware kits, MLSys im, and interview playbook connect" width="760">
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
      <td><a href="https://mlsysbook.ai">Current edition</a> · Vol I + II <i>(Summer 2026)</i></td>
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
      <td><a href="https://mlsysbook.ai/kits">Browse labs</a></td>
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
      <td>1,000+ systems design questions for AI infrastructure roles. Silicon physics, distributed infra, production serving, and ML operations.</td>
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
      <th width="45%">You know...</th>
      <th width="10%" align="center"></th>
      <th width="45%">You will learn...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>How to train a model</td>
      <td align="center">→</td>
      <td><b>How training scales across GPU clusters</b></td>
    </tr>
    <tr>
      <td>That quantization shrinks models</td>
      <td align="center">→</td>
      <td><b>How INT8 math maps to silicon</b></td>
    </tr>
    <tr>
      <td>What a transformer is</td>
      <td align="center">→</td>
      <td><b>Why KV-cache dominates memory at inference</b></td>
    </tr>
    <tr>
      <td>Models run on GPUs</td>
      <td align="center">→</td>
      <td><b>How schedulers balance latency vs throughput</b></td>
    </tr>
    <tr>
      <td>Edge devices have limits</td>
      <td align="center">→</td>
      <td><b>How to co-design models and hardware</b></td>
    </tr>
  </tbody>
</table>

### Book Structure

The textbook follows the Hennessy & Patterson pedagogical model across two volumes:

<table>
  <thead>
    <tr>
      <th width="5%"></th>
      <th width="15%">Volume</th>
      <th width="25%">Theme</th>
      <th width="55%">Scope</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">📗</td>
      <td><b>Volume I</b></td>
      <td>Build, Optimize, Deploy</td>
      <td>Single-machine ML systems (1–8 GPUs). Foundations, optimization, and deployment on one node.</td>
    </tr>
    <tr>
      <td align="center">📘</td>
      <td><b>Volume II</b></td>
      <td>Scale, Distribute, Govern</td>
      <td>Distributed systems at production scale. Multi-machine infrastructure, fault tolerance, and governance.</td>
    </tr>
  </tbody>
</table>

---

## Quick Start

<table>
  <tbody>
    <tr>
      <td width="7%" align="center"><h3>①</h3></td>
      <td width="93%"><b>Read the textbook.</b> Start with the <a href="https://mlsysbook.ai">current edition</a>. It's the foundation for everything else.</td>
    </tr>
    <tr>
      <td align="center"><h3>②</h3></td>
      <td><b>Pick a hands-on path.</b> <a href="https://mlsysbook.ai/tinytorch/">Build a framework</a> (TinyTorch), <a href="labs/README.md">explore trade-offs</a> (Labs), or <a href="https://mlsysbook.ai/kits">deploy to real hardware</a> (Kits).</td>
    </tr>
    <tr>
      <td align="center"><h3>③</h3></td>
      <td><b>Test yourself.</b> Drill the <a href="interviews/README.md">interview playbook</a>: 1,000+ systems design questions across cloud, edge, mobile, and TinyML.</td>
    </tr>
    <tr>
      <td align="center"><h3>④</h3></td>
      <td><b>Teach it.</b> Adopt the curriculum with the <a href="instructors/index.qmd">AI Engineering Blueprint</a> and <a href="slides/README.md">lecture slides</a>.</td>
    </tr>
  </tbody>
</table>

---

## Branch Guide

> [!NOTE]
> **You are on the `dev` branch.** Active development happens here. For the last stable release, see the [`main` branch](https://github.com/harvard-edge/cs249r_book/tree/main).

<table>
  <thead>
    <tr>
      <th width="5%"></th>
      <th width="15%">Branch</th>
      <th width="45%">What's on it</th>
      <th width="35%">Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">🟢</td>
      <td><b><code>main</code></b><br><a href="https://mlsysbook.ai">mlsysbook.ai</a></td>
      <td>Single-volume textbook (current edition)</td>
      <td>Live — this is what readers see today.</td>
    </tr>
    <tr>
      <td align="center">🟡</td>
      <td><b><code>dev</code></b><br><i>← you are here</i></td>
      <td>
        <b>Volume I</b> — two-volume split (content complete, editorial polish)<br>
        <b>Volume II</b> — At Scale (active development)<br>
        <b>Curriculum</b> — TinyTorch, Kits, MLSys·im, Labs, Interview Playbook
      </td>
      <td>
        TinyTorch and Hardware Kits are live.<br>
        MLSys·im, Labs, and Interview Playbook are in development.
      </td>
    </tr>
  </tbody>
</table>

<p align="center"><i>The two-volume split replaces the single-volume edition at launch.</i></p>

---

## Support This Work

<div align="center">

<a href="https://github.com/harvard-edge/cs249r_book/stargazers"><img src="https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold" alt="Stars"></a>
&nbsp;&nbsp;
<a href="https://opencollective.com/mlsysbook"><img src="https://img.shields.io/badge/Fund-Open%20Collective-blue.svg?style=for-the-badge&logo=open-collective" alt="Open Collective"></a>

</div>

<table>
  <tbody>
    <tr>
      <td width="50%" align="center">
        <b>Star the repo</b><br>
        Stars signal to universities and foundations that this work matters. They directly fund workshops and hardware kits for underserved classrooms.<br><br>
        <a href="https://star-history.com/#harvard-edge/cs249r_book&Date"><img src="https://api.star-history.com/svg?repos=harvard-edge/cs249r_book&type=Date" alt="Star History Chart" width="400"></a><br>
        100 → 1,000 → <b>10,000</b> → 100,000 → <b>1M learners by 2030</b>
      </td>
      <td width="50%" align="center">
        <b>Fund the mission</b><br>
        All contributions go to <a href="https://opencollective.com/mlsysbook">Open Collective</a>, a transparent fund for educational outreach. Every dollar goes to reaching more students.<br><br>
        <a href="https://opencollective.com/mlsysbook"><img src="https://opencollective.com/mlsysbook/tiers/badge.svg" alt="Open Collective"></a>
      </td>
    </tr>
  </tbody>
</table>

---

## Contributing

<table>
  <thead>
    <tr>
      <th width="5%"></th>
      <th width="40%">I want to...</th>
      <th width="55%">Go here</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">📖</td>
      <td><b>Fix a typo or improve a chapter</b></td>
      <td><a href="book/docs/CONTRIBUTING.md">Textbook contributing guide</a></td>
    </tr>
    <tr>
      <td align="center">🔥</td>
      <td><b>Add a TinyTorch module or fix a bug</b></td>
      <td><a href="tinytorch/CONTRIBUTING.md">TinyTorch contributing guide</a></td>
    </tr>
    <tr>
      <td align="center">🛠️</td>
      <td><b>Improve hardware labs</b></td>
      <td><a href="kits/README.md">Hardware kits guide</a></td>
    </tr>
    <tr>
      <td align="center">🐛</td>
      <td><b>Report an issue</b></td>
      <td><a href="https://github.com/harvard-edge/cs249r_book/issues">GitHub Issues</a></td>
    </tr>
    <tr>
      <td align="center">💬</td>
      <td><b>Ask a question</b></td>
      <td><a href="https://github.com/harvard-edge/cs249r_book/discussions">GitHub Discussions</a></td>
    </tr>
  </tbody>
</table>

---

## Contributors

Thanks goes to these wonderful people who have contributed to making this resource better for everyone!

**Legend:** 🪲 Bug Hunter · 🧑‍💻 Code Contributor · ✍️ Doc Wizard · 🎨 Design Artist · 🧠 Idea Spark · 🔎 Code Reviewer · 🧪 Test Tinkerer · 🛠️ Tool Builder

### 📖 Textbook Contributors

<!-- BOOK-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧠 🔎 🧪 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Mjrovai"><img src="https://avatars.githubusercontent.com/Mjrovai?v=4?s=50" width="50px;" alt="Marcelo Rovai"/><br /><sub><b>Marcelo Rovai</b></sub></a><br />🧑‍💻 🎨 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/oamazonasgabriel"><img src="https://avatars.githubusercontent.com/oamazonasgabriel?v=4?s=50" width="50px;" alt="Gabriel Amazonas"/><br /><sub><b>Gabriel Amazonas</b></sub></a><br />🪲 ✍️ 🧠</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/hzeljko"><img src="https://avatars.githubusercontent.com/hzeljko?v=4?s=50" width="50px;" alt="Zeljko Hrcek"/><br /><sub><b>Zeljko Hrcek</b></sub></a><br />🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/kai4avaya"><img src="https://avatars.githubusercontent.com/kai4avaya?v=4?s=50" width="50px;" alt="Kai Kleinbard"/><br /><sub><b>Kai Kleinbard</b></sub></a><br />🧑‍💻 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/didier-durand"><img src="https://avatars.githubusercontent.com/didier-durand?v=4?s=50" width="50px;" alt="Didier Durand"/><br /><sub><b>Didier Durand</b></sub></a><br />✍️ 🪲</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jasonjabbour"><img src="https://avatars.githubusercontent.com/jasonjabbour?v=4?s=50" width="50px;" alt="Jason Jabbour"/><br /><sub><b>Jason Jabbour</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/uchendui"><img src="https://avatars.githubusercontent.com/uchendui?v=4?s=50" width="50px;" alt="Ikechukwu Uchendu"/><br /><sub><b>Ikechukwu Uchendu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Naeemkh"><img src="https://avatars.githubusercontent.com/Naeemkh?v=4?s=50" width="50px;" alt="Naeem Khoshnevis"/><br /><sub><b>Naeem Khoshnevis</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Sara-Khosravi"><img src="https://avatars.githubusercontent.com/Sara-Khosravi?v=4?s=50" width="50px;" alt="Sara Khosravi"/><br /><sub><b>Sara Khosravi</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/V0XNIHILI"><img src="https://avatars.githubusercontent.com/V0XNIHILI?v=4?s=50" width="50px;" alt="Douwe den Blanken"/><br /><sub><b>Douwe den Blanken</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/18jeffreyma"><img src="https://avatars.githubusercontent.com/18jeffreyma?v=4?s=50" width="50px;" alt="Jeffrey Ma"/><br /><sub><b>Jeffrey Ma</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/shanzehbatool"><img src="https://avatars.githubusercontent.com/shanzehbatool?v=4?s=50" width="50px;" alt="shanzehbatool"/><br /><sub><b>shanzehbatool</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/eliasab16"><img src="https://avatars.githubusercontent.com/eliasab16?v=4?s=50" width="50px;" alt="Elias"/><br /><sub><b>Elias</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/JaredP94"><img src="https://avatars.githubusercontent.com/JaredP94?v=4?s=50" width="50px;" alt="Jared Ping"/><br /><sub><b>Jared Ping</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/ishapira1"><img src="https://avatars.githubusercontent.com/ishapira1?v=4?s=50" width="50px;" alt="Itai Shapira"/><br /><sub><b>Itai Shapira</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/8863743b4f26c1a20e730fcf7ebc3bc0?d=identicon&s=100?v=4?s=50" width="50px;" alt="Maximilian Lam"/><br /><sub><b>Maximilian Lam</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jaysonzlin"><img src="https://avatars.githubusercontent.com/jaysonzlin?v=4?s=50" width="50px;" alt="Jayson Lin"/><br /><sub><b>Jayson Lin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/sophiacho1"><img src="https://avatars.githubusercontent.com/sophiacho1?v=4?s=50" width="50px;" alt="Sophia Cho"/><br /><sub><b>Sophia Cho</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/andreamurillomtz"><img src="https://avatars.githubusercontent.com/andreamurillomtz?v=4?s=50" width="50px;" alt="Andrea"/><br /><sub><b>Andrea</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/alxrod"><img src="https://avatars.githubusercontent.com/alxrod?v=4?s=50" width="50px;" alt="Alex Rodriguez"/><br /><sub><b>Alex Rodriguez</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/korneelf1"><img src="https://avatars.githubusercontent.com/korneelf1?v=4?s=50" width="50px;" alt="Korneel Van den Berghe"/><br /><sub><b>Korneel Van den Berghe</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/foundingnimo"><img src="https://avatars.githubusercontent.com/foundingnimo?v=4?s=50" width="50px;" alt="Nimo"/><br /><sub><b>Nimo</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/colbybanbury"><img src="https://avatars.githubusercontent.com/colbybanbury?v=4?s=50" width="50px;" alt="Colby Banbury"/><br /><sub><b>Colby Banbury</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/zishenwan"><img src="https://avatars.githubusercontent.com/zishenwan?v=4?s=50" width="50px;" alt="Zishen Wan"/><br /><sub><b>Zishen Wan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/mmaz"><img src="https://avatars.githubusercontent.com/mmaz?v=4?s=50" width="50px;" alt="Mark Mazumder"/><br /><sub><b>Mark Mazumder</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/ma3mool"><img src="https://avatars.githubusercontent.com/ma3mool?v=4?s=50" width="50px;" alt="Abdulrahman Mahmoud"/><br /><sub><b>Abdulrahman Mahmoud</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/DivyaAmirtharaj"><img src="https://avatars.githubusercontent.com/DivyaAmirtharaj?v=4?s=50" width="50px;" alt="Divya Amirtharaj"/><br /><sub><b>Divya Amirtharaj</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/srivatsankrishnan"><img src="https://avatars.githubusercontent.com/srivatsankrishnan?v=4?s=50" width="50px;" alt="Srivatsan Krishnan"/><br /><sub><b>Srivatsan Krishnan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/arnaumarin"><img src="https://avatars.githubusercontent.com/arnaumarin?v=4?s=50" width="50px;" alt="marin-llobet"/><br /><sub><b>marin-llobet</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/aptl26"><img src="https://avatars.githubusercontent.com/aptl26?v=4?s=50" width="50px;" alt="Aghyad Deeb"/><br /><sub><b>Aghyad Deeb</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/James-QiuHaoran"><img src="https://avatars.githubusercontent.com/James-QiuHaoran?v=4?s=50" width="50px;" alt="Haoran Qiu"/><br /><sub><b>Haoran Qiu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Ekhao"><img src="https://avatars.githubusercontent.com/Ekhao?v=4?s=50" width="50px;" alt="Emil Njor"/><br /><sub><b>Emil Njor</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/ELSuitorHarvard"><img src="https://avatars.githubusercontent.com/ELSuitorHarvard?v=4?s=50" width="50px;" alt="ELSuitorHarvard"/><br /><sub><b>ELSuitorHarvard</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/kaiM0ves"><img src="https://avatars.githubusercontent.com/kaiM0ves?v=4?s=50" width="50px;" alt="kaiM0ves"/><br /><sub><b>kaiM0ves</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/oishib"><img src="https://avatars.githubusercontent.com/oishib?v=4?s=50" width="50px;" alt="oishib"/><br /><sub><b>oishib</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jared-ni"><img src="https://avatars.githubusercontent.com/jared-ni?v=4?s=50" width="50px;" alt="Jared Ni"/><br /><sub><b>Jared Ni</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/AditiR-42"><img src="https://avatars.githubusercontent.com/AditiR-42?v=4?s=50" width="50px;" alt="Aditi Raju"/><br /><sub><b>Aditi Raju</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/MichaelSchnebly"><img src="https://avatars.githubusercontent.com/MichaelSchnebly?v=4?s=50" width="50px;" alt="Michael Schnebly"/><br /><sub><b>Michael Schnebly</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/VThuong99"><img src="https://avatars.githubusercontent.com/VThuong99?v=4?s=50" width="50px;" alt="Thuong Duong"/><br /><sub><b>Thuong Duong</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/leo47007"><img src="https://avatars.githubusercontent.com/leo47007?v=4?s=50" width="50px;" alt="Yu-Shun Hsiao"/><br /><sub><b>Yu-Shun Hsiao</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/BaeHenryS"><img src="https://avatars.githubusercontent.com/BaeHenryS?v=4?s=50" width="50px;" alt="Henry Bae"/><br /><sub><b>Henry Bae</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/eimlav"><img src="https://avatars.githubusercontent.com/eimlav?v=4?s=50" width="50px;" alt="Eimhin Laverty"/><br /><sub><b>Eimhin Laverty</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jaywonchung"><img src="https://avatars.githubusercontent.com/jaywonchung?v=4?s=50" width="50px;" alt="Jae-Won Chung"/><br /><sub><b>Jae-Won Chung</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/ShvetankPrakash"><img src="https://avatars.githubusercontent.com/ShvetankPrakash?v=4?s=50" width="50px;" alt="Shvetank Prakash"/><br /><sub><b>Shvetank Prakash</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/marcozennaro"><img src="https://avatars.githubusercontent.com/marcozennaro?v=4?s=50" width="50px;" alt="Marco Zennaro"/><br /><sub><b>Marco Zennaro</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/aryatschand"><img src="https://avatars.githubusercontent.com/aryatschand?v=4?s=50" width="50px;" alt="Arya Tschand"/><br /><sub><b>Arya Tschand</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/arbass22"><img src="https://avatars.githubusercontent.com/arbass22?v=4?s=50" width="50px;" alt="Andrew Bass"/><br /><sub><b>Andrew Bass</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/pongtr"><img src="https://avatars.githubusercontent.com/pongtr?v=4?s=50" width="50px;" alt="Pong Trairatvorakul"/><br /><sub><b>Pong Trairatvorakul</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/euranofshin"><img src="https://avatars.githubusercontent.com/euranofshin?v=4?s=50" width="50px;" alt="Eura Nofshin"/><br /><sub><b>Eura Nofshin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/0c931fcfd03cd548d44c90602dd773ba?d=identicon&s=100?v=4?s=50" width="50px;" alt="Matthew Stewart"/><br /><sub><b>Matthew Stewart</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/af39c27c6090c50a1921a9b6366e81cc?d=identicon&s=100?v=4?s=50" width="50px;" alt="Emeka Ezike"/><br /><sub><b>Emeka Ezike</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jianqingdu"><img src="https://avatars.githubusercontent.com/jianqingdu?v=4?s=50" width="50px;" alt="jianqingdu"/><br /><sub><b>jianqingdu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jzhou1318"><img src="https://avatars.githubusercontent.com/jzhou1318?v=4?s=50" width="50px;" alt="Jennifer Zhou"/><br /><sub><b>Jennifer Zhou</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/vitasam"><img src="https://avatars.githubusercontent.com/vitasam?v=4?s=50" width="50px;" alt="The Random DIY"/><br /><sub><b>The Random DIY</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/468ef35acc69f3266efd700992daa369?d=identicon&s=100?v=4?s=50" width="50px;" alt="Fatima Shah"/><br /><sub><b>Fatima Shah</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/BrunoScaglione"><img src="https://avatars.githubusercontent.com/BrunoScaglione?v=4?s=50" width="50px;" alt="Bruno Scaglione"/><br /><sub><b>Bruno Scaglione</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Allen-Kuang"><img src="https://avatars.githubusercontent.com/Allen-Kuang?v=4?s=50" width="50px;" alt="Allen-Kuang"/><br /><sub><b>Allen-Kuang</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Tess314"><img src="https://www.gravatar.com/avatar/4ad8cdf19eb3b666ace97d3eedb19278?d=identicon&s=100?v=4?s=50" width="50px;" alt="Tess314"/><br /><sub><b>Tess314</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/taunoe"><img src="https://avatars.githubusercontent.com/taunoe?v=4?s=50" width="50px;" alt="Tauno Erik"/><br /><sub><b>Tauno Erik</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/gnodipac886"><img src="https://avatars.githubusercontent.com/gnodipac886?v=4?s=50" width="50px;" alt="gnodipac886"/><br /><sub><b>gnodipac886</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/serco425"><img src="https://avatars.githubusercontent.com/serco425?v=4?s=50" width="50px;" alt="Sercan Aygün"/><br /><sub><b>Sercan Aygün</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/TheHiddenLayer"><img src="https://avatars.githubusercontent.com/TheHiddenLayer?v=4?s=50" width="50px;" alt="TheHiddenLayer"/><br /><sub><b>TheHiddenLayer</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Gjain234"><img src="https://avatars.githubusercontent.com/Gjain234?v=4?s=50" width="50px;" alt="Gauri Jain"/><br /><sub><b>Gauri Jain</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/FinAminToastCrunch"><img src="https://avatars.githubusercontent.com/FinAminToastCrunch?v=4?s=50" width="50px;" alt="Fin Amin"/><br /><sub><b>Fin Amin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/alex-oesterling"><img src="https://avatars.githubusercontent.com/alex-oesterling?v=4?s=50" width="50px;" alt="Alex Oesterling"/><br /><sub><b>Alex Oesterling</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/AbenezerKb"><img src="https://avatars.githubusercontent.com/AbenezerKb?v=4?s=50" width="50px;" alt="Abenezer Angamo"/><br /><sub><b>Abenezer Angamo</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/BravoBaldo"><img src="https://avatars.githubusercontent.com/BravoBaldo?v=4?s=50" width="50px;" alt="Baldassarre Cesarano"/><br /><sub><b>Baldassarre Cesarano</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Jahnic-kb"><img src="https://avatars.githubusercontent.com/Jahnic-kb?v=4?s=50" width="50px;" alt="Jahnic Beck"/><br /><sub><b>Jahnic Beck</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/aethernavshulkraven-allain"><img src="https://avatars.githubusercontent.com/aethernavshulkraven-allain?v=4?s=50" width="50px;" alt="अरनव शुक्ला | Arnav Shukla"/><br /><sub><b>अरनव शुक्ला | Arnav Shukla</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/RinZ27"><img src="https://avatars.githubusercontent.com/RinZ27?v=4?s=50" width="50px;" alt="Rin"/><br /><sub><b>Rin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/bilgeacun"><img src="https://avatars.githubusercontent.com/bilgeacun?v=4?s=50" width="50px;" alt="Bilge Acun"/><br /><sub><b>Bilge Acun</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/atcheng2"><img src="https://avatars.githubusercontent.com/atcheng2?v=4?s=50" width="50px;" alt="Andy Cheng"/><br /><sub><b>Andy Cheng</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/arighosh05"><img src="https://avatars.githubusercontent.com/arighosh05?v=4?s=50" width="50px;" alt="Aritra Ghosh"/><br /><sub><b>Aritra Ghosh</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/abigailswallow"><img src="https://avatars.githubusercontent.com/abigailswallow?v=4?s=50" width="50px;" alt="abigailswallow"/><br /><sub><b>abigailswallow</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/YangZhou1997"><img src="https://avatars.githubusercontent.com/YangZhou1997?v=4?s=50" width="50px;" alt="Yang Zhou"/><br /><sub><b>Yang Zhou</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/XaicuL"><img src="https://avatars.githubusercontent.com/XaicuL?v=4?s=50" width="50px;" alt="JEON HYUNJUN(Luciano)"/><br /><sub><b>JEON HYUNJUN(Luciano)</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/emmanuel2406"><img src="https://avatars.githubusercontent.com/emmanuel2406?v=4?s=50" width="50px;" alt="Emmanuel Rassou"/><br /><sub><b>Emmanuel Rassou</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jasonlyik"><img src="https://avatars.githubusercontent.com/jasonlyik?v=4?s=50" width="50px;" alt="Jason Yik"/><br /><sub><b>Jason Yik</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jessicaquaye"><img src="https://avatars.githubusercontent.com/jessicaquaye?v=4?s=50" width="50px;" alt="Jessica Quaye"/><br /><sub><b>Jessica Quaye</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/cursoragent"><img src="https://avatars.githubusercontent.com/cursoragent?v=4?s=50" width="50px;" alt="Cursor Agent"/><br /><sub><b>Cursor Agent</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/happyappledog"><img src="https://avatars.githubusercontent.com/happyappledog?v=4?s=50" width="50px;" alt="happyappledog"/><br /><sub><b>happyappledog</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/snuggs"><img src="https://avatars.githubusercontent.com/snuggs?v=4?s=50" width="50px;" alt="Snuggs"/><br /><sub><b>Snuggs</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/swilcock0"><img src="https://avatars.githubusercontent.com/swilcock0?v=4?s=50" width="50px;" alt="Sam Wilcock"/><br /><sub><b>Sam Wilcock</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/sjohri20"><img src="https://avatars.githubusercontent.com/sjohri20?v=4?s=50" width="50px;" alt="Shreya Johri"/><br /><sub><b>Shreya Johri</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/skmur"><img src="https://avatars.githubusercontent.com/skmur?v=4?s=50" width="50px;" alt="Sonia Murthy"/><br /><sub><b>Sonia Murthy</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/fc4f3460cdfb9365ab59bdeafb06413e?d=identicon&s=100?v=4?s=50" width="50px;" alt="Costin-Andrei Oncescu"/><br /><sub><b>Costin-Andrei Oncescu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/0d6b8616427d8b19d425c9808692e347?d=identicon&s=100?v=4?s=50" width="50px;" alt="formlsysbookissue"/><br /><sub><b>formlsysbookissue</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/7cd8d5dfd83071f23979019d97655dc5?d=identicon&s=100?v=4?s=50" width="50px;" alt="Annie Laurie Cook"/><br /><sub><b>Annie Laurie Cook</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/5aa037840c0ca11ee42784ed4843c655?d=identicon&s=100?v=4?s=50" width="50px;" alt="Parampreet Singh"/><br /><sub><b>Parampreet Singh</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/b15b6e0e9adf58099905c1a0fd474cb9?d=identicon&s=100?v=4?s=50" width="50px;" alt="Vijay Edupuganti"/><br /><sub><b>Vijay Edupuganti</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/f88052cca4f401d9b0f43aed0a53434a?d=identicon&s=100?v=4?s=50" width="50px;" alt="Jothi Ramaswamy"/><br /><sub><b>Jothi Ramaswamy</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/35a8d9ffd03f05e79a2c6ce6206a56f2?d=identicon&s=100?v=4?s=50" width="50px;" alt="Batur Arslan"/><br /><sub><b>Batur Arslan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/bd53d146aa888548c8db4da02bf81e7a?d=identicon&s=100?v=4?s=50" width="50px;" alt="Curren Iyer"/><br /><sub><b>Curren Iyer</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/8d8410338458e08bd5e4b96f58e1c217?d=identicon&s=100?v=4?s=50" width="50px;" alt="Edward Jin"/><br /><sub><b>Edward Jin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/28c6123d2c9f75578d3ccdedb0df3d11?d=identicon&s=100?v=4?s=50" width="50px;" alt="Tess Watt"/><br /><sub><b>Tess Watt</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/ef139181fe00190f21730f6912532e9e?d=identicon&s=100?v=4?s=50" width="50px;" alt="bluebaer7"/><br /><sub><b>bluebaer7</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/f5d58ba6aa9b00189d4c018d370e8f43?d=identicon&s=100?v=4?s=50" width="50px;" alt="yanjingl"/><br /><sub><b>yanjingl</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/a5a47df988ab1720dd706062e523ca32?d=identicon&s=100?v=4?s=50" width="50px;" alt="a-saraf"/><br /><sub><b>a-saraf</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/c2dc311aa8122d5f5f061e1db14682b1?d=identicon&s=100?v=4?s=50" width="50px;" alt="songhan"/><br /><sub><b>songhan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/4814aad67982ab07a69006a1ce9d2a72?d=identicon&s=100?v=4?s=50" width="50px;" alt="jvijay"/><br /><sub><b>jvijay</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/43b1feff77c8a95fd581774fb8ec891f?d=identicon&s=100?v=4?s=50" width="50px;" alt="Zishen"/><br /><sub><b>Zishen</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/BunningsWarehouseOfficial"><img src="https://avatars.githubusercontent.com/u/49220945?v=4?v=4?s=50" width="50px;" alt="Kristian Radoš"/><br /><sub><b>Kristian Radoš</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/minhdang26403"><img src="https://avatars.githubusercontent.com/u/86156224?v=4?v=4?s=50" width="50px;" alt="Dang Truong"/><br /><sub><b>Dang Truong</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/pipme"><img src="https://avatars.githubusercontent.com/pipme?v=4?s=50" width="50px;" alt="pipme"/><br /><sub><b>pipme</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/salmanmkc"><img src="https://avatars.githubusercontent.com/u/32169182?v=4?v=4?s=50" width="50px;" alt="Salman Chishti"/><br /><sub><b>Salman Chishti</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/paolo-estavillo"><img src="https://avatars.githubusercontent.com/u/95209078?v=4?v=4?s=50" width="50px;" alt="Paolo Estavillo"/><br /><sub><b>Paolo Estavillo</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/GronuJ"><img src="https://avatars.githubusercontent.com/u/152858896?v=4?v=4?s=50" width="50px;" alt="GronuJ"/><br /><sub><b>GronuJ</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Pratham-ja"><img src="https://avatars.githubusercontent.com/u/114498234?v=4?v=4?s=50" width="50px;" alt="Pratham Chaudhary"/><br /><sub><b>Pratham Chaudhary</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/octo-patch"><img src="https://avatars.githubusercontent.com/u/266937838?v=4?v=4?s=50" width="50px;" alt="Octopus"/><br /><sub><b>Octopus</b></sub></a><br />✍️</td>
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
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧠 🔎 🧪 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/kai4avaya"><img src="https://avatars.githubusercontent.com/kai4avaya?v=4?s=50" width="50px;" alt="kai"/><br /><sub><b>kai</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/minhdang26403"><img src="https://avatars.githubusercontent.com/minhdang26403?v=4?s=50" width="50px;" alt="Dang Truong"/><br /><sub><b>Dang Truong</b></sub></a><br />🪲 🧑‍💻 ✍️ 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/didier-durand"><img src="https://avatars.githubusercontent.com/didier-durand?v=4?s=50" width="50px;" alt="Didier Durand"/><br /><sub><b>Didier Durand</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Pratham-ja"><img src="https://avatars.githubusercontent.com/u/114498234?v=4?v=4?s=50" width="50px;" alt="Pratham Chaudhary"/><br /><sub><b>Pratham Chaudhary</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/karthikdani"><img src="https://avatars.githubusercontent.com/karthikdani?v=4?s=50" width="50px;" alt="Karthik Dani"/><br /><sub><b>Karthik Dani</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/avikde"><img src="https://avatars.githubusercontent.com/avikde?v=4?s=50" width="50px;" alt="Avik De"/><br /><sub><b>Avik De</b></sub></a><br />🪲 🧪</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Takosaga"><img src="https://avatars.githubusercontent.com/Takosaga?v=4?s=50" width="50px;" alt="Takosaga"/><br /><sub><b>Takosaga</b></sub></a><br />🪲 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/rnjema"><img src="https://avatars.githubusercontent.com/rnjema?v=4?s=50" width="50px;" alt="rnjema"/><br /><sub><b>rnjema</b></sub></a><br />🧑‍💻 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/joeswagson"><img src="https://avatars.githubusercontent.com/joeswagson?v=4?s=50" width="50px;" alt="joeswagson"/><br /><sub><b>joeswagson</b></sub></a><br />🧑‍💻 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/AndreaMattiaGaravagno"><img src="https://avatars.githubusercontent.com/u/22458187?v=4?v=4?s=50" width="50px;" alt="AndreaMattiaGaravagno"/><br /><sub><b>AndreaMattiaGaravagno</b></sub></a><br />🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Roldao-Neto"><img src="https://avatars.githubusercontent.com/u/148023227?v=4?v=4?s=50" width="50px;" alt="Rolds"/><br /><sub><b>Rolds</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/asgalon"><img src="https://avatars.githubusercontent.com/u/45242704?v=4?v=4?s=50" width="50px;" alt="asgalon"/><br /><sub><b>asgalon</b></sub></a><br />🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/AmirAlasady"><img src="https://avatars.githubusercontent.com/AmirAlasady?v=4?s=50" width="50px;" alt="Amir Alasady"/><br /><sub><b>Amir Alasady</b></sub></a><br />🪲</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jettythek"><img src="https://avatars.githubusercontent.com/jettythek?v=4?s=50" width="50px;" alt="jettythek"/><br /><sub><b>jettythek</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/wz1114841863"><img src="https://avatars.githubusercontent.com/wz1114841863?v=4?s=50" width="50px;" alt="wzz"/><br /><sub><b>wzz</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/ngbolin"><img src="https://avatars.githubusercontent.com/u/9389997?v=4?v=4?s=50" width="50px;" alt="Ng Bo Lin"/><br /><sub><b>Ng Bo Lin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/keo-dara"><img src="https://avatars.githubusercontent.com/u/175544368?v=4?v=4?s=50" width="50px;" alt="keo-dara"/><br /><sub><b>keo-dara</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Kobra299"><img src="https://avatars.githubusercontent.com/u/4283156?v=4?v=4?s=50" width="50px;" alt="Wayne Norman"/><br /><sub><b>Wayne Norman</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/lalalostcode"><img src="https://avatars.githubusercontent.com/u/149884766?v=4?v=4?s=50" width="50px;" alt="Ilham Rafiqin"/><br /><sub><b>Ilham Rafiqin</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/oscarf189"><img src="https://avatars.githubusercontent.com/u/28113740?v=4?v=4?s=50" width="50px;" alt="Oscar Flores"/><br /><sub><b>Oscar Flores</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harishb00a"><img src="https://avatars.githubusercontent.com/harishb00a?v=4?s=50" width="50px;" alt="harishb00a"/><br /><sub><b>harishb00a</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/sotoblanco"><img src="https://avatars.githubusercontent.com/u/46135649?v=4?v=4?s=50" width="50px;" alt="Pastor Soto"/><br /><sub><b>Pastor Soto</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/salmanmkc"><img src="https://avatars.githubusercontent.com/u/32169182?v=4?v=4?s=50" width="50px;" alt="Salman Chishti"/><br /><sub><b>Salman Chishti</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/adityamulik"><img src="https://avatars.githubusercontent.com/u/10626835?v=4?v=4?s=50" width="50px;" alt="Aditya Mulik"/><br /><sub><b>Aditya Mulik</b></sub></a><br />✍️</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- TINYTORCH-CONTRIBUTORS-END -->

---

### 💼 Interview Hub Contributors

<!-- INTERVIEWS-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧠 🎨 ✍️</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- INTERVIEWS-CONTRIBUTORS-END -->

---

### 🛠️ Hardware Kits Contributors

<!-- KITS-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧪 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Mjrovai"><img src="https://avatars.githubusercontent.com/Mjrovai?v=4?s=50" width="50px;" alt="Marcelo Rovai"/><br /><sub><b>Marcelo Rovai</b></sub></a><br />✍️ 🧑‍💻 🎨 </td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/salmanmkc"><img src="https://avatars.githubusercontent.com/u/32169182?v=4?v=4?s=50" width="50px;" alt="Salman Chishti"/><br /><sub><b>Salman Chishti</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Pratham-ja"><img src="https://avatars.githubusercontent.com/u/114498234?v=4?v=4?s=50" width="50px;" alt="Pratham Chaudhary"/><br /><sub><b>Pratham Chaudhary</b></sub></a><br />🧑‍💻</td>
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
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/salmanmkc"><img src="https://avatars.githubusercontent.com/u/32169182?v=4?v=4?s=50" width="50px;" alt="Salman Chishti"/><br /><sub><b>Salman Chishti</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Pratham-ja"><img src="https://avatars.githubusercontent.com/u/114498234?v=4?v=4?s=50" width="50px;" alt="Pratham Chaudhary"/><br /><sub><b>Pratham Chaudhary</b></sub></a><br />🧑‍💻</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- LABS-CONTRIBUTORS-END -->
---

<div align="center">

<b><a href="https://buttondown.email/mlsysbook">✉️ Subscribe</a> • <a href="https://github.com/harvard-edge/cs249r_book/discussions">💬 Join discussions</a> • <a href="https://mlsysbook.ai/">🌐 Visit mlsysbook.ai</a></b>

<b>Made with ❤️ for AI engineers</b><br>
<i>in the making, around the world</i> 🌎
</div>
