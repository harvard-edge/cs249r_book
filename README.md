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
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/book-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/book-validate-dev.yml?branch=dev&event=push&label=Book&logo=githubactions&cacheSeconds=300" alt="Book"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/tinytorch-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/tinytorch-validate-dev.yml?branch=dev&event=push&label=TinyTorch&logo=python&cacheSeconds=300" alt="TinyTorch"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/labs-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/labs-validate-dev.yml?branch=dev&event=push&label=Labs&logo=jupyter&cacheSeconds=300" alt="Labs"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/kits-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/kits-validate-dev.yml?branch=dev&event=push&label=Kits&logo=arduino&cacheSeconds=300" alt="Kits"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/mlsysim-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/mlsysim-validate-dev.yml?branch=dev&event=push&label=MLSys%C2%B7im&logo=python&cacheSeconds=300" alt="MLSys·im"></a></br>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/slides-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/slides-validate-dev.yml?branch=dev&event=push&label=Slides&logo=googleslides&cacheSeconds=300" alt="Slides"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/instructors-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/instructors-validate-dev.yml?branch=dev&event=push&label=Instructors&logo=googleclassroom&cacheSeconds=300" alt="Instructors"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/staffml-preview-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/staffml-preview-dev.yml?branch=dev&event=push&label=StaffML&logo=target&cacheSeconds=300" alt="StaffML"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/sync-newsletter.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/sync-newsletter.yml?branch=dev&event=schedule&label=Newsletter&logo=buttondown&cacheSeconds=300" alt="Newsletter"></a>
  <img src="https://img.shields.io/github/last-commit/harvard-edge/cs249r_book?branch=dev&label=Updated&logo=git&cacheSeconds=300" alt="Updated">
</p>

<!-- Meta -->
<p align="center">
  <a href="https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE.md"><img src="https://img.shields.io/badge/License-CC--BY--NC--SA%204.0-blue.svg" alt="License"></a>
  <a href="CITATION.bib"><img src="https://img.shields.io/badge/Cite-IEEE%202024-blue?logo=ieee" alt="Cite"></a>
  <a href="https://opencollective.com/mlsysbook"><img src="https://img.shields.io/badge/Fund%20Us-Open%20Collective-blue.svg?logo=open-collective" alt="Fund Us"></a>
</p>

<p align="center">
  <b><a href="https://mlsysbook.ai">📘 Textbook</a></b> •
  <b><a href="https://mlsysbook.ai/vol1/">📗 Vol I</a> + <a href="https://mlsysbook.ai/vol2/">📘 Vol II</a></b> •
  <b><a href="https://mlsysbook.ai/tinytorch/">🔥 TinyTorch</a></b> •
  <b><a href="https://mlsysbook.ai/labs/">🔬 Labs</a></b> •
  <b><a href="https://mlsysbook.ai/mlsysim/">🔮 MLSys·im</a></b> •
  <b><a href="https://mlsysbook.ai/staffml/">💼 StaffML</a></b>
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

Every component connects. The textbook gives you the mental models. The labs let you reason through trade-offs interactively, powered by MLSys·im — a modeling engine for infrastructure you can't physically access, and a standalone tool in its own right. TinyTorch makes you build the machinery yourself. The hardware kits put you face-to-face with real deployment constraints. StaffML tests whether you actually understand it. Socratiq adds AI-guided reading, contextual quizzes, and spaced repetition inside the learning experience. And the instructor hub, slides, and newsletter give educators everything they need to bring this into a classroom.

<p align="center">
  <img src="README/curriculum-map.svg?v=4" alt="Curriculum map showing how the textbook, labs, TinyTorch, hardware kits, MLSys·im, and StaffML connect" width="760">
</p>

### For Students

<table width="100%" style="width:100%">
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
      <td><a href="https://mlsysbook.ai/vol1/">Vol I</a> · <a href="https://mlsysbook.ai/vol2/">Vol II</a></td>
    </tr>
    <tr>
      <td align="center">🔬</td>
      <td><b>Labs</b></td>
      <td>Interactive Marimo notebooks where you explore trade-offs from the textbook: change a parameter, see what breaks, build intuition. Powered by MLSys·im under the hood.</td>
      <td><a href="https://mlsysbook.ai/labs/">Launch labs</a> · <a href="labs/README.md">Repo guide</a></td>
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
      <td>Deploy ML to Arduino, Seeed, Grove, and Raspberry Pi devices. Real memory limits, real power budgets, real latency.</td>
      <td><a href="https://mlsysbook.ai/kits">Browse labs</a></td>
    </tr>
    <tr>
      <td align="center">🔮</td>
      <td><b>MLSys·im</b></td>
      <td>Calculate memory bottlenecks, network saturation, and scheduling limits at infrastructure scales you can't physically access.</td>
      <td><a href="https://mlsysbook.ai/mlsysim/">Use simulator</a> · <a href="mlsysim/README.md">Repo guide</a></td>
    </tr>
    <tr>
      <td align="center">💼</td>
      <td><b>StaffML</b></td>
      <td>Physics-grounded interview questions for ML systems roles. Vault, practice drills, mock interviews, and progress tracking.</td>
      <td><a href="https://mlsysbook.ai/staffml/">Practice</a> · <a href="interviews/README.md">Repo guide</a></td>
    </tr>
  </tbody>
</table>

### For Educators

<table width="100%" style="width:100%">
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
      <td>The AI Engineering Blueprint: two 16-week syllabi, pedagogy guide, assessment rubrics, and a TA handbook.</td>
      <td><a href="https://mlsysbook.ai/instructors/">View hub</a> · <a href="instructors/README.md">Repo guide</a></td>
    </tr>
    <tr>
      <td align="center">🎬</td>
      <td><b>Lecture Slides</b></td>
      <td>Beamer slide decks for every chapter, with four theme variants. Drop into your course and teach.</td>
      <td><a href="https://mlsysbook.ai/slides/">Browse decks</a> · <a href="slides/README.md">Repo guide</a></td>
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

## Choose Your Path

The pieces are designed to work together, but you do not need to adopt everything at once.

| If you are... | Start here | Then go deeper |
|---|---|---|
| **A student or self-learner** | [Read Volume I](https://mlsysbook.ai/vol1/) and try [Lab 00](https://mlsysbook.ai/labs/vol1/lab_00_introduction/) | Build [TinyTorch](https://mlsysbook.ai/tinytorch/), use [MLSys·im](https://mlsysbook.ai/mlsysim/), and practice with [StaffML](https://mlsysbook.ai/staffml/) |
| **An instructor** | Open [The AI Engineering Blueprint](https://mlsysbook.ai/instructors/) | Use the [course map](https://mlsysbook.ai/instructors/course-map.html), [slides](https://mlsysbook.ai/slides/), rubrics, and TA guide |
| **A contributor** | Pick the component you use most | Improve chapters, labs, tests, examples, hardware notes, simulator models, or assessment content |

The learning loop is: **Read → Explore → Build → Model → Deploy → Practice → Teach**.

### Adjacent and Experimental Work

Some projects are intentionally earlier-stage than the main curriculum:

- [Socratiq](socratiq/README.md) explores AI-guided reading, contextual quizzes, and spaced repetition for static learning sites.
- [MLPerf EDU](mlperf-edu/README.md) is an under-construction pedagogical benchmark suite aligned with MLCommons MLPerf.
- [ML Systems Design Grammar](design-grammar/README.md) is an experimental framework for reasoning from stable primitives, constraints, and rewrite rules.

---

## What You Will Learn

This textbook teaches you to think at the intersection of machine learning and systems engineering. Each chapter bridges algorithmic concepts with the infrastructure that makes them work in practice.

<table width="100%" style="width:100%">
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

<table width="100%" style="width:100%">
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

## FAQ

<details>
<summary><b>Who is this for, and what should I know first?</b></summary>

<br>

This is for anyone who wants to engineer intelligent systems, not only train models: students, working engineers moving into ML infrastructure, and educators building a course. We assume you can program in Python and have met basic machine learning ideas, but the book builds the systems concepts from the ground up. You do not need a background in computer architecture, distributed systems, or datacenter operations. Volume I starts at the foundations, and the rest of the curriculum (TinyTorch, labs, hardware kits, and the simulator) lets you learn by building rather than only by reading.

</details>

<details>
<summary><b>Do I need Volume I before Volume II? What is the difference?</b></summary>

<br>

The two volumes differ in scope, not depth. Both are equally rigorous. Volume I is the single-machine world: how an ML system works on one node with a handful of accelerators, from data and a single neuron's computation up through training, optimization, and deployment. Volume II is the at-scale world: many machines across a network, distributed training, fault tolerance, fleet orchestration, inference at scale, and governance. Volume II does not assume you have read Volume I, so you can start there if you already have the foundations. The natural path, though, is Volume I to build the mental models and Volume II to apply them across the fleet. The analogy we follow is Hennessy and Patterson: *Computer Organization and Design* first, then *Computer Architecture: A Quantitative Approach*.

</details>

<details>
<summary><b>Do I need to use TinyTorch, the labs, and the kits, or can I just read the book?</b></summary>

<br>

You can just read the book. Each volume stands on its own. The rest of the curriculum (TinyTorch, the labs, the hardware kits, the simulator, and the interview practice) exists to deepen what the book teaches by making you build and measure it, but none of it is required to follow the text. Start with the book, and reach for the hands-on pieces when you want a concept to become muscle memory.

</details>

<details>
<summary><b>Isn't this just a deep learning book?</b></summary>

<br>

Deep learning books (Goodfellow et al.'s *Deep Learning*, Bishop, d2l.ai, fast.ai) teach you to design and train models: architectures, optimization, and the mathematics of learning. They mostly stop at the model. This book starts where they leave off. It treats the model as one component inside a system that has to ingest data, run on real silicon under power and latency budgets, serve predictions reliably, and keep working as the world drifts. You can finish a deep learning course knowing how a transformer learns and still not know why it stalls on a 4,000-accelerator training run, what the KV cache does to your serving memory, or why your accelerator sits idle. That gap is what we teach. Learn the model from a deep learning text, then learn the system here.

</details>

<details>
<summary><b>Isn't this MLOps, or the same as <i>Designing Machine Learning Systems</i>?</b></summary>

<br>

This is the most common mix-up, because "ML systems" and "MLOps" sound interchangeable and several good practitioner books share the words. MLOps books are operations guides: how to wire up a feature store, a pipeline, and a deployment with today's tools. They are valuable, and they age with the tooling. This book teaches the layer underneath: the physics and quantitative reasoning that explain why those tools exist and what they cost. We ask which questions matter, why a design is the way it is, and what it cannot escape (bandwidth, latency, power, failure rates).

Think of the difference between following a recipe and understanding how cooking works. A recipe gives you exact steps for one dish: this temperature, this pan, this many minutes. It works beautifully until the oven, the ingredients, or the kitchen changes. Understanding why heat, salt, acid, and time transform food is different. It lets you cook in any kitchen, rescue a dish that is going wrong, and invent one that no recipe covers.

An MLOps book hands you the recipe for the stack you have today. This book teaches the underlying science, so you can reason about any stack, debug the one that is failing, and design the one that does not exist yet.

</details>

<details>
<summary><b>How is this different from a classic systems reference like <i>The Datacenter as a Warehouse-Scale Computer</i>?</b></summary>

<br>

References like Barroso, Hölzle, and Clidaras's *The Datacenter as a Warehouse-Scale Computer* are excellent. They distill how one organization engineered one canonical system, written by the people who built it. This project is a different kind of artifact, and the two are complementary rather than competing.

1. **Curriculum, not reference.** A synthesis lecture documents a finished design for practitioners who already know the field. This book teaches the discipline from the ground up, with learning objectives, worked quantitative examples, labs, and an AI tutor, and it carries the reader from a single neuron (Volume I) all the way to the warehouse-scale fleet (Volume II).

2. **Vendor neutral, not a single stack.** A vendor reference can say "here is how we do it" and quote real production numbers. We generalize across accelerators (GPU and TPU), across cloud and edge, and teach *why* a design is the way it is, so the reasoning survives the next hardware generation and transfers anywhere.

3. **Living, not a snapshot.** A printed edition is frozen until its next revision. This is open source, continuously updated, and surrounded by code you build yourself (TinyTorch), hardware kits, a simulator, and interview practice.

In short, a warehouse-scale reference tells you how one machine was built. This curriculum teaches you to reason about why, and builds the judgment to design the next one.

</details>

<details>
<summary><b>Why read a textbook in the age of LLMs?</b></summary>

<br>

Because a textbook gives you something an LLM does not: perspective. An LLM is excellent at retrieval, and we are not trying to compete on retrieval. If a paragraph only delivers a fact you could get faster by asking a model, it has not earned its place. What a book builds instead is a structured mental model in the right order, the judgment to know which questions matter, and the reasoning behind why a design is the way it is and what it costs. Much of that comes from what a textbook chooses to leave out, since deciding what is central and what is peripheral is itself a lesson that an encyclopedic pile of facts cannot teach.

Bruce Davie makes this case well in ["Textbooks in Tokenland"](https://systemsapproach.org/2026/06/01/textbooks-in-tokenland/) (Systems Approach): an LLM generates text that is not grounded in communicative intent, while a textbook is written by people trying to convey a model of the world to a reader. We agree, and we go one step further by building an AI tutor (SocratiQ) into the reading experience. The goal is not textbook versus LLM. A good book gives you the perspective to ask meaningful questions, and the LLM helps you answer them. Use each for what it does best.

</details>

<details>
<summary><b>Is it free, and how do I read it?</b></summary>

<br>

Yes. Both volumes are free to read online at [mlsysbook.ai](https://mlsysbook.ai), and the textbook is open source under a Creative Commons license (CC BY-NC-SA 4.0), so you can share and adapt it for non-commercial use with attribution. If you prefer print, a hardcopy edition is coming in 2026 with MIT Press. The surrounding tools are open source too, each under its own license.

</details>

---

## Quick Start

<table width="100%" style="width:100%">
  <tbody>
    <tr>
      <td width="7%" align="center"><h3>①</h3></td>
      <td width="93%"><b>Read the textbook.</b> Start with <a href="https://mlsysbook.ai/vol1/">Volume I</a> or continue to <a href="https://mlsysbook.ai/vol2/">Volume II</a>. It's the foundation for everything else.</td>
    </tr>
    <tr>
      <td align="center"><h3>②</h3></td>
      <td><b>Pick a hands-on path.</b> <a href="https://mlsysbook.ai/tinytorch/">Build a framework</a> (TinyTorch), <a href="https://mlsysbook.ai/labs/">explore trade-offs</a> (Labs), <a href="https://mlsysbook.ai/mlsysim/">model constraints</a> (MLSys·im), or <a href="https://mlsysbook.ai/kits">deploy to real hardware</a> (Kits).</td>
    </tr>
    <tr>
      <td align="center"><h3>③</h3></td>
      <td><b>Test yourself.</b> Drill <a href="https://mlsysbook.ai/staffml/">StaffML</a>: physics-grounded systems design questions across cloud, edge, mobile, and TinyML.</td>
    </tr>
    <tr>
      <td align="center"><h3>④</h3></td>
      <td><b>Teach it.</b> Adopt the curriculum with the <a href="https://mlsysbook.ai/instructors/">AI Engineering Blueprint</a> and <a href="https://mlsysbook.ai/slides/">lecture slides</a>.</td>
    </tr>
  </tbody>
</table>

---

## Branch Guide

> [!NOTE]
> **You are on the `dev` branch.** Active development happens here. For the last stable release, see the [`main` branch](https://github.com/harvard-edge/cs249r_book/tree/main).

<table width="100%" style="width:100%">
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
        <b>Curriculum</b> — TinyTorch, Kits, MLSys·im, Labs, StaffML
      </td>
      <td>
        TinyTorch and Hardware Kits are live.<br>
        MLSys·im, Labs, and StaffML are early-release and actively iterated.
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

<table width="100%" style="width:100%">
  <tbody>
    <tr>
      <td width="50%" align="center">
        <b>Star the repo</b><br>
        Stars signal to universities and foundations that this work matters. They directly fund workshops and hardware kits for underserved classrooms.<br><br>
        <a href="https://star-history.com/#harvard-edge/cs249r_book&Date">
          <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=harvard-edge/cs249r_book&type=date&theme=dark&legend=top-left">
            <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=harvard-edge/cs249r_book&type=date&legend=top-left">
            <img src="https://api.star-history.com/chart?repos=harvard-edge/cs249r_book&type=date&legend=top-left" alt="Star History Chart" width="400">
          </picture>
        </a><br>
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

<table width="100%" style="width:100%">
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
      <td align="center">🔬</td>
      <td><b>Improve interactive labs or simulator models</b></td>
      <td><a href="labs/README.md">Labs guide</a> · <a href="mlsysim/README.md">MLSys·im guide</a></td>
    </tr>
    <tr>
      <td align="center">💼</td>
      <td><b>Improve assessment or career-readiness content</b></td>
      <td><a href="interviews/README.md">StaffML guide</a> · <a href="book/tools/scripts/genai/quiz_refresh/README.md">quiz refresh guide</a></td>
    </tr>
    <tr>
      <td align="center">🧠</td>
      <td><b>Improve AI learning tools</b></td>
      <td><a href="socratiq/README.md">Socratiq guide</a></td>
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

## License

This is a multi-component repository, and **each component is released under its own license** to match its purpose. The file inside each directory (e.g. `tinytorch/LICENSE`, `interviews/staffml/LICENSE`) is authoritative.

| Component | License | What it means |
|---|---|---|
| Textbook (`book/`), Labs (`labs/`), Kits (`kits/`), Slides (`slides/`), Instructors (`instructors/`) | [CC-BY-NC-SA 4.0](LICENSE.md) | Share and adapt for **non-commercial** use, with attribution and same-license sharing. |
| [TinyTorch](tinytorch/LICENSE) | MIT | Permissive — use, modify, redistribute, including commercially. |
| [MLSys·im](mlsysim/LICENSE.md) | Apache 2.0 | Permissive with explicit patent grant. |
| [StaffML](interviews/staffml/LICENSE) | AGPL v3 | Strong copyleft — modifications to deployed services must be published. Commercial licensing available; contact the authors. |
| [StaffML question corpus](interviews/vault/questions/LICENSE) | CC BY-NC 4.0 | Research and educational use; commercial use requires permission. |
| [TinyDigits dataset](tinytorch/datasets/tinydigits/LICENSE) | BSD 3-Clause | Permissive (matches sklearn ancestry). |
| [TinyTalks dataset](tinytorch/datasets/tinytalks/LICENSE) | CC BY 4.0 | Permissive with attribution; commercial use allowed. |

A user-facing summary lives at [mlsysbook.ai/about/license](https://mlsysbook.ai/about/license.html).

If you are an institution considering adoption, or a company interested in commercial terms for a copyleft component, please reach out to [edu@tinyML.org](mailto:edu@tinyML.org).

---

## Contributors

Thanks goes to these wonderful people who have contributed to making this resource better for everyone!

**Legend:** 🪲 Bug Hunter · 🧑‍💻 Code Contributor · ✍️ Doc Wizard · 🎨 Design Artist · 🧠 Idea Spark · 🔎 Code Reviewer · 🧪 Test Tinkerer · 🛠️ Tool Builder

### 📖 Textbook Contributors

<!-- BOOK-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<p><em>Coming soon!</em></p>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- BOOK-CONTRIBUTORS-END -->

---

### 🔥 TinyTorch Contributors

<!-- TINYTORCH-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table width="100%" style="width:100%">
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧠 🔎 🧪 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Shashank-Tripathi-07"><img src="https://avatars.githubusercontent.com/u/178375647?v=4?v=4?s=50" width="50px;" alt="Rocky"/><br /><sub><b>Rocky</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧪 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/kai4avaya"><img src="https://avatars.githubusercontent.com/kai4avaya?v=4?s=50" width="50px;" alt="kai"/><br /><sub><b>kai</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/minhdang26403"><img src="https://avatars.githubusercontent.com/minhdang26403?v=4?s=50" width="50px;" alt="Dang Truong"/><br /><sub><b>Dang Truong</b></sub></a><br />🪲 🧑‍💻 ✍️ 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/ngbolin"><img src="https://avatars.githubusercontent.com/u/9389997?v=4?v=4?s=50" width="50px;" alt="Ng Bo Lin"/><br /><sub><b>Ng Bo Lin</b></sub></a><br />🪲 🧑‍💻 ✍️ 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/farhan523"><img src="https://avatars.githubusercontent.com/u/62025759?v=4?v=4?s=50" width="50px;" alt="Farhan Asghar"/><br /><sub><b>Farhan Asghar</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/vedant-a-joshi"><img src="https://avatars.githubusercontent.com/u/77033374?v=4?v=4?s=50" width="50px;" alt="Vedant Joshi"/><br /><sub><b>Vedant Joshi</b></sub></a><br />🪲 🧑‍💻 ✍️ 🧪</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/didier-durand"><img src="https://avatars.githubusercontent.com/didier-durand?v=4?s=50" width="50px;" alt="Didier Durand"/><br /><sub><b>Didier Durand</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/rnjema"><img src="https://avatars.githubusercontent.com/rnjema?v=4?s=50" width="50px;" alt="rnjema"/><br /><sub><b>rnjema</b></sub></a><br />🧑‍💻 ✍️ 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/AndreaMattiaGaravagno"><img src="https://avatars.githubusercontent.com/u/22458187?v=4?v=4?s=50" width="50px;" alt="AndreaMattiaGaravagno"/><br /><sub><b>AndreaMattiaGaravagno</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Pratham-ja"><img src="https://avatars.githubusercontent.com/u/114498234?v=4?v=4?s=50" width="50px;" alt="Pratham Chaudhary"/><br /><sub><b>Pratham Chaudhary</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/asgalon"><img src="https://avatars.githubusercontent.com/u/45242704?v=4?v=4?s=50" width="50px;" alt="asgalon"/><br /><sub><b>asgalon</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/bdub-1"><img src="https://avatars.githubusercontent.com/u/100250017?v=4?v=4?s=50" width="50px;" alt="bdub"/><br /><sub><b>bdub</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/karthikdani"><img src="https://avatars.githubusercontent.com/karthikdani?v=4?s=50" width="50px;" alt="Karthik Dani"/><br /><sub><b>Karthik Dani</b></sub></a><br />🪲 🧑‍💻</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/avikde"><img src="https://avatars.githubusercontent.com/avikde?v=4?s=50" width="50px;" alt="Avik De"/><br /><sub><b>Avik De</b></sub></a><br />🪲 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Takosaga"><img src="https://avatars.githubusercontent.com/Takosaga?v=4?s=50" width="50px;" alt="Takosaga"/><br /><sub><b>Takosaga</b></sub></a><br />🪲 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/joeswagson"><img src="https://avatars.githubusercontent.com/joeswagson?v=4?s=50" width="50px;" alt="joeswagson"/><br /><sub><b>joeswagson</b></sub></a><br />🧑‍💻 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Roldao-Neto"><img src="https://avatars.githubusercontent.com/u/148023227?v=4?v=4?s=50" width="50px;" alt="Rolds"/><br /><sub><b>Rolds</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/AdemolaAri"><img src="https://avatars.githubusercontent.com/u/49918815?v=4?v=4?s=50" width="50px;" alt="Ademola Arigbabuwo"/><br /><sub><b>Ademola Arigbabuwo</b></sub></a><br />🪲 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harishb00"><img src="https://avatars.githubusercontent.com/u/43300971?v=4?v=4?s=50" width="50px;" alt="Harish"/><br /><sub><b>Harish</b></sub></a><br />🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Chufeng-Jiang"><img src="https://avatars.githubusercontent.com/u/80246982?v=4?v=4?s=50" width="50px;" alt="Chufeng JIANG"/><br /><sub><b>Chufeng JIANG</b></sub></a><br />🪲 🧑‍💻</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Fabio-RibeiroB"><img src="https://avatars.githubusercontent.com/u/74654489?v=4?v=4?s=50" width="50px;" alt="Fábio"/><br /><sub><b>Fábio</b></sub></a><br />🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/aadityansha06"><img src="https://avatars.githubusercontent.com/u/96714228?v=4?v=4?s=50" width="50px;" alt="Aadityansha "/><br /><sub><b>Aadityansha </b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/euwyngoh"><img src="https://avatars.githubusercontent.com/u/211522196?v=4?v=4?s=50" width="50px;" alt="euwyngoh"/><br /><sub><b>euwyngoh</b></sub></a><br />🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/AmirAlasady"><img src="https://avatars.githubusercontent.com/AmirAlasady?v=4?s=50" width="50px;" alt="Amir Alasady"/><br /><sub><b>Amir Alasady</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jettythek"><img src="https://avatars.githubusercontent.com/jettythek?v=4?s=50" width="50px;" alt="jettythek"/><br /><sub><b>jettythek</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/wz1114841863"><img src="https://avatars.githubusercontent.com/wz1114841863?v=4?s=50" width="50px;" alt="wzz"/><br /><sub><b>wzz</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/keo-dara"><img src="https://avatars.githubusercontent.com/u/175544368?v=4?v=4?s=50" width="50px;" alt="keo-dara"/><br /><sub><b>keo-dara</b></sub></a><br />🪲</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Kobra299"><img src="https://avatars.githubusercontent.com/u/4283156?v=4?v=4?s=50" width="50px;" alt="Wayne Norman"/><br /><sub><b>Wayne Norman</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/lalalostcode"><img src="https://avatars.githubusercontent.com/u/149884766?v=4?v=4?s=50" width="50px;" alt="Ilham Rafiqin"/><br /><sub><b>Ilham Rafiqin</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/oscarf189"><img src="https://avatars.githubusercontent.com/u/28113740?v=4?v=4?s=50" width="50px;" alt="Oscar Flores"/><br /><sub><b>Oscar Flores</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/sotoblanco"><img src="https://avatars.githubusercontent.com/u/46135649?v=4?v=4?s=50" width="50px;" alt="Pastor Soto"/><br /><sub><b>Pastor Soto</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/salmanmkc"><img src="https://avatars.githubusercontent.com/u/32169182?v=4?v=4?s=50" width="50px;" alt="Salman Chishti"/><br /><sub><b>Salman Chishti</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/adityamulik"><img src="https://avatars.githubusercontent.com/u/10626835?v=4?v=4?s=50" width="50px;" alt="Aditya Mulik"/><br /><sub><b>Aditya Mulik</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/yarikoptic"><img src="https://avatars.githubusercontent.com/u/39889?v=4?v=4?s=50" width="50px;" alt="Yaroslav Halchenko"/><br /><sub><b>Yaroslav Halchenko</b></sub></a><br />🧑‍💻</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/RinZ27"><img src="https://avatars.githubusercontent.com/u/222222878?v=4?v=4?s=50" width="50px;" alt="Rin"/><br /><sub><b>Rin</b></sub></a><br />🧑‍💻</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- TINYTORCH-CONTRIBUTORS-END -->

---

### 🚀 MLSys·im Contributors

<!-- MLSYSIM-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table width="100%" style="width:100%">
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧑‍💻 🎨 ✍️ 🧠 </td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Shashank-Tripathi-07"><img src="https://avatars.githubusercontent.com/u/178375647?v=4?v=4?s=50" width="50px;" alt="Rocky"/><br /><sub><b>Rocky</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/farhan523"><img src="https://avatars.githubusercontent.com/u/62025759?v=4?v=4?s=50" width="50px;" alt="Farhan Asghar"/><br /><sub><b>Farhan Asghar</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/aadityansha06"><img src="https://avatars.githubusercontent.com/u/96714228?v=4?v=4?s=50" width="50px;" alt="Aadityansha "/><br /><sub><b>Aadityansha </b></sub></a><br />🪲 🧑‍💻 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/ShivtejG236"><img src="https://avatars.githubusercontent.com/u/91597404?v=4?v=4?s=50" width="50px;" alt="Shivtej Gaikwad"/><br /><sub><b>Shivtej Gaikwad</b></sub></a><br />🪲 🧑‍💻 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/asgalon"><img src="https://avatars.githubusercontent.com/u/45242704?v=4?v=4?s=50" width="50px;" alt="Peter Koellner"/><br /><sub><b>Peter Koellner</b></sub></a><br />🪲 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/hzeljko"><img src="https://avatars.githubusercontent.com/hzeljko?v=4?s=50" width="50px;" alt="Zeljko Hrcek"/><br /><sub><b>Zeljko Hrcek</b></sub></a><br />🧑‍💻</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/nyxst4ck"><img src="https://avatars.githubusercontent.com/u/289980115?v=4?v=4?s=50" width="50px;" alt="nyxst4ck"/><br /><sub><b>nyxst4ck</b></sub></a><br />✍️</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- MLSYSIM-CONTRIBUTORS-END -->

---

### 🤖 StaffML Contributors

<!-- STAFFML-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<p><em>Coming soon!</em></p>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- STAFFML-CONTRIBUTORS-END -->

---

### 🛠️ Hardware Kits Contributors

<!-- KITS-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table width="100%" style="width:100%">
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧪 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Mjrovai"><img src="https://avatars.githubusercontent.com/Mjrovai?v=4?s=50" width="50px;" alt="Marcelo Rovai"/><br /><sub><b>Marcelo Rovai</b></sub></a><br />✍️ 🧑‍💻 🎨 </td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Shashank-Tripathi-07"><img src="https://avatars.githubusercontent.com/u/178375647?v=4?v=4?s=50" width="50px;" alt="Rocky"/><br /><sub><b>Rocky</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/farhan523"><img src="https://avatars.githubusercontent.com/u/62025759?v=4?v=4?s=50" width="50px;" alt="Farhan Asghar"/><br /><sub><b>Farhan Asghar</b></sub></a><br />🪲 🧑‍💻 🎨</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/imrehg"><img src="https://avatars.githubusercontent.com/u/38863?v=4?v=4?s=50" width="50px;" alt="Gergely Imreh"/><br /><sub><b>Gergely Imreh</b></sub></a><br />🪲 ✍️</td>
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
<table width="100%" style="width:100%">
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Shashank-Tripathi-07"><img src="https://avatars.githubusercontent.com/u/178375647?v=4?v=4?s=50" width="50px;" alt="Rocky"/><br /><sub><b>Rocky</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/farhan523"><img src="https://avatars.githubusercontent.com/u/62025759?v=4?v=4?s=50" width="50px;" alt="Farhan Asghar"/><br /><sub><b>Farhan Asghar</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/aadityansha06"><img src="https://avatars.githubusercontent.com/u/96714228?v=4?v=4?s=50" width="50px;" alt="Aadityansha "/><br /><sub><b>Aadityansha </b></sub></a><br />🪲 🧑‍💻 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/asgalon"><img src="https://avatars.githubusercontent.com/u/45242704?v=4?v=4?s=50" width="50px;" alt="Peter Koellner"/><br /><sub><b>Peter Koellner</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/salmanmkc"><img src="https://avatars.githubusercontent.com/u/32169182?v=4?v=4?s=50" width="50px;" alt="Salman Chishti"/><br /><sub><b>Salman Chishti</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Pratham-ja"><img src="https://avatars.githubusercontent.com/u/114498234?v=4?v=4?s=50" width="50px;" alt="Pratham Chaudhary"/><br /><sub><b>Pratham Chaudhary</b></sub></a><br />🧑‍💻</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- LABS-CONTRIBUTORS-END -->

---

### 🎞️ Slides Contributors

<!-- SLIDES-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table width="100%" style="width:100%">
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Shashank-Tripathi-07"><img src="https://avatars.githubusercontent.com/u/178375647?v=4?v=4?s=50" width="50px;" alt="Rocky"/><br /><sub><b>Rocky</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/farhan523"><img src="https://avatars.githubusercontent.com/u/62025759?v=4?v=4?s=50" width="50px;" alt="Farhan Asghar"/><br /><sub><b>Farhan Asghar</b></sub></a><br />🪲 🧑‍💻 🎨</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- SLIDES-CONTRIBUTORS-END -->

---

### 🗺️ Instructor Site Contributors

<!-- INSTRUCTORS-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table width="100%" style="width:100%">
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/farhan523"><img src="https://avatars.githubusercontent.com/u/62025759?v=4?v=4?s=50" width="50px;" alt="Farhan Asghar"/><br /><sub><b>Farhan Asghar</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Shashank-Tripathi-07"><img src="https://avatars.githubusercontent.com/u/178375647?v=4?v=4?s=50" width="50px;" alt="Rocky"/><br /><sub><b>Rocky</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🔎</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/coyaSONG"><img src="https://avatars.githubusercontent.com/u/66289470?v=4?v=4?s=50" width="50px;" alt="coyaSONG"/><br /><sub><b>coyaSONG</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- INSTRUCTORS-CONTRIBUTORS-END -->
---

<div align="center">

<b><a href="https://buttondown.email/mlsysbook">✉️ Subscribe</a> • <a href="https://github.com/harvard-edge/cs249r_book/discussions">💬 Join discussions</a> • <a href="https://mlsysbook.ai/">🌐 Visit mlsysbook.ai</a></b>

<b>Made with ❤️ for AI engineers</b><br>
<i>in the making, around the world</i> 🌎
</div>
