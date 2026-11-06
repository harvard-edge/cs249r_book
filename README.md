# Machine Learning Systems

<p align="center">
  <a href="README.md">English</a> •
  <a href="README/README_zh.md">中文</a> •
  <a href="README/README_ja.md">日本語</a> •
  <a href="README/README_ko.md">한국어</a>
  <br>
  <sub>The translations follow an earlier version of this page. Help bringing them up to date is welcome.</sub>
</p>

<div align="center">

<p align="center"><b>An open textbook series on how machine learning systems are built and why they are built that way, with companion tools for building, measuring, and teaching them.</b></p>

<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/License-per%20component-blue.svg" alt="License: per component"></a>
  <a href="CITATION.bib"><img src="https://img.shields.io/badge/Cite-IEEE%202024-blue?logo=ieee" alt="Cite"></a>
  <a href="https://opencollective.com/mlsysbook"><img src="https://img.shields.io/badge/Fund%20Us-Open%20Collective-blue.svg?logo=open-collective" alt="Fund Us"></a>
</p>

<p align="center">
  <a href="https://github.com/harvard-edge/cs249r_book/stargazers"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.github.com%2Frepos%2Fharvard-edge%2Fcs249r_book&query=%24.stargazers_count&label=Stars&logo=github&color=gold&style=for-the-badge&cacheSeconds=3600" alt="GitHub stars"></a>
</p>

<p align="center"><sub><b>Build status on <code>dev</code></b></sub></p>

<p align="center">
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/book-validate-dev.yml"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fharvard-edge%2Fcs249r_book%2Fgh-pages%2Fstatus%2Fvol1.json&logo=book&cacheSeconds=300" alt="Vol I"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/book-validate-dev.yml"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fharvard-edge%2Fcs249r_book%2Fgh-pages%2Fstatus%2Fvol2.json&logo=book&cacheSeconds=300" alt="Vol II"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/book-validate-dev.yml"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fharvard-edge%2Fcs249r_book%2Fgh-pages%2Fstatus%2Fvol3.json&logo=robot&cacheSeconds=300" alt="Vol III"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/book-validate-dev.yml"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fharvard-edge%2Fcs249r_book%2Fgh-pages%2Fstatus%2Fvol4.json&logo=gear&cacheSeconds=300" alt="Vol IV"></a>
  <br>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/tinytorch-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/tinytorch-validate-dev.yml?branch=dev&event=push&label=TinyTorch&logo=python&cacheSeconds=300" alt="TinyTorch"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/mlsysim-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/mlsysim-validate-dev.yml?branch=dev&event=push&label=MLSys%C2%B7im&logo=python&cacheSeconds=300" alt="MLSys·im"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/labs-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/labs-validate-dev.yml?branch=dev&event=push&label=Labs&logo=jupyter&cacheSeconds=300" alt="Labs"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/kits-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/kits-validate-dev.yml?branch=dev&event=push&label=Kits&logo=arduino&cacheSeconds=300" alt="Kits"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/slides-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/slides-validate-dev.yml?branch=dev&event=push&label=Slides&logo=googleslides&cacheSeconds=300" alt="Slides"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/instructors-validate-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/instructors-validate-dev.yml?branch=dev&event=push&label=Instructors&logo=googleclassroom&cacheSeconds=300" alt="Instructors"></a>
  <a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/staffml-preview-dev.yml"><img src="https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/staffml-preview-dev.yml?branch=dev&event=push&label=StaffML&logo=target&cacheSeconds=300" alt="StaffML"></a>
</p>

<p align="center">
  <b><a href="https://mlsysbook.ai">📘 Textbook Series</a></b> •
  <b><a href="https://mlsysbook.ai/vol1/">📗 Vol I</a></b> •
  <b><a href="https://mlsysbook.ai/vol2/">📘 Vol II (Preview)</a></b> •
  <b><a href="books/vol3/">🟣 Vol III (Dev)</a></b> •
  <b><a href="books/vol4/">🌲 Vol IV (Dev)</a></b>
  <br>
  <b><a href="https://mlsysbook.ai/tinytorch/">🔥 TinyTorch</a></b> •
  <b><a href="https://mlsysbook.ai/labs/">🔬 Labs</a></b> •
  <b><a href="https://mlsysbook.ai/kits/">🧰 Kits</a></b> •
  <b><a href="https://mlsysbook.ai/mlsysim/">🔮 MLSys·im</a></b> •
  <b><a href="https://mlsysbook.ai/instructors/">🎓 Instructors</a></b> •
  <b><a href="https://mlsysbook.ai/staffml/">💼 StaffML</a></b> •
  <b><a href="taxonomy/">Taxonomy</a></b>
</p>

<p align="center">📚 <b>Volume I is forthcoming in print from MIT Press.</b></p>

</div>

---

## Why This Curriculum Exists

<div align="center">
  <blockquote>
    <b>The world is rushing to build AI systems. It is not yet engineering them.</b>
  </blockquote>
</div>

**AI engineering** is the discipline of building efficient, reliable, and safe intelligent systems that operate in the real world, not models in isolation. It deserves a place alongside software engineering and computer engineering, and this curriculum teaches it the way computer architecture has long been taught, from first principles and with numbers a student can recompute.

Those numbers come from constraints. A model that scores well on a benchmark still has to fit in memory, meet a latency budget, stay within a power limit, and fail safely in front of real people. These constraints decide the architecture, and they follow laws that outlast any framework or chip. Memory bandwidth, network latency, energy, and failure rates govern a microcontroller with 256 KB of memory and an accelerator in a datacenter alike, even though the two differ in scale by many orders of magnitude. I learned this while working on TinyML at Google, and it became the idea that runs through every volume: **constraints drive architecture.**

Teaching AI engineering this way takes more than a book, which is why everything lives in one repository. Each part teaches something the others cannot. Students who only read do not internalize, and students who only code do not generalize. The textbook builds the mental model, TinyTorch makes it concrete by having students build the machinery themselves, and the hardware kits and the MLSys·im simulator show where the model breaks, on devices students can hold and at scales beyond any one lab.

The work began in a classroom. The HarvardX TinyML course came first and reached more than 100,000 learners. In 2023, the students in CS249r at Harvard wrote the first class notes for this curriculum together, one topic per group, and we published them on GitHub. Since then, I have kept revising the material with help from a growing community of contributors, and instructors at universities around the world now teach from it. The goal is to reach one million learners by 2030, because the world is building AI factories faster than it is training AI engineers.

---

## What You Will Learn

Each chapter connects an idea a reader may already know from machine learning to the systems that make it work in practice.

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
      <td><b>How training scales across accelerator clusters</b></td>
    </tr>
    <tr>
      <td>That quantization shrinks models</td>
      <td align="center">→</td>
      <td><b>How low-precision arithmetic maps to silicon</b></td>
    </tr>
    <tr>
      <td>What a transformer is</td>
      <td align="center">→</td>
      <td><b>Why the KV cache dominates memory at inference</b></td>
    </tr>
    <tr>
      <td>Models run on accelerators</td>
      <td align="center">→</td>
      <td><b>How serving systems batch requests to balance latency against throughput</b></td>
    </tr>
    <tr>
      <td>Models need good data</td>
      <td align="center">→</td>
      <td><b>How data pipelines, selection, and drift decide what a model can learn</b></td>
    </tr>
    <tr>
      <td>Models get deployed</td>
      <td align="center">→</td>
      <td><b>How to benchmark, monitor, and roll back models in production</b></td>
    </tr>
    <tr>
      <td>Hardware fails sometimes</td>
      <td align="center">→</td>
      <td><b>How to checkpoint and recover training jobs that span thousands of accelerators</b></td>
    </tr>
    <tr>
      <td>Edge devices have limits</td>
      <td align="center">→</td>
      <td><b>How to co-design models and hardware</b></td>
    </tr>
    <tr>
      <td>AI uses a lot of energy</td>
      <td align="center">→</td>
      <td><b>How to measure and reduce the energy and carbon cost of a system over its full lifecycle</b></td>
    </tr>
    <tr>
      <td>How to prompt an LLM</td>
      <td align="center">→</td>
      <td><b>How to architect agent loops, tool sandboxes, and context memory</b> <i>(Vol III, in development)</i></td>
    </tr>
    <tr>
      <td>How vision and robotics models infer</td>
      <td align="center">→</td>
      <td><b>How to close real-time sensor-actuator loops under physical constraints</b> <i>(Vol IV, in development)</i></td>
    </tr>
  </tbody>
</table>

---

## The Four Volumes

The textbook spans four volumes. Each one widens the boundary of the system and raises the cost of failure, so the volumes differ in scope, not in depth.

<table width="100%" style="width:100%">
  <thead>
    <tr>
      <th width="5%"></th>
      <th width="27%">Volume</th>
      <th width="13%">Unit of work</th>
      <th width="33%">Core systems question</th>
      <th width="22%">Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">📗</td>
      <td><a href="https://mlsysbook.ai/vol1/"><b>I. Introduction to Machine Learning Systems</b></a></td>
      <td>The model</td>
      <td>How does intelligence execute efficiently on a single machine?</td>
      <td><b>Released</b><br>Forthcoming in print from MIT Press</td>
    </tr>
    <tr>
      <td align="center">📘</td>
      <td><a href="https://mlsysbook.ai/vol2/"><b>II. Scaling Machine Learning Systems</b></a></td>
      <td>The fleet</td>
      <td>How does intelligence scale across distributed clusters and datacenters?</td>
      <td><b>Preview</b><br>In active revision</td>
    </tr>
    <tr>
      <td align="center">🟣</td>
      <td><a href="books/vol3/"><b>III. Agentic Machine Learning Systems</b></a></td>
      <td>The trajectory</td>
      <td>How is intelligence governed when it acts autonomously over long horizons?</td>
      <td><b>In development</b><br>Changes often</td>
    </tr>
    <tr>
      <td align="center">🌲</td>
      <td><a href="books/vol4/"><b>IV. Physical AI: Machine Learning Systems That Sense and Act</b></a></td>
      <td>The physical action</td>
      <td>How can intelligence act safely on matter and physical systems?</td>
      <td><b>In development</b><br>Changes often</td>
    </tr>
  </tbody>
</table>

Each volume answers a question the previous one leaves open.

* **Vol I → Vol II, when one machine is not enough.** Volume I covers single-machine execution, memory walls, and kernel efficiency. Volume II carries those foundations across thousands of accelerators, collective communication, fault-tolerant networks, and datacenter orchestration.
* **Vol II → Vol III, when a stateless response is not enough.** Volume II serves stateless requests at scale. Volume III adds stateful, multi-step loops that manage context memory, call tools, run in sandboxes, and recover from nondeterministic failures.
* **Vol III → Vol IV, when software acts on the physical world.** Volume III governs software environments. Volume IV crosses into physical systems, where computational latency becomes distance traveled and a failure cannot be undone.

I am writing Volumes III and IV because I plan to teach both subjects, and because they are where machine learning systems are heading. The progression is a natural one. Volumes I and II run in an open loop, where a request goes in and a response comes out. Volume III closes the loop in software, where an agent acts, observes the result, and acts again. Volume IV closes it through the physical world. Writing a book is how I learn a subject well enough to teach it.

> ⚠️ *Volumes III and IV are open drafts. Chapters are added, reorganized, and rewritten as my understanding grows, so please do not cite or teach from them yet. Feedback is welcome through the [book feedback](https://github.com/harvard-edge/cs249r_book/issues/new/choose) issue forms.*

---

## What's in the Repository

Around the textbook sit the companion tools that let students build, measure, and practice what they read. The pieces are designed to work together, but none of them requires the others.

<p align="center">
  <img src="README/curriculum-map.svg?v=6" alt="Curriculum map showing how the textbook, labs, TinyTorch, hardware kits, MLSys·im, and StaffML connect" width="760">
</p>

<table width="100%" style="width:100%">
  <thead>
    <tr>
      <th width="5%"></th>
      <th width="15%">Component</th>
      <th width="45%">What it does</th>
      <th width="12%">Status</th>
      <th width="23%">Links</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">📖</td>
      <td><b>Textbook</b></td>
      <td>Four volumes that build the mental models, from one machine to fleets, agents, and physical systems.</td>
      <td>Vol I released</td>
      <td><a href="https://mlsysbook.ai">Read online</a></td>
    </tr>
    <tr>
      <td align="center">🔥</td>
      <td><b>Tiny🔥Torch</b></td>
      <td>Build an ML framework from scratch in 20 progressive modules, from tensors to a transformer.</td>
      <td>Released</td>
      <td><a href="https://mlsysbook.ai/tinytorch/">Get started</a> · <a href="tinytorch/README.md">Repo guide</a></td>
    </tr>
    <tr>
      <td align="center">🔮</td>
      <td><b>MLSys·im</b></td>
      <td>Model memory, compute, network, and energy limits for systems too large to run in a classroom. The textbook's calculations are built on it.</td>
      <td>Released</td>
      <td><a href="https://mlsysbook.ai/mlsysim/">Use simulator</a> · <a href="mlsysim/README.md">Repo guide</a></td>
    </tr>
    <tr>
      <td align="center">🛠️</td>
      <td><b>Hardware Kits</b></td>
      <td>Deploy models to Arduino, Seeed, Grove, and Raspberry Pi devices, with real memory limits, power budgets, and latency.</td>
      <td>Released</td>
      <td><a href="https://mlsysbook.ai/kits/">Browse kits</a> · <a href="kits/README.md">Repo guide</a></td>
    </tr>
    <tr>
      <td align="center">🔬</td>
      <td><b>Labs</b></td>
      <td>Interactive Marimo notebooks, run in the browser, that pair with textbook chapters. A student predicts an outcome, changes a parameter, and sees what breaks.</td>
      <td>Early release, built in the open</td>
      <td><a href="https://mlsysbook.ai/labs/">Try labs</a> · <a href="labs/README.md">Repo guide</a></td>
    </tr>
    <tr>
      <td align="center">💼</td>
      <td><b>StaffML</b></td>
      <td>More than 9,000 systems design questions across cloud, edge, mobile, and TinyML, with practice drills and mock interviews.</td>
      <td>Early release</td>
      <td><a href="https://mlsysbook.ai/staffml/">Practice</a> · <a href="staffml/README.md">Repo guide</a></td>
    </tr>
    <tr>
      <td align="center">🎓</td>
      <td><b>Instructor Hub</b></td>
      <td>A syllabus for each course, with weekly readings and slides: introductory, scaling, TinyML, Edge AI, Physical AI (draft), and Agentic AI (in preparation).</td>
      <td>Released</td>
      <td><a href="https://mlsysbook.ai/instructors/">View hub</a> · <a href="instructors/README.md">Repo guide</a></td>
    </tr>
    <tr>
      <td align="center">🎬</td>
      <td><b>Lecture Slides</b></td>
      <td>Beamer decks for every chapter of the textbook, in one shared theme that is easy to restyle.</td>
      <td>Released</td>
      <td><a href="https://mlsysbook.ai/slides/">Browse decks</a> · <a href="slides/README.md">Repo guide</a></td>
    </tr>
    <tr>
      <td align="center">📬</td>
      <td><b>Newsletter</b></td>
      <td>Updates on new chapters, releases, and what the community is building.</td>
      <td>Ongoing</td>
      <td><a href="https://buttondown.email/mlsysbook">Subscribe</a></td>
    </tr>
  </tbody>
</table>

Two earlier-stage experiments also live here. [SocratiQ](socratiq/README.md) is an AI tutor that adds guided reading, quizzes, and spaced repetition to static learning sites. [MLPerf EDU](mlperf-edu/README.md) is a teaching benchmark suite aligned with MLCommons MLPerf.

---

## Start Here

Where to begin depends on what you want from the curriculum.

<table width="100%" style="width:100%">
  <thead>
    <tr>
      <th width="5%"></th>
      <th width="20%">If you are...</th>
      <th width="30%">Start here</th>
      <th width="45%">Then go deeper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">📖</td>
      <td><b>A student or self-learner</b></td>
      <td><a href="https://mlsysbook.ai/vol1/">Read Volume I</a></td>
      <td>Build <a href="https://mlsysbook.ai/tinytorch/">TinyTorch</a>, model systems with <a href="https://mlsysbook.ai/mlsysim/">MLSys·im</a>, try the <a href="https://mlsysbook.ai/labs/">labs</a> (early release), and practice with <a href="https://mlsysbook.ai/staffml/">StaffML</a></td>
    </tr>
    <tr>
      <td align="center">🎓</td>
      <td><b>An instructor</b></td>
      <td>Open the <a href="https://mlsysbook.ai/instructors/">Instructor Hub</a></td>
      <td>Pick a course and use its syllabus, <a href="https://mlsysbook.ai/slides/">slides</a>, kits, and labs</td>
    </tr>
    <tr>
      <td align="center">🤝</td>
      <td><b>A contributor</b></td>
      <td>Find the guide for your component under <a href="#contributing">Contributing</a></td>
      <td>Report problems through <a href="https://github.com/harvard-edge/cs249r_book/issues">GitHub Issues</a></td>
    </tr>
  </tbody>
</table>

---

## How the Material Is Made

No chapter is written once. Each one goes through a cycle that repeats as the field changes.

1. **Watch the field.** I follow the research and industry venues where new work appears: ISCA, MICRO, HPCA, ASPLOS, ISPASS, MLSys, NeurIPS, ICML, ICLR, Hot Chips, and industry summits.
2. **Distill.** A chapter keeps the principles and leaves out the variations. Quantization, for example, has been implemented in dozens of ways, and a student needs the principles that explain all of them. To decide what belongs, I use a test that David Patterson, who wrote the foreword to Volume I, gave me. A technique earns a place only when several companies use it in practice. That test ties the material to systems in production, so what a student learns carries over to practice.
3. **Review with practitioners.** I review chapters one-on-one with the engineers who build these systems inside large companies, from hyperscale infrastructure engineers to model developers. Practitioners at Arm, Google, Intel, Meta, Microsoft, Qualcomm, STMicroelectronics, and other organizations have reviewed the material, and academic reviewers in Europe, Asia, and the Americas have tested it against their own research and teaching.
4. **Teach it.** Volume I has been taught at Harvard through CS249r since 2023 and adopted by instructors at UC Berkeley, Columbia, Georgia Tech, IIT Bombay, and universities across Europe, Latin America, Africa, and Asia. The classroom shows where students get stuck.
5. **Revise.** Reader issues, reviewer notes, and classroom experience go back into the next draft.

Reviewers do more than check facts, and sometimes they change what the book covers. Sustainable AI and responsible AI both entered the book this way. Engineers building AI datacenters told me that the footprint of these facilities keeps growing and that students need to learn to account for it. Their concern led to a chapter I had not planned, which I wrote in 2023, before sustainable AI was a common topic.

Engineers in industry also urged me to teach responsible AI. It can look like a subject outside traditional engineering, but the systems in this book make decisions that affect people, and more and more of them act on their own. The engineers who build such systems are responsible for how they behave, so the book teaches that responsibility as part of the engineering.

**Readers are part of the review, too.** Every chapter is open, and when something is wrong, unclear, or missing, the [book feedback](https://github.com/harvard-edge/cs249r_book/issues/new/choose) forms send it straight into the next revision.

**Numbers are computed, not typed.** Every derived quantity in the books comes from Python code that ships with the text. I learned the need for this the hard way. In early drafts I typed numbers by hand, made mistakes, and watched the same quantity drift apart from one chapter and one volume to the next. No author can hold every number across the series in mind at once, so I wrote [MLSys·im](https://mlsysbook.ai/mlsysim/) to hold them instead. It is the single source of truth for every chapter and every volume. Each hardware and model specification in it traces to a cited source, and when a value changes there, every chapter that uses it changes with it. A test suite runs on every commit and checks unit consistency, cross-references, notation, and citations.

---

## FAQ

<details>
<summary><b>Why do you do all this?</b></summary>

<br>

When I began teaching machine learning systems, there was no textbook for the course. Books covered the models, and papers covered individual systems, but nothing taught the infrastructure underneath them as one subject. The field was changing too quickly for anyone to have written one. So I started writing it myself, because I have always learned best by building from scratch and teaching others.

When I want to understand an emerging or complex systems area, writing about it and writing code for it force me to slow down. They force me to find the principles that last, work through the math, and separate what is fundamentally true about physics and hardware from what is merely current.

This repository is my open workshop and teaching lab. It is where I work out ideas for my students at Harvard, my research lab, and open benchmarking efforts like MLPerf. Volumes III and IV are that process happening in public.

There is also a deeper reason. In the late 1990s, as an undergraduate, I wrote to a professor at Stanford because I was fascinated by genetic algorithms. I did not expect an answer. Then a brown package arrived at my house with a VHS tape on genetic algorithms inside, sent from the professor's office. I no longer remember the professor's name, so I never got to say thank you, but I have not forgotten what it felt like to have someone I had never met take the time to share what they knew with a curious student. That package sparked my curiosity, and the feeling has stayed with me.

Students write to me now the way I once wrote to that professor, far more than I can answer one by one. This curriculum is my way of sending the package back out, to a much bigger classroom than any lecture hall can hold.

</details>

<details>
<summary><b>Who is this for, and what should I know first?</b></summary>

<br>

It is for anyone who wants to understand how intelligent systems run, not only how models are trained: students, software engineers moving into ML infrastructure, hardware architects, and educators designing courses.

The only assumptions are that the reader can program in Python and has met basic machine learning ideas, such as a neural network or a loss function. The systems concepts are built from the ground up. No prior background in computer architecture, networking, or distributed systems is needed.

</details>

<details>
<summary><b>Do I need Volume I before Volume II?</b></summary>

<br>

No. The two volumes differ in scope, not in depth or rigor.

Volume I covers one machine with a handful of accelerators, from the memory hierarchy and kernel efficiency to quantization and on-device deployment. Volume II covers the fleet, with thousands of accelerators connected by high-speed networks, the parallelism that spans them, fault tolerance, and serving at datacenter scale.

Volume II does not assume that a reader has finished Volume I, so a reader with strong single-machine foundations can start there. Reading Volume I first gives the cleanest mental model from the silicon up, in the same way that Hennessy and Patterson's *Computer Organization and Design* precedes *Computer Architecture: A Quantitative Approach*.

</details>

<details>
<summary><b>Do I need TinyTorch, the labs, and the kits, or can I just read the book?</b></summary>

<br>

Each volume stands on its own as a textbook. The companions are there for readers who want to go further. Reading builds the mental model, and building and measuring make it last. Start with the book and add the hands-on pieces when an idea needs to become practice.

</details>

<details>
<summary><b>Isn't this just a deep learning book?</b></summary>

<br>

Books such as Goodfellow, Bengio, and Courville's *Deep Learning*, Bishop's texts, d2l.ai, and fast.ai teach the mathematics, architectures, and optimization of models, and they teach them well.

This curriculum starts where those books stop. It treats the model as one component inside a system that must ingest data, run on real silicon under power and thermal limits, answer queries within milliseconds, and keep running across thousands of machines. A reader can finish an excellent deep learning course and still not know why a cluster idles at 30% utilization, why serving memory grows with the KV cache, or how gradients are synchronized without stalling the network. Those questions are the subject here.

</details>

<details>
<summary><b>Isn't this MLOps, or the same as <i>Designing Machine Learning Systems</i>?</b></summary>

<br>

MLOps books, such as Chip Huyen's *Designing Machine Learning Systems*, focus on operational pipelines: feature stores, workflow orchestration, and deployment with current industry tools. They are valuable practical guides.

This curriculum focuses on what lies underneath: the hardware limits, quantitative trade-offs, and first principles that explain why those tools are designed as they are, and the limits no tool escapes, such as memory bandwidth, network latency, energy, and failure rates.

The two differ the way a recipe differs from an understanding of how heat, salt, fat, and time interact. A recipe works until the oven changes or the batch grows tenfold. Understanding the chemistry lets a cook work in any kitchen, fix a dish that went wrong, and invent new ones.

</details>

<details>
<summary><b>How is this different from <i>The Datacenter as a Warehouse-Scale Computer</i>?</b></summary>

<br>

Barroso, Hölzle, and Clidaras give a firsthand account of how one organization built hyperscale infrastructure, and I admire that work. This curriculum aims to complement it in three ways:

1. **A progressive curriculum, not a reference.** It starts from a single machine and its memory hierarchy (Volume I) and builds up to multi-node clusters and warehouse-scale fleets (Volume II), with quantitative exercises along the way.
2. **Vendor-neutral foundations.** It generalizes across accelerator architectures (GPUs, TPUs, custom ASICs) and deployment targets (cloud clusters to milliwatt microcontrollers), so the principles carry across hardware generations.
3. **Hands-on and living.** The text is surrounded by code readers build themselves (TinyTorch), open hardware kits, cluster modeling (MLSys·im), and community contributions, and it changes as the field does.

</details>

<details>
<summary><b>Do you use AI tools?</b></summary>

<br>

Yes, and I would rather say so plainly than leave readers to guess. I use generative AI tools to write code and the checks that test it, to survey the literature before I read the papers myself, to draft passages that I then rewrite, to help produce figures that I once drew by hand, and to audit the books for consistency. I use them on the companions as well, from TinyTorch and the labs to the slides and the StaffML question bank. They are also how I learn a new area quickly. The tools take on mechanical work, which leaves my time for the parts that need a teacher: what to include, how to explain it, and whether it is right.

A tool can write text that sounds right and is wrong, so I do not take its output on trust. I check the work in ways that do not depend on the tool. Every number in the books is computed by code, and automated checks run whenever something changes. Practitioners review the chapters, and students test them in class. In StaffML, each question records whether it began as a machine draft and whether a person has reviewed it.

I researched, wrote, and rewrote every chapter. The decisions about what goes in, what stays out, and how an idea is taught are mine, and so is every mistake. Because all of the work is open, anyone can check it and [tell me](https://github.com/harvard-edge/cs249r_book/issues/new/choose) what they find.

Volumes III and IV have not yet been through the classroom or practitioner review, which is why they are marked as drafts. Each volume includes a full [Note on AI Assistance](https://mlsysbook.ai/vol1/frontmatter/ai_use.html).

</details>

<details>
<summary><b>Why read a textbook in the age of LLMs?</b></summary>

<br>

Language models are good at retrieval, code generation, and summarizing facts. A paragraph that only delivers a fact a reader could look up in seconds has not earned its place in a textbook.

What a textbook provides is perspective and a coherent mental model: the order in which to learn ideas, how trade-offs constrain one another, and what not to worry about. Deciding what to leave out is one of the most important choices an author makes.

Bruce Davie made this point in his essay [“Textbooks in Tokenland”](https://systemsapproach.org/2026/06/01/textbooks-in-tokenland/) (Systems Approach). A language model generates text without communicative intent, he writes, while a textbook is a deliberate path from a teacher to a student toward a shared understanding of how the world works.

The two are not in opposition. The book provides the structure and the quantitative argument, and interactive tools let a reader test that understanding. The Marimo labs ask a reader to predict an outcome and then check it, and the SocratiQ tutor answers questions about the page being read.

</details>

<details>
<summary><b>Is it free, and how do I read it?</b></summary>

<br>

Yes. Every volume is free to read online at [mlsysbook.ai](https://mlsysbook.ai).

The textbook is released under a Creative Commons license ([CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)), the same license as the MIT Press print edition. It may be read and shared for non-commercial use with attribution. Translations and other modified versions need permission. The companion software is open source under its own licenses, listed under [License](#license).

</details>

---

## Support This Work

If this curriculum has helped you, there are two simple ways to help it grow.

<p align="center">
  <a href="https://github.com/harvard-edge/cs249r_book/stargazers"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.github.com%2Frepos%2Fharvard-edge%2Fcs249r_book&query=%24.stargazers_count&label=Stars&logo=github&color=gold&style=for-the-badge&cacheSeconds=3600" alt="GitHub stars"></a>
  &nbsp;&nbsp;
  <a href="https://opencollective.com/mlsysbook"><img src="https://img.shields.io/badge/Fund-Open%20Collective-blue.svg?style=for-the-badge&logo=open-collective" alt="Open Collective"></a>
</p>

<table width="100%" style="width:100%">
  <tbody>
    <tr>
      <td width="50%" align="center" valign="top">
        <b>Star the repository</b><br>
        A star takes one click, and each one tells universities, foundations, and other learners that people want to study how machine learning systems are built. That helps AI engineering earn its place in the curriculum, and the support of those institutions pays for workshops and hardware kits in classrooms that could not otherwise afford them.<br><br>
        Stars matter to me as well. This is a long project, and the new stars I see each morning are what keep me going, because each one tells me that a student somewhere is learning from this work.<br><br>
        <a href="https://star-history.com/#harvard-edge/cs249r_book&Date">
          <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=harvard-edge/cs249r_book&type=date&theme=dark&legend=top-left">
            <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=harvard-edge/cs249r_book&type=date&legend=top-left">
            <img src="https://api.star-history.com/chart?repos=harvard-edge/cs249r_book&type=date&legend=top-left" alt="Star History Chart" width="400">
          </picture>
        </a>
      </td>
      <td width="50%" align="center" valign="top">
        <b>Fund the work</b><br>
        Contributions go to <a href="https://opencollective.com/mlsysbook">Open Collective</a>, a transparent fund for educational outreach where every expense is public. The money goes back into the work, to reach more students and to improve the material they learn from.<br><br>
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
      <th width="20%">Area</th>
      <th width="50%">What you can do</th>
      <th width="25%">Where to start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">📖</td>
      <td><b>Textbook</b></td>
      <td>Fix an error, clarify an explanation, or suggest material for any chapter.</td>
      <td><a href="binder/docs/CONTRIBUTING.md">Textbook guide</a></td>
    </tr>
    <tr>
      <td align="center">🔥</td>
      <td><b>TinyTorch</b></td>
      <td>Add a module, fix a bug, or improve the tests and documentation.</td>
      <td><a href="tinytorch/CONTRIBUTING.md">TinyTorch guide</a></td>
    </tr>
    <tr>
      <td align="center">🛠️</td>
      <td><b>Hardware Kits</b></td>
      <td>Improve the deployment guides, examples, and board support.</td>
      <td><a href="kits/README.md">Kits guide</a></td>
    </tr>
    <tr>
      <td align="center">🔬</td>
      <td><b>Labs and MLSys·im</b></td>
      <td>Improve the interactive labs or the simulator&#39;s hardware and model data.</td>
      <td><a href="labs/README.md">Labs guide</a> · <a href="mlsysim/README.md">MLSys·im guide</a></td>
    </tr>
    <tr>
      <td align="center">💼</td>
      <td><b>StaffML</b></td>
      <td>Improve the systems design questions, answers, and interview practice.</td>
      <td><a href="staffml/README.md">StaffML guide</a></td>
    </tr>
    <tr>
      <td align="center">🧠</td>
      <td><b>SocratiQ</b></td>
      <td>Improve the tutor, its quizzes, and its prompts.</td>
      <td><a href="socratiq/README.md">SocratiQ guide</a></td>
    </tr>
    <tr>
      <td align="center">🐛</td>
      <td><b>Report an issue</b></td>
      <td>Point out an error, a broken link, or anything unclear.</td>
      <td><a href="https://github.com/harvard-edge/cs249r_book/issues">GitHub Issues</a></td>
    </tr>
    <tr>
      <td align="center">💬</td>
      <td><b>Ask a question</b></td>
      <td>Ask about the material or share what you are building with it.</td>
      <td><a href="https://github.com/harvard-edge/cs249r_book/discussions">GitHub Discussions</a></td>
    </tr>
  </tbody>
</table>

**Branches.** Development happens on `dev`, which this page shows. The `main` branch holds what is live on [mlsysbook.ai](https://mlsysbook.ai), and the publish workflow promotes `dev` to `main` at each release.

---

## License

This is a multi-component repository, and **each component is released under its own license** to match its purpose. The license file inside each directory, such as `tinytorch/LICENSE` or `staffml/app/LICENSE`, is authoritative.

<table width="100%" style="width:100%">
  <thead>
    <tr>
      <th width="5%"></th>
      <th width="30%">Component</th>
      <th width="15%">License</th>
      <th width="50%">What it means</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">📖</td>
      <td><b>Textbook</b><br><code>books/</code></td>
      <td><a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">CC BY-NC-ND 4.0</a></td>
      <td>The license of the MIT Press edition. Share for <b>non-commercial</b> use with attribution. Modified versions, including translations, need permission.</td>
    </tr>
    <tr>
      <td align="center">🧰</td>
      <td><b>Labs, Kits, Slides, Instructor Hub</b><br><code>labs/</code> · <code>kits/</code> · <code>slides/</code> · <code>instructors/</code></td>
      <td><a href="LICENSE.md">CC BY-NC-SA 4.0</a></td>
      <td>Share and adapt for <b>non-commercial</b> use, with attribution and same-license sharing.</td>
    </tr>
    <tr>
      <td align="center">🔥</td>
      <td><a href="tinytorch/LICENSE"><b>TinyTorch</b></a></td>
      <td>MIT</td>
      <td>Permissive. Use, modify, and redistribute, including commercially.</td>
    </tr>
    <tr>
      <td align="center">🔮</td>
      <td><a href="mlsysim/LICENSE.md"><b>MLSys·im</b></a></td>
      <td>Apache 2.0</td>
      <td>Permissive with explicit patent grant.</td>
    </tr>
    <tr>
      <td align="center">💼</td>
      <td><a href="staffml/app/LICENSE"><b>StaffML</b></a></td>
      <td>AGPL v3</td>
      <td>Strong copyleft. Modifications to deployed services must be published. Commercial licenses are available from the authors.</td>
    </tr>
    <tr>
      <td align="center">🗃️</td>
      <td><a href="staffml/vault/questions/LICENSE"><b>StaffML question corpus</b></a></td>
      <td>CC BY-NC 4.0</td>
      <td>Research and educational use. Commercial use requires permission.</td>
    </tr>
    <tr>
      <td align="center">🔢</td>
      <td><a href="tinytorch/LICENSE"><b>TinyDigits dataset</b></a></td>
      <td>BSD 3-Clause</td>
      <td>Permissive (matches sklearn ancestry).</td>
    </tr>
    <tr>
      <td align="center">🗣️</td>
      <td><a href="tinytorch/LICENSE"><b>TinyTalks dataset</b></a></td>
      <td>CC BY 4.0</td>
      <td>Permissive with attribution. Commercial use is allowed.</td>
    </tr>
  </tbody>
</table>

A shorter summary for readers is at [mlsysbook.ai/about/license](https://mlsysbook.ai/about/license.html).

If you are an institution considering adoption, or a company interested in commercial terms for a copyleft component, please write to [edu@tinyML.org](mailto:edu@tinyML.org).

---

## Acknowledgments

I assembled this curriculum, but I did not invent most of what it teaches. The ideas come from the engineers and practitioners in industry who build these systems every day, and from researchers worldwide who study them and publish what they learn. The books cite their work throughout, and the curriculum would not exist without it. My own research lab has contributed as well.

Many people also gave their time to the curriculum itself. My students in CS249r at Harvard have tested every revision. Reviewers in industry and academia read chapters and told me where I was wrong. Instructors who adopted the material reported what worked in their classrooms and what did not. Readers found errors, and contributors fixed them. The financial contributors on Open Collective helped carry the work to more students.

Whatever is good here owes a great deal to all of them. The mistakes are mine.

---

## Contributors

These people have improved the textbooks, the software, and the teaching materials.

**Legend:** 🪲 Bug Hunter · 🧑‍💻 Code Contributor · ✍️ Doc Wizard · 🎨 Design Artist · 🧠 Idea Spark · 🔎 Code Reviewer · 🧪 Test Tinkerer · 🛠️ Tool Builder

### 📖 Textbook Contributors

<!-- BOOK-CONTRIBUTORS-START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table width="100%" style="width:100%">
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧠 🔎 🧪 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/hzeljko"><img src="https://avatars.githubusercontent.com/hzeljko?v=4?s=50" width="50px;" alt="Zeljko Hrcek"/><br /><sub><b>Zeljko Hrcek</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Shashank-Tripathi-07"><img src="https://avatars.githubusercontent.com/u/178375647?v=4?v=4?s=50" width="50px;" alt="Rocky"/><br /><sub><b>Rocky</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/farhan523"><img src="https://avatars.githubusercontent.com/u/62025759?v=4?v=4?s=50" width="50px;" alt="Farhan Asghar"/><br /><sub><b>Farhan Asghar</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Mjrovai"><img src="https://avatars.githubusercontent.com/Mjrovai?v=4?s=50" width="50px;" alt="Marcelo Rovai"/><br /><sub><b>Marcelo Rovai</b></sub></a><br />🧑‍💻 🎨 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/oamazonasgabriel"><img src="https://avatars.githubusercontent.com/oamazonasgabriel?v=4?s=50" width="50px;" alt="Gabriel Amazonas"/><br /><sub><b>Gabriel Amazonas</b></sub></a><br />🪲 ✍️ 🧠</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/octo-patch"><img src="https://avatars.githubusercontent.com/u/266937838?v=4?v=4?s=50" width="50px;" alt="Octopus"/><br /><sub><b>Octopus</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Tess314"><img src="https://avatars.githubusercontent.com/Tess314?v=4?s=50" width="50px;" alt="Tess Watt"/><br /><sub><b>Tess Watt</b></sub></a><br />🪲 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Jahnic-kb"><img src="https://avatars.githubusercontent.com/Jahnic-kb?v=4?s=50" width="50px;" alt="Jahnic Beck"/><br /><sub><b>Jahnic Beck</b></sub></a><br />🪲 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/bluebaer7"><img src="https://avatars.githubusercontent.com/u/44038795?v=4?v=4?s=50" width="50px;" alt="bluebaer"/><br /><sub><b>bluebaer</b></sub></a><br />🪲 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/kai4avaya"><img src="https://avatars.githubusercontent.com/kai4avaya?v=4?s=50" width="50px;" alt="Kai Kleinbard"/><br /><sub><b>Kai Kleinbard</b></sub></a><br />🧑‍💻 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/didier-durand"><img src="https://avatars.githubusercontent.com/didier-durand?v=4?s=50" width="50px;" alt="Didier Durand"/><br /><sub><b>Didier Durand</b></sub></a><br />✍️ 🪲</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/salmanmkc"><img src="https://avatars.githubusercontent.com/u/32169182?v=4?v=4?s=50" width="50px;" alt="Salman Chishti"/><br /><sub><b>Salman Chishti</b></sub></a><br />🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/GronuJ"><img src="https://avatars.githubusercontent.com/u/152858896?v=4?v=4?s=50" width="50px;" alt="GronuJ"/><br /><sub><b>GronuJ</b></sub></a><br />🪲 ✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/gustaf-hammarberg"><img src="https://avatars.githubusercontent.com/u/46937826?v=4?v=4?s=50" width="50px;" alt="Gustaf Hammarberg"/><br /><sub><b>Gustaf Hammarberg</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/obsessor-ak1"><img src="https://avatars.githubusercontent.com/u/56510780?v=4?v=4?s=50" width="50px;" alt="Aayush Kumar"/><br /><sub><b>Aayush Kumar</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/tddg"><img src="https://avatars.githubusercontent.com/u/1696190?v=4?v=4?s=50" width="50px;" alt="Yue Cheng"/><br /><sub><b>Yue Cheng</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/s3rg3ysh"><img src="https://avatars.githubusercontent.com/u/122926347?v=4?v=4?s=50" width="50px;" alt="s3rg3ysh"/><br /><sub><b>s3rg3ysh</b></sub></a><br />🪲 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/JOhnsonKC201"><img src="https://avatars.githubusercontent.com/u/176641612?v=4?v=4?s=50" width="50px;" alt="Johnson K C"/><br /><sub><b>Johnson K C</b></sub></a><br />🧑‍💻 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/MBK-fr"><img src="https://avatars.githubusercontent.com/u/41642346?v=4?v=4?s=50" width="50px;" alt="mehdi B"/><br /><sub><b>mehdi B</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/ShivtejG236"><img src="https://avatars.githubusercontent.com/u/91597404?v=4?v=4?s=50" width="50px;" alt="Shivtej Gaikwad"/><br /><sub><b>Shivtej Gaikwad</b></sub></a><br />🪲 🧑‍💻</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jasonjabbour"><img src="https://avatars.githubusercontent.com/jasonjabbour?v=4?s=50" width="50px;" alt="Jason Jabbour"/><br /><sub><b>Jason Jabbour</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/uchendui"><img src="https://avatars.githubusercontent.com/uchendui?v=4?s=50" width="50px;" alt="Ikechukwu Uchendu"/><br /><sub><b>Ikechukwu Uchendu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Naeemkh"><img src="https://avatars.githubusercontent.com/Naeemkh?v=4?s=50" width="50px;" alt="Naeem Khoshnevis"/><br /><sub><b>Naeem Khoshnevis</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Sara-Khosravi"><img src="https://avatars.githubusercontent.com/Sara-Khosravi?v=4?s=50" width="50px;" alt="Sara Khosravi"/><br /><sub><b>Sara Khosravi</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/V0XNIHILI"><img src="https://avatars.githubusercontent.com/V0XNIHILI?v=4?s=50" width="50px;" alt="Douwe den Blanken"/><br /><sub><b>Douwe den Blanken</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/18jeffreyma"><img src="https://avatars.githubusercontent.com/18jeffreyma?v=4?s=50" width="50px;" alt="Jeffrey Ma"/><br /><sub><b>Jeffrey Ma</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/shanzehbatool"><img src="https://avatars.githubusercontent.com/shanzehbatool?v=4?s=50" width="50px;" alt="shanzehbatool"/><br /><sub><b>shanzehbatool</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/eliasab16"><img src="https://avatars.githubusercontent.com/eliasab16?v=4?s=50" width="50px;" alt="Elias"/><br /><sub><b>Elias</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/JaredP94"><img src="https://avatars.githubusercontent.com/JaredP94?v=4?s=50" width="50px;" alt="Jared Ping"/><br /><sub><b>Jared Ping</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/ishapira1"><img src="https://avatars.githubusercontent.com/ishapira1?v=4?s=50" width="50px;" alt="Itai Shapira"/><br /><sub><b>Itai Shapira</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/8863743b4f26c1a20e730fcf7ebc3bc0?d=identicon&s=100?v=4?s=50" width="50px;" alt="Maximilian Lam"/><br /><sub><b>Maximilian Lam</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jaysonzlin"><img src="https://avatars.githubusercontent.com/jaysonzlin?v=4?s=50" width="50px;" alt="Jayson Lin"/><br /><sub><b>Jayson Lin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/sophiacho1"><img src="https://avatars.githubusercontent.com/sophiacho1?v=4?s=50" width="50px;" alt="Sophia Cho"/><br /><sub><b>Sophia Cho</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/andreamurillomtz"><img src="https://avatars.githubusercontent.com/andreamurillomtz?v=4?s=50" width="50px;" alt="Andrea"/><br /><sub><b>Andrea</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/alxrod"><img src="https://avatars.githubusercontent.com/alxrod?v=4?s=50" width="50px;" alt="Alex Rodriguez"/><br /><sub><b>Alex Rodriguez</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/korneelf1"><img src="https://avatars.githubusercontent.com/korneelf1?v=4?s=50" width="50px;" alt="Korneel Van den Berghe"/><br /><sub><b>Korneel Van den Berghe</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/foundingnimo"><img src="https://avatars.githubusercontent.com/foundingnimo?v=4?s=50" width="50px;" alt="Nimo"/><br /><sub><b>Nimo</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/colbybanbury"><img src="https://avatars.githubusercontent.com/colbybanbury?v=4?s=50" width="50px;" alt="Colby Banbury"/><br /><sub><b>Colby Banbury</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/zishenwan"><img src="https://avatars.githubusercontent.com/zishenwan?v=4?s=50" width="50px;" alt="Zishen Wan"/><br /><sub><b>Zishen Wan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/mmaz"><img src="https://avatars.githubusercontent.com/mmaz?v=4?s=50" width="50px;" alt="Mark Mazumder"/><br /><sub><b>Mark Mazumder</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/ma3mool"><img src="https://avatars.githubusercontent.com/ma3mool?v=4?s=50" width="50px;" alt="Abdulrahman Mahmoud"/><br /><sub><b>Abdulrahman Mahmoud</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/DivyaAmirtharaj"><img src="https://avatars.githubusercontent.com/DivyaAmirtharaj?v=4?s=50" width="50px;" alt="Divya Amirtharaj"/><br /><sub><b>Divya Amirtharaj</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/srivatsankrishnan"><img src="https://avatars.githubusercontent.com/srivatsankrishnan?v=4?s=50" width="50px;" alt="Srivatsan Krishnan"/><br /><sub><b>Srivatsan Krishnan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/arnaumarin"><img src="https://avatars.githubusercontent.com/arnaumarin?v=4?s=50" width="50px;" alt="marin-llobet"/><br /><sub><b>marin-llobet</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/aptl26"><img src="https://avatars.githubusercontent.com/aptl26?v=4?s=50" width="50px;" alt="Aghyad Deeb"/><br /><sub><b>Aghyad Deeb</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/James-QiuHaoran"><img src="https://avatars.githubusercontent.com/James-QiuHaoran?v=4?s=50" width="50px;" alt="Haoran Qiu"/><br /><sub><b>Haoran Qiu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Ekhao"><img src="https://avatars.githubusercontent.com/Ekhao?v=4?s=50" width="50px;" alt="Emil Njor"/><br /><sub><b>Emil Njor</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/ELSuitorHarvard"><img src="https://avatars.githubusercontent.com/ELSuitorHarvard?v=4?s=50" width="50px;" alt="ELSuitorHarvard"/><br /><sub><b>ELSuitorHarvard</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/kaiM0ves"><img src="https://avatars.githubusercontent.com/kaiM0ves?v=4?s=50" width="50px;" alt="kaiM0ves"/><br /><sub><b>kaiM0ves</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/oishib"><img src="https://avatars.githubusercontent.com/oishib?v=4?s=50" width="50px;" alt="oishib"/><br /><sub><b>oishib</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jared-ni"><img src="https://avatars.githubusercontent.com/jared-ni?v=4?s=50" width="50px;" alt="Jared Ni"/><br /><sub><b>Jared Ni</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/AditiR-42"><img src="https://avatars.githubusercontent.com/AditiR-42?v=4?s=50" width="50px;" alt="Aditi Raju"/><br /><sub><b>Aditi Raju</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/MichaelSchnebly"><img src="https://avatars.githubusercontent.com/MichaelSchnebly?v=4?s=50" width="50px;" alt="Michael Schnebly"/><br /><sub><b>Michael Schnebly</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/VThuong99"><img src="https://avatars.githubusercontent.com/VThuong99?v=4?s=50" width="50px;" alt="Thuong Duong"/><br /><sub><b>Thuong Duong</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/leo47007"><img src="https://avatars.githubusercontent.com/leo47007?v=4?s=50" width="50px;" alt="Yu-Shun Hsiao"/><br /><sub><b>Yu-Shun Hsiao</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/BaeHenryS"><img src="https://avatars.githubusercontent.com/BaeHenryS?v=4?s=50" width="50px;" alt="Henry Bae"/><br /><sub><b>Henry Bae</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/eimlav"><img src="https://avatars.githubusercontent.com/eimlav?v=4?s=50" width="50px;" alt="Eimhin Laverty"/><br /><sub><b>Eimhin Laverty</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jaywonchung"><img src="https://avatars.githubusercontent.com/jaywonchung?v=4?s=50" width="50px;" alt="Jae-Won Chung"/><br /><sub><b>Jae-Won Chung</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/ShvetankPrakash"><img src="https://avatars.githubusercontent.com/ShvetankPrakash?v=4?s=50" width="50px;" alt="Shvetank Prakash"/><br /><sub><b>Shvetank Prakash</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/marcozennaro"><img src="https://avatars.githubusercontent.com/marcozennaro?v=4?s=50" width="50px;" alt="Marco Zennaro"/><br /><sub><b>Marco Zennaro</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/aryatschand"><img src="https://avatars.githubusercontent.com/aryatschand?v=4?s=50" width="50px;" alt="Arya Tschand"/><br /><sub><b>Arya Tschand</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/arbass22"><img src="https://avatars.githubusercontent.com/arbass22?v=4?s=50" width="50px;" alt="Andrew Bass"/><br /><sub><b>Andrew Bass</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/pongtr"><img src="https://avatars.githubusercontent.com/pongtr?v=4?s=50" width="50px;" alt="Pong Trairatvorakul"/><br /><sub><b>Pong Trairatvorakul</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/euranofshin"><img src="https://avatars.githubusercontent.com/euranofshin?v=4?s=50" width="50px;" alt="Eura Nofshin"/><br /><sub><b>Eura Nofshin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/0c931fcfd03cd548d44c90602dd773ba?d=identicon&s=100?v=4?s=50" width="50px;" alt="Matthew Stewart"/><br /><sub><b>Matthew Stewart</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/af39c27c6090c50a1921a9b6366e81cc?d=identicon&s=100?v=4?s=50" width="50px;" alt="Emeka Ezike"/><br /><sub><b>Emeka Ezike</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jianqingdu"><img src="https://avatars.githubusercontent.com/jianqingdu?v=4?s=50" width="50px;" alt="jianqingdu"/><br /><sub><b>jianqingdu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jzhou1318"><img src="https://avatars.githubusercontent.com/jzhou1318?v=4?s=50" width="50px;" alt="Jennifer Zhou"/><br /><sub><b>Jennifer Zhou</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/vitasam"><img src="https://avatars.githubusercontent.com/vitasam?v=4?s=50" width="50px;" alt="The Random DIY"/><br /><sub><b>The Random DIY</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/468ef35acc69f3266efd700992daa369?d=identicon&s=100?v=4?s=50" width="50px;" alt="Fatima Shah"/><br /><sub><b>Fatima Shah</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/BrunoScaglione"><img src="https://avatars.githubusercontent.com/BrunoScaglione?v=4?s=50" width="50px;" alt="Bruno Scaglione"/><br /><sub><b>Bruno Scaglione</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Allen-Kuang"><img src="https://avatars.githubusercontent.com/Allen-Kuang?v=4?s=50" width="50px;" alt="Allen-Kuang"/><br /><sub><b>Allen-Kuang</b></sub></a><br />✍️</td>
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
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/aethernavshulkraven-allain"><img src="https://avatars.githubusercontent.com/aethernavshulkraven-allain?v=4?s=50" width="50px;" alt="अरनव शुक्ला | Arnav Shukla"/><br /><sub><b>अरनव शुक्ला | Arnav Shukla</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/RinZ27"><img src="https://avatars.githubusercontent.com/RinZ27?v=4?s=50" width="50px;" alt="Rin"/><br /><sub><b>Rin</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/bilgeacun"><img src="https://avatars.githubusercontent.com/bilgeacun?v=4?s=50" width="50px;" alt="Bilge Acun"/><br /><sub><b>Bilge Acun</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/atcheng2"><img src="https://avatars.githubusercontent.com/atcheng2?v=4?s=50" width="50px;" alt="Andy Cheng"/><br /><sub><b>Andy Cheng</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/arighosh05"><img src="https://avatars.githubusercontent.com/arighosh05?v=4?s=50" width="50px;" alt="Aritra Ghosh"/><br /><sub><b>Aritra Ghosh</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/abigailswallow"><img src="https://avatars.githubusercontent.com/abigailswallow?v=4?s=50" width="50px;" alt="abigailswallow"/><br /><sub><b>abigailswallow</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/YangZhou1997"><img src="https://avatars.githubusercontent.com/YangZhou1997?v=4?s=50" width="50px;" alt="Yang Zhou"/><br /><sub><b>Yang Zhou</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/XaicuL"><img src="https://avatars.githubusercontent.com/XaicuL?v=4?s=50" width="50px;" alt="JEON HYUNJUN(Luciano)"/><br /><sub><b>JEON HYUNJUN(Luciano)</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/emmanuel2406"><img src="https://avatars.githubusercontent.com/emmanuel2406?v=4?s=50" width="50px;" alt="Emmanuel Rassou"/><br /><sub><b>Emmanuel Rassou</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jasonlyik"><img src="https://avatars.githubusercontent.com/jasonlyik?v=4?s=50" width="50px;" alt="Jason Yik"/><br /><sub><b>Jason Yik</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jessicaquaye"><img src="https://avatars.githubusercontent.com/jessicaquaye?v=4?s=50" width="50px;" alt="Jessica Quaye"/><br /><sub><b>Jessica Quaye</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/cursoragent"><img src="https://avatars.githubusercontent.com/cursoragent?v=4?s=50" width="50px;" alt="Cursor Agent"/><br /><sub><b>Cursor Agent</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/happyappledog"><img src="https://avatars.githubusercontent.com/happyappledog?v=4?s=50" width="50px;" alt="happyappledog"/><br /><sub><b>happyappledog</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/snuggs"><img src="https://avatars.githubusercontent.com/snuggs?v=4?s=50" width="50px;" alt="Snuggs"/><br /><sub><b>Snuggs</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/swilcock0"><img src="https://avatars.githubusercontent.com/swilcock0?v=4?s=50" width="50px;" alt="Sam Wilcock"/><br /><sub><b>Sam Wilcock</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/sjohri20"><img src="https://avatars.githubusercontent.com/sjohri20?v=4?s=50" width="50px;" alt="Shreya Johri"/><br /><sub><b>Shreya Johri</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/skmur"><img src="https://avatars.githubusercontent.com/skmur?v=4?s=50" width="50px;" alt="Sonia Murthy"/><br /><sub><b>Sonia Murthy</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/fc4f3460cdfb9365ab59bdeafb06413e?d=identicon&s=100?v=4?s=50" width="50px;" alt="Costin-Andrei Oncescu"/><br /><sub><b>Costin-Andrei Oncescu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/0d6b8616427d8b19d425c9808692e347?d=identicon&s=100?v=4?s=50" width="50px;" alt="formlsysbookissue"/><br /><sub><b>formlsysbookissue</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/7cd8d5dfd83071f23979019d97655dc5?d=identicon&s=100?v=4?s=50" width="50px;" alt="Annie Laurie Cook"/><br /><sub><b>Annie Laurie Cook</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/5aa037840c0ca11ee42784ed4843c655?d=identicon&s=100?v=4?s=50" width="50px;" alt="Parampreet Singh"/><br /><sub><b>Parampreet Singh</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/b15b6e0e9adf58099905c1a0fd474cb9?d=identicon&s=100?v=4?s=50" width="50px;" alt="Vijay Edupuganti"/><br /><sub><b>Vijay Edupuganti</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/f88052cca4f401d9b0f43aed0a53434a?d=identicon&s=100?v=4?s=50" width="50px;" alt="Jothi Ramaswamy"/><br /><sub><b>Jothi Ramaswamy</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/35a8d9ffd03f05e79a2c6ce6206a56f2?d=identicon&s=100?v=4?s=50" width="50px;" alt="Batur Arslan"/><br /><sub><b>Batur Arslan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/bd53d146aa888548c8db4da02bf81e7a?d=identicon&s=100?v=4?s=50" width="50px;" alt="Curren Iyer"/><br /><sub><b>Curren Iyer</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/8d8410338458e08bd5e4b96f58e1c217?d=identicon&s=100?v=4?s=50" width="50px;" alt="Edward Jin"/><br /><sub><b>Edward Jin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/f5d58ba6aa9b00189d4c018d370e8f43?d=identicon&s=100?v=4?s=50" width="50px;" alt="yanjingl"/><br /><sub><b>yanjingl</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/a5a47df988ab1720dd706062e523ca32?d=identicon&s=100?v=4?s=50" width="50px;" alt="a-saraf"/><br /><sub><b>a-saraf</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/c2dc311aa8122d5f5f061e1db14682b1?d=identicon&s=100?v=4?s=50" width="50px;" alt="songhan"/><br /><sub><b>songhan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/4814aad67982ab07a69006a1ce9d2a72?d=identicon&s=100?v=4?s=50" width="50px;" alt="jvijay"/><br /><sub><b>jvijay</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/43b1feff77c8a95fd581774fb8ec891f?d=identicon&s=100?v=4?s=50" width="50px;" alt="Zishen"/><br /><sub><b>Zishen</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/BunningsWarehouseOfficial"><img src="https://avatars.githubusercontent.com/u/49220945?v=4?v=4?s=50" width="50px;" alt="Kristian Radoš"/><br /><sub><b>Kristian Radoš</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/minhdang26403"><img src="https://avatars.githubusercontent.com/u/86156224?v=4?v=4?s=50" width="50px;" alt="Dang Truong"/><br /><sub><b>Dang Truong</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/pipme"><img src="https://avatars.githubusercontent.com/pipme?v=4?s=50" width="50px;" alt="pipme"/><br /><sub><b>pipme</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/paolo-estavillo"><img src="https://avatars.githubusercontent.com/u/95209078?v=4?v=4?s=50" width="50px;" alt="Paolo Estavillo"/><br /><sub><b>Paolo Estavillo</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Pratham-ja"><img src="https://avatars.githubusercontent.com/u/114498234?v=4?v=4?s=50" width="50px;" alt="Pratham Chaudhary"/><br /><sub><b>Pratham Chaudhary</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/arulk"><img src="https://avatars.githubusercontent.com/u/616070?v=4?v=4?s=50" width="50px;" alt="Arul Kumaravel"/><br /><sub><b>Arul Kumaravel</b></sub></a><br />🪲</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/bl00dny"><img src="https://avatars.githubusercontent.com/u/106136390?v=4?v=4?s=50" width="50px;" alt="STG"/><br /><sub><b>STG</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/xizhuomengcontin"><img src="https://avatars.githubusercontent.com/u/319346922?v=4?v=4?s=50" width="50px;" alt="Xizhuo"/><br /><sub><b>Xizhuo</b></sub></a><br />🧠</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Ben-Home"><img src="https://avatars.githubusercontent.com/u/238533482?v=4?v=4?s=50" width="50px;" alt="CorpusIQ"/><br /><sub><b>CorpusIQ</b></sub></a><br />🧠</td>
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
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/SatnamCodes"><img src="https://avatars.githubusercontent.com/u/139437542?v=4?v=4?s=50" width="50px;" alt="Satnam Singh"/><br /><sub><b>Satnam Singh</b></sub></a><br />🧑‍💻 ✍️ 🧪</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/karthikdani"><img src="https://avatars.githubusercontent.com/karthikdani?v=4?s=50" width="50px;" alt="Karthik Dani"/><br /><sub><b>Karthik Dani</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/avikde"><img src="https://avatars.githubusercontent.com/avikde?v=4?s=50" width="50px;" alt="Avik De"/><br /><sub><b>Avik De</b></sub></a><br />🪲 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Takosaga"><img src="https://avatars.githubusercontent.com/Takosaga?v=4?s=50" width="50px;" alt="Takosaga"/><br /><sub><b>Takosaga</b></sub></a><br />🪲 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/joeswagson"><img src="https://avatars.githubusercontent.com/joeswagson?v=4?s=50" width="50px;" alt="joeswagson"/><br /><sub><b>joeswagson</b></sub></a><br />🧑‍💻 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Roldao-Neto"><img src="https://avatars.githubusercontent.com/u/148023227?v=4?v=4?s=50" width="50px;" alt="Rolds"/><br /><sub><b>Rolds</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/AdemolaAri"><img src="https://avatars.githubusercontent.com/u/49918815?v=4?v=4?s=50" width="50px;" alt="Ademola Arigbabuwo"/><br /><sub><b>Ademola Arigbabuwo</b></sub></a><br />🪲 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/harishb00"><img src="https://avatars.githubusercontent.com/u/43300971?v=4?v=4?s=50" width="50px;" alt="Harish"/><br /><sub><b>Harish</b></sub></a><br />🧑‍💻 ✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Chufeng-Jiang"><img src="https://avatars.githubusercontent.com/u/80246982?v=4?v=4?s=50" width="50px;" alt="Chufeng JIANG"/><br /><sub><b>Chufeng JIANG</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Fabio-RibeiroB"><img src="https://avatars.githubusercontent.com/u/74654489?v=4?v=4?s=50" width="50px;" alt="Fábio"/><br /><sub><b>Fábio</b></sub></a><br />🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/aadityansha06"><img src="https://avatars.githubusercontent.com/u/96714228?v=4?v=4?s=50" width="50px;" alt="Aadityansha "/><br /><sub><b>Aadityansha </b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/euwyngoh"><img src="https://avatars.githubusercontent.com/u/211522196?v=4?v=4?s=50" width="50px;" alt="euwyngoh"/><br /><sub><b>euwyngoh</b></sub></a><br />🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/pratyushadk"><img src="https://avatars.githubusercontent.com/u/184196841?v=4?v=4?s=50" width="50px;" alt="Pratyush Adhikari"/><br /><sub><b>Pratyush Adhikari</b></sub></a><br />🧑‍💻 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/diegosaldonid-lab"><img src="https://avatars.githubusercontent.com/u/301918395?v=4?v=4?s=50" width="50px;" alt="diegosaldonid-lab"/><br /><sub><b>diegosaldonid-lab</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/choyrim"><img src="https://avatars.githubusercontent.com/u/380261?v=4?v=4?s=50" width="50px;" alt="Choy Rim"/><br /><sub><b>Choy Rim</b></sub></a><br />🪲 🧑‍💻</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/hiimnhan"><img src="https://avatars.githubusercontent.com/u/35358825?v=4?v=4?s=50" width="50px;" alt="Nhan Nguyen"/><br /><sub><b>Nhan Nguyen</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/ehhholi"><img src="https://avatars.githubusercontent.com/u/159473234?v=4?v=4?s=50" width="50px;" alt="ehhholi"/><br /><sub><b>ehhholi</b></sub></a><br />✍️ 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/AmirAlasady"><img src="https://avatars.githubusercontent.com/AmirAlasady?v=4?s=50" width="50px;" alt="Amir Alasady"/><br /><sub><b>Amir Alasady</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/jettythek"><img src="https://avatars.githubusercontent.com/jettythek?v=4?s=50" width="50px;" alt="jettythek"/><br /><sub><b>jettythek</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/wz1114841863"><img src="https://avatars.githubusercontent.com/wz1114841863?v=4?s=50" width="50px;" alt="wzz"/><br /><sub><b>wzz</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/keo-dara"><img src="https://avatars.githubusercontent.com/u/175544368?v=4?v=4?s=50" width="50px;" alt="keo-dara"/><br /><sub><b>keo-dara</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Kobra299"><img src="https://avatars.githubusercontent.com/u/4283156?v=4?v=4?s=50" width="50px;" alt="Wayne Norman"/><br /><sub><b>Wayne Norman</b></sub></a><br />🪲</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/lalalostcode"><img src="https://avatars.githubusercontent.com/u/149884766?v=4?v=4?s=50" width="50px;" alt="Ilham Rafiqin"/><br /><sub><b>Ilham Rafiqin</b></sub></a><br />🪲</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/oscarf189"><img src="https://avatars.githubusercontent.com/u/28113740?v=4?v=4?s=50" width="50px;" alt="Oscar Flores"/><br /><sub><b>Oscar Flores</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/sotoblanco"><img src="https://avatars.githubusercontent.com/u/46135649?v=4?v=4?s=50" width="50px;" alt="Pastor Soto"/><br /><sub><b>Pastor Soto</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/salmanmkc"><img src="https://avatars.githubusercontent.com/u/32169182?v=4?v=4?s=50" width="50px;" alt="Salman Chishti"/><br /><sub><b>Salman Chishti</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/adityamulik"><img src="https://avatars.githubusercontent.com/u/10626835?v=4?v=4?s=50" width="50px;" alt="Aditya Mulik"/><br /><sub><b>Aditya Mulik</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/yarikoptic"><img src="https://avatars.githubusercontent.com/u/39889?v=4?v=4?s=50" width="50px;" alt="Yaroslav Halchenko"/><br /><sub><b>Yaroslav Halchenko</b></sub></a><br />🧑‍💻</td>
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
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Shakhtar-Sankur"><img src="https://avatars.githubusercontent.com/u/312389772?v=4?v=4?s=50" width="50px;" alt="Sankur Kundu"/><br /><sub><b>Sankur Kundu</b></sub></a><br />🧑‍💻 ✍️ 🧠 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/aadityansha06"><img src="https://avatars.githubusercontent.com/u/96714228?v=4?v=4?s=50" width="50px;" alt="Aadityansha "/><br /><sub><b>Aadityansha </b></sub></a><br />🪲 🧑‍💻 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/ShivtejG236"><img src="https://avatars.githubusercontent.com/u/91597404?v=4?v=4?s=50" width="50px;" alt="Shivtej Gaikwad"/><br /><sub><b>Shivtej Gaikwad</b></sub></a><br />🪲 🧑‍💻 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/kjlintong"><img src="https://avatars.githubusercontent.com/u/40327910?v=4?v=4?s=50" width="50px;" alt="Tong Lin"/><br /><sub><b>Tong Lin</b></sub></a><br />🧑‍💻 ✍️ 🧪</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/lynnyulinlin-debug"><img src="https://avatars.githubusercontent.com/u/252566641?v=4?v=4?s=50" width="50px;" alt="lynnyulinlin-debug"/><br /><sub><b>lynnyulinlin-debug</b></sub></a><br />🪲 🧑‍💻 🧠</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Nishitha-Rajesh"><img src="https://avatars.githubusercontent.com/u/222974421?v=4?v=4?s=50" width="50px;" alt="Nishitha-Rajesh"/><br /><sub><b>Nishitha-Rajesh</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/asgalon"><img src="https://avatars.githubusercontent.com/u/45242704?v=4?v=4?s=50" width="50px;" alt="Peter Koellner"/><br /><sub><b>Peter Koellner</b></sub></a><br />🪲 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/hzeljko"><img src="https://avatars.githubusercontent.com/hzeljko?v=4?s=50" width="50px;" alt="Zeljko Hrcek"/><br /><sub><b>Zeljko Hrcek</b></sub></a><br />🧑‍💻</td>
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
<table width="100%" style="width:100%">
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🎨 ✍️ 🧠</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Shashank-Tripathi-07"><img src="https://avatars.githubusercontent.com/u/178375647?v=4?v=4?s=50" width="50px;" alt="Rocky"/><br /><sub><b>Rocky</b></sub></a><br />🪲 🧑‍💻 🎨 🧪 🛠️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/farhan523"><img src="https://avatars.githubusercontent.com/u/62025759?v=4?v=4?s=50" width="50px;" alt="Farhan Asghar"/><br /><sub><b>Farhan Asghar</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Nishitha-Rajesh"><img src="https://avatars.githubusercontent.com/u/222974421?v=4?v=4?s=50" width="50px;" alt="Nishitha-Rajesh"/><br /><sub><b>Nishitha-Rajesh</b></sub></a><br />🪲 🧑‍💻 ✍️ 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/pelinbalci"><img src="https://avatars.githubusercontent.com/u/26349040?v=4?v=4?s=50" width="50px;" alt="Pelin Balcı"/><br /><sub><b>Pelin Balcı</b></sub></a><br />🎨 🧠 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/torrresagus"><img src="https://avatars.githubusercontent.com/u/69726650?v=4?v=4?s=50" width="50px;" alt="Agustin Torres"/><br /><sub><b>Agustin Torres</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/DIYA73"><img src="https://avatars.githubusercontent.com/u/174887077?v=4?v=4?s=50" width="50px;" alt="didi"/><br /><sub><b>didi</b></sub></a><br />🧑‍💻 🎨</td>
    </tr>
  </tbody>
</table>

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
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Nishitha-Rajesh"><img src="https://avatars.githubusercontent.com/u/222974421?v=4?v=4?s=50" width="50px;" alt="Nishitha-Rajesh"/><br /><sub><b>Nishitha-Rajesh</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/imrehg"><img src="https://avatars.githubusercontent.com/u/38863?v=4?v=4?s=50" width="50px;" alt="Gergely Imreh"/><br /><sub><b>Gergely Imreh</b></sub></a><br />🪲 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/salmanmkc"><img src="https://avatars.githubusercontent.com/u/32169182?v=4?v=4?s=50" width="50px;" alt="Salman Chishti"/><br /><sub><b>Salman Chishti</b></sub></a><br />🧑‍💻</td>
    </tr>
    <tr>
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
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Shashank-Tripathi-07"><img src="https://avatars.githubusercontent.com/u/178375647?v=4?v=4?s=50" width="50px;" alt="Rocky"/><br /><sub><b>Rocky</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/farhan523"><img src="https://avatars.githubusercontent.com/u/62025759?v=4?v=4?s=50" width="50px;" alt="Farhan Asghar"/><br /><sub><b>Farhan Asghar</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Shakhtar-Sankur"><img src="https://avatars.githubusercontent.com/u/312389772?v=4?v=4?s=50" width="50px;" alt="Sankur Kundu"/><br /><sub><b>Sankur Kundu</b></sub></a><br />🧑‍💻 ✍️ 🧠 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/aadityansha06"><img src="https://avatars.githubusercontent.com/u/96714228?v=4?v=4?s=50" width="50px;" alt="Aadityansha "/><br /><sub><b>Aadityansha </b></sub></a><br />🪲 🧑‍💻 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/asgalon"><img src="https://avatars.githubusercontent.com/u/45242704?v=4?v=4?s=50" width="50px;" alt="Peter Koellner"/><br /><sub><b>Peter Koellner</b></sub></a><br />🪲 🧑‍💻</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/salmanmkc"><img src="https://avatars.githubusercontent.com/u/32169182?v=4?v=4?s=50" width="50px;" alt="Salman Chishti"/><br /><sub><b>Salman Chishti</b></sub></a><br />🧑‍💻</td>
    </tr>
    <tr>
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
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Shashank-Tripathi-07"><img src="https://avatars.githubusercontent.com/u/178375647?v=4?v=4?s=50" width="50px;" alt="Rocky"/><br /><sub><b>Rocky</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️</td>
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
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=50" width="50px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/farhan523"><img src="https://avatars.githubusercontent.com/u/62025759?v=4?v=4?s=50" width="50px;" alt="Farhan Asghar"/><br /><sub><b>Farhan Asghar</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧪</td>
      <td align="center" valign="top" width="14.29%"><a href="https://github.com/Shashank-Tripathi-07"><img src="https://avatars.githubusercontent.com/u/178375647?v=4?v=4?s=50" width="50px;" alt="Rocky"/><br /><sub><b>Rocky</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🔎</td>
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

<b>Made with ❤️ for AI engineers in the making, around the world 🌎</b><br>
<i>If this material helped you, <a href="https://github.com/harvard-edge/cs249r_book/stargazers">star the repository</a> and pass it on to the next curious student.</i>
</div>
