<!-- DEV-BANNER-START -->
<div align="center">
<table>
<tr><td>
<h3>🚧 Under Active Development</h3>
<p>This component is being built on the <code>dev</code> branch and is <b>not yet available</b> on the live site.<br>
Content may be incomplete or change without notice. The published curriculum lives at <a href="https://mlsysbook.ai"><b>mlsysbook.ai</b></a>.</p>
<p>
<a href="https://github.com/harvard-edge/cs249r_book/tree/dev"><img src="https://img.shields.io/badge/branch-dev-orange?logo=git&logoColor=white" alt="dev branch"></a>
<a href="https://mlsysbook.ai"><img src="https://img.shields.io/badge/live_site-mlsysbook.ai-blue?logo=safari&logoColor=white" alt="live site"></a>
<a href="https://github.com/harvard-edge/cs249r_book/actions/workflows/staffml-preview-dev.yml"><img src="https://github.com/harvard-edge/cs249r_book/actions/workflows/staffml-preview-dev.yml/badge.svg?branch=dev" alt="StaffML Build"></a>
</p>
</td></tr>
</table>
</div>
<!-- DEV-BANNER-END -->

# The ML Systems Interview Playbook

<p align="center">
  <b>5,700+ systems design questions across Cloud, Edge, Mobile & TinyML tracks.</b><br>
  <i>You can generate the code, but you cannot prompt your way out of a silicon bottleneck.</i>
</p>

<p align="center">
  <a href="cloud/README.md">☁️ Cloud</a> ·
  <a href="edge/README.md">🤖 Edge</a> ·
  <a href="mobile/README.md">📱 Mobile</a> ·
  <a href="tinyml/README.md">🔬 TinyML</a> ·
  <a href="NUMBERS.md">📊 Numbers</a> ·
  <a href="00_The_Architects_Rubric.md">📋 Rubric</a>
</p>

---

## 🚀 NEW: The Gauntlet (Interactive)

We are building a next-generation interview platform based on this playbook. You can now use our **[Mock Interview Prompts](MOCK_INTERVIEWS.md)** to turn any LLM into a Principal Engineer who will grill you on hardware physics and system design.

**Platform Status:**
- ✅ **Content Corpus:** 5,700+ questions across 1,100+ chains, 6 Bloom levels.
- ✅ **StaffML App:** [staffml.ai](https://mlsysbook.ai/staffml) — vault, practice, mock interviews, progress tracking.
- ✅ **CI/CD:** Auto-build and deploy on push to dev.
- 🏗️ **MCP Server:** Developer extension coming soon.

---

## Why This Exists

In the age of GenAI, writing a training loop is trivial. Anyone can ask an LLM for PyTorch syntax. But an LLM cannot fix a fragmented KV-cache, it cannot un-choke a saturated InfiniBand switch, and it cannot cool a melting Edge NPU. **Code is generated; physics is enforced.**

Students often ask me: *"How do I prepare for ML systems interviews?"* This playbook is the answer. These questions test your **Mechanical Sympathy**: the ability to see past the framework abstractions and engineer the metal underneath. You must learn to reason about the physical constraints of keeping 10,000 GPUs fed and 1 million users served. This is exactly what companies like Meta, Google, and OpenAI test for.

---

## Quick Start

Pick your level and start drilling:

<table>
  <thead>
    <tr>
      <th width="35%">You are...</th>
      <th width="65%">Start here</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Preparing for a screen</b> (Intern/New Grad)</td>
      <td>🟢 L1/L2 and L3 Green-tagged questions</td>
    </tr>
    <tr>
      <td><b>Building applied skills</b> (Mid)</td>
      <td>🔵 Blue-tagged questions — diagnose real systems</td>
    </tr>
    <tr>
      <td><b>Targeting Senior (L5)</b></td>
      <td>🟡 Yellow-tagged questions + <a href="cloud/01_single_machine.md">1. Single Machine</a> & <a href="cloud/03_serving_stack.md">3. Serving Stack</a></td>
    </tr>
    <tr>
      <td><b>Targeting Staff+ (L6+)</b></td>
      <td>🔴 Red-tagged questions + <a href="cloud/05_visual_debugging.md">5. Visual Debugging</a></td>
    </tr>
    <tr>
      <td><b>Practice under pressure</b></td>
      <td><b><a href="MOCK_INTERVIEWS.md">The Staff Gauntlet (LLM Prompts)</a></b></td>
    </tr>
  </tbody>
</table>

---

## Choose Your Track

Each track targets a different deployment regime — different physics, different constraints, different interview questions. Pick the one that matches the roles you're interviewing for, or study multiple tracks to build breadth.

<table>
  <thead>
    <tr>
      <th width="15%">Track</th>
      <th width="25%">Focus</th>
      <th width="20%">Primary Constraint</th>
      <th width="10%">Questions</th>
      <th width="15%">Topics</th>
      <th width="15%">Scale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b><a href="cloud/README.md">☁️ Cloud</a></b></td>
      <td>Data center training & serving</td>
      <td>Memory bandwidth / network</td>
      <td>819</td>
      <td>5</td>
      <td>PFLOPS, 80 GB HBM</td>
    </tr>
    <tr>
      <td><b><a href="edge/README.md">🤖 Edge</a></b></td>
      <td>Autonomous vehicles, robotics</td>
      <td>Thermal envelope / real-time</td>
      <td>811</td>
      <td>4</td>
      <td>TOPS, 8–32 GB</td>
    </tr>
    <tr>
      <td><b><a href="mobile/README.md">📱 Mobile</a></b></td>
      <td>On-device AI for smartphones</td>
      <td>Battery life / shared resources</td>
      <td>751</td>
      <td>4</td>
      <td>TOPS, 6–12 GB</td>
    </tr>
    <tr>
      <td><b><a href="tinyml/README.md">🔬 TinyML</a></b></td>
      <td>Microcontroller & ultra-low-power</td>
      <td>SRAM capacity / hard real-time</td>
      <td>760</td>
      <td>4</td>
      <td>MFLOPS, 256 KB–2 MB</td>
    </tr>
  </tbody>
</table>

> **📊 [Numbers Every ML Systems Engineer Should Know](NUMBERS.md)** — The physics constants, scaling rules, and hardware specs behind every question in this playbook.

---

## Mastery Levels

Every question is tagged with a mastery level. These levels mirror engineering ladders at major tech companies (Google, Meta, etc.) but represent **cognitive thresholds**: each level tests a different kind of reasoning, mapped to [Bloom's taxonomy](https://en.wikipedia.org/wiki/Bloom%27s_taxonomy) and the **scope of ownership** expected at that career stage.

<table>
  <thead>
    <tr>
      <th width="15%">Level</th>
      <th width="15%">Scope</th>
      <th width="20%">Cognitive Skill</th>
      <th width="50%">What the interviewer hears</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🟢 <b>L1/L2 — Intern</b></td>
      <td>Own a <b>task</b></td>
      <td><b>Recall & Identify</b></td>
      <td>"HBM is 300x slower than L1 cache." (Physics literacy)</td>
    </tr>
    <tr>
      <td>🟢 <b>L3 — New Grad</b></td>
      <td>Own a <b>task</b></td>
      <td><b>Define & Apply</b></td>
      <td>"The Roofline model relates compute to memory bandwidth."</td>
    </tr>
    <tr>
      <td>🔵 <b>L4 — Practitioner</b></td>
      <td>Own a <b>component</b></td>
      <td><b>Apply & Identify</b></td>
      <td>"This workload is memory-bound because its arithmetic intensity is below the ridge point."</td>
    </tr>
    <tr>
      <td>🟡 <b>L5 — Senior</b></td>
      <td>Own a <b>system</b></td>
      <td><b>Analyze & Predict</b></td>
      <td>"Switching from A100 to H100 won't help because the ridge point shifts right while our intensity stays at ~1."</td>
    </tr>
    <tr>
      <td>🔴 <b>L6+ — Staff</b></td>
      <td>Own the <b>architecture</b></td>
      <td><b>Synthesize & Derive</b></td>
      <td>"Let me derive the optimal parallelism dimensions from the NVLink topology, memory capacity, and pipeline bubble cost."</td>
    </tr>
  </tbody>
</table>

---

### How This Maps to Industry (Proxy)

| Level | Google | Meta | Amazon | What systems interviews test |
| :--- | :--- | :--- | :--- | :--- |
| **L1/L2** | Intern | Intern | Intern | Can you handle the basic physics and unit math? |
| **L3** | L3 (SWE II) | E3 (IC3) | SDE I | Do you know the vocabulary and basic concepts? |
| **L4** | L4 (SWE III) | E4 (IC4) | SDE II | Given a broken system, can you diagnose the root cause? |
| **L5** | L5 (Senior) | E5 (IC5) | Senior SDE | Given a changing constraint, can you predict what breaks? |
| **L6+** | L6 (Staff) | E6 (Staff) | Principal | Given a whiteboard, can you derive the architecture from physics? |

---

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=80" width="80px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧠 🎨 ✍️</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

---

<p align="center">
  <i>Wishing you all the best in your interviews and your engineering journey.</i><br>
  — <b>Vijay Janapa Reddi</b>
</p>
