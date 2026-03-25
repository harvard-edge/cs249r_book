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

# StaffML: ML Systems Interview Playbook

<p align="center">
  <b>Physics-grounded systems design questions across Cloud, Edge, Mobile & TinyML tracks.</b><br>
  <i>You can generate the code, but you cannot prompt your way out of a silicon bottleneck.</i>
</p>

---

## What is StaffML?

StaffML is an interview prep platform for ML systems engineers. It provides a curated vault of questions organized by competency area, difficulty level (Bloom's Taxonomy L1–L6+), and deployment track.

**Key features:**
- **Vault** — Browse questions by area, topic, and difficulty
- **Practice** — Drill with spaced repetition and daily challenges
- **Mock Interview (Gauntlet)** — Timed sessions with self-assessment
- **Progress** — Track coverage across competency areas and tracks
- **Chains** — Deepening question sequences (L1 Recall → L6+ Architect)

Built on [MLSysBook.ai](https://mlsysbook.ai) by Prof. Vijay Janapa Reddi, Harvard University.

**App:** [staffml/](staffml/) · **Corpus data:** `corpus.json` · **Taxonomy:** `taxonomy.json`

---

## Deployment Tracks

Each track targets a different deployment regime — different physics, different constraints, different interview questions.

| Track | Focus | Primary Constraint |
|-------|-------|-------------------|
| ☁️ **Cloud** | Data center training & serving | Memory bandwidth / network |
| 🤖 **Edge** | Autonomous vehicles, robotics | Thermal envelope / real-time |
| 📱 **Mobile** | On-device AI for smartphones | Battery life / shared resources |
| 🔬 **TinyML** | Microcontroller & ultra-low-power | SRAM capacity / hard real-time |

---

## Mastery Levels

Every question is tagged with a mastery level mapped to [Bloom's taxonomy](https://en.wikipedia.org/wiki/Bloom%27s_taxonomy):

| Level | Name | Scope | What the interviewer hears |
|-------|------|-------|---------------------------|
| 🔵 **L1** | Recall | Own a task | "HBM is 300x slower than L1 cache." |
| 🟢 **L2** | Understand | Own a task | "The Roofline model relates compute to memory bandwidth." |
| 🟡 **L3** | Apply | Own a component | "This workload is memory-bound because its arithmetic intensity is below the ridge point." |
| 🟠 **L4** | Analyze | Own a system | "Switching from A100 to H100 won't help because the ridge point shifts." |
| 🔴 **L5** | Evaluate | Own the architecture | "Let me derive the optimal parallelism from the NVLink topology." |
| 🟣 **L6+** | Architect | Own the org | "Here's a fault-tolerant training architecture for 1T params across 3 data centers." |

---

## Development

```bash
# Run the StaffML app locally
cd interviews/staffml
npm install
npm run dev         # → http://localhost:3000

# Regenerate vault manifest after corpus updates
python3 scripts/generate-manifest.py
```

**CI/CD:** Pushes to `dev` auto-build and deploy via [GitHub Actions](https://github.com/harvard-edge/cs249r_book/actions/workflows/staffml-preview-dev.yml).

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
