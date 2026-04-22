<!-- EARLY-RELEASE-CALLOUT:START -->
> [!NOTE]
> **📌 Early release (2026)**
>
> MLSys·im shipped with the **2026** MLSysBook refresh. The modeling platform, APIs, and lab integrations are **actively iterated** as we harden the simulator and teaching workflows.
>
> **Feedback** — [GitHub issues](https://github.com/harvard-edge/cs249r_book/issues) or pull requests.
>
> [![dev branch](https://img.shields.io/badge/branch-dev-orange?logo=git&logoColor=white)](https://github.com/harvard-edge/cs249r_book/tree/dev) [![live site](https://img.shields.io/badge/live_site-mlsysbook.ai-blue?logo=safari&logoColor=white)](https://mlsysbook.ai)
<!-- EARLY-RELEASE-CALLOUT:END -->

<div align="center">
  <h1>🚀 MLSys·im: The Modeling Platform</h1>
  <blockquote>
    <b>The physics-grounded analytical simulator powering the Machine Learning Systems ecosystem.</b><br>
    Provides a unified "Single Source of Truth" (SSoT) for modeling systems from sub-watt microcontrollers to exaflop-scale global fleets.
  </blockquote>
</div>

---

## 🏗 The 5-Layer Analytical Stack

`mlsysim` implements a "Progressive Lowering" architecture, separating high-level workloads from the physical infrastructure that executes them.

<table width="100%">
  <thead>
    <tr>
      <th width="20%">Layer</th>
      <th width="30%">Domain</th>
      <th width="50%">Key Components</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>Layer A</b></td>
      <td><b>Workload Representation</b><br><code>mlsysim.models</code></td>
      <td>FLOPs, parameters, and intensity.<br><i>e.g., Llama3_70B, ResNet50</i></td>
    </tr>
    <tr>
      <td align="center"><b>Layer B</b></td>
      <td><b>Hardware Registry</b><br><code>mlsysim.hardware</code></td>
      <td>Concrete specs for real-world silicon.<br><i>e.g., H100, TPUv5p, Jetson</i></td>
    </tr>
    <tr>
      <td align="center"><b>Layer C</b></td>
      <td><b>Infrastructure</b><br><code>mlsysim.infra</code></td>
      <td>Grid profiles and datacenter sustainability.<br><i>e.g., PUE, Carbon Intensity, WUE</i></td>
    </tr>
    <tr>
      <td align="center"><b>Layer D</b></td>
      <td><b>Systems & Topology</b><br><code>mlsysim.systems</code></td>
      <td>Fleet configurations and network fabrics.<br><i>e.g., Doorbell, AutoDrive Scenarios</i></td>
    </tr>
    <tr>
      <td align="center"><b>Layer E</b></td>
      <td><b>Execution & Resolvers</b><br><code>mlsysim.core.solver</code></td>
      <td>The 3-tier math engine: Models, Solvers, and Optimizers (Design space search).</td>
    </tr>
  </tbody>
</table>

---

## 🚀 Quick Usage: The Agent-Ready CLI

`mlsysim` is a **first-principles analytical calculator** for ML systems. It provides a terminal UI for humans and a strict JSON API for CI/CD pipelines and AI agents.

> **Accuracy note:** mlsysim predictions are typically within 2–5× of measured performance for well-characterized workloads. For production capacity planning, always validate with benchmarks. This tool formalizes the back-of-envelope math that senior engineers do intuitively — it is not a substitute for profiling or load testing.

### 1. Explore the Registry (The Zoo)
Discover built-in hardware, models, and infrastructure without reading source code:
<kbd>mlsysim zoo hardware</kbd><br>
<kbd>mlsysim zoo models</kbd>

### 2. Quick Evaluation (CLI Flags)
Evaluate the physics of a workload on a specific hardware node instantly:
<kbd>mlsysim eval Llama3_8B H100 --batch-size 32</kbd>

### 3. Deep Simulation (Infrastructure as Code)
Define your entire cluster and SLA constraints in a declarative `mlsys.yaml` file:

```yaml
# example_cluster.yaml
version: "1.0"
workload:
  name: "Llama3_70B"
  batch_size: 4096
hardware:
  name: "H100"
  nodes: 64
ops:
  region: "Quebec"
  duration_days: 14.0
constraints:
  assert:
    - metric: "performance.latency"
      max: 50.0
```

Then compile and evaluate the 3-lens scorecard (Feasibility, Performance, Macro):
<kbd>mlsysim eval example_cluster.yaml</kbd>

### 4. CI/CD & Agentic Automation
Every command supports strict, schema-validated JSON output. If an `assert` constraint is violated, the CLI returns a semantic `Exit Code 3`.
```bash
# Export the JSON Schema for your IDE or AI Agent
mlsysim schema > schema.json

# Run an evaluation in a CI pipeline
tco=$(mlsysim --output json eval example_cluster.yaml | jq .macro.metrics.tco_usd)
```

### 5. Design Space Search (Optimizers)
Use the Tier 3 Engineering Engine to automatically find the optimal configuration:
<kbd>mlsysim optimize parallelism example_cluster.yaml</kbd><br>
<kbd>mlsysim optimize placement example_cluster.yaml --carbon-tax 150</kbd>

---

## 🛡 Stability & Integrity
Because this core powers a printed textbook, we enforce strict **Invariant Verification**. Every physical constant is traceable to a primary source (datasheet or paper), and dimensional integrity is enforced via `pint`.

## ⚠️ What This Tool Does Not Model

MLSysim is an **analytical hardware calculator**, not a production deployment simulator.
The 22 walls model physical and economic constraints that bound ML system performance.
Several critical production concerns are deliberately **out of scope**:

<table width="100%">
  <thead>
    <tr>
      <th align="left" width="22%">Concern</th>
      <th align="left" width="38%">Why it matters</th>
      <th align="left" width="40%">Where to learn more</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><b>Data drift / distribution shift</b></td><td>The #1 cause of production ML failures — model accuracy degrades silently as input distributions change</td><td>Sculley et al. (2015), &quot;Hidden Technical Debt in ML Systems&quot;</td></tr>
    <tr><td><b>Model versioning &amp; rollback</b></td><td>Production requires running multiple versions, A/B testing, and safe rollback</td><td>Huyen (2022), <i>Designing Machine Learning Systems</i></td></tr>
    <tr><td><b>Monitoring &amp; observability</b></td><td>You cannot manage what you cannot measure — prediction distributions, latency percentiles, error rates</td><td>Google SRE Book (2016); Huyen (2022)</td></tr>
    <tr><td><b>Feature store freshness</b></td><td>Stale features silently degrade real-time models (recommendations, fraud detection)</td><td>Uber Michelangelo (2017)</td></tr>
    <tr><td><b>Software bugs &amp; misconfigurations</b></td><td>Most outages are caused by software, not hardware</td><td>Barroso et al. (2018)</td></tr>
    <tr><td><b>Human factors</b></td><td>Team velocity, on-call burden, and organizational alignment often dominate outcomes</td><td>Brooks (1975), <i>The Mythical Man-Month</i></td></tr>
  </tbody>
</table>

**Passing all 22 walls is necessary but not sufficient for a successful production deployment.**

Students using this tool should understand that infrastructure physics (what mlsysim models)
is one dimension of a multi-dimensional engineering challenge.

## 📖 How to Cite

If you use mlsysim in your research or teaching, please cite:

```bibtex
@software{mlsysim2026,
  author       = {Janapa Reddi, Vijay},
  title        = {{MLSysim}: A Composable Analytical Framework for Machine Learning Systems},
  year         = {2026},
  url          = {https://mlsysbook.ai/mlsysim},
  version      = {0.1.0},
  institution  = {Harvard University}
}
```

## 🛠 Installation

MLSys·im is designed to be highly modular. Install only what you need:

```bash
# Core physics engine only (fastest, smallest footprint)
pip install mlsysim

# Install with the beautiful Terminal UI & YAML support
pip install "mlsysim[cli]"

# Install with dependencies for interactive labs (Marimo, Plotly)
pip install "mlsysim[labs]"
```

## 🐍 Python API Usage

The framework is just as powerful inside a Python script or Jupyter Notebook. The `SystemEvaluator` provides a clean, unified entry point for full-stack analysis:

```python
import mlsysim

# 1. Define the scenario
model = mlsysim.Models.Language.Llama3_8B
hardware = mlsysim.Hardware.Cloud.H100

# 2. Run the evaluation
evaluation = mlsysim.SystemEvaluator.evaluate(
    scenario_name="Llama-3 8B on H100",
    model_obj=model,
    hardware_obj=hardware,
    batch_size=32,
    precision="fp16",
    efficiency=0.45
)

# 3. View the beautifully formatted scorecard
print(evaluation.scorecard())
```

### Efficiency Parameter Guide

The `efficiency` parameter (0.0–1.0) captures the gap between peak hardware performance and what your software stack actually achieves. Use these guidelines:

<table width="100%">
  <thead>
    <tr>
      <th align="left" width="40%">Scenario</th>
      <th align="left" width="18%">Efficiency</th>
      <th align="left" width="42%">Rationale</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Training (Megatron-LM, large Transformer)</td><td align="center">0.40–0.55</td><td>Well-optimized GEMM + FlashAttention</td></tr>
    <tr><td>Training (PyTorch eager, small model)</td><td align="center">0.08–0.15</td><td>Kernel launch overhead dominates</td></tr>
    <tr><td>Inference decode, batch=1</td><td align="center">0.01–0.05</td><td>Memory-bound; compute nearly idle</td></tr>
    <tr><td>Inference decode, batch=32+</td><td align="center">0.15–0.35</td><td>Batch amortizes weight loading</td></tr>
    <tr><td>Inference prefill, long context</td><td align="center">0.30–0.50</td><td>Compute-bound GEMM + attention</td></tr>
    <tr><td>TinyML (TFLite Micro on ESP32)</td><td align="center">0.05–0.15</td><td>Interpreter overhead, no tensor cores</td></tr>
  </tbody>
</table>

---

## Contributors

Thanks to these wonderful people for helping improve MLSys·im!

**Legend:** 🪲 Bug Hunter · ⚡ Code Warrior · 📚 Documentation Hero · 🎨 Design Artist · 🧠 Idea Generator · 🔎 Code Reviewer · 🧪 Test Engineer · 🛠️ Tool Builder

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table width="100%">
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=80" width="80px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧑‍💻 🎨 ✍️ 🧠 maintenance</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/asgalon"><img src="https://avatars.githubusercontent.com/u/45242704?v=4?v=4?s=80" width="80px;" alt="Peter Koellner"/><br /><sub><b>Peter Koellner</b></sub></a><br />🪲 ✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hzeljko"><img src="https://avatars.githubusercontent.com/hzeljko?v=4?s=80" width="80px;" alt="Zeljko Hrcek"/><br /><sub><b>Zeljko Hrcek</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Shashank-Tripathi-07"><img src="https://avatars.githubusercontent.com/u/178375647?v=4?v=4?s=80" width="80px;" alt="Rocky"/><br /><sub><b>Rocky</b></sub></a><br />🧑‍💻</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

**Recognize a contributor:** Comment on any issue or PR:
```
@all-contributors please add @username for code, doc, ideas, or bug
```
