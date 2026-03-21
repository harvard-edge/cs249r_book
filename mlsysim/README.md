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
</p>
</td></tr>
</table>
</div>
<!-- DEV-BANNER-END -->

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

<table>
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

`mlsysim` is designed as an **Infrastructure-as-Code (IaC) Compiler** for ML systems. It features a stunning terminal UI for humans and a strict JSON API for CI/CD pipelines and AI agents.

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
