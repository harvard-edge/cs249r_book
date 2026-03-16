# ☁️ Cloud Track — Data Center & Distributed Systems

<div align="center">
  <a href="../README.md">🏠 Playbook Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <b>☁️ Cloud</b> ·
  <a href="../edge/README.md">🤖 Edge</a> ·
  <a href="../mobile/README.md">📱 Mobile</a> ·
  <a href="../tinyml/README.md">🔬 TinyML</a>
</div>

---

The Cloud track covers ML systems that run in data centers — from a single H100 to 10,000-GPU training clusters to production serving fleets handling millions of requests per second.

### The Constraint Regime

| Dimension | Cloud Reality |
|---|---|
| **Compute** | PFLOPS (H100, TPU, B200) |
| **Memory** | 80 GB HBM per chip, terabytes across a cluster |
| **Interconnect** | NVLink (900 GB/s intra-node), InfiniBand (400 Gbps inter-node) |
| **Power budget** | 700W–1000W per chip, megawatts per cluster |
| **Primary bottleneck** | Memory bandwidth (single node), network (multi-node) |
| **Failure mode** | Silent data corruption at scale, straggler nodes, MTBF collapse |

### The Rounds

| Round | Focus | Questions |
|---|---|---|
| [**1. Silicon Physics**](01_Single_Node_Physics.md) | What happens inside a single server | 9 |
| [**2. Distributed Infra**](02_Distributed_Infrastructure.md) | What happens when you exceed one node | 9 |
| [**3. Production Serving**](03_Production_Serving.md) | Surviving real user traffic | 9 |
| [**4. Ops & Economics**](04_Operations_and_Economics.md) | Keeping systems healthy over time | 9 |
| [**5. Visual Debugging**](05_Visual_Architecture_Debugging.md) | Spotting bottlenecks in diagrams | 7 |
| [**6. Advanced Systems**](06_Advanced_Systems.md) | Compute analysis, power, architecture cost, security & fairness | 14 |

### Who This Track Is For

Engineers interviewing at frontier labs and cloud infrastructure companies — Meta, Google, OpenAI, Anthropic, NVIDIA, Amazon, Microsoft, and similar organizations building or operating large-scale ML systems.
