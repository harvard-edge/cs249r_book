<div align="center">
  <h1>Co-Labs</h1>
  <blockquote>
    <b>33 Interactive Labs Powered by MLSys·IM</b><br>
    <i>Predict → Discover → Explain</i>
  </blockquote>
</div>

---

## What Are Co-Labs?

Co-Labs are interactive [Marimo](https://marimo.io) notebooks that bridge the gap between **reading about ML systems** (the textbook) and **building them from scratch** (TinyTorch). Every lab runs in your browser via WebAssembly — no installation required.

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│    Textbook     │────▶│    Co-Labs      │────▶│    TinyTorch    │
│                 │     │                 │     │                 │
│  Concepts &     │     │  Predict &      │     │  Build from     │
│  Theory         │     │  Discover       │     │  Scratch        │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
      READ                    EXPLORE                  BUILD
```

## How Labs Work

Each lab follows a consistent structure:

1. **Briefing** — Learning objectives, prerequisites, and the core question
2. **Parts A–E** — Tabbed explorations, each with a prediction lock, interactive instruments, and a reveal
3. **Synthesis** — Key takeaways, textbook connections, and a pointer to the next lab

Every prediction is structured (radio buttons or numeric entry, never free text). You predict first, then explore the instruments to discover whether you were right. The gap between prediction and reality is the learning moment.

## Lab Inventory

### Volume I: Foundations (17 labs · Single-machine ML systems)

| # | Slug | Title |
|---|------|-------|
| 00 | `lab_00_introduction` | The Architect's Portal (orientation) |
| 01 | `lab_01_ml_intro` | The Magnitude Awakening |
| 02 | `lab_02_ml_systems` | The Iron Law |
| 03 | `lab_03_ml_workflow` | The Silent Degradation Loop |
| 04 | `lab_04_data_engr` | The Data Gravity Trap |
| 05 | `lab_05_nn_compute` | The Activation Tax |
| 06 | `lab_06_nn_arch` | The Quadratic Wall |
| 07 | `lab_07_ml_frameworks` | The Kernel Fusion Dividend |
| 08 | `lab_08_model_train` | The Training Memory Budget |
| 09 | `lab_09_data_selection` | The Data Selection Tradeoff |
| 10 | `lab_10_model_compress` | The Compression Frontier |
| 11 | `lab_11_hw_accel` | The Roofline |
| 12 | `lab_12_perf_bench` | The Speedup Ceiling |
| 13 | `lab_13_model_serving` | The Tail Latency Trap |
| 14 | `lab_14_ml_ops` | The Silent Degradation Problem |
| 15 | `lab_15_responsible_engr` | There Is No Free Fairness |
| 16 | `lab_16_ml_conclusion` | The Architect's Audit (capstone) |

### Volume II: At Scale (16 labs · Distributed ML systems)

| # | Slug | Title |
|---|------|-------|
| 01 | `lab_01_introduction` | The Scale Illusion |
| 02 | `lab_02_compute_infra` | The Compute Infrastructure Wall |
| 03 | `lab_03_communication` | Communication at Scale |
| 04 | `lab_04_data_storage` | The Data Pipeline Wall |
| 05 | `lab_05_dist_train` | The Parallelism Puzzle |
| 06 | `lab_06_fault_tolerance` | When Failure Is Routine |
| 07 | `lab_07_fleet_orch` | The Scheduling Trap |
| 08 | `lab_08_inference` | The Inference Economy |
| 09 | `lab_09_perf_engineering` | The Optimization Trap |
| 10 | `lab_10_edge_intelligence` | The Edge Thermodynamics Lab |
| 11 | `lab_11_ops_scale` | The Silent Fleet |
| 12 | `lab_12_security_privacy` | The Price of Privacy |
| 13 | `lab_13_robust_ai` | The Robustness Budget |
| 14 | `lab_14_sustainable_ai` | The Carbon Budget |
| 15 | `lab_15_responsible_ai` | The Fairness Budget |
| 16 | `lab_16_fleet_synthesis` | The Fleet Synthesis (capstone) |

## The Design Ledger

Every lab saves your predictions and design decisions to the **Design Ledger** — a persistence layer in your browser's localStorage. Later labs read earlier decisions: Lab 08's training memory budget builds on Lab 05's activation analysis, which builds on Lab 01's magnitude calibration. The capstone labs synthesize your full Design Ledger into a portfolio.

## Running Labs

### In the Browser (Recommended)

Visit the [Co-Labs site](https://mlsysbook.ai/labs/) and click any lab. They run via Marimo + WebAssembly with zero setup.

### Locally

```bash
pip install mlsysim[labs]
git clone https://github.com/harvard-edge/cs249r_book.git
cd cs249r_book/labs
marimo run vol1/lab_01_ml_intro.py
```

## Development

See [PROTOCOL.md](PROTOCOL.md) for the lab development specification and [TEMPLATE.md](TEMPLATE.md) for the cell architecture and quality checklist.

### Running Tests

```bash
pytest tests/test_static.py -v
```

---

## Related Resources

| Resource | Description |
|----------|-------------|
| [Textbook](https://mlsysbook.ai) | ML Systems principles and practices |
| [TinyTorch](https://mlsysbook.ai/tinytorch) | Build your own ML framework from scratch |
| [Discussions](https://github.com/harvard-edge/cs249r_book/discussions) | Ask questions, share feedback |

---

## Contributors

Thanks to these wonderful people who helped build the labs!

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=80" width="80px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/salmanmkc"><img src="https://avatars.githubusercontent.com/u/32169182?v=4?v=4?s=80" width="80px;" alt="Salman Chishti"/><br /><sub><b>Salman Chishti</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Pratham-ja"><img src="https://avatars.githubusercontent.com/u/114498234?v=4?v=4?s=80" width="80px;" alt="Pratham Chaudhary"/><br /><sub><b>Pratham Chaudhary</b></sub></a><br />🧑‍💻</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

**Recognize a contributor:** Comment on any issue or PR:
```
@all-contributors please add @username for code, tutorial, test, or doc
```

---

<div align="center">

<b>Predict. Discover. Explain.</b>

</div>
