<!-- MLPERF-EDU-STATUS:START -->
<div align="center">
<table border="0" cellspacing="0" cellpadding="0" width="92%">

<tr><td align="center" bgcolor="#fff4e5">

<table border="0" cellspacing="0" cellpadding="20" width="100%">
<tr><td align="center">

<h2>🚧 Under construction</h2>
<p align="center">This tree is <b>not</b> polished end-to-end yet: APIs, CLI flags, workload manifests, and documentation are still being wired for classroom use. <b>Do not rely on it for production benchmarking</b>—expect breaking changes until we publish a stable “1.0” teaching release.</p>

</td></tr>
</table>

</td></tr>

<tr><td align="center" bgcolor="#f6f8fa">

<table border="0" cellspacing="0" cellpadding="18" width="100%">
<tr><td align="center">

<h3>📌 Early work (2026)</h3>
<p align="center">MLPerf EDU is being developed in public alongside the <b>2026</b> MLSysBook ecosystem. Harness scripts, compliance checks, and teaching notes will keep moving as we align workloads with the core curriculum.</p>
<p align="center"><b>Feedback</b> — <a href="https://github.com/harvard-edge/cs249r_book/issues">GitHub issues</a> or pull requests (especially if something in this README is wrong or outdated).</p>

</td></tr>
</table>

</td></tr>

</table>
</div>
<!-- MLPERF-EDU-STATUS:END -->

# MLPerf EDU 🎓

**A 16-workload pedagogical ML systems benchmark suite aligned with [MLCommons MLPerf](https://mlcommons.org/benchmarks/).**

MLPerf EDU brings industry-standard ML benchmarking to the classroom. Every model is a self-contained, white-box PyTorch `nn.Module` — no `torchvision.models`, no HuggingFace model cards, no opaque C++ bindings. Students read, modify, and optimize every layer.

📄 **Paper**: See [`paper/paper.tex`](paper/paper.tex) — "MLPerf EDU: Bridging Industry Benchmarking and ML Systems Education"

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-org/mlperf-edu.git
cd mlperf-edu
pip install -e .

# Train a single workload (25 epochs, ~89 seconds)
mlperf cloud --task nanogpt-12m

# Train ALL 16 workloads (~8 minutes)
mlperf train --all

# Run inference with a student SUT plugin
mlperf cloud --task nanogpt-12m \
    --sut my_optimized_sut.py \
    --scenario Server --division closed

# Check compliance
python scripts/compliance_checker.py \
    --workload nanogpt --log results.json
```

## Benchmark Suite (16 Workloads)

<div align="center">
<table border="0" cellspacing="0" cellpadding="0" width="92%">
<tr><td align="center" bgcolor="#f6f8fa">
<table border="0" cellspacing="0" cellpadding="14" width="100%">
<tr><td align="center">
<p align="center"><b>Source of truth</b> — Row counts and targets stay in sync with <code>workloads.yaml</code>. When you change a workload, update the YAML; this table is regenerated from it.</p>
</td></tr>
</table>
</td></tr>
</table>
</div>

<div align="center">
<table width="98%" border="0" cellspacing="0" cellpadding="1" bgcolor="#cfd6dd" role="presentation"><tr><td bgcolor="#ffffff" align="left">
<table width="100%" border="0" cellspacing="0" cellpadding="12" bgcolor="#ffffff">
  <thead>
    <tr>
      <th bgcolor="#eef2f7" align="left" valign="top" width="10%">Division</th>
      <th bgcolor="#eef2f7" align="left" valign="top" width="12%">Task</th>
      <th bgcolor="#eef2f7" align="left" valign="top" width="14%">Model</th>
      <th bgcolor="#eef2f7" align="left" valign="top" width="10%">Params</th>
      <th bgcolor="#eef2f7" align="left" valign="top" width="26%">Dataset</th>
      <th bgcolor="#eef2f7" align="left" valign="top" width="28%">Quality target</th>
    </tr>
  </thead>
  <tbody>
    <tr bgcolor="#fafbfc"><td>Cloud</td><td>Language</td><td>NanoGPT</td><td>11.1M</td><td>TinyShakespeare (char)</td><td>Loss &lt; 2.3</td></tr>
    <tr><td>Cloud</td><td>Sparse MoE</td><td>Nano-MoE</td><td>17.4M</td><td>TinyShakespeare (char)</td><td>Loss &lt; 0.05</td></tr>
    <tr bgcolor="#fafbfc"><td>Cloud</td><td>Rec.</td><td>Micro-DLRM</td><td>23K</td><td>MovieLens-100K</td><td>Acc &gt; 0.70</td></tr>
    <tr><td>Cloud</td><td>Generation</td><td>Micro-Diff.</td><td>2.0M</td><td>CIFAR-10</td><td>MSE &lt; 0.002</td></tr>
    <tr bgcolor="#fafbfc"><td>Cloud</td><td>Graph</td><td>Micro-GCN</td><td>5.6K</td><td>Cora</td><td>Acc &gt; 0.78</td></tr>
    <tr><td>Cloud</td><td>Text Cls.</td><td>Micro-BERT</td><td>432K</td><td>SST-2</td><td>Acc &gt; 0.78</td></tr>
    <tr bgcolor="#fafbfc"><td>Cloud</td><td>Time Series</td><td>Micro-LSTM</td><td>51K</td><td>ETTh1</td><td>MSE &lt; 0.13</td></tr>
    <tr><td>Cloud</td><td>RL</td><td>Micro-RL</td><td>17K</td><td>CartPole (local)</td><td>Reward &gt; 195</td></tr>
    <tr bgcolor="#fafbfc"><td>Edge</td><td>Img. Cls.</td><td>ResNet-18</td><td>11.2M</td><td>CIFAR-100</td><td>Top1 &gt; 36%</td></tr>
    <tr><td>Edge</td><td>Mobile</td><td>MobileNetV2</td><td>2.4M</td><td>CIFAR-100</td><td>Top1 &gt; 40%</td></tr>
    <tr bgcolor="#fafbfc"><td>Tiny</td><td>KWS</td><td>DS-CNN</td><td>20K</td><td>Speech Commands v2</td><td>Top1 &gt; 90%</td></tr>
    <tr><td>Tiny</td><td>Anomaly</td><td>Autoencoder</td><td>0.3M</td><td>MNIST</td><td>MSE &lt; 0.04</td></tr>
    <tr bgcolor="#fafbfc"><td>Tiny</td><td>Person Det.</td><td>MicroNet</td><td>8.5K</td><td>Wake Vision</td><td>Acc &gt; 85%</td></tr>
    <tr><td>Agent</td><td>RAG</td><td>NanoRAG</td><td>20.1M</td><td>ReAct Traces</td><td>Retr.+Gen</td></tr>
    <tr bgcolor="#fafbfc"><td>Agent</td><td>CodeGen</td><td>NanoCodeGen</td><td>13.7M</td><td>MBPP (20 tasks)</td><td>pass@1 &gt; 0.15</td></tr>
    <tr><td>Agent</td><td>ReAct</td><td>NanoReAct</td><td>13.7M</td><td>ReAct Traces</td><td>Trace acc &gt; 0.60</td></tr>
  </tbody>
</table>
</td></tr>
</table>
</div>

All models are pure PyTorch. All training times measured on Apple M1 MPS. Total supervised suite: ~9 minutes.

## Project Structure

```
mlperf-edu/
├── paper/                      # Publication source (LaTeX)
│   ├── paper.tex               # Main paper
│   ├── refs.bib                # Bibliography
│   └── figures/                # TikZ + pgfplots figures
├── reference/                  # Reference implementations
│   ├── cloud/                  # NanoGPT, MoE, DLRM, Diffusion, GNN, BERT, LSTM, RL, Agents
│   ├── edge/                   # ResNet-18, MobileNetV2  (fully local)
│   ├── tiny/                   # DS-CNN, Autoencoder, MicroNet
│   ├── dataset_factory.py      # Unified data loading (deterministic, seed=42)
│   └── agent_datasets.py       # MBPP + ReAct trace datasets
├── src/mlperf/                 # Core harness
│   ├── cli.py                  # CLI entry point
│   ├── loadgen.py              # LoadGen proxy (Offline/Server/SingleStream/MultiStream)
│   ├── power.py                # Power profiler (powermetrics / nvidia-smi)
│   └── sut.py                  # System Under Test interface
├── scripts/
│   └── compliance_checker.py   # Quality target validation
├── examples/                   # Student lab exercises
│   ├── lab1_optimization.py    # Systems optimization challenge
│   ├── lab2_inference_sut.py   # Inference SUT plugin
│   └── lab3_arch_comparison.py # Dense vs. sparse architectures
├── workloads.yaml              # Workload registry (single source of truth)
└── data/                       # Local datasets (TinyShakespeare, MBPP, etc.)
```

## Lab Exercises

### Lab 1: Systems Optimization Challenge
Students receive a "broken baseline" ResNet-18 (batch_size=8, no workers, no schedule, no augmentation) and must reach >50% accuracy within a 30-second wall-clock budget.

```bash
python examples/lab1_optimization.py
```

### Lab 2: Inference Latency Optimization
Students implement a System Under Test (SUT) plugin for NanoGPT inference. Optimize with KV-cache, `torch.compile()`, or FP16 while the LoadGen measures p90 latency.

```bash
mlperf cloud --task nanogpt-12m --sut examples/lab2_inference_sut.py --scenario SingleStream
```

### Lab 3: Architecture Comparison
Students train NanoGPT (dense) and Nano-MoE (sparse) side-by-side, comparing convergence, memory, and throughput.

```bash
python examples/lab3_arch_comparison.py
```

## How It Works

**Students are "submitters."** They modify model code, training loops, or inference pipelines. The harness measures everything:

1. **Train** → Quality target validation (loss/accuracy thresholds)
2. **Infer** → LoadGen proxy generates Poisson/bulk arrivals, measures latency percentiles
3. **Profile** → Power measurement via `powermetrics` (macOS) or `nvidia-smi` (Linux)
4. **Submit** → JSON artifact with hardware fingerprint, metrics, and SHA-256 hash
5. **Check** → Compliance checker validates quality, parameter counts, convergence bounds

## Dataset Strategy

<div align="center">
<table width="98%" border="0" cellspacing="0" cellpadding="1" bgcolor="#cfd6dd" role="presentation"><tr><td bgcolor="#ffffff" align="left">
<table width="100%" border="0" cellspacing="0" cellpadding="14" bgcolor="#ffffff">
  <thead>
    <tr>
      <th bgcolor="#eef2f7" align="left" valign="top" width="22%">Strategy</th>
      <th bgcolor="#eef2f7" align="left" valign="top" width="53%">Datasets</th>
      <th bgcolor="#eef2f7" align="left" valign="top" width="25%">Download</th>
    </tr>
  </thead>
  <tbody>
    <tr bgcolor="#fafbfc"><td><b>Shipped with repo</b></td><td>TinyShakespeare, MBPP, ReAct Traces</td><td align="center">0 B</td></tr>
    <tr><td><b>Deterministic synthetic</b></td><td>GCN, BERT, LSTM, DLRM, CartPole, RL</td><td align="center">0 B</td></tr>
    <tr bgcolor="#fafbfc"><td><b>Auto-download</b></td><td>CIFAR-10/100, MNIST, Speech Commands v2, Wake Vision</td><td align="center">On first run</td></tr>
  </tbody>
</table>
</td></tr>
</table>
</div>

8 of 13 datasets require zero network access. All use seed=42 for reproducibility.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- `torchvision` (for CIFAR/MNIST)
- `torchaudio` (for Speech Commands)

```bash
pip install torch torchvision torchaudio
pip install -e .
```

For Apple Silicon: set `PYTORCH_ENABLE_MPS_FALLBACK=1` for full MPS compatibility.

## Citation

```bibtex
@inproceedings{mlperfedu2026,
  title={{MLPerf EDU}: Bridging Industry Benchmarking and {ML} Systems Education},
  author={[Authors]},
  year={2026}
}
```

---

*Built for [Machine Learning Systems](https://mlsysbook.ai) education.*
