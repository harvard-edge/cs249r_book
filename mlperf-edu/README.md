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

> Numbers below are kept in sync with `workloads.yaml` (the single source of truth).
> If you are updating a workload, update the YAML; this table is regenerated from it.

| Division | Task | Model | Params | Dataset | Quality Target |
|----------|------|-------|--------|---------|----------------|
| Cloud | Language | NanoGPT | 11.1M | TinyShakespeare (char) | Loss < 2.3 |
| Cloud | Sparse MoE | Nano-MoE | 17.4M | TinyShakespeare (char) | Loss < 0.05 |
| Cloud | Rec. | Micro-DLRM | 23K | MovieLens-100K | Acc > 0.70 |
| Cloud | Generation | Micro-Diff. | 2.0M | CIFAR-10 | MSE < 0.002 |
| Cloud | Graph | Micro-GCN | 5.6K | Cora | Acc > 0.78 |
| Cloud | Text Cls. | Micro-BERT | 432K | SST-2 | Acc > 0.78 |
| Cloud | Time Series | Micro-LSTM | 51K | ETTh1 | MSE < 0.13 |
| Cloud | RL | Micro-RL | 17K | CartPole (local) | Reward > 195 |
| Edge | Img. Cls. | ResNet-18 | 11.2M | CIFAR-100 | Top1 > 36% |
| Edge | Mobile | MobileNetV2 | 2.4M | CIFAR-100 | Top1 > 40% |
| Tiny | KWS | DS-CNN | 20K | Speech Commands v2 | Top1 > 90% |
| Tiny | Anomaly | Autoencoder | 0.3M | MNIST | MSE < 0.04 |
| Tiny | Person Det. | MicroNet | 8.5K | Wake Vision | Acc > 85% |
| Agent | RAG | NanoRAG | 20.1M | ReAct Traces | Retr.+Gen |
| Agent | CodeGen | NanoCodeGen | 13.7M | MBPP (20 tasks) | pass@1 > 0.15 |
| Agent | ReAct | NanoReAct | 13.7M | ReAct Traces | Trace acc > 0.60 |

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

| Strategy | Datasets | Download |
|----------|----------|----------|
| **Shipped with repo** | TinyShakespeare, MBPP, ReAct Traces | 0 B |
| **Deterministic synthetic** | GCN, BERT, LSTM, DLRM, CartPole, RL | 0 B |
| **Auto-download** | CIFAR-10/100, MNIST, Speech Commands v2, Wake Vision | On first run |

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
