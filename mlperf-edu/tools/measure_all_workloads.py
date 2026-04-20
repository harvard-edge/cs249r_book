#!/usr/bin/env python3
"""
Iter-5.6: run every workload through measure_roofline once to populate
the sidecar directory used by check_taxonomy.py --verify-against-sidecars.

This closes the loop the user asked about: every workload gets evidence
for at least one axis (working_set + arithmetic_intensity + dispatch),
sourced from real measurement on this host, not from static analysis or
guessing.

Workloads that need special data (the 3 agents, KWS with audio) are
skipped with a noted reason; they will need bespoke harnesses to
emit sidecars (logged for iter 9).
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from mlperf.roofline import measure_roofline  # noqa: E402


def _device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Each entry: workload_name -> (module, class, kwargs, build_input_fn, n_iter, note)
# build_input_fn(model, device) -> (args_tuple, analytic_flops_fn, analytic_bytes_fn)
def _input_lm(model, device, ctx=64, batch=4):
    """Inputs for char-level LM workloads (NanoGPT, Nano-MoE).

    Reads dimensions from model.config when present (NanoGPT exposes it
    after iter-1); otherwise falls back to typical Nano-MoE defaults.
    """
    cfg = getattr(model, "config", None)
    if cfg is not None:
        vocab = cfg["vocab_size"]
        n_embd = cfg["n_embd"]
        n_layer = cfg["n_layer"]
    else:
        vocab = 128
        n_embd = 384
        n_layer = 6
    ids = torch.randint(0, vocab, (batch, ctx), device=device)
    n_params = sum(p.numel() for p in model.parameters())
    return (
        (ids,),
        lambda: 2 * n_params * ctx * batch,
        lambda: n_params * 4 + batch * ctx * n_embd * 4 * n_layer,
    )


def _input_image(model, device, batch=8, channels=3, hw=32):
    """Inputs for vision workloads (ResNet, MobileNet, MicroNet, Diffusion)."""
    x = torch.randn(batch, channels, hw, hw, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    return (
        (x,),
        lambda: 2 * n_params * batch,
        lambda: n_params * 4 + batch * channels * hw * hw * 4 * 8,  # rough activation
    )


def _input_dlrm(model, device, batch=256):
    """Inputs for DLRM cache variant."""
    dense = torch.randn(batch, 16, device=device)
    sparse_indices = [
        torch.randint(0, 943, (batch,), device=device),
        torch.randint(0, 1682, (batch,), device=device),
        torch.randint(0, 21, (batch,), device=device),
    ]
    sparse_offsets = [torch.arange(batch, device=device) for _ in range(3)]
    n_params = sum(p.numel() for p in model.parameters())
    return (
        (dense, sparse_indices, sparse_offsets),
        lambda: 2 * n_params * batch,
        lambda: 64 * 1024,  # tiny cache-resident lookups dominate
    )


def _input_dlrm_dram(model, device, batch=8192):
    """Inputs for DLRM DRAM variant — large random IDs to force DRAM access."""
    dense = torch.randn(batch, 16, device=device)
    sparse_indices = [
        torch.randint(0, 50_000_000, (batch,), device=device, dtype=torch.long) for _ in range(3)
    ]
    sparse_offsets = [torch.arange(batch, device=device) for _ in range(3)]
    n_params_trainable = 23_000  # MLPs only; the 512M virtual table is read-only here
    return (
        (dense, sparse_indices, sparse_offsets),
        lambda: 2 * n_params_trainable * batch,
        # 3 features * batch * m_spa * 4 bytes + MLPs
        lambda: 3 * batch * 256 * 4 + n_params_trainable * 4,
    )


def _input_gnn(model, device):
    """Cora has 2708 nodes; build a small adjacency."""
    n = 2708
    x = torch.randn(n, 1433, device=device)  # Cora feature dim
    adj = torch.eye(n, device=device)  # placeholder identity adjacency
    n_params = sum(p.numel() for p in model.parameters())
    return ((x, adj), lambda: 2 * n_params, lambda: n_params * 4 + n * 1433 * 4 * 2)


def _input_lstm(model, device, batch=8, seq_len=96, in_dim=7):
    x = torch.randn(batch, seq_len, in_dim, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    return ((x,), lambda: 2 * n_params * batch * seq_len,
            lambda: n_params * 4 + batch * seq_len * in_dim * 4)


def _input_anomaly_ae(model, device, batch=64):
    x = torch.randn(batch, 1, 28, 28, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    return ((x,), lambda: 2 * n_params * batch, lambda: n_params * 4 + batch * 784 * 4)


def _input_bert(model, device, batch=16, seq_len=64):
    """Char-level BERT inputs."""
    ids = torch.randint(0, 128, (batch, seq_len), device=device)
    n_params = sum(p.numel() for p in model.parameters())
    return ((ids,), lambda: 2 * n_params * batch * seq_len,
            lambda: n_params * 4 + batch * seq_len * 64 * 4 * 2)


WORKLOAD_RUNNERS = {
    "nanogpt-train": ("reference.cloud.nanogpt_train", "NanoGPTWhiteBox", {}, _input_lm, 50),
    "nano-moe-train": ("reference.cloud.nano_moe", "NanoMoEWhiteBox", {}, _input_lm, 50),
    "micro-dlrm-train": ("reference.cloud.micro_dlrm", "MicroDLRMWhiteBox", {}, _input_dlrm, 100),
    "micro-dlrm-dram-train": ("reference.cloud.micro_dlrm_dram", "MicroDLRMDRAM", {}, _input_dlrm_dram, 50),
    "micro-diffusion-train": ("reference.cloud.micro_diffusion", "MicroDiffusionUNet", {}, _input_image, 30),
    "micro-bert-train": ("reference.cloud.micro_bert", "MicroBERT", {}, _input_bert, 50),
    "micro-lstm-train": ("reference.cloud.micro_lstm", "MicroLSTM", {}, _input_lstm, 50),
    "anomaly-ae-train": ("reference.tiny.anomaly_detection_ae", "AnomalyDetectionAE", {}, _input_anomaly_ae, 50),
    "wake-vision-vww": ("reference.tiny.wake_vision_vww", "MicroNet", {}, lambda m, d: _input_image(m, d, batch=16, hw=64), 50),
    "resnet18-train": ("reference.edge.resnet_train", "ResNet18WhiteBox", {"num_classes": 100}, _input_image, 30),
    "mobilenetv2-train": ("reference.mobile.mobilenet_core", "MobileNetV2Local", {"num_classes": 100}, _input_image, 30),
}

SKIPPED = {
    "micro-gnn-train": "needs Cora adjacency — bespoke loader",
    "micro-rl-train": "needs CartPole rollout — bespoke environment harness",
    "dscnn-kws-train": "needs Speech Commands spectrogram — torchaudio dataset",
    "nanogpt-prefill": "covered by smoke_nanogpt_phases.py",
    "nanogpt-decode": "covered by smoke_nanogpt_phases.py",
    "nano-rag-agent": "needs RAG corpus + retriever — bespoke harness",
    "nano-codegen-agent": "needs MBPP problems + verifier — bespoke harness",
    "nano-react-agent": "needs ReAct trace decoder — bespoke harness; also iter-7 KV-cache fix pending",
    "nano-toolcall-agent": "needs structured-output decoder — bespoke harness",
}


def _instantiate(module_name: str, class_name: str, kwargs: dict):
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)(**kwargs)


def measure_one(workload: str, runner: tuple, device: str) -> tuple[bool, str]:
    """Run one workload through measure_roofline. Returns (success, message)."""
    module_name, class_name, kwargs, input_fn, n_iter = runner
    try:
        model = _instantiate(module_name, class_name, kwargs).to(device).eval()
    except Exception as e:
        return False, f"instantiate failed: {type(e).__name__}: {e}"

    try:
        args_tuple, flops_fn, bytes_fn = input_fn(model, device)
    except Exception as e:
        return False, f"input_fn failed: {type(e).__name__}: {e}"

    # Warm up.
    try:
        with torch.no_grad():
            for _ in range(3):
                model(*args_tuple)
    except Exception as e:
        return False, f"warmup forward failed: {type(e).__name__}: {e}"

    try:
        with torch.no_grad():
            with measure_roofline(
                workload,
                analytic_flops=lambda: flops_fn() * n_iter,
                analytic_bytes=lambda: bytes_fn() * n_iter,
                n_iter=n_iter,
            ):
                for _ in range(n_iter):
                    model(*args_tuple)
    except Exception as e:
        return False, f"timed forward failed: {type(e).__name__}: {e}"

    return True, "ok"


def main() -> int:
    device = _device()
    print(f"Device: {device}")
    print(f"Workloads to measure: {len(WORKLOAD_RUNNERS)}")
    print(f"Workloads skipped (need bespoke harness): {len(SKIPPED)}")
    print()

    n_ok, n_fail = 0, 0
    for workload, runner in WORKLOAD_RUNNERS.items():
        success, msg = measure_one(workload, runner, device)
        marker = "OK  " if success else "FAIL"
        print(f"  [{marker}] {workload:<25} {msg}")
        if success:
            n_ok += 1
        else:
            n_fail += 1

    print()
    print(f"Skipped (need bespoke harness):")
    for w, reason in SKIPPED.items():
        print(f"  [SKIP] {w:<25} {reason}")

    print()
    print(f"Summary: {n_ok} measured, {n_fail} failed, {len(SKIPPED)} skipped.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
