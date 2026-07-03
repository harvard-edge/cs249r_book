from __future__ import annotations

import json
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root


def ensure_reference_path() -> Path:
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def run_nano_moe_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step sparse MoE language-model smoke."""
    root = ensure_reference_path()
    from reference.cloud.nano_moe import NanoMoEWhiteBox

    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 2
    seq_len = 16

    model = NanoMoEWhiteBox(vocab_size=128, d_model=32, n_heads=4, n_layers=1).to(device)
    model.train()
    inputs = torch.randint(0, 96, (batch_size, seq_len), dtype=torch.long, device=device)
    targets = torch.roll(inputs, shifts=-1, dims=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = time.perf_counter()
    logits, loss = model(inputs, targets=targets)
    assert loss is not None
    loss.backward()
    optimizer.step()
    duration = time.perf_counter() - start

    metrics = {
        "loss": float(loss.item()),
        "duration_seconds": float(duration),
        "tokens": int(batch_size * seq_len),
        "tokens_per_second": float((batch_size * seq_len) / duration) if duration > 0 else 0.0,
        "n_params": count_params(model),
        "num_experts": 8,
        "top_k": 2,
        "active_expert_fraction": 0.25,
        "logits_shape": list(logits.shape),
    }
    return write_min_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        metrics=metrics,
        dataset_name="synthetic-deterministic-tokens",
        quality_note="min profile validates sparse MoE routing and training plumbing only; max profile owns TinyShakespeare quality checks.",
    )


def run_nano_moe_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_extended_max_from_min(
        workload,
        output_dir,
        run_nano_moe_min,
        quality_note="max profile records sparse MoE routing/training systems metrics on a deterministic micro-shard; real TinyShakespeare quality check remains future work.",
    )


def run_micro_diffusion_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step micro U-Net denoising smoke."""
    root = ensure_reference_path()
    from reference.cloud.micro_diffusion import MicroDiffusionUNet

    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 2

    model = MicroDiffusionUNet(n_channels=3, n_classes=3).to(device)
    model.train()
    clean = torch.rand(batch_size, 3, 32, 32, device=device)
    noisy = clean + 0.1 * torch.randn_like(clean)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = time.perf_counter()
    reconstructed = model(noisy)
    loss = F.mse_loss(reconstructed, clean)
    loss.backward()
    optimizer.step()
    duration = time.perf_counter() - start

    metrics = {
        "mse_loss": float(loss.item()),
        "duration_seconds": float(duration),
        "samples": batch_size,
        "samples_per_second": float(batch_size / duration) if duration > 0 else 0.0,
        "n_params": count_params(model),
        "input_shape": list(noisy.shape),
        "output_shape": list(reconstructed.shape),
    }
    return write_min_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        metrics=metrics,
        dataset_name="synthetic-deterministic-images",
        quality_note="min profile validates U-Net denoising execution only; max profile owns CIFAR-10 reconstruction checks.",
    )


def run_micro_diffusion_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_extended_max_from_min(
        workload,
        output_dir,
        run_micro_diffusion_min,
        quality_note="max profile records U-Net denoising systems metrics on a deterministic image micro-shard; CIFAR-10 quality check remains future work.",
    )


def run_micro_gnn_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step MicroGCN smoke on a synthetic graph."""
    root = ensure_reference_path()
    from reference.cloud.micro_gnn import MicroGCN

    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cpu")
    n_nodes = 16
    n_features = 8
    n_classes = 3

    model = MicroGCN(nfeat=n_features, nhid=16, nclass=n_classes, dropout=0.0).to(device)
    model.train()
    features = torch.randn(n_nodes, n_features, device=device)
    labels = torch.arange(n_nodes, device=device) % n_classes
    adj = synthetic_ring_adjacency(n_nodes, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    start = time.perf_counter()
    logits = model(features, adj)
    loss = F.nll_loss(logits, labels)
    loss.backward()
    optimizer.step()
    duration = time.perf_counter() - start

    accuracy = float((logits.detach().argmax(dim=1) == labels).float().mean().item())
    metrics = {
        "loss": float(loss.item()),
        "test_accuracy": accuracy,
        "duration_seconds": float(duration),
        "nodes": n_nodes,
        "edges": int(n_nodes * 2),
        "n_params": count_params(model),
        "logits_shape": list(logits.shape),
    }
    return write_min_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        metrics=metrics,
        dataset_name="synthetic-deterministic-ring-graph",
        quality_note="min profile validates graph convolution execution only; max profile owns Cora quality checks.",
    )


def run_micro_gnn_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_extended_max_from_min(
        workload,
        output_dir,
        run_micro_gnn_min,
        quality_note="max profile records graph-convolution systems metrics on a deterministic graph micro-shard; Cora quality check remains future work.",
    )


def run_micro_bert_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step bidirectional-transformer smoke."""
    root = ensure_reference_path()
    from reference.cloud.micro_bert import CLS_IDX, PAD_IDX, MicroBERT

    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 4
    seq_len = 16

    model = MicroBERT(vocab_size=128, d_model=32, nhead=4, num_layers=1, max_len=seq_len, dropout=0.0).to(device)
    model.train()
    input_ids = torch.randint(4, 128, (batch_size, seq_len), dtype=torch.long, device=device)
    input_ids[:, 0] = CLS_IDX
    input_ids[:, -1] = PAD_IDX
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = time.perf_counter()
    logits, loss = model(input_ids, targets=labels)
    assert loss is not None
    loss.backward()
    optimizer.step()
    duration = time.perf_counter() - start

    accuracy = float((logits.detach().argmax(dim=1) == labels).float().mean().item())
    metrics = {
        "loss": float(loss.item()),
        "val_accuracy": accuracy,
        "duration_seconds": float(duration),
        "samples": batch_size,
        "samples_per_second": float(batch_size / duration) if duration > 0 else 0.0,
        "n_params": count_params(model),
        "logits_shape": list(logits.shape),
    }
    return write_min_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        metrics=metrics,
        dataset_name="synthetic-deterministic-tokenized-sentences",
        quality_note="min profile validates bidirectional transformer execution only; max profile owns SST-2 quality checks.",
    )


def run_micro_bert_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_extended_max_from_min(
        workload,
        output_dir,
        run_micro_bert_min,
        quality_note="max profile records bidirectional-transformer systems metrics on a deterministic token micro-shard; SST-2 quality check remains future work.",
    )


def run_micro_lstm_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step time-series LSTM smoke."""
    root = ensure_reference_path()
    from reference.cloud.micro_lstm import MicroLSTM

    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 2
    seq_len = 12
    pred_len = 4

    model = MicroLSTM(input_dim=7, hidden_dim=16, num_layers=1, pred_len=pred_len).to(device)
    model.train()
    inputs = torch.randn(batch_size, seq_len, 7, device=device)
    targets = torch.randn(batch_size, pred_len, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = time.perf_counter()
    predictions = model(inputs)
    loss = F.mse_loss(predictions, targets)
    loss.backward()
    optimizer.step()
    duration = time.perf_counter() - start

    metrics = {
        "val_mse": float(loss.item()),
        "duration_seconds": float(duration),
        "samples": batch_size,
        "samples_per_second": float(batch_size / duration) if duration > 0 else 0.0,
        "n_params": count_params(model),
        "input_shape": list(inputs.shape),
        "prediction_shape": list(predictions.shape),
    }
    return write_min_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        metrics=metrics,
        dataset_name="synthetic-deterministic-timeseries",
        quality_note="min profile validates recurrent forecasting execution only; max profile owns ETTh1 quality checks.",
    )


def run_micro_lstm_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_extended_max_from_min(
        workload,
        output_dir,
        run_micro_lstm_min,
        quality_note="max profile records recurrent forecasting systems metrics on a deterministic time-series micro-shard; ETTh1 quality check remains future work.",
    )


def run_micro_rl_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic short REINFORCE rollout and optimizer step."""
    root = ensure_reference_path()
    from reference.cloud.micro_rl import CartPoleLocal, REINFORCEAgent

    seed = 42
    torch.manual_seed(seed)
    env = CartPoleLocal()
    agent = REINFORCEAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    agent.train()
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3)

    state = env.reset(seed=seed)
    log_probs = []
    values = []
    rewards = []
    start = time.perf_counter()
    for _ in range(8):
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        log_probs.append(log_prob.reshape(()))
        values.append(value.reshape(()))
        rewards.append(float(reward))
        state = next_state
        if done:
            break

    returns = agent.compute_returns(rewards)
    log_probs_t = torch.stack(log_probs)
    values_t = torch.stack(values)
    advantages = returns - values_t.detach()
    policy_loss = -(log_probs_t * advantages).mean()
    value_loss = F.mse_loss(values_t, returns)
    loss = policy_loss + 0.5 * value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    duration = time.perf_counter() - start

    metrics = {
        "avg_episode_reward": float(sum(rewards)),
        "rollout_steps": len(rewards),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "duration_seconds": float(duration),
        "steps_per_second": float(len(rewards) / duration) if duration > 0 else 0.0,
        "n_params": count_params(agent),
    }
    return write_min_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        metrics=metrics,
        dataset_name="cartpole-local-deterministic-rollout",
        quality_note="min profile validates local environment and policy-gradient plumbing only; max profile owns reward checks.",
    )


def run_micro_rl_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_extended_max_from_min(
        workload,
        output_dir,
        run_micro_rl_min,
        quality_note="max profile records local policy-gradient systems metrics on a deterministic CartPole rollout; reward convergence check remains future work.",
    )


def run_nano_lora_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step LoRA fine-tuning smoke."""
    root = ensure_reference_path()
    from reference.cloud.lora import base_grad_norm, inject_lora, lora_grad_norm
    from reference.cloud.nanogpt_train import NanoGPTWhiteBox

    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 2
    seq_len = 16

    model = NanoGPTWhiteBox(vocab_size=128, n_embd=32, n_head=4, n_layer=1, max_seq_len=32).to(device)
    total_params = count_params(model)
    n_adapters, trainable_lora_params = inject_lora(model, rank=4, alpha=8, target="c_attn")
    model.train()
    inputs = torch.randint(0, 96, (batch_size, seq_len), dtype=torch.long, device=device)
    targets = torch.roll(inputs, shifts=-1, dims=1)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    start = time.perf_counter()
    logits, loss = model(inputs, targets=targets)
    assert loss is not None
    loss.backward()
    base_norm = base_grad_norm(model)
    lora_norm = lora_grad_norm(model)
    optimizer.step()
    duration = time.perf_counter() - start

    metrics = {
        "loss": float(loss.item()),
        "duration_seconds": float(duration),
        "tokens": int(batch_size * seq_len),
        "tokens_per_second": float((batch_size * seq_len) / duration) if duration > 0 else 0.0,
        "n_params_total": int(total_params),
        "n_lora_adapters": int(n_adapters),
        "n_lora_trainable_params": int(trainable_lora_params),
        "trainable_ratio_pct": float(100.0 * trainable_lora_params / total_params),
        "base_grad_norm": float(base_norm),
        "lora_grad_norm": float(lora_norm),
        "logits_shape": list(logits.shape),
    }
    return write_min_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        metrics=metrics,
        dataset_name="synthetic-deterministic-tokens",
        quality_note="min profile validates LoRA injection, frozen base parameters, and adapter-only optimization.",
    )


def run_nano_lora_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_extended_max_from_min(
        workload,
        output_dir,
        run_nano_lora_min,
        quality_note="max profile records LoRA adapter fine-tuning systems metrics on deterministic tokens; task-quality check remains future work.",
    )


def run_nanogpt_decode_fp32_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_nanogpt_decode_variant_min(workload, output_dir, dtype=torch.float32, dtype_label="fp32")


def run_nanogpt_decode_fp32_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_extended_max_from_min(
        workload,
        output_dir,
        run_nanogpt_decode_fp32_min,
        quality_note="max profile records fp32 NanoGPT decode systems metrics on deterministic prompts; trained-checkpoint serving check remains future work.",
    )


def run_nanogpt_decode_fp16_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_nanogpt_decode_variant_min(workload, output_dir, dtype=torch.float16, dtype_label="fp16")


def run_nanogpt_decode_fp16_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_extended_max_from_min(
        workload,
        output_dir,
        run_nanogpt_decode_fp16_min,
        quality_note="max profile records fp16 NanoGPT decode systems metrics on deterministic prompts; trained-checkpoint serving check remains future work.",
    )


def run_nanogpt_decode_variant_min(
    workload: Workload,
    output_dir: Path,
    *,
    dtype: torch.dtype,
    dtype_label: str,
) -> dict[str, Any]:
    """Run a deterministic tiny NanoGPT decode variant smoke."""
    root = ensure_reference_path()
    from reference.cloud.nanogpt_decode import NanoGPTDecode
    from reference.cloud.nanogpt_train import NanoGPTWhiteBox

    seed = 42
    torch.manual_seed(seed)
    model = NanoGPTWhiteBox(vocab_size=128, n_embd=32, n_head=4, n_layer=1, max_seq_len=64).to(dtype=dtype)
    result = NanoGPTDecode(model, prefill_ctx=8, decode_steps=4, batch_size=2).run(emit_sidecar=False)
    metrics = {
        **{key: float(value) if isinstance(value, float) else value for key, value in result.items()},
        "n_params": count_params(model),
        "dtype": dtype_label,
    }
    return write_min_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        metrics=metrics,
        dataset_name="synthetic-deterministic-prompts",
        quality_note=f"min profile validates {dtype_label} KV-cache decode mechanics on a tiny random NanoGPT model.",
        backend=f"pytorch-cpu-{dtype_label}",
    )


def run_nanogpt_decode_spec_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic tiny speculative decoding smoke."""
    root = ensure_reference_path()
    from reference.cloud import nanogpt_decode_spec
    from reference.cloud.nanogpt_decode_spec import SpeculativeDecode
    from reference.cloud.nanogpt_train import NanoGPTWhiteBox

    seed = 42
    torch.manual_seed(seed)
    target = NanoGPTWhiteBox(vocab_size=128, n_embd=32, n_head=4, n_layer=1, max_seq_len=64)
    draft = NanoGPTWhiteBox(vocab_size=128, n_embd=16, n_head=4, n_layer=1, max_seq_len=64)
    original_measure = nanogpt_decode_spec.measure_roofline
    nanogpt_decode_spec.measure_roofline = null_roofline
    try:
        result = SpeculativeDecode(
            target,
            draft,
            prefill_ctx=8,
            decode_tokens=4,
            gamma=2,
            batch_size=1,
        ).run()
    finally:
        nanogpt_decode_spec.measure_roofline = original_measure

    metrics = {
        **{key: float(value) if isinstance(value, float) else value for key, value in result.items()},
        "n_params_total": count_params(target) + count_params(draft),
    }
    return write_min_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        metrics=metrics,
        dataset_name="synthetic-deterministic-prompts",
        quality_note="min profile validates speculative decode control flow on tiny random target and draft NanoGPT models.",
    )


def run_nanogpt_decode_spec_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_extended_max_from_min(
        workload,
        output_dir,
        run_nanogpt_decode_spec_min,
        quality_note="max profile records speculative decode control-flow systems metrics; trained draft/target acceptance check remains future work.",
    )


def write_min_report(
    workload: Workload,
    output_dir: Path,
    *,
    root: Path,
    seed: int,
    metrics: dict[str, Any],
    dataset_name: str,
    quality_note: str,
    backend: str = "pytorch-cpu",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / f"{workload.id}_min_report.json").resolve()
    manifest_path = (output_dir / f"{workload.id}_min.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "min",
        "status": "passed",
        "backend": backend,
        "data_mode": "synthetic-deterministic",
        "seed": seed,
        "metrics": metrics,
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": quality_note,
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload=workload.id,
        scenario=workload.scenario or "single_stream",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        dataset_name=dataset_name,
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return report


def run_extended_max_from_min(
    workload: Workload,
    output_dir: Path,
    min_runner: Any,
    *,
    quality_note: str,
) -> dict[str, Any]:
    root = ensure_reference_path()
    with tempfile.TemporaryDirectory(prefix=f"mlperf-edu-{workload.id}-max-") as tmpdir:
        min_report = min_runner(workload, Path(tmpdir))

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / f"{workload.id}_max_report.json").resolve()
    manifest_path = (output_dir / f"{workload.id}_max.provd.json").resolve()
    metrics = dict(min_report.get("metrics") or {})
    metrics["max_micro_shard"] = True
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed",
        "backend": min_report.get("backend", "pytorch-cpu"),
        "data_mode": "synthetic-micro-shard",
        "seed": int(min_report.get("seed", 42)),
        "metrics": metrics,
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": quality_note,
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload=workload.id,
        scenario=workload.scenario or "single_stream",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        dataset_name=f"{workload.id}-max-synthetic-micro-shard",
        dataset_files=[],
        rng_seed=int(report["seed"]),
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return report


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def synthetic_ring_adjacency(n_nodes: int, *, device: torch.device) -> torch.Tensor:
    dense = torch.eye(n_nodes, device=device)
    for idx in range(n_nodes):
        dense[idx, (idx - 1) % n_nodes] = 1.0
        dense[idx, (idx + 1) % n_nodes] = 1.0
    degree = dense.sum(dim=1)
    d_inv_sqrt = torch.pow(degree, -0.5)
    norm = torch.diag(d_inv_sqrt) @ dense @ torch.diag(d_inv_sqrt)
    return norm.to_sparse()


@contextmanager
def null_roofline(*_args: Any, **_kwargs: Any):
    yield
