from __future__ import annotations

import json
import math
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import configured_seed


def ensure_reference_path() -> Path:
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def snapshot_trainable_parameters(model: torch.nn.Module) -> list[torch.Tensor]:
    return [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def snapshot_frozen_parameters(model: torch.nn.Module) -> list[torch.Tensor]:
    return [
        parameter.detach().clone()
        for parameter in model.parameters()
        if not parameter.requires_grad
    ]


def trainable_parameter_delta_l2(
    model: torch.nn.Module, initial: list[torch.Tensor]
) -> float:
    current = [
        parameter.detach()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if len(current) != len(initial):
        raise ValueError("trainable parameter structure changed during the max run")
    squared = sum(
        float((after - before).float().pow(2).sum().item())
        for before, after in zip(initial, current)
    )
    return math.sqrt(squared)


def frozen_parameter_delta_l2(
    model: torch.nn.Module, initial: list[torch.Tensor]
) -> float:
    current = [
        parameter.detach()
        for parameter in model.parameters()
        if not parameter.requires_grad
    ]
    if len(current) != len(initial):
        raise ValueError("frozen parameter structure changed during the max run")
    squared = sum(
        float((after - before).float().pow(2).sum().item())
        for before, after in zip(initial, current)
    )
    return math.sqrt(squared)


def finite_series(values: list[float]) -> bool:
    return bool(values) and all(math.isfinite(value) for value in values)


def run_nano_moe_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step sparse MoE language-model smoke."""
    root = ensure_reference_path()
    from mlperf.reference.cloud.nano_moe import NanoMoEWhiteBox

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 2
    seq_len = 16

    model = NanoMoEWhiteBox(vocab_size=128, d_model=32, n_heads=4, n_layers=1).to(
        device
    )
    model.train()
    inputs = torch.randint(
        0, 96, (batch_size, seq_len), dtype=torch.long, device=device
    )
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
        "tokens_per_second": float((batch_size * seq_len) / duration)
        if duration > 0
        else 0.0,
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
        quality_note="min profile validates sparse MoE routing and training plumbing only; task-quality calibration remains future work.",
    )


def run_nano_moe_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a bounded multi-step MoE training systems workload."""
    root = ensure_reference_path()
    from mlperf.reference.cloud.nano_moe import NanoMoEWhiteBox

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    config = {
        "vocab_size": 256,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "num_experts": 8,
        "top_k": 2,
        "batch_size": 4,
        "seq_len": 32,
        "train_steps": 3,
        "lr": 0.001,
    }
    model = NanoMoEWhiteBox(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
    ).to(device)
    model.train()
    inputs = torch.randint(
        0,
        config["vocab_size"],
        (config["batch_size"], config["seq_len"]),
        dtype=torch.long,
        device=device,
    )
    targets = torch.roll(inputs, shifts=-1, dims=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    initial = snapshot_trainable_parameters(model)
    losses: list[float] = []
    start = time.perf_counter()
    logits = None
    for _ in range(config["train_steps"]):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(inputs, targets=targets)
        if loss is None:
            raise ValueError("Nano-MoE max training did not return a loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    duration = time.perf_counter() - start
    assert logits is not None
    parameter_delta = trainable_parameter_delta_l2(model, initial)
    tokens = config["batch_size"] * config["seq_len"] * config["train_steps"]
    checks = {
        "completed_configured_train_steps": len(losses) == config["train_steps"],
        "finite_losses": finite_series(losses),
        "parameters_updated": parameter_delta > 0,
        "positive_throughput": duration > 0 and tokens / duration > 0,
    }
    return write_max_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        config=config,
        model_metadata={
            "architecture": "NanoMoEWhiteBox",
            "n_params": count_params(model),
            "dtype": "float32",
        },
        metrics={
            "loss": losses[-1],
            "losses": losses,
            "duration_seconds": float(duration),
            "train_steps": len(losses),
            "tokens": int(tokens),
            "tokens_per_second": float(tokens / duration) if duration > 0 else 0.0,
            "n_params": count_params(model),
            "num_experts": config["num_experts"],
            "top_k": config["top_k"],
            "active_expert_fraction": config["top_k"] / config["num_experts"],
            "parameter_delta_l2": parameter_delta,
            "logits_shape": list(logits.shape),
        },
        functional_checks=checks,
        dataset_name="synthetic-deterministic-token-training-shard",
        quality_note="max profile records sparse MoE routing/training systems metrics on a deterministic micro-shard; real TinyShakespeare quality check remains future work.",
    )


def run_micro_diffusion_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step micro U-Net denoising smoke."""
    root = ensure_reference_path()
    from mlperf.reference.cloud.micro_diffusion import MicroDiffusionUNet

    seed = configured_seed()
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
        quality_note="min profile validates U-Net denoising execution only; task-quality calibration remains future work.",
    )


def run_micro_diffusion_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run several deterministic denoising optimizer steps."""
    root = ensure_reference_path()
    from mlperf.reference.cloud.micro_diffusion import MicroDiffusionUNet

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    config = {
        "image_size": 32,
        "channels": 3,
        "batch_size": 4,
        "train_steps": 3,
        "noise_std": 0.1,
        "lr": 0.001,
    }
    model = MicroDiffusionUNet(
        n_channels=config["channels"], n_classes=config["channels"]
    ).to(device)
    model.train()
    clean = torch.rand(
        config["batch_size"],
        config["channels"],
        config["image_size"],
        config["image_size"],
        device=device,
    )
    noisy = clean + config["noise_std"] * torch.randn_like(clean)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    initial = snapshot_trainable_parameters(model)
    losses: list[float] = []
    reconstructed = None
    start = time.perf_counter()
    for _ in range(config["train_steps"]):
        optimizer.zero_grad(set_to_none=True)
        reconstructed = model(noisy)
        loss = F.mse_loss(reconstructed, clean)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    duration = time.perf_counter() - start
    assert reconstructed is not None
    parameter_delta = trainable_parameter_delta_l2(model, initial)
    samples = config["batch_size"] * config["train_steps"]
    checks = {
        "completed_configured_train_steps": len(losses) == config["train_steps"],
        "finite_losses": finite_series(losses),
        "parameters_updated": parameter_delta > 0,
        "output_shape_matches_input": reconstructed.shape == clean.shape,
    }
    return write_max_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        config=config,
        model_metadata={
            "architecture": "MicroDiffusionUNet",
            "n_params": count_params(model),
            "dtype": "float32",
        },
        metrics={
            "mse_loss": losses[-1],
            "losses": losses,
            "duration_seconds": float(duration),
            "train_steps": len(losses),
            "samples": int(samples),
            "samples_per_second": float(samples / duration) if duration > 0 else 0.0,
            "n_params": count_params(model),
            "parameter_delta_l2": parameter_delta,
            "input_shape": list(noisy.shape),
            "output_shape": list(reconstructed.shape),
        },
        functional_checks=checks,
        dataset_name="synthetic-deterministic-denoising-shard",
        quality_note="max profile records U-Net denoising systems metrics on a deterministic image micro-shard; CIFAR-10 quality check remains future work.",
    )


def run_micro_gnn_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step MicroGCN smoke on a synthetic graph."""
    root = ensure_reference_path()
    from mlperf.reference.cloud.micro_gnn import MicroGCN

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    n_nodes = 16
    n_features = 8
    n_classes = 3

    model = MicroGCN(nfeat=n_features, nhid=16, nclass=n_classes, dropout=0.0).to(
        device
    )
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
        quality_note="min profile validates graph convolution execution only; task-quality calibration remains future work.",
    )


def run_micro_gnn_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a larger deterministic ring-graph training workload."""
    root = ensure_reference_path()
    from mlperf.reference.cloud.micro_gnn import MicroGCN

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    config = {
        "n_nodes": 64,
        "n_features": 32,
        "hidden_dim": 32,
        "n_classes": 4,
        "train_steps": 4,
        "lr": 0.01,
    }
    model = MicroGCN(
        nfeat=config["n_features"],
        nhid=config["hidden_dim"],
        nclass=config["n_classes"],
        dropout=0.0,
    ).to(device)
    model.train()
    features = torch.randn(config["n_nodes"], config["n_features"], device=device)
    labels = torch.arange(config["n_nodes"], device=device) % config["n_classes"]
    adjacency = synthetic_ring_adjacency(config["n_nodes"], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    initial = snapshot_trainable_parameters(model)
    losses: list[float] = []
    logits = None
    start = time.perf_counter()
    for _ in range(config["train_steps"]):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, adjacency)
        loss = F.nll_loss(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    duration = time.perf_counter() - start
    assert logits is not None
    accuracy = float((logits.detach().argmax(dim=1) == labels).float().mean().item())
    parameter_delta = trainable_parameter_delta_l2(model, initial)
    checks = {
        "completed_configured_train_steps": len(losses) == config["train_steps"],
        "finite_losses": finite_series(losses),
        "parameters_updated": parameter_delta > 0,
        "logits_cover_every_node": logits.shape
        == (config["n_nodes"], config["n_classes"]),
    }
    return write_max_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        config=config,
        model_metadata={
            "architecture": "MicroGCN",
            "n_params": count_params(model),
            "dtype": "float32",
        },
        metrics={
            "loss": losses[-1],
            "losses": losses,
            "test_accuracy": accuracy,
            "duration_seconds": float(duration),
            "train_steps": len(losses),
            "nodes": config["n_nodes"],
            "edges": config["n_nodes"] * 2,
            "node_steps": config["n_nodes"] * config["train_steps"],
            "n_params": count_params(model),
            "parameter_delta_l2": parameter_delta,
            "logits_shape": list(logits.shape),
        },
        functional_checks=checks,
        dataset_name="synthetic-deterministic-ring-graph-max",
        quality_note="max profile records graph-convolution systems metrics on a deterministic graph micro-shard; Cora quality check remains future work.",
    )


def run_micro_bert_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step bidirectional-transformer smoke."""
    root = ensure_reference_path()
    from mlperf.reference.cloud.micro_bert import CLS_IDX, PAD_IDX, MicroBERT

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 4
    seq_len = 16

    model = MicroBERT(
        vocab_size=128, d_model=32, nhead=4, num_layers=1, max_len=seq_len, dropout=0.0
    ).to(device)
    model.train()
    input_ids = torch.randint(
        4, 128, (batch_size, seq_len), dtype=torch.long, device=device
    )
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
        quality_note="min profile validates bidirectional transformer execution only; task-quality calibration remains future work.",
    )


def run_micro_bert_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a bounded multi-step bidirectional-transformer workload."""
    root = ensure_reference_path()
    from mlperf.reference.cloud.micro_bert import CLS_IDX, PAD_IDX, MicroBERT

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    config = {
        "vocab_size": 512,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "batch_size": 8,
        "seq_len": 32,
        "train_steps": 4,
        "lr": 0.001,
    }
    model = MicroBERT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["n_heads"],
        num_layers=config["n_layers"],
        max_len=config["seq_len"],
        dropout=0.0,
    ).to(device)
    model.train()
    input_ids = torch.randint(
        4,
        config["vocab_size"],
        (config["batch_size"], config["seq_len"]),
        dtype=torch.long,
        device=device,
    )
    input_ids[:, 0] = CLS_IDX
    input_ids[:, -1] = PAD_IDX
    labels = torch.arange(config["batch_size"], device=device) % 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    initial = snapshot_trainable_parameters(model)
    losses: list[float] = []
    logits = None
    start = time.perf_counter()
    for _ in range(config["train_steps"]):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(input_ids, targets=labels)
        if loss is None:
            raise ValueError("Micro-BERT max training did not return a loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    duration = time.perf_counter() - start
    assert logits is not None
    accuracy = float((logits.detach().argmax(dim=1) == labels).float().mean().item())
    parameter_delta = trainable_parameter_delta_l2(model, initial)
    samples = config["batch_size"] * config["train_steps"]
    checks = {
        "completed_configured_train_steps": len(losses) == config["train_steps"],
        "finite_losses": finite_series(losses),
        "parameters_updated": parameter_delta > 0,
        "binary_logits_emitted": logits.shape == (config["batch_size"], 2),
    }
    return write_max_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        config=config,
        model_metadata={
            "architecture": "MicroBERT",
            "n_params": count_params(model),
            "dtype": "float32",
        },
        metrics={
            "loss": losses[-1],
            "losses": losses,
            "val_accuracy": accuracy,
            "duration_seconds": float(duration),
            "train_steps": len(losses),
            "samples": int(samples),
            "samples_per_second": float(samples / duration) if duration > 0 else 0.0,
            "n_params": count_params(model),
            "parameter_delta_l2": parameter_delta,
            "logits_shape": list(logits.shape),
        },
        functional_checks=checks,
        dataset_name="synthetic-deterministic-token-classification-shard",
        quality_note="max profile records bidirectional-transformer systems metrics on a deterministic token micro-shard; SST-2 quality check remains future work.",
    )


def run_micro_lstm_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step time-series LSTM smoke."""
    root = ensure_reference_path()
    from mlperf.reference.cloud.micro_lstm import MicroLSTM

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 2
    seq_len = 12
    pred_len = 4

    model = MicroLSTM(input_dim=7, hidden_dim=16, num_layers=1, pred_len=pred_len).to(
        device
    )
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
        quality_note="min profile validates recurrent forecasting execution only; task-quality calibration remains future work.",
    )


def run_micro_lstm_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a larger deterministic recurrent forecasting workload."""
    root = ensure_reference_path()
    from mlperf.reference.cloud.micro_lstm import MicroLSTM

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    config = {
        "input_dim": 7,
        "hidden_dim": 32,
        "n_layers": 2,
        "batch_size": 8,
        "seq_len": 48,
        "pred_len": 12,
        "train_steps": 4,
        "lr": 0.001,
    }
    model = MicroLSTM(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["n_layers"],
        pred_len=config["pred_len"],
    ).to(device)
    model.train()
    inputs = torch.randn(
        config["batch_size"],
        config["seq_len"],
        config["input_dim"],
        device=device,
    )
    targets = torch.randn(config["batch_size"], config["pred_len"], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    initial = snapshot_trainable_parameters(model)
    losses: list[float] = []
    predictions = None
    start = time.perf_counter()
    for _ in range(config["train_steps"]):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = F.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    duration = time.perf_counter() - start
    assert predictions is not None
    parameter_delta = trainable_parameter_delta_l2(model, initial)
    samples = config["batch_size"] * config["train_steps"]
    checks = {
        "completed_configured_train_steps": len(losses) == config["train_steps"],
        "finite_losses": finite_series(losses),
        "parameters_updated": parameter_delta > 0,
        "prediction_shape_matches_contract": predictions.shape
        == (config["batch_size"], config["pred_len"]),
    }
    return write_max_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        config=config,
        model_metadata={
            "architecture": "MicroLSTM",
            "n_params": count_params(model),
            "dtype": "float32",
        },
        metrics={
            "val_mse": losses[-1],
            "losses": losses,
            "duration_seconds": float(duration),
            "train_steps": len(losses),
            "samples": int(samples),
            "samples_per_second": float(samples / duration) if duration > 0 else 0.0,
            "n_params": count_params(model),
            "parameter_delta_l2": parameter_delta,
            "input_shape": list(inputs.shape),
            "prediction_shape": list(predictions.shape),
        },
        functional_checks=checks,
        dataset_name="synthetic-deterministic-timeseries-max",
        quality_note="max profile records recurrent forecasting systems metrics on a deterministic time-series micro-shard; ETTh1 quality check remains future work.",
    )


def run_micro_rl_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic short REINFORCE rollout and optimizer step."""
    root = ensure_reference_path()
    from mlperf.reference.cloud.micro_rl import CartPoleLocal, REINFORCEAgent

    seed = configured_seed()
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
        quality_note="min profile validates local environment and policy-gradient plumbing only; reward-convergence calibration remains future work.",
    )


def run_micro_rl_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run multiple deterministic CartPole rollouts and optimizer steps."""
    root = ensure_reference_path()
    from mlperf.reference.cloud.micro_rl import CartPoleLocal, REINFORCEAgent

    seed = configured_seed()
    torch.manual_seed(seed)
    environment = CartPoleLocal()
    agent = REINFORCEAgent(
        state_dim=environment.state_dim, n_actions=environment.n_actions
    )
    agent.train()
    config = {
        "episodes": 4,
        "max_steps_per_episode": 32,
        "lr": 0.001,
        "gamma": float(agent.gamma),
    }
    optimizer = torch.optim.AdamW(agent.parameters(), lr=config["lr"])
    initial = snapshot_trainable_parameters(agent)
    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    episode_losses: list[float] = []
    start = time.perf_counter()
    for episode in range(config["episodes"]):
        state = environment.reset(seed=seed + episode)
        log_probabilities: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[float] = []
        for _ in range(config["max_steps_per_episode"]):
            action, log_probability, value = agent.select_action(state)
            state, reward, done, _ = environment.step(action)
            log_probabilities.append(log_probability.reshape(()))
            values.append(value.reshape(()))
            rewards.append(float(reward))
            if done:
                break
        returns = agent.compute_returns(rewards)
        log_probabilities_t = torch.stack(log_probabilities)
        values_t = torch.stack(values)
        advantages = returns - values_t.detach()
        policy_loss = -(log_probabilities_t * advantages).mean()
        value_loss = F.mse_loss(values_t, returns)
        loss = policy_loss + 0.5 * value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        episode_rewards.append(float(sum(rewards)))
        episode_steps.append(len(rewards))
        episode_losses.append(float(loss.item()))
    duration = time.perf_counter() - start
    parameter_delta = trainable_parameter_delta_l2(agent, initial)
    total_steps = sum(episode_steps)
    checks = {
        "completed_configured_episodes": len(episode_rewards) == config["episodes"],
        "every_episode_executed_steps": all(steps > 0 for steps in episode_steps),
        "finite_losses": finite_series(episode_losses),
        "parameters_updated": parameter_delta > 0,
    }
    return write_max_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        config=config,
        model_metadata={
            "architecture": "REINFORCEAgent",
            "n_params": count_params(agent),
            "dtype": "float32",
        },
        metrics={
            "avg_episode_reward": float(statistics.mean(episode_rewards)),
            "episode_rewards": episode_rewards,
            "episode_steps": episode_steps,
            "episode_losses": episode_losses,
            "episodes": len(episode_rewards),
            "optimizer_steps": len(episode_losses),
            "rollout_steps": int(total_steps),
            "duration_seconds": float(duration),
            "steps_per_second": float(total_steps / duration) if duration > 0 else 0.0,
            "n_params": count_params(agent),
            "parameter_delta_l2": parameter_delta,
        },
        functional_checks=checks,
        dataset_name="cartpole-local-deterministic-multi-episode",
        quality_note="max profile records local policy-gradient systems metrics on a deterministic CartPole rollout; reward convergence check remains future work.",
    )


def run_nano_lora_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step LoRA fine-tuning smoke."""
    root = ensure_reference_path()
    from mlperf.reference.cloud.lora import base_grad_norm, inject_lora, lora_grad_norm
    from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 2
    seq_len = 16

    model = NanoGPTWhiteBox(
        vocab_size=128, n_embd=32, n_head=4, n_layer=1, max_seq_len=32
    ).to(device)
    total_params = count_params(model)
    n_adapters, trainable_lora_params = inject_lora(
        model, rank=4, alpha=8, target="c_attn"
    )
    model.train()
    inputs = torch.randint(
        0, 96, (batch_size, seq_len), dtype=torch.long, device=device
    )
    targets = torch.roll(inputs, shifts=-1, dims=1)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3
    )
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
        "tokens_per_second": float((batch_size * seq_len) / duration)
        if duration > 0
        else 0.0,
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
    """Run multi-step LoRA training on a larger local NanoGPT geometry."""
    root = ensure_reference_path()
    from mlperf.reference.cloud.lora import base_grad_norm, inject_lora, lora_grad_norm
    from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    config = {
        "vocab_size": 256,
        "n_embd": 64,
        "n_head": 4,
        "n_layer": 2,
        "max_seq_len": 64,
        "batch_size": 4,
        "seq_len": 32,
        "rank": 8,
        "alpha": 16,
        "target_module": "c_attn",
        "train_steps": 3,
        "lr": 0.001,
    }
    model = NanoGPTWhiteBox(
        vocab_size=config["vocab_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        max_seq_len=config["max_seq_len"],
    ).to(device)
    base_params = count_params(model)
    adapters, trainable_lora_params = inject_lora(
        model,
        rank=config["rank"],
        alpha=config["alpha"],
        target=config["target_module"],
    )
    model.train()
    total_params = count_params(model)
    inputs = torch.randint(
        0,
        config["vocab_size"],
        (config["batch_size"], config["seq_len"]),
        dtype=torch.long,
        device=device,
    )
    targets = torch.roll(inputs, shifts=-1, dims=1)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["lr"],
    )
    initial = snapshot_trainable_parameters(model)
    initial_frozen = snapshot_frozen_parameters(model)
    losses: list[float] = []
    base_gradient_norms: list[float] = []
    lora_gradient_norms: list[float] = []
    logits = None
    start = time.perf_counter()
    for _ in range(config["train_steps"]):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(inputs, targets=targets)
        if loss is None:
            raise ValueError("Nano-LoRA max training did not return a loss")
        loss.backward()
        base_gradient_norms.append(float(base_grad_norm(model)))
        lora_gradient_norms.append(float(lora_grad_norm(model)))
        optimizer.step()
        losses.append(float(loss.item()))
    duration = time.perf_counter() - start
    assert logits is not None
    parameter_delta = trainable_parameter_delta_l2(model, initial)
    frozen_parameter_delta = frozen_parameter_delta_l2(model, initial_frozen)
    tokens = config["batch_size"] * config["seq_len"] * config["train_steps"]
    checks = {
        "completed_configured_train_steps": len(losses) == config["train_steps"],
        "base_parameters_remained_frozen": frozen_parameter_delta == 0.0
        and all(norm == 0.0 for norm in base_gradient_norms),
        "lora_gradients_nonzero": all(norm > 0.0 for norm in lora_gradient_norms),
        "lora_parameters_updated": parameter_delta > 0,
        "finite_losses": finite_series(losses),
    }
    return write_max_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        config=config,
        model_metadata={
            "architecture": "NanoGPTWhiteBox+LoRA",
            "n_params": total_params,
            "base_n_params": base_params,
            "trainable_n_params": trainable_lora_params,
            "dtype": "float32",
        },
        metrics={
            "loss": losses[-1],
            "losses": losses,
            "duration_seconds": float(duration),
            "train_steps": len(losses),
            "tokens": int(tokens),
            "tokens_per_second": float(tokens / duration) if duration > 0 else 0.0,
            "n_params_total": total_params,
            "n_params_base": base_params,
            "n_lora_adapters": int(adapters),
            "n_lora_trainable_params": int(trainable_lora_params),
            "trainable_ratio_pct": float(100 * trainable_lora_params / total_params),
            "base_grad_norm": max(base_gradient_norms),
            "base_gradient_norms": base_gradient_norms,
            "lora_grad_norm": min(lora_gradient_norms),
            "lora_gradient_norms": lora_gradient_norms,
            "parameter_delta_l2": parameter_delta,
            "frozen_parameter_delta_l2": frozen_parameter_delta,
            "logits_shape": list(logits.shape),
        },
        functional_checks=checks,
        dataset_name="synthetic-deterministic-token-lora-shard",
        quality_note="max profile records LoRA adapter fine-tuning systems metrics on deterministic tokens; task-quality check remains future work.",
    )


def run_nanogpt_decode_fp32_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_nanogpt_decode_variant_min(
        workload, output_dir, dtype=torch.float32, dtype_label="fp32"
    )


def run_nanogpt_decode_fp32_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_nanogpt_decode_variant_max(
        workload, output_dir, dtype=torch.float32, dtype_label="fp32"
    )


def run_nanogpt_decode_fp16_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_nanogpt_decode_variant_min(
        workload, output_dir, dtype=torch.float16, dtype_label="fp16"
    )


def run_nanogpt_decode_fp16_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_nanogpt_decode_variant_max(
        workload, output_dir, dtype=torch.float16, dtype_label="fp16"
    )


def run_nanogpt_decode_variant_max(
    workload: Workload,
    output_dir: Path,
    *,
    dtype: torch.dtype,
    dtype_label: str,
) -> dict[str, Any]:
    """Run repeated batch-16 decode on a larger deterministic local model."""
    root = ensure_reference_path()
    from mlperf.reference.cloud.nanogpt_decode import NanoGPTDecode
    from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox

    seed = configured_seed()
    torch.manual_seed(seed)
    config = {
        "vocab_size": 256,
        "n_embd": 64,
        "n_head": 4,
        "n_layer": 2,
        "max_seq_len": 128,
        "batch_size": 16,
        "prefill_ctx": 32,
        "decode_steps": 8,
        "measured_requests": 3,
        "dtype": dtype_label,
    }
    model = NanoGPTWhiteBox(
        vocab_size=config["vocab_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        max_seq_len=config["max_seq_len"],
    ).to(dtype=dtype)
    model.eval()
    results = [
        NanoGPTDecode(
            model,
            prefill_ctx=config["prefill_ctx"],
            decode_steps=config["decode_steps"],
            batch_size=config["batch_size"],
        ).run(emit_sidecar=False)
        for _ in range(config["measured_requests"])
    ]
    throughput_samples = [float(result["output_tokens_per_sec"]) for result in results]
    request_latency_samples = [
        float(result["request_end_to_end_latency_s"]) for result in results
    ]
    last = results[-1]
    generated_ids_by_request = [
        result.get("generated_token_ids") or [] for result in results
    ]
    checks = {
        "completed_configured_requests": len(results) == config["measured_requests"],
        "positive_throughput_each_request": all(
            throughput > 0 for throughput in throughput_samples
        ),
        "decode_steps_match_contract": all(
            result.get("decode_steps") == config["decode_steps"] for result in results
        ),
        "batch_size_matches_contract": all(
            len(generated_ids) == config["batch_size"]
            for generated_ids in generated_ids_by_request
        ),
        "every_request_emitted_all_tokens": all(
            len(tokens) == config["decode_steps"]
            for generated_ids in generated_ids_by_request
            for tokens in generated_ids
        ),
    }
    metrics = {
        **{
            key: float(value) if isinstance(value, float) else value
            for key, value in last.items()
        },
        "n_params": count_params(model),
        "dtype": dtype_label,
        "measured_requests": len(results),
        "output_tokens_per_sec": float(statistics.median(throughput_samples)),
        "output_tokens_per_sec_samples": throughput_samples,
        "request_end_to_end_latency_samples_s": request_latency_samples,
        "total_output_tokens": int(
            config["batch_size"] * config["decode_steps"] * config["measured_requests"]
        ),
    }
    return write_max_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        config=config,
        model_metadata={
            "architecture": "NanoGPTWhiteBox",
            "n_params": count_params(model),
            "dtype": dtype_label,
        },
        metrics=metrics,
        functional_checks=checks,
        dataset_name="synthetic-deterministic-batch16-prompts",
        quality_note=(
            f"max profile records {dtype_label} batch-16 NanoGPT decode systems "
            "metrics on deterministic prompts; random weights provide no task-quality "
            "or checkpoint-inheritance claim."
        ),
        backend=f"pytorch-cpu-{dtype_label}",
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
    from mlperf.reference.cloud.nanogpt_decode import NanoGPTDecode
    from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox

    seed = configured_seed()
    torch.manual_seed(seed)
    model = NanoGPTWhiteBox(
        vocab_size=128, n_embd=32, n_head=4, n_layer=1, max_seq_len=64
    ).to(dtype=dtype)
    result = NanoGPTDecode(model, prefill_ctx=8, decode_steps=4, batch_size=2).run(
        emit_sidecar=False
    )
    metrics = {
        **{
            key: float(value) if isinstance(value, float) else value
            for key, value in result.items()
        },
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
    from mlperf.reference.cloud import nanogpt_decode_spec
    from mlperf.reference.cloud.nanogpt_decode_spec import SpeculativeDecode
    from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox

    seed = configured_seed()
    torch.manual_seed(seed)
    target = NanoGPTWhiteBox(
        vocab_size=128, n_embd=32, n_head=4, n_layer=1, max_seq_len=64
    )
    draft = NanoGPTWhiteBox(
        vocab_size=128, n_embd=16, n_head=4, n_layer=1, max_seq_len=64
    )
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
        **{
            key: float(value) if isinstance(value, float) else value
            for key, value in result.items()
        },
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
    """Run repeated speculative decoding on a larger local target/draft pair."""
    root = ensure_reference_path()
    from mlperf.reference.cloud import nanogpt_decode_spec
    from mlperf.reference.cloud.nanogpt_decode_spec import SpeculativeDecode
    from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox

    seed = configured_seed()
    torch.manual_seed(seed)
    config = {
        "vocab_size": 256,
        "target_n_embd": 64,
        "target_n_layer": 2,
        "draft_n_embd": 32,
        "draft_n_layer": 1,
        "n_head": 4,
        "max_seq_len": 128,
        "batch_size": 4,
        "prefill_ctx": 32,
        "decode_tokens": 8,
        "gamma": 4,
        "measured_requests": 2,
        "dtype": "float32",
    }
    target = NanoGPTWhiteBox(
        vocab_size=config["vocab_size"],
        n_embd=config["target_n_embd"],
        n_head=config["n_head"],
        n_layer=config["target_n_layer"],
        max_seq_len=config["max_seq_len"],
    )
    draft = NanoGPTWhiteBox(
        vocab_size=config["vocab_size"],
        n_embd=config["draft_n_embd"],
        n_head=config["n_head"],
        n_layer=config["draft_n_layer"],
        max_seq_len=config["max_seq_len"],
    )
    target.eval()
    draft.eval()
    original_measure = nanogpt_decode_spec.measure_roofline
    nanogpt_decode_spec.measure_roofline = null_roofline
    try:
        results = [
            SpeculativeDecode(
                target,
                draft,
                prefill_ctx=config["prefill_ctx"],
                decode_tokens=config["decode_tokens"],
                gamma=config["gamma"],
                batch_size=config["batch_size"],
            ).run()
            for _ in range(config["measured_requests"])
        ]
    finally:
        nanogpt_decode_spec.measure_roofline = original_measure
    throughput_samples = [float(result["output_tokens_per_sec"]) for result in results]
    acceptance_samples = [float(result["acceptance_rate"]) for result in results]
    last = results[-1]
    checks = {
        "completed_configured_requests": len(results) == config["measured_requests"],
        "every_request_emitted_all_tokens": all(
            result.get("tokens_emitted") == config["decode_tokens"]
            for result in results
        ),
        "positive_throughput_each_request": all(
            throughput > 0 for throughput in throughput_samples
        ),
        "acceptance_rate_is_bounded": all(
            0.0 <= acceptance <= 1.0 for acceptance in acceptance_samples
        ),
        "decode_used_multiple_cycles": all(
            int(result.get("cycles", 0)) > 0 for result in results
        ),
    }
    target_params = count_params(target)
    draft_params = count_params(draft)
    metrics = {
        **{
            key: float(value) if isinstance(value, float) else value
            for key, value in last.items()
        },
        "target_params": target_params,
        "draft_params": draft_params,
        "n_params_total": target_params + draft_params,
        "measured_requests": len(results),
        "output_tokens_per_sec": float(statistics.median(throughput_samples)),
        "output_tokens_per_sec_samples": throughput_samples,
        "acceptance_rate": float(statistics.median(acceptance_samples)),
        "acceptance_rate_samples": acceptance_samples,
        "total_tokens_emitted": config["decode_tokens"] * config["measured_requests"],
    }
    return write_max_report(
        workload,
        output_dir,
        root=root,
        seed=seed,
        config=config,
        model_metadata={
            "architecture": "NanoGPTWhiteBox speculative target/draft pair",
            "n_params": target_params + draft_params,
            "target_n_params": target_params,
            "draft_n_params": draft_params,
            "dtype": "float32",
        },
        metrics=metrics,
        functional_checks=checks,
        dataset_name="synthetic-deterministic-speculative-prompts",
        quality_note=(
            "max profile records speculative decode control-flow systems metrics on "
            "deterministic prompts; random target and draft weights provide no task-quality "
            "or checkpoint-inheritance claim."
        ),
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
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def write_max_report(
    workload: Workload,
    output_dir: Path,
    *,
    root: Path,
    seed: int,
    config: dict[str, Any],
    model_metadata: dict[str, Any],
    metrics: dict[str, Any],
    functional_checks: dict[str, bool],
    dataset_name: str,
    quality_note: str,
    backend: str = "pytorch-cpu",
) -> dict[str, Any]:
    if not functional_checks:
        raise ValueError("max systems workload must declare functional checks")
    invalid_checks = {
        name: value
        for name, value in functional_checks.items()
        if not isinstance(value, bool)
    }
    if invalid_checks:
        raise TypeError(f"functional checks must be boolean: {invalid_checks}")
    functional_met = all(functional_checks.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / f"{workload.id}_max_report.json").resolve()
    manifest_path = (output_dir / f"{workload.id}_max.provd.json").resolve()
    metrics = dict(metrics)
    metrics["max_micro_shard"] = True
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if functional_met else "quality_failed",
        "backend": backend,
        "data_mode": "synthetic-micro-shard",
        "seed": seed,
        "config": config,
        "model": model_metadata,
        "metrics": metrics,
        "functional_check": {
            "passed": functional_met,
            "checks": functional_checks,
            "contract": workload.raw.get("functional_check"),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "target_met": None,
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
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
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
