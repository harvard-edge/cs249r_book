from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import (
    configured_seed,
    select_torch_device,
    synchronize_device,
)


def _write_functional_report(
    workload: Workload,
    output_dir: Path,
    *,
    profile: str,
    device: torch.device,
    duration: float,
    metrics: dict[str, Any],
    probe: str,
) -> dict[str, Any]:
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"{workload.id} duration must be finite and positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / f"{workload.id}_{profile}_report.json").resolve()
    manifest_path = (output_dir / f"{workload.id}_{profile}.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": profile,
        "status": "passed",
        "backend": f"pytorch-{device.type}",
        "data_mode": "synthetic-deterministic-functional-probe",
        "seed": configured_seed(),
        "metrics": {"duration_seconds": duration, **metrics},
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "target_met": None,
            "note": (
                "This functional-setup probe does not execute the authoritative "
                "model, complete dataset, or published quality evaluator."
            ),
        },
        "functional_readiness": {
            "schema": "mlperf-edu-functional-readiness/0.1",
            "stage": "functional",
            "probe": probe,
            "end_to_end_execution": True,
            "authoritative_quality_contract_executed": False,
            "repeatability_verified": False,
            "promotion_eligible": False,
            "next_stage": "quality-conformance",
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload=workload.id,
        scenario=workload.scenario or "offline",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        dataset_name=f"{workload.id}-synthetic-functional-probe",
        dataset_files=[],
        rng_seed=configured_seed(),
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=find_project_root(),
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def _run_probe(
    workload: Workload,
    output_dir: Path,
    *,
    profile: str,
    probe: Callable[[torch.device, str], tuple[float, dict[str, Any]]],
    label: str,
) -> dict[str, Any]:
    seed = configured_seed()
    torch.manual_seed(seed)
    device = select_torch_device()
    duration, metrics = probe(device, profile)
    return _write_functional_report(
        workload,
        output_dir,
        profile=profile,
        device=device,
        duration=duration,
        metrics=metrics,
        probe=label,
    )


def _causal_generation_probe(
    device: torch.device, profile: str
) -> tuple[float, dict[str, Any]]:
    from transformers import GPT2Config, GPT2LMHeadModel

    generated_tokens = 8 if profile == "min" else 16
    config = GPT2Config(
        vocab_size=128,
        bos_token_id=1,
        eos_token_id=2,
        n_positions=64,
        n_ctx=64,
        n_embd=32 if profile == "min" else 64,
        n_layer=1 if profile == "min" else 2,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config).to(device).eval()
    sequence = torch.tensor([[2, 11, 7, 19, 5, 23, 3, 29]], device=device)
    synchronize_device(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(generated_tokens):
            logits = model(input_ids=sequence).logits[:, -1, :]
            sequence = torch.cat((sequence, logits.argmax(dim=-1, keepdim=True)), dim=1)
    synchronize_device(device)
    duration = time.perf_counter() - start
    output = sequence[:, -generated_tokens:].detach().cpu().numpy().tobytes()
    return duration, {
        "prompt_tokens": 8,
        "generated_tokens": generated_tokens,
        "tokens_per_second": generated_tokens / duration,
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "output_token_sha256": hashlib.sha256(output).hexdigest(),
        "functional_check": "autoregressive-decode-completed",
    }


def _function_calling_probe(
    device: torch.device, profile: str
) -> tuple[float, dict[str, Any]]:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    fixture = {
        "name": "weather_lookup",
        "arguments": {"city": "Zurich", "unit": "celsius"},
    }
    encoded_fixture = json.dumps(fixture, separators=(",", ":")).encode("utf-8")
    sequence = torch.tensor([[1, 11, 7, 19]], device=device)
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=32 if profile == "min" else 64,
        intermediate_size=64 if profile == "min" else 128,
        num_hidden_layers=1 if profile == "min" else 2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    model = Qwen2ForCausalLM(config).to(device).eval()
    generated: list[int] = []
    synchronize_device(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for required_token in encoded_fixture:
            logits = model(input_ids=sequence).logits[:, -1, :]
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("function-calling decoder produced nonfinite logits")
            constrained = torch.tensor([[required_token]], device=device)
            sequence = torch.cat((sequence, constrained), dim=1)
            generated.append(required_token)
    synchronize_device(device)
    duration = time.perf_counter() - start
    generated_bytes = bytes(generated)
    parsed = json.loads(generated_bytes)
    ast_valid = (
        parsed.get("name") == "weather_lookup"
        and isinstance(parsed.get("arguments"), dict)
        and set(parsed["arguments"]) == {"city", "unit"}
    )
    if not ast_valid:
        raise RuntimeError("function-calling functional probe failed")
    return duration, {
        "prompt_tokens": 4,
        "generated_tokens": len(generated),
        "decoder_logits": len(generated) * config.vocab_size,
        "grammar_constraint_steps": len(generated),
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "ast_fixture_valid": True,
        "functional_check": "grammar-constrained-generation-and-ast-evaluator-completed",
    }


class _DlrmProbe(nn.Module):
    def __init__(self, *, table_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(table_size, embedding_dim) for _ in range(4)]
        )
        self.dense = nn.Sequential(nn.Linear(13, 32), nn.ReLU(), nn.Linear(32, 16))
        self.top = nn.Sequential(
            nn.Linear(16 + 4 * embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        embedded = [
            table(sparse[:, index]) for index, table in enumerate(self.embeddings)
        ]
        features = torch.cat((self.dense(dense), *embedded), dim=1)
        return self.top(features).squeeze(1)


def _recommendation_probe(
    device: torch.device, profile: str
) -> tuple[float, dict[str, Any]]:
    batch_size = 32 if profile == "min" else 128
    table_size = 256 if profile == "min" else 1024
    model = _DlrmProbe(table_size=table_size, embedding_dim=8).to(device).eval()
    dense = torch.randn(batch_size, 13, device=device)
    sparse = torch.randint(0, table_size, (batch_size, 4), device=device)
    synchronize_device(device)
    start = time.perf_counter()
    with torch.inference_mode():
        probabilities = model(dense, sparse).sigmoid()
    synchronize_device(device)
    duration = time.perf_counter() - start
    if not torch.isfinite(probabilities).all().item():
        raise RuntimeError("recommendation functional probe produced nonfinite values")
    return duration, {
        "samples": batch_size,
        "samples_per_second": batch_size / duration,
        "embedding_tables": len(model.embeddings),
        "embedding_table_rows": table_size,
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "prediction_min": float(probabilities.min().item()),
        "prediction_max": float(probabilities.max().item()),
        "functional_check": "dense-sparse-interaction-completed",
    }


def _image_generation_probe(
    device: torch.device, profile: str
) -> tuple[float, dict[str, Any]]:
    batch_size = 2 if profile == "min" else 8
    steps = 4 if profile == "min" else 12
    denoiser = (
        nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )
        .to(device)
        .eval()
    )
    images = torch.randn(batch_size, 3, 16, 16, device=device)
    synchronize_device(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for step in range(steps):
            images = images - denoiser(images) / float(steps - step + 1)
        images = images.sigmoid()
    synchronize_device(device)
    duration = time.perf_counter() - start
    if not torch.isfinite(images).all().item():
        raise RuntimeError(
            "image-generation functional probe produced nonfinite pixels"
        )
    digest = hashlib.sha256(images.detach().cpu().numpy().tobytes()).hexdigest()
    return duration, {
        "images": batch_size,
        "sampler_steps": steps,
        "network_evaluations": batch_size * steps,
        "images_per_second": batch_size / duration,
        "n_params": sum(parameter.numel() for parameter in denoiser.parameters()),
        "image_tensor_sha256": digest,
        "pixel_min": float(images.min().item()),
        "pixel_max": float(images.max().item()),
        "functional_check": "iterative-denoising-completed",
    }


class _MiniGoProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.policy = nn.Linear(16 * 9 * 9, 9 * 9 + 1)
        self.value = nn.Sequential(
            nn.Linear(16 * 9 * 9, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh()
        )

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(board).flatten(1)
        return self.policy(hidden), self.value(hidden).squeeze(1)


def _reinforcement_learning_probe(
    device: torch.device, profile: str
) -> tuple[float, dict[str, Any]]:
    readouts = 8 if profile == "min" else 32
    model = _MiniGoProbe().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    board = torch.zeros(1, 2, 9, 9, device=device)
    selected_moves: list[int] = []
    synchronize_device(device)
    start = time.perf_counter()
    for turn in range(readouts):
        policy, value = model(board)
        occupied = board.sum(dim=1).flatten(1).bool()
        masked = policy[:, : 9 * 9].masked_fill(occupied, float("-inf"))
        move = int(masked.argmax(dim=1).item())
        selected_moves.append(move)
        row, column = divmod(move, 9)
        board[0, turn % 2, row, column] = 1.0
    target_policy = torch.tensor([selected_moves[-1]], device=device)
    policy, value = model(board)
    loss = nn.functional.cross_entropy(policy, target_policy) + value.square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    synchronize_device(device)
    duration = time.perf_counter() - start
    if len(set(selected_moves)) != len(selected_moves) or not math.isfinite(
        float(loss.item())
    ):
        raise RuntimeError("reinforcement-learning functional probe failed")
    return duration, {
        "board_size": 9,
        "self_play_moves": readouts,
        "unique_legal_moves": len(set(selected_moves)),
        "training_steps": 1,
        "training_loss": float(loss.item()),
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "functional_check": "self-play-policy-value-training-step-completed",
    }


def run_code_generation_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return _run_probe(
        workload,
        output_dir,
        profile="min",
        probe=_causal_generation_probe,
        label="autoregressive-code-generation",
    )


def run_code_generation_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return _run_probe(
        workload,
        output_dir,
        profile="max",
        probe=_causal_generation_probe,
        label="autoregressive-code-generation",
    )


def run_function_calling_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return _run_probe(
        workload,
        output_dir,
        profile="min",
        probe=_function_calling_probe,
        label="grammar-constrained-generation-and-ast-evaluation",
    )


def run_function_calling_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return _run_probe(
        workload,
        output_dir,
        profile="max",
        probe=_function_calling_probe,
        label="grammar-constrained-generation-and-ast-evaluation",
    )


def run_recommendation_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return _run_probe(
        workload,
        output_dir,
        profile="min",
        probe=_recommendation_probe,
        label="dlrm-dense-sparse-interaction",
    )


def run_recommendation_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return _run_probe(
        workload,
        output_dir,
        profile="max",
        probe=_recommendation_probe,
        label="dlrm-dense-sparse-interaction",
    )


def run_image_generation_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return _run_probe(
        workload,
        output_dir,
        profile="min",
        probe=_image_generation_probe,
        label="iterative-image-denoising",
    )


def run_image_generation_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return _run_probe(
        workload,
        output_dir,
        profile="max",
        probe=_image_generation_probe,
        label="iterative-image-denoising",
    )


def run_reinforcement_learning_min(
    workload: Workload, output_dir: Path
) -> dict[str, Any]:
    return _run_probe(
        workload,
        output_dir,
        profile="min",
        probe=_reinforcement_learning_probe,
        label="minigo-policy-value-self-play",
    )


def run_reinforcement_learning_max(
    workload: Workload, output_dir: Path
) -> dict[str, Any]:
    return _run_probe(
        workload,
        output_dir,
        profile="max",
        probe=_reinforcement_learning_probe,
        label="minigo-policy-value-self-play",
    )
