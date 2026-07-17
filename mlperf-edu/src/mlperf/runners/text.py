from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from mlperf.assets import ensure_sst2, sha256_file, sst2_paths
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import (
    configured_seed,
    select_torch_device,
    synchronize_device,
)


DISTILBERT_REPO_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
DISTILBERT_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
DISTILBERT_HASHES = {
    "config.json": "582122c8f414793d131e10022ce9ba04e3811a9da6389137ee2f18665b4f4d15",
    "model.safetensors": "7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa",
    "tokenizer_config.json": "5ab9097b4149371c5fd52b2d6e26cb6f9c07c0d19fcfdda895b1adad6b57c3e0",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def _model_file_records(snapshot: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": snapshot / filename,
            "logical_path": filename,
            "role": "weights" if filename == "model.safetensors" else "model-config",
        }
        for filename in DISTILBERT_HASHES
    ]


def run_text_classification_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a tiny DistilBERT configuration as a non-quality smoke test."""
    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    root = find_project_root()
    seed = configured_seed()
    torch.manual_seed(seed)
    model = DistilBertForSequenceClassification(
        DistilBertConfig(
            vocab_size=128,
            max_position_embeddings=32,
            n_layers=1,
            n_heads=4,
            dim=32,
            hidden_dim=64,
            dropout=0.0,
            attention_dropout=0.0,
            num_labels=2,
        )
    ).eval()
    inputs = torch.randint(0, 128, (2, 16), dtype=torch.long)
    attention_mask = torch.ones_like(inputs)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_ids=inputs, attention_mask=attention_mask).logits
    duration = time.perf_counter() - start

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
        "backend": "pytorch-cpu",
        "data_mode": "synthetic-deterministic",
        "seed": seed,
        "metrics": {
            "duration_seconds": duration,
            "samples": 2,
            "samples_per_second": 2 / duration,
            "n_params": sum(parameter.numel() for parameter in model.parameters()),
            "logits_shape": list(logits.shape),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": "The min profile validates the DistilBERT execution interface only. It does not use the source checkpoint or SST-2 quality set.",
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
        dataset_name="synthetic-deterministic-tokenized-sentences",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_text_classification_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Evaluate the pinned official DistilBERT checkpoint on GLUE SST-2 dev."""
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    root = find_project_root()
    seed = configured_seed()
    torch.manual_seed(seed)
    device = select_torch_device()
    batch_size = int(
        os.environ.get("MLPERF_EDU_TEXT_CLASSIFICATION_MAX_BATCH_SIZE", 32)
    )
    repetitions = int(
        os.environ.get("MLPERF_EDU_TEXT_CLASSIFICATION_MAX_REPETITIONS", 5)
    )
    max_length = int(os.environ.get("MLPERF_EDU_TEXT_CLASSIFICATION_MAX_LENGTH", 128))
    if batch_size < 1 or repetitions < 1 or max_length < 1:
        raise ValueError(
            "text classification requires positive batch, repetition, and length values"
        )

    asset = ensure_sst2(download=True)
    snapshot = Path(
        snapshot_download(
            repo_id=DISTILBERT_REPO_ID,
            revision=DISTILBERT_REVISION,
            allow_patterns=list(DISTILBERT_HASHES),
            local_files_only=os.environ.get("MLPERF_EDU_HF_LOCAL_ONLY", "0") == "1",
        )
    )
    for filename, expected in DISTILBERT_HASHES.items():
        path = snapshot / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"pinned DistilBERT artifact mismatch: {filename}")

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            snapshot, local_files_only=True
        )
        .to(device)
        .eval()
    )
    sentences, labels = _load_sst2_validation(sst2_paths()["validation"])
    encoded = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    with torch.inference_mode():
        for warmup_size in sorted({batch_size, len(sentences) % batch_size} - {0}):
            warmup = {name: value[:warmup_size] for name, value in encoded.items()}
            model(**warmup).logits
    synchronize_device(device)
    first_outputs: list[torch.Tensor] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for repetition in range(repetitions):
            for index in range(0, len(sentences), batch_size):
                batch = {
                    name: value[index : index + batch_size]
                    for name, value in encoded.items()
                }
                logits = model(**batch).logits
                if repetition == 0:
                    first_outputs.append(logits.detach())
    synchronize_device(device)
    duration = time.perf_counter() - start
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(
            "text-classification inference duration must be finite and positive"
        )

    predictions = torch.cat(first_outputs).argmax(dim=1).cpu()
    accuracy = float((predictions == labels_tensor).float().mean().item())
    target = float(workload.quality_value or 0.9105504587155964)
    target_met = accuracy + float(workload.quality_tolerance or 0.0) >= target
    total_samples = len(sentences) * repetitions
    n_params = sum(parameter.numel() for parameter in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / f"{workload.id}_max_report.json").resolve()
    manifest_path = (output_dir / f"{workload.id}_max.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if target_met else "quality_failed",
        "backend": f"pytorch-{device.type}",
        "model": {
            "id": DISTILBERT_REPO_ID,
            "revision": DISTILBERT_REVISION,
            "n_params": n_params,
        },
        "data_mode": "real",
        "dataset": {
            "name": asset.name,
            "source": asset.source,
            "root": str(asset.root),
            "sha256": asset.sha256,
            "n_bytes": asset.n_bytes,
            "split": "validation",
        },
        "model_source": {
            "repo_id": DISTILBERT_REPO_ID,
            "revision": DISTILBERT_REVISION,
            "files": {
                name: f"sha256:{value}" for name, value in DISTILBERT_HASHES.items()
            },
            "snapshot": str(snapshot),
        },
        "seed": seed,
        "measurement_protocol": workload.raw.get("measurement_protocol", {}),
        "config": {
            "batch_size": batch_size,
            "repetitions": repetitions,
            "samples": len(sentences),
            "max_length": max_length,
            "padding": "max_length",
            "truncation": True,
            "execution_dtype": "float32",
        },
        "metrics": {
            "accuracy": accuracy,
            "evaluation_samples": len(sentences),
            "duration_seconds": duration,
            "inference_seconds": duration,
            "samples": total_samples,
            "samples_per_second": total_samples / duration,
            "n_params": n_params,
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "accuracy",
            "target": target,
            "tolerance": workload.quality_tolerance,
            "direction": "higher",
            "target_met": target_met,
            "quality_required": True,
            "override": False,
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
            "source_weights": str(snapshot),
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
        weights_files=_model_file_records(snapshot),
        weights_name=DISTILBERT_REPO_ID,
        weights_revision=DISTILBERT_REVISION,
        weights_n_params=n_params,
        weights_dtype="float32",
        dataset_name=asset.name,
        dataset_files=list(asset.files),
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def _load_sst2_validation(path: Path) -> tuple[list[str], list[int]]:
    sentences: list[str] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sentences.append(row["sentence"])
            labels.append(int(row["label"]))
    if len(sentences) != 872:
        raise ValueError(
            f"GLUE SST-2 validation split expected 872 rows, found {len(sentences)}"
        )
    return sentences, labels
