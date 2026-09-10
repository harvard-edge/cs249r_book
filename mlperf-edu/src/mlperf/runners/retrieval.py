from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from mlperf.assets import ensure_nanobeir_reranking, sha256_file
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import (
    configured_seed,
    select_torch_device,
    synchronize_device,
)


MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
MODEL_FILES = {
    "config.json": "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc",
    "model.safetensors": "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
    "special_tokens_map.json": "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "tokenizer_config.json": "a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
SENTENCE_TRANSFORMERS_VERSION = "5.5.1"
DATASET_NAMES = ("MSMARCO", "NFCorpus", "NQ")
PUBLISHED_MEAN_NDCG_AT_10 = 0.60716840988382


def _model_file_records(snapshot: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": snapshot / filename,
            "logical_path": filename,
            "role": "weights" if filename == "model.safetensors" else "model-config",
        }
        for filename in MODEL_FILES
    ]


def run_information_retrieval_min(
    workload: Workload, output_dir: Path
) -> dict[str, Any]:
    from transformers import BertConfig, BertForSequenceClassification

    root = find_project_root()
    seed = configured_seed()
    torch.manual_seed(seed)
    config = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        num_labels=1,
    )
    model = BertForSequenceClassification(config)
    input_ids = torch.randint(0, config.vocab_size, (4, 32))
    attention_mask = torch.ones_like(input_ids)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
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
            "pairs": int(input_ids.shape[0]),
            "pairs_per_second": float(input_ids.shape[0] / duration),
            "n_params": sum(parameter.numel() for parameter in model.parameters()),
            "logits_shape": list(logits.shape),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": "The min profile validates compact BERT pair-scoring execution only. It does not use NanoBEIR or support a retrieval-quality claim.",
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
        dataset_name="synthetic-deterministic-query-document-pairs",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_information_retrieval_max(
    workload: Workload, output_dir: Path
) -> dict[str, Any]:
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import (
        CrossEncoderRerankingEvaluator,
    )

    root = find_project_root()
    seed = configured_seed()
    torch.manual_seed(seed)
    device = select_torch_device()
    asset = ensure_nanobeir_reranking(download=True)
    snapshot = _snapshot_model(workload)
    model = CrossEncoder(str(snapshot), device=str(device))
    n_params = sum(parameter.numel() for parameter in model.model.parameters())
    batch_size = int(os.environ.get("MLPERF_EDU_RETRIEVAL_BATCH_SIZE", 32))
    rerank_k = 100
    at_k = 10
    samples_by_dataset = {
        dataset_name: _load_reranking_samples(asset.root, dataset_name, rerank_k)
        for dataset_name in DATASET_NAMES
    }
    canonical_config = workload.raw.get("canonical_max_contract", {}).get("config", {})
    measurement_repetitions = int(canonical_config.get("measurement_repetitions", 1))
    performance_aggregate = str(
        canonical_config.get("performance_aggregate", "single-complete-evaluation")
    )
    if measurement_repetitions != 1:
        raise ValueError("the retrieval contract requires one complete evaluation")
    if performance_aggregate != "single-complete-evaluation":
        raise ValueError(
            "the retrieval contract requires the single-complete-evaluation aggregate"
        )

    representative = samples_by_dataset[DATASET_NAMES[0]][0]
    warmup_pairs = [
        [representative["query"], document] for document in representative["documents"]
    ]
    for warmup_size in sorted({batch_size, len(warmup_pairs) % batch_size} - {0}):
        model.predict(
            warmup_pairs[:warmup_size],
            batch_size=warmup_size,
            show_progress_bar=False,
        )

    def evaluate_once() -> tuple[dict[str, dict[str, float]], dict[str, int]]:
        dataset_metrics: dict[str, dict[str, float]] = {}
        samples_per_dataset: dict[str, int] = {}
        for dataset_name in DATASET_NAMES:
            samples = samples_by_dataset[dataset_name]
            evaluator = CrossEncoderRerankingEvaluator(
                samples=samples,
                at_k=at_k,
                name=f"Nano{dataset_name}_R{rerank_k}",
                write_csv=False,
                show_progress_bar=False,
                batch_size=batch_size,
                always_rerank_positives=True,
            )
            evaluation = evaluator(model)
            prefix = f"Nano{dataset_name}_R{rerank_k}_"
            dataset_metrics[dataset_name] = {
                key.removeprefix(prefix): float(value)
                for key, value in evaluation.items()
                if key.startswith(prefix)
            }
            samples_per_dataset[dataset_name] = len(samples)
        return dataset_metrics, samples_per_dataset

    synchronize_device(device)
    start = time.perf_counter()
    dataset_metrics, samples_per_dataset = evaluate_once()
    synchronize_device(device)
    duration = float(time.perf_counter() - start)
    repetition_seconds = [duration]

    mean_map = float(np.mean([metrics["map"] for metrics in dataset_metrics.values()]))
    mean_mrr = float(
        np.mean([metrics[f"mrr@{at_k}"] for metrics in dataset_metrics.values()])
    )
    mean_ndcg = float(
        np.mean([metrics[f"ndcg@{at_k}"] for metrics in dataset_metrics.values()])
    )
    target = float(workload.quality_value or PUBLISHED_MEAN_NDCG_AT_10)
    tolerance = float((workload.raw.get("quality_target") or {}).get("tolerance", 0.0))
    target_met = mean_ndcg >= target - tolerance

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
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "n_params": n_params,
        },
        "data_mode": "real",
        "dataset": {
            "name": "sentence-transformers/NanoBEIR-en",
            "asset_name": asset.name,
            "source": asset.source,
            "root": str(asset.root),
            "sha256": asset.sha256,
            "n_bytes": asset.n_bytes,
            "splits": [f"Nano{name}" for name in DATASET_NAMES],
        },
        "model_source": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot": str(snapshot),
            "files": {name: f"sha256:{digest}" for name, digest in MODEL_FILES.items()},
            "sentence_transformers_version": SENTENCE_TRANSFORMERS_VERSION,
        },
        "seed": seed,
        "measurement_protocol": workload.raw.get("measurement_protocol", {}),
        "config": {
            "queries_per_dataset": 50,
            "rerank_k": rerank_k,
            "at_k": at_k,
            "batch_size": batch_size,
            "always_rerank_positives": True,
            "measurement_repetitions": measurement_repetitions,
            "performance_aggregate": performance_aggregate,
        },
        "metrics": {
            "mean_ndcg_at_10": mean_ndcg,
            "mean_mrr_at_10": mean_mrr,
            "mean_map": mean_map,
            "dataset_metrics": dataset_metrics,
            "samples_per_dataset": samples_per_dataset,
            "pairs": sum(samples_per_dataset.values()) * rerank_k,
            "duration_seconds": duration,
            "inference_and_evaluation_seconds": duration,
            "inference_and_evaluation_repetition_seconds": repetition_seconds,
            "total_measured_seconds": float(sum(repetition_seconds)),
            "pairs_per_second": (sum(samples_per_dataset.values()) * rerank_k)
            / duration,
            "n_params": n_params,
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "mean_ndcg_at_10",
            "target": target,
            "tolerance": tolerance,
            "direction": "higher",
            "target_met": target_met,
            "quality_required": True,
            "override": False,
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
            "weights": str(snapshot),
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
        weights_name=MODEL_ID,
        weights_revision=MODEL_REVISION,
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


def _load_reranking_samples(
    root: Path, dataset_name: str, rerank_k: int
) -> list[dict[str, Any]]:
    def frame(kind: str) -> pd.DataFrame:
        path = root / kind / f"Nano{dataset_name}-00000-of-00001.parquet"
        return pd.read_parquet(path)

    corpus = frame("corpus")
    queries = frame("queries")
    qrels = frame("qrels")
    bm25 = frame("bm25")
    corpus_mapping = dict(zip(corpus["_id"], corpus["text"]))
    query_mapping = dict(zip(queries["_id"], queries["text"]))
    qrels_mapping: dict[str, set[str]] = {}
    for row in qrels.to_dict("records"):
        corpus_ids = row["corpus-id"]
        relevant = qrels_mapping.setdefault(row["query-id"], set())
        if isinstance(corpus_ids, np.ndarray):
            corpus_ids = corpus_ids.tolist()
        if isinstance(corpus_ids, list):
            relevant.update(str(value) for value in corpus_ids)
        else:
            relevant.add(str(corpus_ids))
    samples = []
    for row in bm25.to_dict("records"):
        corpus_ids = row["corpus-ids"]
        if isinstance(corpus_ids, np.ndarray):
            corpus_ids = corpus_ids.tolist()
        samples.append(
            {
                "query": query_mapping[row["query-id"]],
                "positive": [
                    corpus_mapping[corpus_id]
                    for corpus_id in qrels_mapping[row["query-id"]]
                ],
                "documents": [
                    corpus_mapping[corpus_id]
                    for corpus_id in list(corpus_ids)[:rerank_k]
                ],
            }
        )
    return samples


def _snapshot_model(workload: Workload) -> Path:
    from huggingface_hub import snapshot_download

    model_source = workload.raw.get("model_source") or {}
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(MODEL_FILES),
            local_files_only=os.environ.get("MLPERF_EDU_HF_LOCAL_ONLY", "0") == "1",
        )
    ).resolve()
    for name, expected in MODEL_FILES.items():
        path = snapshot / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(
                f"pinned retrieval model file failed SHA-256 verification: {name}"
            )
    if model_source.get("revision") != MODEL_REVISION:
        raise ValueError("registry retrieval-model revision does not match the runner")
    return snapshot
