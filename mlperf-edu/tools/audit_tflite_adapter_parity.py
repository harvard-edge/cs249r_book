#!/usr/bin/env python3
"""Audit MLPerf Tiny PyTorch adapters against their pinned TFLite graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlperf.assets import (
    cifar10_paths,
    ensure_cifar10,
    ensure_mlperf_tiny_image,
    ensure_mlperf_tiny_kws,
    ensure_mlperf_tiny_vww,
    load_cifar10_dataset,
    mlperf_tiny_image_paths,
    mlperf_tiny_kws_paths,
    mlperf_tiny_vww_paths,
    sha256_file,
)
from mlperf.reference.tiny.mlperf_tiny_kws import load_mlperf_tiny_kws
from mlperf.reference.tiny.mlperf_tiny_resnet import load_mlperf_tiny_float_resnet
from mlperf.reference.tiny.mlperf_tiny_vww import load_mlperf_tiny_vww
from mlperf.registry import load_registry
from mlperf.runners.tiny import (
    _load_mlperf_tiny_kws_accuracy_set,
    _load_mlperf_tiny_vww_accuracy_set,
)
from mlperf.runners.vision import _cifar10_raw_float_tensor


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCHEMA = "mlperf-edu-tflite-adapter-audit/0.2"
WORKLOADS = (
    "image-classification",
    "keyword-spotting",
    "visual-wake-words",
)


def sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def source_status() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": None, "git_dirty": None}
    return {"git_sha": head, "git_dirty": bool(status.strip())}


def resolver_type(name: str):
    try:
        from ai_edge_litert.interpreter import OpResolverType
    except ImportError as exc:
        raise RuntimeError(
            "LiteRT is required. Run `uv run --extra parity python "
            "tools/audit_tflite_adapter_parity.py`."
        ) from exc
    return {
        "auto": OpResolverType.AUTO,
        "builtin": OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        "builtin-ref": OpResolverType.BUILTIN_REF,
    }[name]


def tflite_probabilities(
    model_path: Path,
    inputs_nhwc: np.ndarray,
    *,
    batch_size: int,
    resolver: str,
) -> np.ndarray:
    from ai_edge_litert.interpreter import Interpreter

    outputs: list[np.ndarray] = []
    interpreter = Interpreter(
        model_path=str(model_path),
        experimental_op_resolver_type=resolver_type(resolver),
    )
    allocated_batch_size: int | None = None
    for start in range(0, len(inputs_nhwc), batch_size):
        batch = inputs_nhwc[start : start + batch_size]
        if allocated_batch_size != len(batch):
            input_index = interpreter.get_input_details()[0]["index"]
            interpreter.resize_tensor_input(
                input_index,
                [len(batch), *inputs_nhwc.shape[1:]],
                strict=False,
            )
            interpreter.allocate_tensors()
            allocated_batch_size = len(batch)
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        interpreter.set_tensor(
            input_detail["index"], batch.astype(input_detail["dtype"], copy=False)
        )
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(output_detail["index"]).copy())
    return np.concatenate(outputs)


def pytorch_probabilities(
    model: torch.nn.Module,
    inputs_nchw: torch.Tensor,
    *,
    batch_size: int,
    apply_softmax: bool,
) -> np.ndarray:
    outputs: list[torch.Tensor] = []
    model = model.cpu().eval()
    with torch.inference_mode():
        for start in range(0, len(inputs_nchw), batch_size):
            output = model(inputs_nchw[start : start + batch_size])
            outputs.append(torch.softmax(output, dim=1) if apply_softmax else output)
    return torch.cat(outputs).cpu().numpy()


def comparison_summary(
    pytorch_output: np.ndarray,
    tflite_output: np.ndarray,
    labels: np.ndarray,
    *,
    quality_target: float,
) -> dict[str, Any]:
    pytorch_predictions = pytorch_output.argmax(axis=1)
    tflite_predictions = tflite_output.argmax(axis=1)
    disagreements = np.flatnonzero(pytorch_predictions != tflite_predictions)
    pytorch_accuracy = float(np.mean(pytorch_predictions == labels))
    tflite_accuracy = float(np.mean(tflite_predictions == labels))
    return {
        "sample_count": int(len(labels)),
        "prediction_disagreement_count": int(len(disagreements)),
        "prediction_disagreement_indices": disagreements.astype(int).tolist(),
        "exact_prediction_parity": len(disagreements) == 0,
        "pytorch_accuracy": pytorch_accuracy,
        "tflite_accuracy": tflite_accuracy,
        "quality_target": quality_target,
        "pytorch_quality_pass": pytorch_accuracy >= quality_target,
        "tflite_quality_pass": tflite_accuracy >= quality_target,
        "max_absolute_probability_error": float(
            np.max(np.abs(pytorch_output - tflite_output))
        ),
        "mean_absolute_probability_error": float(
            np.mean(np.abs(pytorch_output - tflite_output))
        ),
    }


def image_classification_inputs() -> tuple[
    torch.nn.Module, torch.Tensor, np.ndarray, Path, dict[str, Any]
]:
    dataset_asset = ensure_cifar10(download=False)
    evaluation_asset = ensure_mlperf_tiny_image(download=False)
    paths = mlperf_tiny_image_paths()
    indices = np.load(paths["performance_indices"], allow_pickle=False)
    dataset = load_cifar10_dataset(
        root=cifar10_paths()["root"],
        train=False,
        download=False,
        transform=_cifar10_raw_float_tensor,
    )
    samples = [dataset[int(index)] for index in indices]
    inputs = torch.stack([image for image, _label in samples])
    labels = np.asarray([label for _image, label in samples], dtype=np.int64)
    return (
        load_mlperf_tiny_float_resnet(paths["float_model"]),
        inputs,
        labels,
        paths["float_model"],
        {
            "dataset_sha256": dataset_asset.sha256,
            "evaluation_bundle_sha256": evaluation_asset.sha256,
            "performance_indices_sha256": f"sha256:{sha256_file(paths['performance_indices'])}",
        },
    )


def keyword_spotting_inputs() -> tuple[
    torch.nn.Module, torch.Tensor, np.ndarray, Path, dict[str, Any]
]:
    dataset_asset = ensure_mlperf_tiny_kws(download=False)
    paths = mlperf_tiny_kws_paths()
    model, metadata = load_mlperf_tiny_kws(paths["float_model"], paths["int8_model"])
    inputs, labels = _load_mlperf_tiny_kws_accuracy_set(
        dataset_asset.root,
        scale=float(metadata["input_scale"]),
        zero_point=int(metadata["input_zero_point"]),
    )
    return (
        model,
        inputs,
        labels.numpy(),
        paths["float_model"],
        {
            "dataset_sha256": dataset_asset.sha256,
            "source_int8_model_sha256": f"sha256:{sha256_file(paths['int8_model'])}",
            "adapter": metadata["adapter"],
        },
    )


def visual_wake_words_inputs() -> tuple[
    torch.nn.Module, torch.Tensor, np.ndarray, Path, dict[str, Any]
]:
    dataset_asset = ensure_mlperf_tiny_vww(download=False)
    paths = mlperf_tiny_vww_paths()
    inputs, labels = _load_mlperf_tiny_vww_accuracy_set(dataset_asset.root)
    return (
        load_mlperf_tiny_vww(paths["float_model"]),
        inputs,
        labels.numpy(),
        paths["float_model"],
        {
            "dataset_sha256": dataset_asset.sha256,
            "source_int8_model_sha256": f"sha256:{sha256_file(paths['int8_model'])}",
        },
    )


def audit_workload(
    workload: str, *, batch_size: int, resolvers: list[str]
) -> dict[str, Any]:
    loaders = {
        "image-classification": (image_classification_inputs, True),
        "keyword-spotting": (keyword_spotting_inputs, True),
        "visual-wake-words": (visual_wake_words_inputs, False),
    }
    loader, apply_softmax = loaders[workload]
    contract = load_registry()[workload]
    if contract.quality_direction != "higher" or not isinstance(
        contract.quality_value, (int, float)
    ):
        raise ValueError(f"{workload} does not declare a higher-is-better target")
    quality_target = float(contract.quality_value)
    model, inputs, labels, model_path, provenance = loader()
    pytorch_output = pytorch_probabilities(
        model,
        inputs,
        batch_size=batch_size,
        apply_softmax=apply_softmax,
    )
    inputs_nhwc = inputs.numpy().transpose(0, 2, 3, 1)
    runtime_results: dict[str, Any] = {}
    for resolver in resolvers:
        tflite_output = tflite_probabilities(
            model_path,
            inputs_nhwc,
            batch_size=batch_size,
            resolver=resolver,
        )
        runtime_results[resolver] = comparison_summary(
            pytorch_output,
            tflite_output,
            labels,
            quality_target=quality_target,
        )
    strict_pass = all(
        result["exact_prediction_parity"]
        and result["pytorch_quality_pass"]
        and result["tflite_quality_pass"]
        for result in runtime_results.values()
    )
    return {
        "status": "passed" if strict_pass else "failed",
        "strict_policy": "exact top-1 prediction parity and quality pass on every selected LiteRT resolver",
        "batch_size": batch_size,
        # Keep committed audit summaries portable and free of workstation paths.
        "model_file": model_path.name,
        "model_sha256": f"sha256:{sha256_file(model_path)}",
        "provenance": provenance,
        "runtime_results": runtime_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        action="append",
        choices=WORKLOADS,
        help="Workload to audit; repeat the option to select multiple (default: all).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--resolver",
        action="append",
        choices=("auto", "builtin", "builtin-ref"),
        help="LiteRT resolver to audit; repeat to select multiple (default: auto).",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    workloads = list(dict.fromkeys(args.workload or WORKLOADS))
    resolvers = list(dict.fromkeys(args.resolver or ["auto"]))
    try:
        results = {
            workload: audit_workload(
                workload, batch_size=args.batch_size, resolvers=resolvers
            )
            for workload in workloads
        }
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    payload = {
        "schema": AUDIT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed"
        if all(result["status"] == "passed" for result in results.values())
        else "failed",
        "source": source_status(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ai_edge_litert": version("ai-edge-litert"),
            "numpy": np.__version__,
        },
        "workloads": results,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(f"wrote {args.output} ({sha256_bytes(serialized.encode())})")
    else:
        print(serialized, end="")
    for workload, result in results.items():
        detail = ", ".join(
            f"{resolver}: {summary['prediction_disagreement_count']} disagreement(s), "
            f"PyTorch {summary['pytorch_accuracy']:.3%}, TFLite {summary['tflite_accuracy']:.3%}"
            for resolver, summary in result["runtime_results"].items()
        )
        print(f"{workload}: {result['status']} ({detail})")
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
