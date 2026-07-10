from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root


def ensure_reference_path() -> Path:
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def run_rag_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_rag(workload, output_dir, profile="min", d_model=32, n_layers=1, n_passages=32, top_k=2, batch_size=1, seq_len=16)


def run_rag_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    n_passages = int(os.environ.get("MLPERF_EDU_RAG_MAX_PASSAGES", "128"))
    return run_rag(workload, output_dir, profile="max", d_model=48, n_layers=2, n_passages=n_passages, top_k=4, batch_size=2, seq_len=32)


def run_rag(
    workload: Workload,
    output_dir: Path,
    *,
    profile: str,
    d_model: int,
    n_layers: int,
    n_passages: int,
    top_k: int,
    batch_size: int,
    seq_len: int,
) -> dict[str, Any]:
    ensure_reference_path()
    from mlperf.reference.cloud.nano_rag_agent import NanoRAGAgent

    seed = 42
    torch.manual_seed(seed)
    model = NanoRAGAgent(
        vocab_size=512,
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        max_seq_len=128,
        n_passages=n_passages,
        top_k=top_k,
    )
    model.retriever.passage_tokens.copy_(
        torch.randint(0, model.vocab_size, model.retriever.passage_tokens.shape)
    )
    input_ids = torch.randint(0, 512, (batch_size, seq_len))
    logits, timings = model.forward_with_timing(input_ids)
    metrics = {
        "encode_latency_ms": float(timings["encode_ms"]),
        "retrieve_latency_ms": float(timings["retrieve_ms"]),
        "generate_latency_ms": float(timings["generate_ms"]),
        "total_latency_ms": float(timings["total_ms"]),
        "queries_per_second": float(batch_size * 1000.0 / timings["total_ms"]) if timings["total_ms"] else 0.0,
        "n_params": count_params(model),
        "n_passages": n_passages,
        "top_k": top_k,
        "logits_shape": list(logits.shape),
    }
    return write_agent_report(workload, output_dir, profile=profile, seed=seed, metrics=metrics, output_shape=list(logits.shape))


def run_codegen_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    retries = int(os.environ.get("MLPERF_EDU_CODEGEN_MIN_RETRIES", "2"))
    return run_codegen(workload, output_dir, profile="min", retries=retries, d_model=32, n_layers=1, batch_size=1, seq_len=24)


def run_codegen_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    retries = int(os.environ.get("MLPERF_EDU_CODEGEN_MAX_RETRIES", "4"))
    return run_codegen(workload, output_dir, profile="max", retries=retries, d_model=48, n_layers=2, batch_size=2, seq_len=32)


def run_codegen(
    workload: Workload,
    output_dir: Path,
    *,
    profile: str,
    retries: int,
    d_model: int,
    n_layers: int,
    batch_size: int,
    seq_len: int,
) -> dict[str, Any]:
    ensure_reference_path()
    from mlperf.reference.cloud.nano_codegen_agent import NanoCodeGenAgent

    seed = 42
    torch.manual_seed(seed)
    model = NanoCodeGenAgent(vocab_size=512, d_model=d_model, n_heads=4, n_layers=n_layers, max_seq_len=192)
    input_ids = torch.randint(0, 512, (batch_size, seq_len))
    results = model.forward_with_timing(input_ids, max_retries=retries)
    metrics = {
        "iterations": len(results["iterations"]),
        "total_tokens_generated": int(results["total_tokens_generated"]),
        "total_latency_ms": float(results["total_ms"]),
        "tokens_per_second": float(results["total_tokens_generated"] * 1000.0 / results["total_ms"]) if results["total_ms"] else 0.0,
        "context_growth_factor": float(results["iterations"][-1]["context_length"] / results["iterations"][0]["context_length"]),
        "n_params": count_params(model),
    }
    return write_agent_report(workload, output_dir, profile=profile, seed=seed, metrics=metrics, details={"iterations": results["iterations"]})


def run_react_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    steps = int(os.environ.get("MLPERF_EDU_REACT_MIN_STEPS", "2"))
    return run_react(workload, output_dir, profile="min", steps=steps, d_model=32, n_layers=1, batch_size=1, seq_len=24)


def run_react_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    steps = int(os.environ.get("MLPERF_EDU_REACT_MAX_STEPS", "4"))
    return run_react(workload, output_dir, profile="max", steps=steps, d_model=48, n_layers=2, batch_size=2, seq_len=32)


def run_react(
    workload: Workload,
    output_dir: Path,
    *,
    profile: str,
    steps: int,
    d_model: int,
    n_layers: int,
    batch_size: int,
    seq_len: int,
) -> dict[str, Any]:
    ensure_reference_path()
    from mlperf.reference.cloud.nano_react_agent import NanoReActAgent

    seed = 42
    torch.manual_seed(seed)
    model = NanoReActAgent(vocab_size=512, d_model=d_model, n_heads=4, n_layers=n_layers, max_seq_len=192, n_tools=4)
    input_ids = torch.randint(0, 512, (batch_size, seq_len))
    results = model.forward_with_timing(input_ids, max_steps=steps)
    metrics = {
        "steps": len(results["steps"]),
        "total_reasoning_ms": float(results["total_reasoning_ms"]),
        "total_tool_ms": float(results["total_tool_ms"]),
        "total_latency_ms": float(results["total_ms"]),
        "final_context_length": int(results["final_context_length"]),
        "final_memory_bytes": int(results["final_memory_bytes"]),
        "n_params": count_params(model),
    }
    return write_agent_report(workload, output_dir, profile=profile, seed=seed, metrics=metrics, details={"steps": results["steps"]})


def run_toolcall_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    n_queries = int(os.environ.get("MLPERF_EDU_TOOLCALL_MIN_QUERIES", "2"))
    return run_toolcall(workload, output_dir, profile="min", n_queries=n_queries, d_model=32, n_layers=1, batch_size=1, seq_len=24)


def run_toolcall_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    n_queries = int(os.environ.get("MLPERF_EDU_TOOLCALL_MAX_QUERIES", "8"))
    return run_toolcall(workload, output_dir, profile="max", n_queries=n_queries, d_model=48, n_layers=2, batch_size=2, seq_len=32)


def run_toolcall(
    workload: Workload,
    output_dir: Path,
    *,
    profile: str,
    n_queries: int,
    d_model: int,
    n_layers: int,
    batch_size: int,
    seq_len: int,
) -> dict[str, Any]:
    ensure_reference_path()
    from mlperf.reference.cloud.nano_toolcall_agent import NanoToolCallAgent

    seed = 42
    torch.manual_seed(seed)
    model = NanoToolCallAgent(vocab_size=512, d_model=d_model, n_heads=4, n_layers=n_layers, max_seq_len=128, n_functions=10)
    input_ids = torch.randint(0, 512, (batch_size, seq_len))
    results = model.forward_with_timing(input_ids, n_queries=n_queries)
    metrics = {
        "total_queries": int(results["total_queries"]),
        "valid_calls": int(results["valid_calls"]),
        "valid_call_rate": float(results["valid_calls"] / max(results["total_queries"], 1)),
        "total_generation_ms": float(results["total_generation_ms"]),
        "total_classification_ms": float(results["total_classification_ms"]),
        "total_dispatch_ms": float(results["total_dispatch_ms"]),
        "total_latency_ms": float(results["total_ms"]),
        "n_params": count_params(model),
    }
    return write_agent_report(workload, output_dir, profile=profile, seed=seed, metrics=metrics, details={"queries": results["queries"]})


def write_agent_report(
    workload: Workload,
    output_dir: Path,
    *,
    profile: str = "min",
    seed: int,
    metrics: dict[str, Any],
    output_shape: list[int] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = ensure_reference_path()
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
        "backend": "pytorch-cpu",
        "data_mode": "synthetic-deterministic",
        "seed": seed,
        "metrics": metrics,
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": f"{profile} profile measures the local agent systems loop; task quality is not score-bearing yet.",
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
        },
    }
    if output_shape is not None:
        report["output_shape"] = output_shape
    if details:
        report["details"] = details
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload=workload.id,
        scenario=workload.scenario or "agent",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        dataset_name="synthetic-deterministic-agent-prompts",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return report


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
