from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from mlperf.assets import (
    MINIGO_COMMIT,
    MINIGO_SOURCE_FILES,
    minigo_environment_handoff_contract,
    ensure_minigo_reference,
    minigo_reference_paths,
    sha256_file,
)
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import configured_seed


TARGET_PROFESSIONAL_MOVE_ACCURACY = 0.40
SELF_PLAY_GAMES_PER_GENERATION = 2_000
SELF_PLAY_WORKERS = 16
SEARCH_READOUTS = 200
PLAYOFF_GAMES = 100
PLAYOFF_WIN_RATE = 0.55
IMAGE_ENV = "MLPERF_EDU_MINIGO_IMAGE"
RUNTIME_ENV = "MLPERF_EDU_MINIGO_CONTAINER_RUNTIME"
PRO_GAMES_REVIEW_ENV = "MLPERF_EDU_MINIGO_PRO_GAMES_REVIEWED"
IMAGE_PATTERN = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


def environment_handoff_contract() -> dict[str, Any]:
    """Describe the complete external environment needed for MiniGo quality."""
    return minigo_environment_handoff_contract()


RESUMABLE_WRAPPER = """#!/usr/bin/env bash
set -euo pipefail

seed="$1"
cd /research/reinforcement/minigo
export GOPARAMS=params/final.json
results=/research/results/minigo/final
progress="$results/mlperf_edu_next_iteration"

if [[ ! -d "$results/models" ]]; then
  python3 loop_init.py
  printf '0\n' > "$progress"
fi

if [[ ! -f "$progress" ]]; then
  echo "MiniGo state exists without $progress; refusing an ambiguous resume" >&2
  exit 2
fi

iteration="$(cat "$progress")"
while (( iteration <= 1000 )); do
  python3 loop_selfplay.py "$seed" "$iteration"
  python3 loop_train_eval.py "$seed" "$iteration"
  iteration=$((iteration + 1))
  printf '%s\n' "$iteration" > "$progress"
  if [[ -f TERMINATE_FLAG ]]; then
    exit 0
  fi
done
"""


def preflight_environment() -> dict[str, str]:
    """Validate the immutable legacy GPU runtime without starting a run."""
    if os.environ.get(PRO_GAMES_REVIEW_ENV) != "1":
        raise RuntimeError(
            "MiniGo's four professional SGF inputs are pinned in the MLCommons "
            "archive but still require a release/terms decision. Review them for "
            f"the intended classroom or research use, then set {PRO_GAMES_REVIEW_ENV}=1."
        )
    image = os.environ.get(IMAGE_ENV, "")
    if not image:
        raise RuntimeError(
            "The historical MiniGo runner needs a prepared CUDA/TensorFlow 1.x "
            f"container. Set {IMAGE_ENV} to an immutable image@sha256 digest."
        )
    if not IMAGE_PATTERN.fullmatch(image):
        raise ValueError(
            f"{IMAGE_ENV} must use an immutable image@sha256:<64 hex> reference"
        )

    runtime_value = os.environ.get(RUNTIME_ENV, "docker")
    runtime_candidate = Path(runtime_value).expanduser()
    if runtime_candidate.is_file():
        runtime = str(runtime_candidate.resolve())
    else:
        runtime = shutil.which(runtime_value) or ""
    if not runtime:
        raise FileNotFoundError(
            f"{RUNTIME_ENV} does not resolve to a Docker-compatible executable"
        )

    daemon = subprocess.run(
        [runtime, "info"], check=False, capture_output=True, text=True
    )
    if daemon.returncode != 0:
        raise RuntimeError(
            "The MiniGo container runtime is unavailable. Start the daemon and "
            f"verify GPU support. Details: {daemon.stderr.strip()}"
        )
    inspection = subprocess.run(
        [runtime, "image", "inspect", image],
        check=False,
        capture_output=True,
        text=True,
    )
    if inspection.returncode != 0:
        raise RuntimeError(
            f"The pinned MiniGo runtime image is not present locally: {image}. "
            "Pull or build that exact digest before running the benchmark."
        )
    return {"runtime": runtime, "image": image}


def build_container_command(
    *,
    runtime: str,
    image: str,
    execution_root: Path,
    results_mount: Path,
    seed: int,
) -> list[str]:
    return [
        runtime,
        "run",
        "--rm",
        "--gpus",
        "all",
        "--env",
        "HOME=/research",
        "--mount",
        f"type=bind,src={execution_root},dst=/research/reinforcement",
        "--mount",
        f"type=bind,src={results_mount},dst=/research/results",
        "--workdir",
        "/research/reinforcement/minigo",
        image,
        "bash",
        "/research/reinforcement/mlperf_edu_resume.sh",
        str(seed),
    ]


def parse_quality_events(stats_root: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for path in sorted(stats_root.glob("*.json")):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid MiniGo qmeas JSON at {path}:{line_number}"
                ) from exc
            if event.get("name") in {"puzzle_summary", "eval_summary"}:
                events.append(event)
    puzzle_events = sorted(
        (event for event in events if event.get("name") == "puzzle_summary"),
        key=lambda event: float(event.get("wall_time", 0.0)),
    )
    if not puzzle_events:
        raise ValueError("MiniGo run emitted no professional-move evaluation")
    latest_puzzle = puzzle_events[-1]
    puzzle_value = latest_puzzle.get("value") or {}
    latest_accuracy = float(puzzle_value["total_pct"])
    max_accuracy = max(
        float((event.get("value") or {})["total_pct"]) for event in puzzle_events
    )
    model = str(puzzle_value["model"])

    evaluation_events = sorted(
        (event for event in events if event.get("name") == "eval_summary"),
        key=lambda event: float(event.get("wall_time", 0.0)),
    )
    latest_evaluation = (
        evaluation_events[-1].get("value", {}) if evaluation_events else {}
    )
    return {
        "professional_move_prediction": latest_accuracy,
        "max_professional_move_prediction": max_accuracy,
        "evaluated_models": len(puzzle_events),
        "evaluated_model": model,
        "playoff_win_rate": float(latest_evaluation["win_pct"])
        if "win_pct" in latest_evaluation
        else None,
        "model_promoted": bool(latest_evaluation.get("keep"))
        if latest_evaluation
        else None,
    }


def build_checkpoint_manifest(
    *, results_root: Path, model: str, destination: Path
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    candidates: list[Path] = []
    for directory in (results_root / "models", results_root / "bury_models"):
        if directory.is_dir():
            candidates.extend(
                path for path in directory.glob(f"{model}*") if path.is_file()
            )
    files = tuple(sorted(set(candidates)))
    if not files:
        raise FileNotFoundError(f"MiniGo checkpoint files not found for {model}")
    records = [
        {
            "path": str(path.relative_to(results_root)),
            "sha256": f"sha256:{sha256_file(path)}",
            "n_bytes": path.stat().st_size,
        }
        for path in files
    ]
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True).encode("utf-8") + b"\n")
    payload = {
        "schema": "mlperf-edu-checkpoint-set/0.1",
        "model": model,
        "files": records,
        "merkle_root": f"sha256:{digest.hexdigest()}",
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload, files


def _prepare_execution_tree(source: Path, destination: Path) -> Path:
    if destination.exists():
        wrapper = destination / "mlperf_edu_resume.sh"
        if not wrapper.is_file() or wrapper.read_text() != RESUMABLE_WRAPPER:
            raise RuntimeError(
                f"MiniGo execution tree already exists with unknown state: {destination}"
            )
        return wrapper
    shutil.copytree(source, destination)
    wrapper = destination / "mlperf_edu_resume.sh"
    wrapper.write_text(RESUMABLE_WRAPPER)
    wrapper.chmod(0o755)
    return wrapper


def run_reinforcement_learning_max(
    workload: Workload, output_dir: Path
) -> dict[str, Any]:
    """Run the complete historical MLPerf Training v0.5 MiniGo contract."""
    contract = workload.raw.get("canonical_max_contract") or {}
    config = contract.get("config") or {}
    if (
        int(config.get("board_size", 0)) != 9
        or int(config.get("self_play_games_per_generation", 0))
        != SELF_PLAY_GAMES_PER_GENERATION
        or int(config.get("self_play_workers", 0)) != SELF_PLAY_WORKERS
        or int(config.get("search_readouts", 0)) != SEARCH_READOUTS
        or int(config.get("playoff_games", 0)) != PLAYOFF_GAMES
        or float(config.get("playoff_win_rate", 0.0)) != PLAYOFF_WIN_RATE
    ):
        raise ValueError("registry MiniGo quality contract does not match the runner")

    source_asset = ensure_minigo_reference(download=True)
    paths = minigo_reference_paths()
    environment = preflight_environment()
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_root = (output_dir / "minigo-execution").resolve()
    results_mount = (output_dir / "minigo-container-results").resolve()
    results_mount.mkdir(parents=True, exist_ok=True)
    wrapper_path = _prepare_execution_tree(paths["tensorflow"], execution_root)
    command = build_container_command(
        runtime=environment["runtime"],
        image=environment["image"],
        execution_root=execution_root,
        results_mount=results_mount,
        seed=configured_seed(default=1),
    )
    command_path = (output_dir / "reinforcement-learning_max_command.json").resolve()
    log_path = (output_dir / "reinforcement-learning_max_console.log").resolve()
    report_path = (output_dir / "reinforcement-learning_max_report.json").resolve()
    provenance_path = (output_dir / "reinforcement-learning_max.provd.json").resolve()
    checkpoint_manifest_path = (
        output_dir / "reinforcement-learning_max_checkpoint.json"
    ).resolve()
    command_path.write_text(json.dumps(command, indent=2) + "\n")

    start = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    wall_seconds = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"MiniGo container exited with {completed.returncode}; resumable state "
            f"and logs remain under {output_dir}"
        )

    results_root = results_mount / "minigo" / "final"
    metrics = parse_quality_events(results_root / "stats")
    checkpoint_manifest, checkpoint_files = build_checkpoint_manifest(
        results_root=results_root,
        model=metrics["evaluated_model"],
        destination=checkpoint_manifest_path,
    )
    target = float(workload.quality_value or TARGET_PROFESSIONAL_MOVE_ACCURACY)
    tolerance = float(workload.quality_tolerance or 0.0)
    target_met = metrics["professional_move_prediction"] >= target - tolerance
    playoff_value = metrics["playoff_win_rate"]
    promotion_gate_met = playoff_value is not None and playoff_value >= PLAYOFF_WIN_RATE
    generated_selfplay = tuple(
        sorted(
            path
            for path in (results_root / "data" / "selfplay").rglob("*")
            if path.is_file()
        )
    )
    stats_files = tuple(
        sorted(
            path for path in (results_root / "stats").glob("*.json") if path.is_file()
        )
    )
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if target_met else "quality_failed",
        "backend": "mlperf-training-tensorflow-container",
        "data_mode": "run-generated-self-play-plus-pinned-professional-games",
        "model": {
            "id": "MLPerf-Training-v0.5-MiniGo",
            "revision": MINIGO_COMMIT,
            "checkpoint_model": metrics["evaluated_model"],
            "checkpoint_merkle_root": checkpoint_manifest["merkle_root"],
        },
        "model_source": {
            "repository": "https://github.com/mlcommons/training",
            "revision": MINIGO_COMMIT,
            "training_entrypoint": str(paths["minigo"] / "loop_train_eval.py"),
            "self_play_entrypoint": str(paths["minigo"] / "loop_selfplay.py"),
            "inference_entrypoint": str(paths["minigo"] / "predict_games.py"),
            "checkpoint_manifest": str(checkpoint_manifest_path),
            "critical_files": {
                name: f"sha256:{digest}" for name, digest in MINIGO_SOURCE_FILES.items()
            },
        },
        "dataset": {
            "name": "minigo-self-play",
            "source": source_asset.source,
            "professional_games": 4,
            "generated_selfplay_files": len(generated_selfplay),
            "professional_game_review_asserted": True,
        },
        "evaluator": {
            "repository": "https://github.com/mlcommons/training",
            "revision": MINIGO_COMMIT,
            "professional_move_evaluator": str(paths["minigo"] / "predict_games.py"),
            "tries_per_position": 2,
        },
        "seed": configured_seed(default=1),
        "measurement_protocol": workload.raw.get("measurement_protocol", {}),
        "config": {
            "board_size": 9,
            "self_play_games_per_generation": SELF_PLAY_GAMES_PER_GENERATION,
            "self_play_workers": SELF_PLAY_WORKERS,
            "search_readouts": SEARCH_READOUTS,
            "playoff_games": PLAYOFF_GAMES,
            "playoff_win_rate": PLAYOFF_WIN_RATE,
            "container_image": environment["image"],
            "resumable_orchestration": True,
        },
        "metrics": {**metrics, "wall_seconds": wall_seconds},
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "professional_move_prediction",
            "target": target,
            "tolerance": tolerance,
            "direction": "higher",
            "target_met": target_met,
            "quality_required": True,
            "override": False,
            "secondary_gates": {
                "model_promotion_playoff": {
                    "metric_key": "playoff_win_rate",
                    "value": playoff_value,
                    "target": PLAYOFF_WIN_RATE,
                    "target_met": promotion_gate_met,
                }
            },
        },
        "functional_readiness": {
            "schema": "mlperf-edu-functional-readiness/0.1",
            "stage": "quality-conformance",
            "end_to_end_execution": True,
            "authoritative_quality_contract_executed": True,
            "repeatability_verified": False,
            "promotion_eligible": False,
            "next_stage": "stability" if target_met else "quality-target-review",
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(provenance_path),
            "weights": str(checkpoint_manifest_path),
            "checkpoint_files": [str(path) for path in checkpoint_files],
            "container_results": str(results_mount),
            "execution_tree": str(execution_root),
            "resumable_wrapper": str(wrapper_path),
            "console_log": str(log_path),
            "command": str(command_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload=workload.id,
        scenario=workload.scenario or "training",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        weights_path=checkpoint_manifest_path,
        weights_dtype="float32",
        dataset_name="minigo-self-play-and-professional-games",
        dataset_files=[
            *source_asset.files,
            *generated_selfplay,
            *stats_files,
            command_path,
            log_path,
            wrapper_path,
        ],
        rng_seed=configured_seed(default=1),
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=find_project_root(),
    )
    provenance_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report
