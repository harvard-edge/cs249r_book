from __future__ import annotations

import json
import math
import os
import shutil
import sys
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from mlperf.assets import (
    EVALPLUS_COMMIT,
    HUMANEVAL_PLUS_VERSION,
    ensure_evalplus_evaluator,
    ensure_humaneval_plus,
    humaneval_plus_paths,
    sha256_file,
)
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import (
    configured_seed,
    select_torch_device,
    synchronize_device,
)


MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
MODEL_REVISION = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
MODEL_FILES = {
    "config.json": "b1e58593cd31852f7da5c2fc31ddf6135b9c066c0fd9177a4bbe95717083adff",
    "generation_config.json": "fdaccbcb02f3e1e7914ccb0f69ebe899071ffd27cf825166d823163e156870f2",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model.safetensors": "f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
QWEN_EVALUATOR_REVISION = "dc5af295350a48a4dd04625d3b8f348dc3c4f218"
MAX_NEW_TOKENS = 2048
MINIMUM_PASSING_TASKS = 94
EVALPLUS_REFERENCE_FAILING_TASKS = ("HumanEval/32",)
EVALPLUS_RUNTIME_REVISION = "python3.10-runtime-v1"
EVALPLUS_BASE_IMAGE = (
    "python:3.10-slim@sha256:"
    "c1e4e6c01eb489c422288b2de34b0761ca316f7a2d98e2c33f47659a73ed108a"
)
EVALPLUS_RUNTIME_PACKAGES = (
    "appdirs==1.4.4",
    "numpy==1.26.4",
    "tempdir==0.7.1",
    "termcolor==2.4.0",
    "tqdm==4.66.4",
    "wget==3.2",
)
EVALPLUS_IMAGE = (
    f"mlperf-edu-evalplus:{EVALPLUS_COMMIT[:12]}-{EVALPLUS_RUNTIME_REVISION}"
)
STOP_STRINGS = (
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\n#",
    "\n```",
)
EVALPLUS_DOCKERFILE = f"""FROM {EVALPLUS_BASE_IMAGE}
RUN pip install --no-cache-dir {" ".join(EVALPLUS_RUNTIME_PACKAGES)}
COPY . /evalplus
ENV PYTHONPATH=/evalplus PYTHONHASHSEED=0 HOME=/tmp/evalplus-home
WORKDIR /workspace
ENTRYPOINT [\"python3\", \"-m\", \"evalplus.evaluate\"]
"""


def _model_file_records(snapshot: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": snapshot / filename,
            "logical_path": filename,
            "role": "weights" if filename == "model.safetensors" else "model-config",
        }
        for filename in MODEL_FILES
    ]


def official_qwen_chatml_prompt(prompt: str) -> str:
    """Reproduce Qwen2.5-Coder's published instruct-model evaluation prompt."""
    return f"""<|im_start|>system
You are an intelligent programming assistant to produce Python algorithmic solutions<|im_end|>
<|im_start|>user
Can you complete the following Python function?
```python
{prompt}
```
<|im_end|>
<|im_start|>assistant
```python
"""


def truncate_generation(text: str) -> str:
    boundary = len(text)
    for stop in STOP_STRINGS:
        position = text.find(stop)
        if position >= 0:
            boundary = min(boundary, position)
    return text[:boundary].replace("\t", "    ")


def load_humaneval_plus_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line]
    task_ids = [task.get("task_id") for task in tasks]
    if len(tasks) != 164 or len(set(task_ids)) != 164 or None in task_ids:
        raise ValueError(
            "HumanEval+ must contain exactly 164 uniquely identified tasks"
        )
    if any(not isinstance(task.get("prompt"), str) for task in tasks):
        raise ValueError("every HumanEval+ task must contain a string prompt")
    return tasks


def parse_evalplus_results(
    path: Path, *, expected_task_ids: set[str]
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    evaluation = payload.get("eval")
    if not isinstance(evaluation, dict) or set(evaluation) != expected_task_ids:
        raise ValueError("EvalPlus results do not cover the exact HumanEval+ task set")

    passing: list[str] = []
    failing: list[str] = []
    for task_id in sorted(expected_task_ids):
        task_results = evaluation[task_id]
        if not isinstance(task_results, list) or len(task_results) != 1:
            raise ValueError(f"EvalPlus expected one greedy sample for {task_id}")
        result = task_results[0]
        if result.get("base_status") == result.get("plus_status") == "pass":
            passing.append(task_id)
        else:
            failing.append(task_id)
    return {
        "dataset_hash": payload.get("hash"),
        "passing_task_ids": passing,
        "failing_task_ids": failing,
        "passing_tasks": len(passing),
        "evaluation_tasks": len(evaluation),
        "pass_at_1": len(passing) / len(evaluation),
    }


def evalplus_docker_command(
    *,
    image: str,
    workspace: Path,
    dataset_archive: Path,
    workers: int,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--init",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "512",
        "--ulimit",
        "core=0:0",
        "--ulimit",
        "nofile=1024:1024",
        "--memory",
        "6g",
        "--cpus",
        str(workers),
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=2g",
        "--mount",
        f"type=bind,src={workspace.resolve()},dst=/workspace",
        "--mount",
        (
            f"type=bind,src={dataset_archive.resolve()},"
            "dst=/input/HumanEvalPlus.jsonl.gz,readonly"
        ),
        "--env",
        "HUMANEVAL_OVERRIDE_PATH=/input/HumanEvalPlus.jsonl.gz",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONNOUSERSITE=1",
        image,
        "--dataset",
        "humaneval",
        "--samples",
        "/workspace/samples.jsonl",
        "--parallel",
        str(workers),
    ]


def _snapshot_model(workload: Workload) -> Path:
    from huggingface_hub import snapshot_download

    source = workload.raw.get("model_source") or {}
    if source.get("repo_id") != MODEL_ID or source.get("revision") != MODEL_REVISION:
        raise ValueError("registry code-generation model pin does not match the runner")
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(MODEL_FILES),
            local_files_only=os.environ.get("MLPERF_EDU_HF_LOCAL_ONLY", "0") == "1",
        )
    ).resolve()
    for filename, expected in MODEL_FILES.items():
        path = snapshot / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"pinned Qwen model file failed SHA-256: {filename}")
    return snapshot


HOST_ENGINE_SENTINEL = "host-no-sandbox"


def evalplus_source_root() -> Path:
    from mlperf.assets import ensure_evalplus_evaluator

    return ensure_evalplus_evaluator(download=True).root


def docker_available() -> bool:
    """True when a Docker engine is reachable."""
    try:
        subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def evalplus_host_command(
    *,
    source_root: Path,
    workspace: Path,
    workers: int,
) -> list[str]:
    """Run the pinned EvalPlus source directly, without a container.

    The evaluator commit still matches the image because it is the same
    downloaded source tree. What is lost is the OS sandbox, not comparability.
    """
    return [
        sys.executable,
        "-m",
        "mlperf.runners.evalplus_darwin",
        "--dataset",
        "humaneval",
        "--samples",
        str((workspace / "samples.jsonl").resolve()),
        "--parallel",
        str(workers),
    ]


def _ensure_evalplus_image(source_root: Path) -> tuple[str, bool]:
    if not docker_available():
        raise RuntimeError(
            "The authoritative code-generation quality run requires a running "
            "Docker engine so generated code is never executed on the host. Start "
            "Docker Desktop and rerun the command."
        )

    inspection = subprocess.run(
        ["docker", "image", "inspect", EVALPLUS_IMAGE, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
    )
    built = inspection.returncode != 0
    if built:
        subprocess.run(
            [
                "docker",
                "build",
                "--label",
                f"org.mlperf-edu.evalplus-revision={EVALPLUS_COMMIT}",
                "--label",
                f"org.mlperf-edu.evalplus-runtime={EVALPLUS_RUNTIME_REVISION}",
                "--tag",
                EVALPLUS_IMAGE,
                "--file",
                "-",
                str(source_root),
            ],
            check=True,
            input=EVALPLUS_DOCKERFILE,
            text=True,
        )
        inspection = subprocess.run(
            ["docker", "image", "inspect", EVALPLUS_IMAGE, "--format", "{{.Id}}"],
            check=True,
            capture_output=True,
            text=True,
        )
    return inspection.stdout.strip(), built


def _evaluate_in_sandbox(
    *,
    samples_path: Path,
    dataset_archive: Path,
    output_dir: Path,
    image_id: str,
    image_built: bool,
) -> tuple[Path, dict[str, Any]]:
    workspace = output_dir / ".evalplus"
    workspace.mkdir(parents=True, exist_ok=True)
    container_samples = workspace / "samples.jsonl"
    shutil.copy2(samples_path, container_samples)
    results_path = workspace / "samples_eval_results.json"
    results_path.unlink(missing_ok=True)
    workers = max(1, min(int(os.environ.get("MLPERF_EDU_EVALPLUS_WORKERS", "4")), 8))

    if image_id == HOST_ENGINE_SENTINEL:
        source_root = evalplus_source_root()
        command = evalplus_host_command(
            source_root=source_root, workspace=workspace, workers=workers
        )
        # The evaluator source and this package both have to be importable:
        # the pinned evalplus tree supplies the harness, and mlperf supplies the
        # Darwin rlimit shim that wraps it.
        package_root = Path(__file__).resolve().parents[2]
        env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join([str(source_root), str(package_root)]),
            HUMANEVAL_OVERRIDE_PATH=str(dataset_archive.resolve()),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONNOUSERSITE="1",
        )
        subprocess.run(command, check=True, cwd=workspace, env=env)
    else:
        command = evalplus_docker_command(
            image=EVALPLUS_IMAGE,
            workspace=workspace,
            dataset_archive=dataset_archive,
            workers=workers,
        )
        subprocess.run(command, check=True)
    if not results_path.is_file():
        raise FileNotFoundError("EvalPlus did not produce its expected result artifact")
    if sha256_file(container_samples) != sha256_file(samples_path):
        raise RuntimeError("the sandbox modified the generated sample input")

    if image_id == HOST_ENGINE_SENTINEL:
        # The evaluator commit still matches the container image because this is
        # the same pinned source tree. The OS sandbox is what is missing, so the
        # result is recorded as non-conformant and cannot become a baseline.
        return results_path, {
            "engine": "host",
            "conformant": False,
            "evidence_eligible": False,
            "evalplus_commit": EVALPLUS_COMMIT,
            "sandbox": "none",
            "reason": (
                "Docker was unavailable, so generated code ran directly on the "
                "host without network, filesystem, capability, or resource "
                "isolation. Use the Docker engine for any result that is "
                "intended to be comparable or published."
            ),
            "python": sys.version.split()[0],
            "workers": workers,
        }

    return results_path, {
        "engine": "docker",
        "image": EVALPLUS_IMAGE,
        "image_id": image_id,
        "image_built_for_run": image_built,
        "network": "none",
        "read_only_root": True,
        "capabilities": "all-dropped",
        "no_new_privileges": True,
        "host_user": f"{os.getuid()}:{os.getgid()}",
        "pids_limit": 512,
        "core_limit": 0,
        "open_file_limit": 1024,
        "memory_limit": "6g",
        "workers": workers,
    }


def _validate_evalplus_reference(
    *,
    tasks: list[dict[str, Any]],
    dataset_archive: Path,
    output_dir: Path,
    image_id: str,
) -> dict[str, Any]:
    reference_dir = output_dir / ".evalplus_reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    samples_path = reference_dir / "canonical_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(
                json.dumps(
                    {
                        "task_id": task["task_id"],
                        "solution": task["prompt"] + task["canonical_solution"],
                    }
                )
                + "\n"
            )
    results_path, sandbox = _evaluate_in_sandbox(
        samples_path=samples_path,
        dataset_archive=dataset_archive,
        output_dir=reference_dir,
        image_id=image_id,
        image_built=False,
    )
    result = parse_evalplus_results(
        results_path, expected_task_ids={task["task_id"] for task in tasks}
    )
    if tuple(result["failing_task_ids"]) != EVALPLUS_REFERENCE_FAILING_TASKS:
        raise RuntimeError(
            "EvalPlus reference self-check changed: expected only the pinned "
            f"upstream {EVALPLUS_REFERENCE_FAILING_TASKS[0]} failure, found "
            f"{result['failing_task_ids']}"
        )
    return {
        "passing_tasks": result["passing_tasks"],
        "evaluation_tasks": result["evaluation_tasks"],
        "expected_upstream_failing_tasks": result["failing_task_ids"],
        "results_sha256": f"sha256:{sha256_file(results_path)}",
        "samples_path": str(samples_path.resolve()),
        "results_path": str(results_path.resolve()),
        "sandbox": sandbox,
    }


def _generate_samples(
    *,
    model: Any,
    tokenizer: Any,
    tasks: list[dict[str, Any]],
    device: torch.device,
    samples_path: Path,
) -> tuple[float, int]:
    from transformers import StoppingCriteriaList, StopStringCriteria

    stop_criteria = StoppingCriteriaList(
        [StopStringCriteria(tokenizer, list(STOP_STRINGS))]
    )
    generated_tokens = 0
    synchronize_device(device)
    start = time.perf_counter()
    with samples_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for task in tasks:
            prompt = official_qwen_chatml_prompt(task["prompt"].strip())
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            output = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stop_criteria,
                use_cache=True,
            )
            output_ids = output[0, input_ids.shape[-1] :]
            generated_tokens += int(output_ids.numel())
            solution = truncate_generation(
                tokenizer.decode(output_ids, skip_special_tokens=True)
            )
            handle.write(
                json.dumps(
                    {"task_id": task["task_id"], "solution": solution},
                    ensure_ascii=False,
                )
                + "\n"
            )
    synchronize_device(device)
    duration = time.perf_counter() - start
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("code-generation duration must be finite and positive")
    return duration, generated_tokens


def run_code_generation_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run the pinned Qwen recipe and score all HumanEval+ tasks in Docker."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    contract = workload.raw.get("canonical_max_contract") or {}
    config = contract.get("config") or {}
    if (
        int(config.get("evaluation_tasks", 0)) != 164
        or int(config.get("minimum_passing_tasks", 0)) != MINIMUM_PASSING_TASKS
    ):
        raise ValueError(
            "registry HumanEval+ task-count contract does not match the runner"
        )

    root = find_project_root()
    seed = configured_seed()
    torch.manual_seed(seed)
    device = select_torch_device()
    dataset_asset = ensure_humaneval_plus(download=True)
    evaluator_asset = ensure_evalplus_evaluator(download=True)
    if docker_available():
        image_id, image_built = _ensure_evalplus_image(evaluator_asset.root)
    else:
        # Fall back loudly. The run still uses the pinned evaluator source, but
        # without the container it has no sandbox and is not evidence-eligible.
        print(
            "WARNING: Docker is unavailable. Running EvalPlus directly on the "
            "host.\n"
            "         Model-generated code will execute with no network, "
            "filesystem,\n"
            "         capability, or resource isolation.\n"
            "         This result is marked non-conformant and cannot become a "
            "baseline.",
            file=sys.stderr,
        )
        image_id, image_built = HOST_ENGINE_SENTINEL, False
    dataset_path = humaneval_plus_paths()["dataset"]
    dataset_archive = humaneval_plus_paths()["archive"]
    tasks = load_humaneval_plus_tasks(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_self_check = _validate_evalplus_reference(
        tasks=tasks,
        dataset_archive=dataset_archive,
        output_dir=output_dir,
        image_id=image_id,
    )
    snapshot = _snapshot_model(workload)

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    n_params = sum(parameter.numel() for parameter in model.parameters())

    samples_path = (output_dir / f"{workload.id}_max_samples.jsonl").resolve()
    report_path = (output_dir / f"{workload.id}_max_report.json").resolve()
    manifest_path = (output_dir / f"{workload.id}_max.provd.json").resolve()
    generation_seconds, generated_tokens = _generate_samples(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        device=device,
        samples_path=samples_path,
    )
    results_path, sandbox = _evaluate_in_sandbox(
        samples_path=samples_path,
        dataset_archive=dataset_archive,
        output_dir=output_dir,
        image_id=image_id,
        image_built=image_built,
    )
    evaluation = parse_evalplus_results(
        results_path, expected_task_ids={task["task_id"] for task in tasks}
    )
    score = float(evaluation["pass_at_1"])
    target = float(workload.quality_value or 0.573)
    tolerance = float(workload.quality_tolerance or 0.0)
    target_met = (
        evaluation["passing_tasks"] >= MINIMUM_PASSING_TASKS
        and score + tolerance >= target
    )

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
            "name": dataset_asset.name,
            "version": HUMANEVAL_PLUS_VERSION,
            "source": dataset_asset.source,
            "root": str(dataset_asset.root),
            "sha256": dataset_asset.sha256,
            "n_bytes": dataset_asset.n_bytes,
            "tasks": len(tasks),
        },
        "model_source": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot": str(snapshot),
            "files": {name: f"sha256:{digest}" for name, digest in MODEL_FILES.items()},
        },
        "evaluator": {
            "repository": "https://github.com/evalplus/evalplus",
            "revision": EVALPLUS_COMMIT,
            "source_sha256": evaluator_asset.sha256,
            "qwen_recipe_revision": QWEN_EVALUATOR_REVISION,
            "runtime_revision": EVALPLUS_RUNTIME_REVISION,
            "base_image": EVALPLUS_BASE_IMAGE,
            "runtime_packages": list(EVALPLUS_RUNTIME_PACKAGES),
            "reference_self_check": reference_self_check,
            "transformers_version": transformers.__version__,
            "results_sha256": f"sha256:{sha256_file(results_path)}",
            "sandbox": sandbox,
        },
        "seed": seed,
        "measurement_protocol": workload.raw.get("measurement_protocol", {}),
        "config": {
            "decoding": "greedy",
            "samples_per_task": 1,
            "prompt_format": "qwen2.5-coder-official-chatml",
            "max_new_tokens": MAX_NEW_TOKENS,
            "stop_strings": list(STOP_STRINGS),
            "execution_dtype": "float32",
            "attention_implementation": "eager",
            "evaluation_tasks": len(tasks),
            "minimum_passing_tasks": MINIMUM_PASSING_TASKS,
            "execution": "sandboxed-evalplus",
            "evaluator_reference_self_check": (
                "163-of-164-with-HumanEval-32-exception"
            ),
        },
        "metrics": {
            "humaneval_plus_pass_at_1": score,
            "passing_tasks": evaluation["passing_tasks"],
            "failing_tasks": len(evaluation["failing_task_ids"]),
            "failing_task_ids": evaluation["failing_task_ids"],
            "evaluation_tasks": evaluation["evaluation_tasks"],
            "generated_tokens": generated_tokens,
            "generation_seconds": generation_seconds,
            "duration_seconds": generation_seconds,
            "tokens_per_second": generated_tokens / generation_seconds,
            "n_params": n_params,
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "humaneval_plus_pass_at_1",
            "target": target,
            "minimum_passing_tasks": MINIMUM_PASSING_TASKS,
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
            "samples": str(samples_path),
            "evaluation_results": str(results_path.resolve()),
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
        weights_dtype="bfloat16-source/float32-execution",
        dataset_name="humaneval-plus-and-pinned-evaluator",
        dataset_files=[
            *dataset_asset.files,
            *evaluator_asset.files,
            samples_path,
            results_path,
            reference_self_check["samples_path"],
            reference_self_check["results_path"],
        ],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report
