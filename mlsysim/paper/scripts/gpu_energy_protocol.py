"""Plan, record, and integrity rules shared by the GPU energy runner and its analysis.

Standard library only, so the same file runs on a GPU worker that has neither
mlsysim nor its dependencies installed. ``measure_gpu_energy.py`` writes trial
records (one JSON line each, in ``trials.jsonl``) and a manifest with these
helpers; ``analyze_gpu_energy.py`` reads them back with the same helpers and
refuses anything that does not verify.

Energy comes from the driver's cumulative board-energy counter
(``nvmlDeviceGetTotalEnergyConsumption``, millijoules). Polled power samples are
kept as a cross-check: on recent GPUs ``nvmlDeviceGetPowerUsage`` returns a
moving average that refreshes more slowly than a fast polling loop, so a high
sample rate does not mean independent readings. Each run also records a short
probe of how often the counter and the power reading actually change, so the
analysis reports the refresh interval it measured instead of assuming one.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

PLAN_SCHEMA = "mlsysim-gpu-energy-plan/1"
RECORD_SCHEMA = "mlsysim-gpu-energy-trial/1"
MANIFEST_SCHEMA = "mlsysim-gpu-energy-manifest/1"
# One JSON Lines file per run. The paper's timing receipt hashes every .json and
# .csv under paper/data, so per-trial files would add hundreds of entries to it;
# the receipt hashes manifest.json, and the manifest hashes this file.
TRIALS_FILE = "trials.jsonl"

# Phases each trial records, and the phases the summary reports per unit of work.
PHASES = {"generate": ("idle", "prefill", "generate"), "train": ("idle", "train")}
REPORTED = {"generate": ("prefill", "decode"), "train": ("train",)}
SPLITS = ("calibration", "heldout")
SEQ_LEN_RULES = ("mean_context",)

# Two-sided 95% Student-t quantiles by degrees of freedom; df > 30 uses the normal value.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
         23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
         30: 2.042}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def load_plan(path: Path) -> dict[str, Any]:
    """Read and validate a measurement plan; raise ValueError listing every problem."""
    plan = json.loads(Path(path).read_text())
    problems = plan_problems(plan)
    if problems:
        raise ValueError("invalid plan:\n  " + "\n  ".join(problems))
    return plan


def plan_problems(plan: dict[str, Any]) -> list[str]:
    """Everything wrong with a plan, empty when it is usable."""
    problems = []
    if plan.get("schema") != PLAN_SCHEMA:
        problems.append(f"schema must be {PLAN_SCHEMA!r}")
    for key in ("trials", "warmup_trials"):
        if not isinstance(plan.get(key), int) or plan[key] < (2 if key == "trials" else 0):
            problems.append(f"{key} must be an integer (trials >= 2 for an interval)")
    for key in ("idle_seconds", "phase_min_seconds", "sample_interval_s", "counter_probe_seconds"):
        value = plan.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            problems.append(f"{key} must be a positive number")
    devices = plan.get("devices", {})
    for name, device in devices.items():
        for key in ("registry", "modal_gpu", "expected_name"):
            if not device.get(key):
                problems.append(f"device {name}: missing {key}")
    workloads = plan.get("workloads", {})
    for name, workload in workloads.items():
        if workload.get("kind") not in PHASES:
            problems.append(f"workload {name}: kind must be one of {sorted(PHASES)}")
        if not workload.get("registry"):
            problems.append(f"workload {name}: missing registry")
        if workload.get("kind") == "generate":
            problems += [f"workload {name}: {p}" for p in _generate_spec_problems(workload)]
        if workload.get("kind") == "train" and not (
            isinstance(workload.get("steps"), int) and workload["steps"] > 0
        ):
            problems.append(f"workload {name}: train needs a positive integer steps")
    seen = set()
    for config in plan.get("configs", []):
        cid = config.get("id")
        if not cid or cid in seen:
            problems.append(f"config id {cid!r} is missing or duplicated")
        seen.add(cid)
        if config.get("device") not in devices:
            problems.append(f"config {cid}: unknown device {config.get('device')!r}")
        workload = workloads.get(config.get("workload"))
        if workload is None:
            problems.append(f"config {cid}: unknown workload {config.get('workload')!r}")
            continue
        if config.get("split") not in SPLITS:
            problems.append(f"config {cid}: split must be one of {SPLITS}")
        if not isinstance(config.get("batch"), int) or config["batch"] < 1:
            problems.append(f"config {cid}: batch must be a positive integer")
        efficiency = config.get("efficiency", workload.get("efficiency"))
        if not isinstance(efficiency, (int, float)) or not 0 < efficiency <= 1:
            problems.append(f"config {cid}: efficiency must be declared in (0, 1] before measuring")
        if workload["kind"] == "generate":
            for key in ("prompt_tokens", "output_tokens"):
                if not isinstance(config.get(key), int) or config[key] < (2 if key == "output_tokens" else 1):
                    problems.append(f"config {cid}: {key} must be a positive integer (output_tokens >= 2)")
            problems += [f"config {cid}: {p}" for p in _engine_capacity_problems(workload, config)]
    if not seen:
        problems.append("plan has no configs")
    return problems


def _generate_spec_problems(workload: dict[str, Any]) -> list[str]:
    problems = []
    if not workload.get("hf_model"):
        problems.append("generate needs hf_model")
    if not re.fullmatch(r"[0-9a-f]{40}", str(workload.get("hf_revision", ""))):
        problems.append("hf_revision must be a 40-character commit hash, so the weights are pinned")
    weights = workload.get("weights_sha256")
    if not isinstance(weights, dict) or not weights or not all(
        re.fullmatch(r"[0-9a-f]{64}", str(v)) for v in weights.values()
    ):
        problems.append("weights_sha256 must map each weight file to its 64-character SHA-256")
    if workload.get("model_seq_len") not in SEQ_LEN_RULES:
        problems.append(f"model_seq_len must be one of {SEQ_LEN_RULES}")
    if not workload.get("attention_backend"):
        problems.append("generate needs attention_backend")
    engine = workload.get("vllm")
    if not isinstance(engine, dict):
        return problems + ["generate needs a vllm settings object"]
    for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens"):
        if not isinstance(engine.get(key), int) or engine[key] < 1:
            problems.append(f"vllm.{key} must be a positive integer")
    # Every trial repeats the same prompts, so a prefix cache would skip the prefill being measured.
    if engine.get("enable_prefix_caching") is not False:
        problems.append("vllm.enable_prefix_caching must be false")
    # Chunked prefill would interleave prefill with decode steps inside one generation.
    if engine.get("enable_chunked_prefill") is not False:
        problems.append("vllm.enable_chunked_prefill must be false")
    return problems


def _engine_capacity_problems(workload: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """The engine must hold the whole batch at once and prefill it in one step."""
    engine = workload.get("vllm")
    if not isinstance(engine, dict) or not all(
        isinstance(config.get(k), int) for k in ("batch", "prompt_tokens", "output_tokens")
    ):
        return []
    problems = []
    if isinstance(engine.get("max_num_seqs"), int) and config["batch"] > engine["max_num_seqs"]:
        problems.append("batch exceeds vllm.max_num_seqs")
    tokens = config["batch"] * config["prompt_tokens"]
    if isinstance(engine.get("max_num_batched_tokens"), int) and tokens > engine["max_num_batched_tokens"]:
        problems.append("batch x prompt_tokens exceeds vllm.max_num_batched_tokens")
    length = config["prompt_tokens"] + config["output_tokens"]
    if isinstance(engine.get("max_model_len"), int) and length > engine["max_model_len"]:
        problems.append("prompt_tokens + output_tokens exceeds vllm.max_model_len")
    return problems


def config_by_id(plan: dict[str, Any], config_id: str) -> dict[str, Any]:
    for config in plan["configs"]:
        if config["id"] == config_id:
            return config
    raise KeyError(config_id)


def efficiency_of(plan: dict[str, Any], config: dict[str, Any]) -> float:
    return float(config.get("efficiency", plan["workloads"][config["workload"]].get("efficiency")))


def model_seq_len(rule: str, prompt_tokens: int, output_tokens: int) -> int:
    """Context length at which the decode estimate is evaluated."""
    if rule == "mean_context":
        return prompt_tokens + output_tokens // 2
    raise ValueError(f"unknown model_seq_len rule {rule!r}")


def counter_delta_mj(start: int, end: int, bits: int = 64) -> int:
    """Energy between two cumulative counter readings, allowing one wraparound."""
    if start < 0 or end < 0:
        raise ValueError("energy counter readings must be nonnegative")
    return end - start if end >= start else end + (1 << bits) - start


def integrate_samples(samples: Sequence[Sequence[float]], t0: float, t1: float) -> float | None:
    """Trapezoidal energy in joules from ``(seconds, milliwatts)`` samples inside [t0, t1].

    Returns None when fewer than two samples fall inside the window. The result
    only covers the span between the first and last sample in the window, so it
    is a cross-check on the counter, not a replacement for it.
    """
    window = [(float(t), float(mw)) for t, mw in samples if t0 <= t <= t1]
    if len(window) < 2:
        return None
    if any(b[0] < a[0] for a, b in zip(window, window[1:])):
        raise ValueError("power samples must be in time order")
    joules = 0.0
    for (ta, pa), (tb, pb) in zip(window, window[1:]):
        joules += (tb - ta) * (pa + pb) / 2.0 / 1000.0
    return joules


def refresh_interval_s(changes: Sequence[Sequence[float]]) -> float | None:
    """Median time between successive changes of a polled reading.

    ``changes`` lists ``(seconds, value)`` each time the value differed from the
    previous poll; the first entry is the initial reading, not a change. Returns
    None when fewer than two changes were seen. The result cannot be shorter
    than the polling period, so a value near it means "at most this".
    """
    times = [float(t) for t, _ in changes[1:]]
    if len(times) < 2:
        return None
    return statistics.median(b - a for a, b in zip(times, times[1:]))


def mean_ci95(values: Iterable[float]) -> tuple[float, float, int]:
    """Mean, half-width of a two-sided 95% t interval, and sample count."""
    data = [float(v) for v in values]
    if not data:
        raise ValueError("no values")
    n = len(data)
    mean = statistics.fmean(data)
    if n == 1:
        return mean, math.nan, 1
    t = _T975.get(n - 1, 1.960)
    return mean, t * statistics.stdev(data) / math.sqrt(n), n


def phase_summary(record: dict[str, Any], name: str) -> dict[str, Any]:
    """Seconds, counter energy, sampled energy, and mean power of one recorded phase."""
    phase = record["phases"][name]
    seconds = phase["t1"] - phase["t0"]
    if seconds <= 0:
        raise ValueError(f"{record.get('config_id')} trial {record.get('trial')}: {name} has no duration")
    joules = counter_delta_mj(phase["energy_mj_start"], phase["energy_mj_end"]) / 1000.0
    sampled = integrate_samples(record.get("samples", []), phase["t0"], phase["t1"])
    return {
        "seconds": seconds,
        "energy_j": joules,
        "sampled_energy_j": sampled,
        "avg_power_w": joules / seconds,
        "units": phase.get("units", 1),
    }


def reported_summary(record: dict[str, Any], name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Summary of one reported phase; ``decode`` is derived, the others are recorded.

    A full generation is one prefill plus ``output_tokens - 1`` decode steps. So
    one decode step costs (generation - one-token generation) / (output_tokens - 1),
    in time and in energy. This is the same difference that defines inter-token
    latency as (end-to-end latency - TTFT) / (output_tokens - 1).
    """
    if name != "decode":
        return phase_summary(record, name)
    prefill, generate = phase_summary(record, "prefill"), phase_summary(record, "generate")
    steps = (config["output_tokens"] - 1) * generate["units"]

    def total(key):
        if prefill[key] is None or generate[key] is None:
            return None
        return generate[key] - prefill[key] / prefill["units"] * generate["units"]

    seconds, joules = total("seconds"), total("energy_j")
    if seconds <= 0 or joules <= 0:
        raise ValueError(f"{record.get('config_id')} trial {record.get('trial')}: generation does not "
                         "outlast its prefill, so decode cannot be derived")
    return {"seconds": seconds, "energy_j": joules, "sampled_energy_j": total("sampled_energy_j"),
            "avg_power_w": joules / seconds, "units": steps}


def record_problems(record: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    """Structural problems in one trial record; an empty list means it can be analyzed."""
    problems = []
    if record.get("schema") != RECORD_SCHEMA:
        problems.append(f"schema must be {RECORD_SCHEMA!r}")
    try:
        config = config_by_id(plan, record.get("config_id"))
    except KeyError:
        return problems + [f"config {record.get('config_id')!r} is not in the plan"]
    if record.get("status") != "ok":
        return problems
    kind = plan["workloads"][config["workload"]]["kind"]
    for name in PHASES[kind]:
        phase = record.get("phases", {}).get(name)
        if phase is None:
            problems.append(f"missing phase {name}")
            continue
        for key in ("t0", "t1", "energy_mj_start", "energy_mj_end"):
            if key not in phase:
                problems.append(f"phase {name}: missing {key}")
        units = phase.get("units", 1)
        if not isinstance(units, int) or units < 1:
            problems.append(f"phase {name}: units must be a positive integer")
    return problems


def device_problems(device: dict[str, Any], expected_name: str) -> list[str]:
    """Check the GPU the worker actually reported against the plan's expectation."""
    problems = []
    if device.get("name") != expected_name:
        problems.append(f"device reported {device.get('name')!r}, plan expects {expected_name!r}")
    if not device.get("uuid"):
        problems.append("device uuid missing")
    return problems


def read_trials(path: Path) -> list[dict[str, Any]]:
    """Trial records from a run's JSON Lines file, one record per line."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def append_trial(run_dir: Path, record: dict[str, Any]) -> None:
    """Add one trial record to the run's trials file as soon as it finishes."""
    with open(Path(run_dir) / TRIALS_FILE, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def write_manifest(run_dir: Path, header: dict[str, Any]) -> Path:
    """Hash the run's trials file into ``manifest.json`` and index every trial in it."""
    run_dir = Path(run_dir)
    path = run_dir / TRIALS_FILE
    records = read_trials(path) if path.exists() else []
    trials = [{"line": i, "config_id": r["config_id"], "trial": r["trial"], "warmup": r["warmup"],
               "status": r["status"]} for i, r in enumerate(records)]
    manifest = {"schema": MANIFEST_SCHEMA, **header, "trials_file": TRIALS_FILE,
                "trials_sha256": sha256_file(path) if path.exists() else None, "trials": trials}
    out = run_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def manifest_problems(run_dir: Path, plan_sha256: str, plan: dict[str, Any]) -> list[str]:
    """Reasons a run directory cannot be used; empty when every file verifies."""
    run_dir = Path(run_dir)
    path = run_dir / "manifest.json"
    if not path.exists():
        return ["manifest.json missing"]
    manifest = json.loads(path.read_text())
    problems = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"schema must be {MANIFEST_SCHEMA!r}")
    if manifest.get("plan_sha256") != plan_sha256:
        problems.append("measured under a different plan (plan_sha256 mismatch)")
    device_key = manifest.get("device_key")
    if device_key not in plan["devices"]:
        problems.append(f"unknown device_key {device_key!r}")
    else:
        problems += device_problems(manifest.get("device", {}), plan["devices"][device_key]["expected_name"])
    expected = ("manifest.json", TRIALS_FILE)
    for extra in sorted(p.name for p in run_dir.iterdir() if p.name not in expected and not p.name.startswith(".")):
        problems.append(f"{extra} is not part of the run")
    entries = manifest.get("trials", [])
    trials_path = run_dir / TRIALS_FILE
    if not trials_path.exists():
        return problems + ([f"{TRIALS_FILE} is listed but missing"] if entries else [])
    if sha256_file(trials_path) != manifest.get("trials_sha256"):
        return problems + [f"{TRIALS_FILE} does not match its recorded sha256"]
    records = read_trials(trials_path)
    if len(records) != len(entries):
        problems.append(f"{TRIALS_FILE} has {len(records)} trials, the manifest lists {len(entries)}")
    config_ids = {c["id"] for c in plan["configs"]}
    for entry, record in zip(entries, records):
        name = f"line {entry['line']} ({record.get('config_id')} trial {record.get('trial')})"
        if any(entry[k] != record.get(k) for k in ("config_id", "trial", "warmup", "status")):
            problems.append(f"{name} differs from its manifest entry")
        if record.get("device", {}).get("uuid") != manifest.get("device", {}).get("uuid"):
            problems.append(f"{name} was measured on a different GPU than the manifest names")
        if record.get("config_id") in config_ids:
            config = config_by_id(plan, record["config_id"])
            if config["device"] != device_key:
                problems.append(f"{name}: config {config['id']} belongs to device {config['device']}")
        problems += [f"{name}: {p}" for p in record_problems(record, plan)]
    return problems
