"""Run measure_gpu_energy.py on Modal and copy each run into paper/data/gpu-energy/runs/.

The measurement does not depend on Modal. The runner works the same on any
single NVIDIA GPU. Requirements: ``pip install modal`` and
``modal setup``. The plan's weights come from an ungated repository, so no
Hugging Face token is needed.

    modal run paper/scripts/modal_gpu_energy.py                       # every device, twice
    modal run paper/scripts/modal_gpu_energy.py --devices H100 --repeats 1 \\
        --configs h100-llama-b16-p2048 --label pilot                  # a pilot

One command does the whole matrix without spending GPU time on anything but
measurement. A CPU-only container first downloads the pinned weights into a
Modal volume (a no-op once they are there) and checks every shard against the
plan's SHA-256. Then every requested run starts at once, each in its own GPU
container, and reads the weights offline (``HF_HUB_OFFLINE=1``), so no GPU
container downloads anything.

GPU strings below must equal ``devices.<key>.modal_gpu`` in the plan (a test
checks this). ``H100!`` stops Modal from substituting an H200 for an H100
request. The runner also refuses to measure if the reported device name differs
from the plan's ``expected_name``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PLAN = HERE.parent / "data" / "gpu-energy" / "plan.json"
RUNS = HERE.parent / "data" / "gpu-energy" / "runs"
REMOTE = "/root/gpu-energy"
CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # vllm 0.6.4.post1 requires exactly this torch and torchvision.
        "vllm==0.6.4.post1",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.46.3",
        "nvidia-ml-py==12.560.30",
    )
    .env({"HF_HOME": f"{CACHE}/huggingface"})
    .add_local_file(HERE / "measure_gpu_energy.py", f"{REMOTE}/measure_gpu_energy.py")
    .add_local_file(HERE / "gpu_energy_protocol.py", f"{REMOTE}/gpu_energy_protocol.py")
    .add_local_file(PLAN, f"{REMOTE}/plan.json")
)
app = modal.App("mlsysim-gpu-energy", image=image)
cache = modal.Volume.from_name("mlsysim-hf-cache", create_if_missing=True)
GPU_OPTIONS = dict(
    timeout=4 * 3600,
    # Reserve host CPU: vLLM schedules every decode step on the CPU, so a starved
    # container would slow decode. The manifest records the CPU limit the run saw.
    cpu=8.0,
    memory=32768,
    volumes={CACHE: cache},
)


@app.function(cpu=2.0, memory=4096, timeout=3600, volumes={CACHE: cache})
def prefetch(models: list[dict]) -> list[str]:
    """Download pinned weights, config, and tokenizer into the volume and verify the shards, on a CPU container."""
    import hashlib

    from huggingface_hub import snapshot_download

    paths = []
    for model in models:
        # *.json covers config, generation config, tokenizer, and the weight index.
        # A consolidated original/ checkpoint, if the repo has one, is skipped.
        path = Path(snapshot_download(model["repo"], revision=model["revision"],
                                      allow_patterns=["*.json", "*.safetensors"]))
        for name, expected in model["weights_sha256"].items():
            digest = hashlib.sha256()
            with open(path / name, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 24), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise SystemExit(f"{model['repo']}/{name}: sha256 {digest.hexdigest()} is not the plan's {expected}")
        paths.append(f"{path} ({len(model['weights_sha256'])} shards verified)")
    cache.commit()
    return paths


def _measure(device_key: str, configs: list[str]) -> dict[str, bytes]:
    out = Path("/tmp/run")
    command = ["python", f"{REMOTE}/measure_gpu_energy.py", "--plan", f"{REMOTE}/plan.json",
               "--device", device_key, "--out", str(out)]
    for config in configs:
        command += ["--config", config]
    offline = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    process = subprocess.Popen(command, env=offline)
    manifest, finished_at = out / "manifest.json", None
    while process.poll() is None:
        time.sleep(5)
        if finished_at is None and manifest.exists() and "finished_utc" in manifest.read_text():
            finished_at = time.monotonic()
        # The run is complete once its manifest says so. A process that then hangs
        # on exit (vLLM sometimes does) is killed so it does not run to the timeout.
        if finished_at is not None and time.monotonic() - finished_at > 120:
            process.kill()
            process.wait()
    code = 0 if finished_at is not None else process.returncode
    files = {p.name: p.read_bytes() for p in sorted(out.iterdir()) if p.is_file()} if out.exists() else {}
    files["exit_code.txt"] = f"{code}\n".encode()
    return files


@app.function(gpu="H100!", **GPU_OPTIONS)
def measure_h100(configs: list[str]) -> dict[str, bytes]:
    return _measure("H100", configs)


@app.function(gpu="A100-80GB", **GPU_OPTIONS)
def measure_a100(configs: list[str]) -> dict[str, bytes]:
    return _measure("A100", configs)


FUNCTIONS = {"H100": measure_h100, "A100": measure_a100}


@app.local_entrypoint()
def main(devices: str = "H100,A100", repeats: int = 2, configs: str = "", label: str = ""):
    plan = json.loads(PLAN.read_text())
    selected = [c for c in configs.split(",") if c]
    unknown = sorted(set(selected) - {c["id"] for c in plan["configs"]})
    if unknown:
        raise SystemExit(f"unknown configs: {', '.join(unknown)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jobs = []
    for key in [d for d in devices.split(",") if d]:
        mine = [c for c in plan["configs"] if c["device"] == key and (not selected or c["id"] in selected)]
        if not mine:
            continue
        for k in range(1, repeats + 1):
            target = RUNS / "-".join(p for p in (label, key.lower(), stamp, str(k)) if p)
            if target.exists():  # checked before any GPU time, so a name clash costs nothing
                raise SystemExit(f"{target} already exists")
            jobs.append((key, [c["id"] for c in mine] if selected else [], mine, target))
    if not jobs:
        raise SystemExit("nothing to run")
    llm_workloads = {c["workload"] for _, _, mine, _ in jobs for c in mine
                     if plan["workloads"][c["workload"]]["kind"] == "generate"}
    if llm_workloads:
        models = [{"repo": plan["workloads"][w]["hf_model"], "revision": plan["workloads"][w]["hf_revision"],
                   "weights_sha256": plan["workloads"][w]["weights_sha256"]} for w in sorted(llm_workloads)]
        print("weights in the volume:", prefetch.remote(models))
    for _, _, _, target in jobs:
        target.mkdir(parents=True)
    calls = [(FUNCTIONS[key].spawn(ids), target) for key, ids, _, target in jobs]
    print(f"started {len(calls)} runs:", ", ".join(target.name for _, target in calls))
    failed = 0
    for call, target in calls:
        try:
            files = call.get()
        except Exception as exc:  # the container died and returned nothing
            target.rmdir()
            print(f"{target.name}: no result ({exc!r})")
            failed += 1
            continue
        code = int(files.pop("exit_code.txt").decode())
        for name, data in files.items():
            (target / name).write_bytes(data)
        print(f"{target.name}: wrote {len(files)} files (runner exit code {code})")
        failed += code != 0
    if failed:
        raise SystemExit(f"{failed} run(s) did not finish cleanly")
