#!/usr/bin/env python3
"""Measure GPU board energy for the configurations in paper/data/gpu-energy/plan.json.

Runs on any machine with one NVIDIA GPU, PyTorch, vLLM, and ``nvidia-ml-py``; it
does not import mlsysim. ``modal_gpu_energy.py`` is one way to launch it. Every trial
of every requested configuration is appended to ``<out>/trials.jsonl`` as it
finishes, including warm-up and failed trials, and ``<out>/manifest.json``
hashes that file and indexes every trial in it.

    python paper/scripts/measure_gpu_energy.py --device H100 --out runs/h100-a \\
        [--config h100-llama-b1-p128 ...]

Each trial records, on one monotonic clock: an idle phase with the workload
resident, then the workload phases. Language models run in vLLM's offline engine
and record two phases, repeated one-token generations (prefill) and repeated
full generations; the analysis derives decode steps from their difference.
ResNet-50 records training steps. At every phase boundary the runner synchronizes
the GPU and reads the cumulative energy counter, clocks, temperature, and the
clock-event (throttle) reasons. A background thread polls board power for a
sampled cross-check. Before any workload loads, a short probe polls the energy
counter and the power reading as fast as possible and records when each value
changes, which shows how often the driver refreshes them.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import random
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_energy_protocol as protocol  # noqa: E402

DEFAULT_PLAN = Path(__file__).resolve().parents[1] / "data" / "gpu-energy" / "plan.json"


class Meter:
    """NVML access for one GPU: identity, counters, and a background power sampler."""

    def __init__(self, index: int, interval_s: float):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.interval_s = interval_s
        self.origin = time.perf_counter()
        self.samples: list[list[float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def now(self) -> float:
        return time.perf_counter() - self.origin

    def energy_mj(self) -> int:
        return int(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))

    def identity(self) -> dict:
        n, h = self.nvml, self.handle

        def text(value):
            return value.decode() if isinstance(value, bytes) else value

        def maybe(fn, *args):
            try:
                return fn(*args)
            except n.NVMLError:
                return None

        cuda = maybe(n.nvmlSystemGetCudaDriverVersion)
        return {
            "name": text(n.nvmlDeviceGetName(h)),
            "uuid": text(n.nvmlDeviceGetUUID(h)),
            "pci_bus_id": text(n.nvmlDeviceGetPciInfo(h).busId),
            "driver_version": text(n.nvmlSystemGetDriverVersion()),
            "cuda_driver_version": None if cuda is None else f"{cuda // 1000}.{(cuda % 1000) // 10}",
            "memory_total_bytes": n.nvmlDeviceGetMemoryInfo(h).total,
            "power_limit_mw": maybe(n.nvmlDeviceGetPowerManagementLimit, h),
            "enforced_power_limit_mw": maybe(n.nvmlDeviceGetEnforcedPowerLimit, h),
            "default_power_limit_mw": maybe(n.nvmlDeviceGetPowerManagementDefaultLimit, h),
            "max_sm_clock_mhz": maybe(n.nvmlDeviceGetMaxClockInfo, h, n.NVML_CLOCK_SM),
            "max_mem_clock_mhz": maybe(n.nvmlDeviceGetMaxClockInfo, h, n.NVML_CLOCK_MEM),
            "persistence_mode": maybe(n.nvmlDeviceGetPersistenceMode, h),
        }

    def state(self) -> dict:
        n, h = self.nvml, self.handle
        reasons = getattr(n, "nvmlDeviceGetCurrentClocksEventReasons", None) or getattr(
            n, "nvmlDeviceGetCurrentClocksThrottleReasons"
        )
        try:
            processes = len(n.nvmlDeviceGetComputeRunningProcesses(h))
        except n.NVMLError:
            processes = None
        return {
            "sm_clock_mhz": n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
            "mem_clock_mhz": n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM),
            "temperature_c": n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
            "clock_event_reasons": int(reasons(h)),
            "compute_processes": processes,
        }

    def refresh_probe(self, seconds: float) -> dict:
        """Poll the energy counter and power reading in a tight loop; keep each change.

        Run while the GPU is idle. The first entry of each list is the initial
        reading; later entries are the times the value changed.
        """
        energy: list[list[float]] = []
        power: list[list[float]] = []
        polls, end = 0, time.perf_counter() + seconds
        while time.perf_counter() < end:
            t = self.now()
            mj = self.energy_mj()
            mw = float(self.nvml.nvmlDeviceGetPowerUsage(self.handle))
            polls += 1
            if not energy or energy[-1][1] != mj:
                energy.append([t, mj])
            if not power or power[-1][1] != mw:
                power.append([t, mw])
        return {"seconds": seconds, "polls": polls, "energy_changes": energy, "power_changes": power}

    def start_sampling(self) -> None:
        self.samples = []
        self._stop.clear()

        def loop():
            # 0.1 ms and whole milliwatts are finer than the 50 ms cross-check needs;
            # phase boundaries keep full precision.
            while not self._stop.is_set():
                self.samples.append([round(self.now(), 4), int(self.nvml.nvmlDeviceGetPowerUsage(self.handle))])
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_sampling(self) -> list[list[float]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return self.samples

    @contextmanager
    def phase(self, record: dict, name: str, sync):
        sync()
        entry = {"t0": self.now(), "energy_mj_start": self.energy_mj(), "state_start": self.state()}
        yield entry
        sync()
        entry.update(t1=self.now(), energy_mj_end=self.energy_mj(), state_end=self.state())
        record["phases"][name] = entry


class GenerateWorkload:
    """Greedy generation from random prompt tokens in vLLM's offline engine.

    One engine per workload serves every configuration of a run, as a server
    would. vLLM does not reliably release GPU memory, so the engine is built once
    and never rebuilt; ``measure`` runs training configurations first for the
    same reason.
    """

    _engines: dict[str, object] = {}

    def __init__(self, spec: dict, config: dict, seed: int):
        import torch
        from vllm import SamplingParams

        key = json.dumps(spec, sort_keys=True)
        if key not in self._engines:
            try:
                self._engines[key] = self._start(spec, seed)
            except Exception:
                self._engines[key] = traceback.format_exc()
                raise
        engine = self._engines[key]
        if isinstance(engine, str):
            raise RuntimeError("the vLLM engine failed to start earlier in this run:\n" + engine)
        self.llm, self.details = engine
        self.torch = torch
        self.batch, self.output_tokens = config["batch"], config["output_tokens"]
        vocab = self.llm.get_tokenizer().vocab_size  # base vocabulary, no special tokens
        rng = random.Random(seed)
        self.prompts = [{"prompt_token_ids": [rng.randrange(vocab) for _ in range(config["prompt_tokens"])]}
                        for _ in range(config["batch"])]
        greedy = dict(temperature=0.0, ignore_eos=True, detokenize=False)
        self.params = {"prefill": SamplingParams(max_tokens=1, **greedy),
                       "generate": SamplingParams(max_tokens=config["output_tokens"], **greedy)}

    @staticmethod
    def _start(spec: dict, seed: int):
        os.environ["VLLM_ATTENTION_BACKEND"] = spec["attention_backend"]
        import vllm

        dtype = {"fp16": "float16", "bf16": "bfloat16"}[spec["precision"]]
        llm = vllm.LLM(
            model=spec["hf_model"], revision=spec["hf_revision"], tokenizer_revision=spec["hf_revision"],
            dtype=dtype, seed=seed, **spec["vllm"],
        )
        details = {
            "engine": "vllm",
            "vllm_version": vllm.__version__,
            "hf_model": spec["hf_model"],
            "hf_revision": spec["hf_revision"],
            "dtype": dtype,
            "attention_backend": os.environ["VLLM_ATTENTION_BACKEND"],
            "engine_args": spec["vllm"],
            "kv_cache_gpu_blocks": llm.llm_engine.cache_config.num_gpu_blocks,
        }
        return llm, details

    def sync(self):
        self.torch.cuda.synchronize()

    def run(self, meter: Meter, record: dict, plan: dict) -> None:
        with meter.phase(record, "idle", self.sync):
            time.sleep(plan["idle_seconds"])
        for name, params in self.params.items():
            tokens = params.max_tokens
            with meter.phase(record, name, self.sync) as phase:
                count, start = 0, time.perf_counter()
                while True:
                    outputs = self.llm.generate(self.prompts, params, use_tqdm=False)
                    count += 1
                    lengths = sorted({len(o.outputs[0].token_ids) for o in outputs})
                    if len(outputs) != self.batch or lengths != [tokens]:
                        raise RuntimeError(f"{name}: {len(outputs)} outputs of lengths {lengths}, "
                                           f"expected {self.batch} of {tokens}")
                    if time.perf_counter() - start >= plan["phase_min_seconds"]:
                        break
                phase["units"] = count
                phase["output_tokens"] = tokens


class TrainWorkload:
    """ResNet-50 training steps on synthetic on-device data with fp16 autocast."""

    def __init__(self, spec: dict, config: dict, seed: int):
        import torch
        import torchvision

        torch.manual_seed(seed)
        self.torch = torch
        self.model = torchvision.models.resnet50().cuda().to(memory_format=torch.channels_last)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.scaler = torch.amp.GradScaler("cuda")
        self.loss = torch.nn.CrossEntropyLoss()
        batch = config["batch"]
        self.x = torch.randn(batch, 3, 224, 224, device="cuda").to(memory_format=torch.channels_last)
        self.y = torch.randint(0, 1000, (batch,), device="cuda")
        self.warmup_steps = spec.get("warmup_steps", 0)
        self.steps = spec["steps"]
        self.details = {"torchvision_version": torchvision.__version__, "autocast": "float16"}

    def sync(self):
        self.torch.cuda.synchronize()

    def step(self):
        torch = self.torch
        with torch.autocast("cuda", dtype=torch.float16):
            loss = self.loss(self.model(self.x), self.y)
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def run(self, meter: Meter, record: dict, plan: dict) -> None:
        with meter.phase(record, "idle", self.sync):
            time.sleep(plan["idle_seconds"])
        for _ in range(self.warmup_steps):
            self.step()
        with meter.phase(record, "train", self.sync) as phase:
            for _ in range(self.steps):
                self.step()
            phase["units"] = self.steps


WORKLOADS = {"generate": GenerateWorkload, "train": TrainWorkload}


def environment() -> dict:
    try:
        import torch
    except ImportError:
        torch = None
    try:
        cpus_usable = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        cpus_usable = None
    try:  # a container's CPU limit is a cgroup quota, which the affinity mask does not show
        cgroup_cpu_max = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    except OSError:
        cgroup_cpu_max = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        # Host CPU limits the engine's scheduling overhead, which is part of every decode step.
        "cpu_count": os.cpu_count(),
        "cpus_usable": cpus_usable,
        "cgroup_cpu_max": cgroup_cpu_max,
        "torch_version": None if torch is None else torch.__version__,
        "torch_cuda_version": None if torch is None else torch.version.cuda,
        "cudnn_version": None if torch is None else torch.backends.cudnn.version(),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "modal_region": os.environ.get("MODAL_REGION"),
        "modal_cloud_provider": os.environ.get("MODAL_CLOUD_PROVIDER"),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                     if d.metadata["Name"]},
    }


def measure(plan_path: Path, device_key: str, config_ids: list[str], out: Path, gpu_index: int = 0) -> int:
    plan = protocol.load_plan(plan_path)
    plan_sha = protocol.sha256_file(plan_path)
    expected = plan["devices"][device_key]["expected_name"]
    configs = [protocol.config_by_id(plan, cid) for cid in config_ids] if config_ids else [
        c for c in plan["configs"] if c["device"] == device_key
    ]
    wrong = [c["id"] for c in configs if c["device"] != device_key]
    if wrong:
        raise SystemExit(f"configs for another device: {', '.join(wrong)}")
    # Training first: the vLLM engine keeps its GPU memory until the process exits.
    configs.sort(key=lambda c: plan["workloads"][c["workload"]]["kind"] != "train")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give each run a new directory")
    out.mkdir(parents=True, exist_ok=True)
    meter = Meter(gpu_index, plan["sample_interval_s"])
    device = meter.identity()
    header = {
        "device_key": device_key,
        "device": device,
        "plan_sha256": plan_sha,
        "harness_sha256": {
            name: protocol.sha256_file(Path(__file__).resolve().parent / name)
            for name in ("measure_gpu_energy.py", "gpu_energy_protocol.py")
        },
        "environment": environment(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
    }
    mismatch = protocol.device_problems(device, expected)
    if mismatch:
        header["aborted"] = mismatch
        protocol.write_manifest(out, header)
        print("refusing to measure:", "; ".join(mismatch), file=sys.stderr)
        return 2
    header["counter_probe"] = meter.refresh_probe(plan["counter_probe_seconds"])

    total = plan["warmup_trials"] + plan["trials"]
    for config in configs:
        spec = plan["workloads"][config["workload"]]
        try:
            workload = WORKLOADS[spec["kind"]](spec, config, plan["seed"])
            load_error = None
        except Exception:  # keep a record of configurations that never ran
            workload, load_error = None, traceback.format_exc()
        for trial in range(total):
            record = {
                "schema": protocol.RECORD_SCHEMA,
                "config_id": config["id"],
                "trial": trial,
                "warmup": trial < plan["warmup_trials"],
                "device": {"name": device["name"], "uuid": device["uuid"]},
                "workload": None if workload is None else workload.details,
                "phases": {},
                "status": "ok",
                "error": None,
            }
            if workload is None:
                record.update(status="failed", error=load_error)
            else:
                meter.start_sampling()
                try:
                    workload.run(meter, record, plan)
                except Exception:
                    record.update(status="failed", error=traceback.format_exc())
                record["samples"] = meter.stop_sampling()
            protocol.append_trial(out, record)
            print(f"{config['id']} trial {trial}: {record['status']}", flush=True)
        del workload
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # no torch, or no CUDA context to clear
            pass
    header["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    protocol.write_manifest(out, header)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--device", required=True, help="device key in the plan, e.g. H100")
    parser.add_argument("--config", action="append", default=[], help="config id; repeatable; default all for the device")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args(argv)
    return measure(args.plan, args.device, args.config, args.out, args.gpu_index)


if __name__ == "__main__":
    raise SystemExit(main())
