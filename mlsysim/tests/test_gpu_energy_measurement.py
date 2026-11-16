"""GPU energy measurement protocol: plan rules, energy accounting, integrity checks, and analysis.

No GPU is needed. Measured-looking records here are constructed fixtures with
round numbers, so each expected ratio can be checked by hand. They are not data.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import io
import json
import math
import os
import re
import sys
import time
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "paper/scripts"
PLAN_PATH = Path(__file__).resolve().parents[1] / "paper/data/gpu-energy/plan.json"
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


protocol = _load("gpu_energy_protocol")
analysis = _load("analyze_gpu_energy")
runner = _load("measure_gpu_energy")

UUID = "GPU-constructed-0000"


def _plan(**overrides):
    plan = {
        "schema": protocol.PLAN_SCHEMA,
        "boundary": "GPU board only (constructed fixture)",
        "trials": 3, "warmup_trials": 1, "idle_seconds": 0.05,
        "phase_min_seconds": 0.05, "sample_interval_s": 0.01, "counter_probe_seconds": 0.2, "seed": 1,
        "devices": {"H100": {"registry": "Hardware.Cloud.H100", "modal_gpu": "H100!",
                             "expected_name": "NVIDIA H100 80GB HBM3"}},
        "workloads": {
            "llama": {"kind": "generate", "registry": "Models.Language.Llama3_8B", "hf_model": "x/y",
                      "hf_revision": "0" * 40, "weights_sha256": {"model.safetensors": "0" * 64},
                      "precision": "fp16", "efficiency": 0.5,
                      "model_seq_len": "mean_context", "attention_backend": "FLASH_ATTN",
                      "vllm": {"max_model_len": 2560, "max_num_seqs": 16, "max_num_batched_tokens": 32768,
                               "enable_prefix_caching": False, "enable_chunked_prefill": False}},
            "resnet": {"kind": "train", "registry": "Models.Vision.ResNet50", "precision": "fp16", "steps": 50},
        },
        "configs": [
            {"id": "llama-b1", "device": "H100", "workload": "llama", "batch": 1,
             "prompt_tokens": 128, "output_tokens": 101, "split": "calibration"},
            {"id": "resnet-b256", "device": "H100", "workload": "resnet", "batch": 256,
             "efficiency": 0.13, "split": "heldout"},
        ],
    }
    plan.update(overrides)
    return plan


def _phase(t0, seconds, start_mj, units=1, watts=None, joules=None):
    joules = watts * seconds if joules is None else joules
    return {"t0": t0, "t1": t0 + seconds, "energy_mj_start": start_mj,
            "energy_mj_end": start_mj + round(joules * 1000), "units": units}


def _decode_record(trial, warmup=False, status="ok", decode_seconds=10.0, uuid=UUID, config_id="llama-b1",
                   device_name="NVIDIA H100 80GB HBM3", prefill_watts=500.0, decode_watts=350.0):
    """One-token calls take 0.01 s (5 J at 500 W). A full call adds 100 decode steps (350 W by default)."""
    idle = _phase(0.0, 5.0, 1_000_000, watts=80.0)
    prefill = _phase(5.0, 3.0, idle["energy_mj_end"], units=300, watts=prefill_watts)
    generate = _phase(8.0, 0.01 + decode_seconds, prefill["energy_mj_end"], units=1,
                      joules=prefill_watts * 0.01 + decode_watts * decode_seconds)
    samples = [[t / 10, 80_000.0 if t < 50 else 500_000.0 if t < 80 else 350_000.0]
               for t in range(int((8.01 + decode_seconds) * 10) + 1)]
    return {"schema": protocol.RECORD_SCHEMA, "config_id": config_id, "trial": trial, "warmup": warmup,
            "device": {"name": device_name, "uuid": uuid}, "workload": {},
            "phases": {} if status != "ok" else {"idle": idle, "prefill": prefill, "generate": generate},
            "samples": samples, "status": status, "error": None if status == "ok" else "boom"}


def _probe(refresh_s):
    """A counter that changed every ``refresh_s`` seconds for one second."""
    steps = int(round(1 / refresh_s))
    return {"seconds": 1.0, "polls": 1000,
            "energy_changes": [[i * refresh_s, 1_000_000 + 80 * i] for i in range(steps + 1)],
            "power_changes": [[0.0, 80_000.0]]}


def _write_run(root, plan, records, name="run-a", device_name="NVIDIA H100 80GB HBM3", uuid=UUID, refresh_s=0.1,
               device_key="H100", power_limit_mw=None):
    (root / "plan.json").write_text(json.dumps(plan))
    run = root / "runs" / name
    run.mkdir(parents=True)
    for r in records:
        protocol.append_trial(run, r)
    device = {"name": device_name, "uuid": uuid, "enforced_power_limit_mw": power_limit_mw}
    protocol.write_manifest(run, {"device_key": device_key, "device": device,
                                  "plan_sha256": protocol.sha256_file(root / "plan.json"),
                                  "counter_probe": _probe(refresh_s)})
    return run


@pytest.fixture
def measured_root(tmp_path):
    records = [_decode_record(0, warmup=True, decode_seconds=99.0)]
    records += [_decode_record(i) for i in (1, 2, 3)]
    records += [_decode_record(4, status="failed")]
    _write_run(tmp_path, _plan(), records)
    return tmp_path


def _rows(root):
    text, provenance, _ = analysis.summarize(root)
    lines = text.strip().splitlines()
    header = lines[0].split(",")
    return [dict(zip(header, line.split(","))) for line in lines[1:]], provenance


def test_committed_plan_is_valid_and_declares_every_efficiency():
    plan = protocol.load_plan(PLAN_PATH)
    assert {c["split"] for c in plan["configs"]} == {"calibration", "heldout"}
    for config in plan["configs"]:
        assert 0 < protocol.efficiency_of(plan, config) <= 1
        analysis.resolve(plan["devices"][config["device"]]["registry"])
        analysis.resolve(plan["workloads"][config["workload"]]["registry"])


def test_modal_launcher_requests_the_gpu_the_plan_names():
    source = (SCRIPTS / "modal_gpu_energy.py").read_text()
    requested = dict(re.findall(r'@app\.function\(gpu="([^"]+)".*\)\ndef measure_(\w+)', source))
    plan = protocol.load_plan(PLAN_PATH)
    assert {v.lower(): k for k, v in requested.items()} == {
        key.lower(): device["modal_gpu"] for key, device in plan["devices"].items()
    }


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["configs"].append(copy.deepcopy(p["configs"][0])), "duplicated"),
    (lambda p: p["workloads"]["resnet"].pop("steps"), "steps"),
    (lambda p: p["configs"][1].pop("efficiency"), "efficiency must be declared"),
    (lambda p: p["configs"][0].update(split="test"), "split"),
    (lambda p: p["configs"][0].update(output_tokens=1), "output_tokens"),
    (lambda p: p.update(trials=1), "trials"),
    (lambda p: p["workloads"]["llama"].update(model_seq_len="prompt"), "model_seq_len"),
    (lambda p: p["workloads"]["llama"].pop("attention_backend"), "attention_backend"),
    (lambda p: p["workloads"]["llama"].update(hf_revision="main"), "commit hash"),
    (lambda p: p["workloads"]["llama"].update(weights_sha256={"model.safetensors": "abc"}), "weights_sha256"),
    (lambda p: p["workloads"]["llama"]["vllm"].update(enable_prefix_caching=True), "prefix_caching"),
    (lambda p: p["workloads"]["llama"]["vllm"].pop("enable_chunked_prefill"), "chunked_prefill"),
    (lambda p: p["configs"][0].update(batch=32), "max_num_seqs"),
    (lambda p: p["configs"][0].update(batch=16, prompt_tokens=2100), "max_num_batched_tokens"),
    (lambda p: p.pop("counter_probe_seconds"), "counter_probe_seconds"),
])
def test_plan_rejects_undeclared_or_ambiguous_protocol(mutate, message):
    plan = _plan()
    mutate(plan)
    assert any(message in problem for problem in protocol.plan_problems(plan))


def test_energy_counter_delta_allows_one_wraparound():
    assert protocol.counter_delta_mj(100, 350) == 250
    assert protocol.counter_delta_mj((1 << 64) - 10, 5) == 15
    with pytest.raises(ValueError):
        protocol.counter_delta_mj(-1, 5)


def test_sampled_energy_is_trapezoidal_inside_the_window():
    ramp = [[t, 1000.0 * t] for t in range(11)]  # 0 to 10 W over 10 s
    assert protocol.integrate_samples(ramp, 0, 10) == pytest.approx(50.0)
    assert protocol.integrate_samples(ramp, 2, 4) == pytest.approx(6.0)
    assert protocol.integrate_samples(ramp, 2.5, 2.9) is None
    with pytest.raises(ValueError):
        protocol.integrate_samples([[1, 0], [0, 0]], 0, 1)


def test_refresh_interval_is_the_median_gap_between_changes():
    changes = [[0.0, 5], [0.03, 6], [0.13, 7], [0.23, 8], [0.38, 9]]  # first entry is the initial reading
    assert protocol.refresh_interval_s(changes) == pytest.approx(0.1)
    assert protocol.refresh_interval_s([[0.0, 5], [0.1, 6]]) is None


def test_interval_uses_student_t():
    mean, half, n = protocol.mean_ci95([1.0, 2.0, 3.0])
    assert (mean, n) == (2.0, 3)
    assert half == pytest.approx(4.303 / math.sqrt(3))
    assert math.isnan(protocol.mean_ci95([5.0])[1])


def test_device_identity_must_match_the_plan():
    assert protocol.device_problems({"name": "NVIDIA H200", "uuid": "x"}, "NVIDIA H100 80GB HBM3")
    assert not protocol.device_problems({"name": "NVIDIA H100 80GB HBM3", "uuid": "x"}, "NVIDIA H100 80GB HBM3")


def test_analysis_compares_board_energy_with_rated_power_accounting(measured_root):
    rows, provenance = _rows(measured_root)
    assert [(r["phase"], r["unit"]) for r in rows] == [("prefill", "prefill"), ("decode", "decode step")]
    decode = rows[1]
    assert decode["trials_used"] == "3" and decode["trials_failed"] == "1"
    assert float(decode["measured_s_per_unit"]) == pytest.approx(0.1)
    assert float(decode["measured_j_per_unit"]) == pytest.approx(35.0)
    assert float(decode["measured_power_w"]) == pytest.approx(350.0)
    assert float(decode["idle_power_w"]) == pytest.approx(80.0)
    assert float(decode["measured_power_over_rated"]) == pytest.approx(350 / 700, rel=1e-5)
    # Rated power over the measured time overstates board energy by TDP / measured power.
    assert float(decode["rated_measured_time_over_measured"]) == pytest.approx(700 / 350, rel=1e-5)
    assert float(decode["rated_measured_time_j_per_unit"]) == pytest.approx(70.0, rel=1e-5)
    assert decode["measured_j_per_unit_ci95"] == "0"
    assert 0.9 < float(decode["sampled_over_counter"]) <= 1.0
    model_s = analysis.modeled_seconds(_plan(), _plan()["configs"][0], "decode")
    assert float(decode["rated_modeled_time_j_per_unit"]) == pytest.approx(700 * model_s, rel=1e-5)
    prefill = rows[0]
    assert float(prefill["measured_j_per_unit"]) == pytest.approx(500 * 3 / 300)
    assert provenance["runs_excluded"] == [] and provenance["rows"] == 2
    assert decode["runs"] == "1" and decode["gpus"] == "1"
    assert float(decode["counter_refresh_ms"]) == pytest.approx(100.0)
    assert provenance["runs_used"][0]["harness_matches_committed"] is False  # fixture has no runner hashes


def test_package_power_rule_at_measured_and_modeled_time(measured_root):
    rows, provenance = _rows(measured_root)
    decode = rows[1]
    assert float(decode["idle_over_rated"]) == pytest.approx(80 / 700, rel=1e-5)
    plan, config = _plan(), _plan()["configs"][0]
    work = analysis.work_seconds(plan, config, "decode")
    u = work / 0.1
    assert float(decode["utilization_at_measured_time"]) == pytest.approx(u, rel=1e-5)
    assert float(decode["proportional_measured_time_j_per_unit"]) == pytest.approx(
        700 * (0.30 + 0.70 * u) * 0.1, rel=1e-5)
    assert float(decode["proportional_measured_time_over_measured"]) == pytest.approx(
        700 * (0.30 + 0.70 * u) / 350, rel=1e-5)
    model_s = analysis.modeled_seconds(plan, config, "decode")
    assert float(decode["proportional_modeled_time_j_per_unit"]) == pytest.approx(
        700 * (0.30 + 0.70 * work / model_s) * model_s, rel=1e-5)
    assert provenance["package_power_rule"]["idle_fraction"] == 0.30


def test_work_seconds_count_flops_as_serving_model_does():
    from mlsysim.solvers import ServingModel

    plan = _plan()
    config = plan["configs"][0]
    model = analysis.resolve("Models.Language.Llama3_8B")
    hardware = analysis.resolve("Hardware.Cloud.H100")
    ttft = ServingModel().solve(model, hardware, seq_len=128, batch_size=1, precision="fp16", efficiency=0.5).ttft
    work = analysis.work_seconds(plan, config, "prefill")
    assert work / 0.5 + hardware.dispatch_tax.m_as("s") == pytest.approx(ttft.m_as("s"), rel=1e-9)
    # A decode step is one token per sequence: about 2 x parameters FLOPs at batch 1.
    decode_flops = analysis.work_seconds(plan, config, "decode") * hardware.compute.peak_flops.m_as("flop/s")
    assert decode_flops == pytest.approx(2 * model.parameters.magnitude, rel=0.01)


def test_training_proportional_estimate_is_single_node_models_energy():
    from mlsysim.solvers import SingleNodeModel

    plan = _plan()
    config = plan["configs"][1]
    profile = SingleNodeModel().solve(
        analysis.resolve("Models.Vision.ResNet50"), analysis.resolve("Hardware.Cloud.H100"),
        batch_size=256, precision="fp16", efficiency=0.13, is_training=True,
    )
    model_s = analysis.modeled_seconds(plan, config, "train")
    work = analysis.work_seconds(plan, config, "train")
    tdp = analysis.resolve("Hardware.Cloud.H100").tdp
    energy = analysis.board_energy_j(model_s, analysis.proportional_power(tdp, work / model_s))
    assert energy == pytest.approx(profile.energy.m_as("J"), rel=1e-9)


def test_runs_and_gpus_are_counted_and_the_slowest_refresh_is_reported(tmp_path):
    _write_run(tmp_path, _plan(), [_decode_record(i) for i in (1, 2)], name="run-a", power_limit_mw=700_000)
    other = [_decode_record(i, uuid="GPU-constructed-0001") for i in (1, 2)]
    _write_run(tmp_path, _plan(), other, name="run-b", uuid="GPU-constructed-0001", refresh_s=0.25,
               power_limit_mw=650_000)
    rows, provenance = _rows(tmp_path)
    decode = rows[1]
    assert (decode["runs"], decode["gpus"], decode["trials_used"]) == ("2", "2", "4")
    assert decode["power_limits_w"] == "650/700"  # boards of one model with different limits
    assert float(decode["counter_refresh_ms"]) == pytest.approx(250.0)
    assert [r["counter_refresh_ms"] for r in provenance["runs_used"]] == pytest.approx([100.0, 250.0])


def test_decode_is_the_generation_minus_its_prefill():
    record = _decode_record(1)
    config = _plan()["configs"][0]
    decode = protocol.reported_summary(record, "decode", config)
    # 10.01 s and 3505 J per generation, minus one 0.01 s, 5 J prefill, over 100 steps.
    assert decode["units"] == 100
    assert decode["seconds"] / decode["units"] == pytest.approx(0.1)
    assert decode["energy_j"] / decode["units"] == pytest.approx(35.0)
    short = _decode_record(1, decode_seconds=-0.005)  # a generation faster than its own prefill
    with pytest.raises(ValueError, match="cannot be derived"):
        protocol.reported_summary(short, "decode", config)


def test_warmup_trials_are_never_averaged(measured_root):
    rows, _ = _rows(measured_root)
    # The warm-up decode took 99 s, so including it would move the mean.
    assert float(rows[1]["measured_s_per_unit"]) == pytest.approx(0.1)


def _edit_first_line(run, root):
    path = run / protocol.TRIALS_FILE
    lines = path.read_text().splitlines(keepends=True)
    assert '"trial":0' in lines[0]
    path.write_text(lines[0].replace('"trial":0', '"trial":9') + "".join(lines[1:]))


@pytest.mark.parametrize("tamper, reason", [
    (_edit_first_line, "sha256"),
    (lambda run, root: (run / "trial-llama-b1-09.json").write_text("{}"), "not part of the run"),
    (lambda run, root: (run / protocol.TRIALS_FILE).unlink(), "missing"),
    (lambda run, root: (root / "plan.json").write_text(json.dumps(_plan(trials=4))), "different plan"),
])
def test_runs_that_do_not_verify_are_excluded(measured_root, tamper, reason):
    tamper(measured_root / "runs/run-a", measured_root)
    rows, provenance = _rows(measured_root)
    assert rows == []
    assert any(reason in p for p in provenance["runs_excluded"][0]["problems"])


def test_substituted_gpu_is_excluded(tmp_path):
    _write_run(tmp_path, _plan(), [_decode_record(1)], device_name="NVIDIA H200")
    rows, provenance = _rows(tmp_path)
    assert rows == [] and "NVIDIA H200" in provenance["runs_excluded"][0]["problems"][0]


def test_check_mode_detects_a_stale_summary(measured_root):
    assert analysis.main(["--check", "--root", str(measured_root)]) == 1
    assert analysis.main(["--root", str(measured_root)]) == 0
    assert analysis.main(["--check", "--root", str(measured_root)]) == 0
    (measured_root / "summary.csv").write_text("edited\n")
    assert analysis.main(["--check", "--root", str(measured_root)]) == 1


def test_one_trials_file_per_run_keeps_the_paper_receipt_small(measured_root):
    run = measured_root / "runs/run-a"
    assert sorted(p.name for p in run.iterdir()) == ["manifest.json", protocol.TRIALS_FILE]
    # paper_integrity hashes .json and .csv under paper/data. trials.jsonl is reached through the manifest.
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["trials_sha256"] == protocol.sha256_file(run / protocol.TRIALS_FILE)
    assert [t["line"] for t in manifest["trials"]] == list(range(5))
    (run / ".DS_Store").write_text("")  # Finder metadata must not exclude a run
    assert _rows(measured_root)[1]["runs_excluded"] == []


def test_runner_refuses_a_directory_that_already_has_a_run(tmp_path):
    out = tmp_path / "runs" / "used"
    out.mkdir(parents=True)
    (out / protocol.TRIALS_FILE).write_text("{}\n")
    with pytest.raises(SystemExit, match="not empty"):
        runner.measure(PLAN_PATH, "H100", [], out)


def test_device_choice_reports_where_an_estimate_picks_the_other_device(tmp_path):
    plan = _plan()
    plan["devices"]["A100"] = {"registry": "Hardware.Cloud.A100", "modal_gpu": "A100-80GB",
                               "expected_name": "NVIDIA A100-SXM4-80GB"}
    plan["configs"].append(dict(plan["configs"][0], id="llama-b1-a100", device="A100"))
    _write_run(tmp_path, plan, [_decode_record(i) for i in (1, 2, 3)], name="h100")
    a100 = dict(config_id="llama-b1-a100", device_name="NVIDIA A100-SXM4-80GB", uuid="GPU-a100")
    records = [_decode_record(i, prefill_watts=400.0, decode_watts=200.0, decode_seconds=15.0, **a100)
               for i in (1, 2, 3)]
    _write_run(tmp_path, plan, records, name="a100", device_key="A100",
               device_name=a100["device_name"], uuid=a100["uuid"])
    _, provenance, choice_text = analysis.summarize(tmp_path)
    choices = {row["phase"]: row for row in csv.DictReader(io.StringIO(choice_text))}
    assert provenance["device_choice_rows"] == 2
    decode, prefill = choices["decode"], choices["prefill"]
    # Decode: 35 J per step on the H100 (350 W x 0.1 s), 30 J on the A100 (200 W x 0.15 s).
    assert (decode["H100_config"], decode["A100_config"]) == ("llama-b1", "llama-b1-a100")
    assert decode["measured_preferred"] == "A100" and decode["measured_clear"] == "True"
    assert decode["measured_every_run_pair"] == "True"
    assert float(decode["measured_margin"]) == pytest.approx(35 / 30 - 1, rel=1e-5)
    # Prefill: 5 J vs 4 J measured, but TDP over modeled TTFT (700 W x 4.2 ms vs 400 W x 13.2 ms) picks the H100.
    assert prefill["measured_preferred"] == "A100"
    assert prefill["rated_modeled_time_preferred"] == "H100" and prefill["rated_modeled_time_agrees"] == "False"


def test_check_mode_without_runs_or_summary_passes(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps(_plan()))
    assert analysis.main(["--check", "--root", str(tmp_path)]) == 0


class _FakeNvml(types.ModuleType):
    """A board that draws a constant 250 W, with an energy counter that refreshes every 20 ms."""

    NVML_CLOCK_SM, NVML_CLOCK_MEM, NVML_TEMPERATURE_GPU = 0, 1, 0

    class NVMLError(Exception):
        pass

    def __init__(self, name):
        super().__init__("pynvml")
        self._name = name
        self._t0 = time.perf_counter()

    def nvmlInit(self): pass
    def nvmlDeviceGetHandleByIndex(self, i): return i
    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        return int((time.perf_counter() - self._t0) // 0.02 * 0.02 * 250_000)
    def nvmlDeviceGetPowerUsage(self, h): return 250_000
    def nvmlDeviceGetName(self, h): return self._name
    def nvmlDeviceGetUUID(self, h): return UUID
    def nvmlDeviceGetPciInfo(self, h): return types.SimpleNamespace(busId=b"00000000:01:00.0")
    def nvmlSystemGetDriverVersion(self): return "560.35.03"
    def nvmlSystemGetCudaDriverVersion(self): return 12060
    def nvmlDeviceGetMemoryInfo(self, h): return types.SimpleNamespace(total=80 * 2**30)
    def nvmlDeviceGetPowerManagementLimit(self, h): return 700_000
    def nvmlDeviceGetEnforcedPowerLimit(self, h): return 700_000
    def nvmlDeviceGetPowerManagementDefaultLimit(self, h): return 700_000
    def nvmlDeviceGetMaxClockInfo(self, h, c): return 1980
    def nvmlDeviceGetPersistenceMode(self, h): raise self.NVMLError("not supported")
    def nvmlDeviceGetClockInfo(self, h, c): return 1755
    def nvmlDeviceGetTemperature(self, h, s): return 40
    def nvmlDeviceGetCurrentClocksEventReasons(self, h): return 0
    def nvmlDeviceGetComputeRunningProcesses(self, h): return []


class _FakeGenerate:
    started: list[str] = []

    def __init__(self, spec, config, seed):
        self.started.append(config["id"])
        if config["batch"] > 1:
            raise RuntimeError("constructed load failure")
        self.details = {"fake": True}

    def sync(self):
        pass

    def run(self, meter, record, plan):
        with meter.phase(record, "idle", self.sync):
            time.sleep(plan["idle_seconds"])
        # Long against the fake counter's 20 ms refresh, so the derived decode
        # (0.4 s - 0.2 s / 2) is within about 10% even before averaging.
        with meter.phase(record, "prefill", self.sync) as phase:
            time.sleep(0.2)
            phase["units"] = 2
        with meter.phase(record, "generate", self.sync) as phase:
            time.sleep(0.4)
            phase["units"] = 1


class _FakeTrain(_FakeGenerate):
    def run(self, meter, record, plan):
        with meter.phase(record, "idle", self.sync):
            time.sleep(plan["idle_seconds"])
        with meter.phase(record, "train", self.sync) as phase:
            time.sleep(0.2)
            phase["units"] = 4


@pytest.mark.parametrize("gpu_name, code", [("NVIDIA H100 80GB HBM3", 0), ("NVIDIA H200", 2)])
def test_runner_writes_verifiable_runs_and_refuses_a_substituted_gpu(tmp_path, monkeypatch, gpu_name, code):
    plan = _plan()
    plan["configs"].append(dict(plan["configs"][0], id="llama-b8", batch=8))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml(gpu_name))
    monkeypatch.setitem(runner.WORKLOADS, "generate", _FakeGenerate)
    out = tmp_path / "runs" / "fake"
    assert runner.measure(plan_path, "H100", ["llama-b1", "llama-b8"], out) == code
    manifest = json.loads((out / "manifest.json").read_text())
    if code:
        assert manifest["trials"] == [] and manifest["aborted"]
        return
    assert protocol.manifest_problems(out, protocol.sha256_file(plan_path), plan) == []
    statuses = {(t["config_id"], t["status"]) for t in manifest["trials"]}
    assert statuses == {("llama-b1", "ok"), ("llama-b8", "failed")}
    assert len(manifest["trials"]) == 2 * (plan["warmup_trials"] + plan["trials"])
    assert manifest["environment"]["packages"]["pytest"] == pytest.__version__
    rows, provenance = _rows(tmp_path)
    assert provenance["runs_used"][0]["harness_matches_committed"] is True
    assert provenance["runs_used"][0]["counter_refresh_ms"] == pytest.approx(20, abs=2)
    decode = next(r for r in rows if r["config_id"] == "llama-b1" and r["phase"] == "decode")
    assert float(decode["measured_power_w"]) == pytest.approx(250, rel=0.15)
    assert decode["trials_failed"] == "0"
    assert next(r for r in rows if r["phase"] == "decode")["trials_used"] == "3"


def test_runner_measures_training_before_starting_the_language_model(tmp_path, monkeypatch):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan(trials=2, warmup_trials=0)))
    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml("NVIDIA H100 80GB HBM3"))
    monkeypatch.setitem(runner.WORKLOADS, "generate", _FakeGenerate)
    monkeypatch.setitem(runner.WORKLOADS, "train", _FakeTrain)
    monkeypatch.setattr(_FakeGenerate, "started", [])
    assert runner.measure(plan_path, "H100", [], tmp_path / "runs" / "order") == 0
    # The plan lists llama-b1 first. vLLM keeps its memory, so training has to run first.
    assert _FakeGenerate.started == ["resnet-b256", "llama-b1"]


def _fake_vllm(monkeypatch, short_by=0, fail=False):
    """Stand-ins for vllm and torch.cuda. Returns the keyword arguments of every engine built."""
    built = []
    ns = types.SimpleNamespace

    class LLM:
        def __init__(self, **kwargs):
            if fail:
                raise RuntimeError("constructed engine failure")
            built.append(kwargs)
            self.llm_engine = ns(cache_config=ns(num_gpu_blocks=1234))

        def get_tokenizer(self):
            return ns(vocab_size=128000)

        def generate(self, prompts, params, use_tqdm=True):
            assert use_tqdm is False and params.ignore_eos and not params.detokenize
            time.sleep(0.002)
            n = params.max_tokens - (short_by if params.max_tokens > 1 else 0)
            return [ns(outputs=[ns(token_ids=list(range(n)))]) for _ in prompts]

    vllm = types.ModuleType("vllm")
    vllm.__version__, vllm.LLM, vllm.SamplingParams = "0.6.4.post1", LLM, lambda **kw: ns(**kw)
    torch = types.ModuleType("torch")
    torch.cuda = ns(synchronize=lambda: None)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml("NVIDIA H100 80GB HBM3"))
    monkeypatch.setattr(runner.GenerateWorkload, "_engines", {})
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "unset")  # restored after the test
    return built


QUICK = {"idle_seconds": 0.01, "phase_min_seconds": 0.02}


def test_vllm_workload_records_prefill_and_full_generations_from_the_committed_plan(monkeypatch):
    built = _fake_vllm(monkeypatch)
    plan = protocol.load_plan(PLAN_PATH)
    spec = plan["workloads"]["llama31_8b"]
    large = protocol.config_by_id(plan, "h100-llama-b16-p2048")
    workload = runner.GenerateWorkload(spec, large, plan["seed"])
    assert len(workload.prompts) == 16
    assert {len(p["prompt_token_ids"]) for p in workload.prompts} == {2048}
    assert max(max(p["prompt_token_ids"]) for p in workload.prompts) < 128000
    record = {"phases": {}}
    workload.run(runner.Meter(0, 0.01), record, QUICK)
    phases = record["phases"]
    assert list(phases) == ["idle", "prefill", "generate"]
    assert (phases["prefill"]["output_tokens"], phases["generate"]["output_tokens"]) == (1, 512)
    assert phases["prefill"]["units"] >= 1 and phases["generate"]["units"] >= 1
    small = protocol.config_by_id(plan, "h100-llama-b1-p128")
    first = runner.GenerateWorkload(spec, small, plan["seed"])
    assert first.prompts == runner.GenerateWorkload(spec, small, plan["seed"]).prompts  # fixed seed
    assert len(built) == 1  # one engine runs every configuration of a run
    assert built[0] == {"model": spec["hf_model"], "revision": spec["hf_revision"],
                        "tokenizer_revision": spec["hf_revision"], "dtype": "float16",
                        "seed": plan["seed"], **spec["vllm"]}
    assert os.environ["VLLM_ATTENTION_BACKEND"] == "FLASH_ATTN"
    assert workload.details["kv_cache_gpu_blocks"] == 1234


def test_vllm_workload_fails_a_trial_that_stops_early(monkeypatch):
    _fake_vllm(monkeypatch, short_by=3)
    plan = protocol.load_plan(PLAN_PATH)
    config = protocol.config_by_id(plan, "h100-llama-b1-p128")
    workload = runner.GenerateWorkload(plan["workloads"]["llama31_8b"], config, plan["seed"])
    with pytest.raises(RuntimeError, match="expected 1 of 512"):
        workload.run(runner.Meter(0, 0.01), {"phases": {}}, QUICK)


def test_vllm_start_failure_is_reported_for_every_later_configuration(monkeypatch):
    _fake_vllm(monkeypatch, fail=True)
    plan = protocol.load_plan(PLAN_PATH)
    spec = plan["workloads"]["llama31_8b"]
    with pytest.raises(RuntimeError, match="constructed engine failure"):
        runner.GenerateWorkload(spec, protocol.config_by_id(plan, "h100-llama-b1-p128"), plan["seed"])
    with pytest.raises(RuntimeError, match="failed to start earlier"):
        runner.GenerateWorkload(spec, protocol.config_by_id(plan, "h100-llama-b8-p128"), plan["seed"])
