#!/usr/bin/env python3
"""Compare measured GPU board energy with MLSys·im's two device-power assumptions.

Reads every run under paper/data/gpu-energy/runs/, verifies it against its
manifest and the committed plan, and writes paper/data/gpu-energy/summary.csv
and summary.provenance.json. Each row is one configuration and phase, per unit
of work: one prefill, one decode step or one training step. A decode step is
the difference between a full generation and a one-token generation, divided
by ``output_tokens - 1``. Next to the measured board energy (counter, mean and
95% t interval over trials) each row gives four estimates, all computed as
``RunFootprintModel`` accelerator energy:

* rated, measured time: registry TDP over the measured time.
* rated, modeled time: registry TDP over the time ``ServingModel`` or
  ``SingleNodeModel`` predicts at the plan's declared efficiency.
* proportional, measured time: TDP x (0.30 + 0.70 u) over the measured time,
  with u the utilization that time implies.
* proportional, modeled time: the same rule over the modeled time.

Rated TDP is what the sustainability scenarios pass as device power. The
proportional rule is ``Engine.solve``'s energy model (``ENERGY_IDLE_FRACTION``
and ``ENERGY_DYNAMIC_FRACTION`` in ``engine/calibration.py``). Here u is the
phase's FLOPs, counted as the package counts them, over the time and the peak
rate. For training, "proportional, modeled" equals ``SingleNodeModel``'s own
energy. Comparing the measured-time and modeled-time columns separates the
power assumption's error from the time model's.

The script also writes paper/data/gpu-energy/device_choice.csv. For every
configuration measured on more than one device, it gives the device with the
lowest energy per unit of work by the meter and by each estimate, and whether
they agree. That is the decision an energy (or, at one site, carbon) estimate
would drive, so it shows whether an estimate's error changes the choice as
well as how large the error is.

Everything is at the board boundary (``it_ratio = 1``, ``pue = 1``). Warm-up
and failed trials are counted and never averaged. Runs that do not verify are
listed with the reason and excluded.

    PYTHONPATH=. python paper/scripts/analyze_gpu_energy.py          # write
    PYTHONPATH=. python paper/scripts/analyze_gpu_energy.py --check  # compare
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_energy_protocol as protocol  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "gpu-energy"

COLUMNS = [
    "config_id", "split", "device", "hardware", "workload", "phase", "unit",
    "batch", "prompt_tokens", "output_tokens", "efficiency",
    "runs", "gpus", "power_limits_w", "trials_used", "trials_failed", "units_per_trial",
    "measured_s_per_unit", "measured_s_per_unit_ci95",
    "measured_j_per_unit", "measured_j_per_unit_ci95",
    "measured_power_w", "measured_power_w_ci95",
    "rated_power_w", "measured_power_over_rated", "idle_power_w", "idle_over_rated",
    "sampled_over_counter", "counter_refresh_ms",
    "modeled_s_per_unit", "modeled_over_measured_time",
    "utilization_at_measured_time", "utilization_at_modeled_time",
    "rated_measured_time_j_per_unit", "rated_measured_time_over_measured",
    "rated_modeled_time_j_per_unit", "rated_modeled_time_over_measured",
    "proportional_measured_time_j_per_unit", "proportional_measured_time_over_measured",
    "proportional_modeled_time_j_per_unit", "proportional_modeled_time_over_measured",
]
UNITS = {"prefill": "prefill", "decode": "decode step", "train": "training step"}
ESTIMATES = ("rated_measured_time", "rated_modeled_time", "proportional_measured_time", "proportional_modeled_time")


def resolve(path: str):
    """Look up a registry path such as ``Hardware.Cloud.H100``."""
    import mlsysim

    obj = mlsysim
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def board_energy_j(seconds: float, power):
    """Accelerator energy over ``seconds`` at ``power``, through RunFootprintModel."""
    from mlsysim import Q_
    from mlsysim.solvers import RunFootprintModel

    result = RunFootprintModel().solve(
        Q_(seconds, "second"), 1, power, 1.0, 1.0, {"board": Q_(1, "g/kWh")}, Q_(0, "kg"), Q_(1, "year")
    )
    return result.accelerator_energy.m_as("J")


def proportional_power(tdp, utilization: float):
    """``Engine.solve``'s device power: TDP x (idle fraction + dynamic fraction x utilization)."""
    from mlsysim.engine import calibration

    u = max(0.0, min(utilization, 1.0))  # Engine.solve clamps HFU the same way
    return tdp * (calibration.ENERGY_IDLE_FRACTION + calibration.ENERGY_DYNAMIC_FRACTION * u)


def work_seconds(plan: dict, config: dict, phase: str) -> float:
    """Seconds one unit of work takes at the device's full peak FLOP rate.

    Utilization over a phase that lasts ``t`` is ``work_seconds / t``. FLOPs are
    counted as the package counts them: training from ``SingleNodeModel``'s
    hardware utilization (HFU x modeled step time), prefill and decode from the
    decode-step helper ServingModel uses (with ``query_tokens = prompt_tokens``
    it reproduces ServingModel's prefill FLOPs exactly, which a test checks).
    """
    from mlsysim import Q_
    from mlsysim.core.units import resolve_precision
    from mlsysim.engine.solvers._decode import decode_step
    from mlsysim.solvers import SingleNodeModel

    workload = plan["workloads"][config["workload"]]
    model = resolve(workload["registry"])
    hardware = resolve(plan["devices"][config["device"]]["registry"])
    efficiency = protocol.efficiency_of(plan, config)
    if phase == "train":
        profile = SingleNodeModel().solve(
            model, hardware, batch_size=config["batch"], precision=workload["precision"],
            efficiency=efficiency, is_training=True,
        )
        return profile.hfu * profile.latency.m_as("s")
    precision, bytes_per_element = resolve_precision(workload["precision"])
    if phase == "prefill":
        query, context = config["prompt_tokens"], config["prompt_tokens"]
    else:
        query = 1
        context = protocol.model_seq_len(workload["model_seq_len"], config["prompt_tokens"], config["output_tokens"])
    step = decode_step(
        model, hardware, batch_size=config["batch"], context_tokens=context, precision=precision,
        bytes_per_element=bytes_per_element, kv_bytes=Q_(0, "B"), efficiency=efficiency, query_tokens=query,
    )
    # step.compute = FLOPs / (peak x efficiency)
    return step.compute.m_as("s") * efficiency


def modeled_seconds(plan: dict, config: dict, phase: str) -> float:
    """Time per unit of work that the package predicts for one phase."""
    from mlsysim.solvers import ServingModel, SingleNodeModel

    workload = plan["workloads"][config["workload"]]
    model = resolve(workload["registry"])
    hardware = resolve(plan["devices"][config["device"]]["registry"])
    efficiency = protocol.efficiency_of(plan, config)
    batch = config["batch"]
    if phase == "train":
        profile = SingleNodeModel().solve(
            model, hardware, batch_size=batch, precision=workload["precision"],
            efficiency=efficiency, is_training=True,
        )
        return batch / profile.throughput.m_as("1/s")
    if phase == "prefill":
        seq_len = config["prompt_tokens"]
    else:
        seq_len = protocol.model_seq_len(workload["model_seq_len"], config["prompt_tokens"], config["output_tokens"])
    result = ServingModel().solve(
        model, hardware, seq_len=seq_len, batch_size=batch, precision=workload["precision"], efficiency=efficiency
    )
    return (result.ttft if phase == "prefill" else result.itl).m_as("s")


def _ms(seconds):
    return None if seconds is None else seconds * 1000.0


def _watts(milliwatts):
    return None if milliwatts is None else milliwatts / 1000.0


def load_runs(root: Path, plan: dict, plan_sha: str):
    """Verified trial records by config id, the run each came from, and excluded runs."""
    records: dict[str, list[dict]] = {}
    sources: dict[str, set[str]] = {}
    failed: dict[str, int] = {}
    excluded, used = [], []
    harness = {name: protocol.sha256_file(Path(__file__).resolve().parent / name)
               for name in ("measure_gpu_energy.py", "gpu_energy_protocol.py")}
    for run_dir in sorted(p for p in (root / "runs").glob("*") if p.is_dir()):
        problems = protocol.manifest_problems(run_dir, plan_sha, plan)
        manifest_path = run_dir / "manifest.json"
        if problems:
            excluded.append({"run": run_dir.name, "problems": problems})
            continue
        manifest = json.loads(manifest_path.read_text())
        probe = manifest.get("counter_probe") or {}
        used.append({
            "run": run_dir.name,
            "manifest_sha256": protocol.sha256_file(manifest_path),
            "gpu_uuid": manifest["device"]["uuid"],
            # Boards of one model can ship with different limits, which move both power and TDP's meaning.
            "power_limit_w": _watts(manifest["device"].get("enforced_power_limit_mw")),
            # A run measured with an older runner still verifies, and this field says so.
            "harness_matches_committed": manifest.get("harness_sha256") == harness,
            "counter_refresh_ms": _ms(protocol.refresh_interval_s(probe.get("energy_changes", []))),
            "power_refresh_ms": _ms(protocol.refresh_interval_s(probe.get("power_changes", []))),
            "probe_polls": probe.get("polls"),
        })
        for record in protocol.read_trials(run_dir / protocol.TRIALS_FILE):
            if record["warmup"]:
                continue
            if record["status"] != "ok":
                failed[record["config_id"]] = failed.get(record["config_id"], 0) + 1
                continue
            record["_run"] = run_dir.name  # in memory only, to compare runs
            records.setdefault(record["config_id"], []).append(record)
            sources.setdefault(record["config_id"], set()).add(run_dir.name)
    return records, sources, failed, excluded, used


def fmt(value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def device_choices(plan: dict, rows: list[dict]) -> tuple[list[str], list[dict]]:
    """For each configuration measured on more than one device, the device each energy figure prefers.

    Configurations match on workload, phase, batch, and token counts. The meter's
    choice is "clear" when its 95% interval lies wholly below every other
    device's interval.
    """
    devices = list(plan["devices"])
    groups: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row["workload"], row["phase"], row["batch"], row["prompt_tokens"], row["output_tokens"])
        groups.setdefault(key, {})[row["device"]] = row
    columns = ["workload", "phase", "unit", "batch", "prompt_tokens", "output_tokens"]
    for device in devices:
        columns += [f"{device}_config", f"{device}_measured_j_per_unit", f"{device}_measured_j_per_unit_ci95"]
    columns += ["measured_preferred", "measured_margin", "measured_clear", "measured_every_run_pair"]
    for estimate in ESTIMATES:
        columns += [f"{estimate}_preferred", f"{estimate}_agrees"]
    choices = []
    for (workload, phase, batch, prompt, output), by_device in groups.items():
        present = [d for d in devices if d in by_device]
        if len(present) < 2:
            continue

        def cheapest(field):
            return min(present, key=lambda d: by_device[d][field])

        best = cheapest("measured_j_per_unit")
        low, low_ci = by_device[best]["measured_j_per_unit"], by_device[best]["measured_j_per_unit_ci95"]
        others = [by_device[d] for d in present if d != best]
        choice = {"workload": workload, "phase": phase, "unit": UNITS[phase], "batch": batch,
                  "prompt_tokens": prompt, "output_tokens": output}
        for device in present:
            row = by_device[device]
            choice.update({f"{device}_config": row["config_id"],
                           f"{device}_measured_j_per_unit": row["measured_j_per_unit"],
                           f"{device}_measured_j_per_unit_ci95": row["measured_j_per_unit_ci95"]})
        choice["measured_preferred"] = best
        choice["measured_margin"] = min(r["measured_j_per_unit"] for r in others) / low - 1
        choice["measured_clear"] = all(
            low + low_ci < r["measured_j_per_unit"] - r["measured_j_per_unit_ci95"] for r in others
        )
        # Trial intervals hide board-to-board spread. This asks whether the same device wins
        # for every pairing of one run per device (each run is one physical GPU).
        best_runs = by_device[best]["_j_per_unit_by_run"].values()
        choice["measured_every_run_pair"] = all(
            mine < theirs for r in others for theirs in r["_j_per_unit_by_run"].values() for mine in best_runs
        )
        for estimate in ESTIMATES:
            picked = cheapest(f"{estimate}_j_per_unit")
            choice[f"{estimate}_preferred"] = picked
            choice[f"{estimate}_agrees"] = picked == best
        choices.append(choice)
    return columns, choices


def _csv(columns: list[str], rows: list[dict]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: fmt(v) for k, v in row.items()})
    return stream.getvalue()


def summarize(root: Path = DATA) -> tuple[str, dict, str]:
    """Summary CSV text, provenance, and device-choice CSV text for every verified run under ``root``."""
    from mlsysim.engine import calibration

    plan_path = root / "plan.json"
    plan = protocol.load_plan(plan_path)
    plan_sha = protocol.sha256_file(plan_path)
    records, sources, failed, excluded, used = load_runs(root, plan, plan_sha)
    run_info = {run["run"]: run for run in used}
    rows = []
    for config in plan["configs"]:
        trials = records.get(config["id"], [])
        if not trials:
            continue
        workload = plan["workloads"][config["workload"]]
        hardware = resolve(plan["devices"][config["device"]]["registry"])
        tdp = hardware.tdp
        rated_w = tdp.m_as("W")
        idle_w, _, _ = protocol.mean_ci95(protocol.phase_summary(r, "idle")["avg_power_w"] for r in trials)
        refresh = [run_info[name]["counter_refresh_ms"] for name in sources[config["id"]]]
        # The slowest refresh among contributing runs bounds every phase boundary's error.
        refresh_ms = None if None in refresh else max(refresh)
        limits = sorted({run_info[name]["power_limit_w"] for name in sources[config["id"]]} - {None})
        power_limits = "/".join(fmt(limit) for limit in limits)
        for phase in protocol.REPORTED[workload["kind"]]:
            summaries = [protocol.reported_summary(r, phase, config) for r in trials]
            sec, sec_ci, n = protocol.mean_ci95(s["seconds"] / s["units"] for s in summaries)
            joule, joule_ci, _ = protocol.mean_ci95(s["energy_j"] / s["units"] for s in summaries)
            power, power_ci, _ = protocol.mean_ci95(s["avg_power_w"] for s in summaries)
            units, _, _ = protocol.mean_ci95(s["units"] for s in summaries)
            per_run = {}
            for record, s in zip(trials, summaries):
                per_run.setdefault(record["_run"], []).append(s["energy_j"] / s["units"])
            ratios = [s["sampled_energy_j"] / s["energy_j"] for s in summaries
                      if s["sampled_energy_j"] is not None and s["energy_j"] > 0]
            sampled = protocol.mean_ci95(ratios)[0] if ratios else None
            model_s = modeled_seconds(plan, config, phase)
            work_s = work_seconds(plan, config, phase)
            rated_measured = board_energy_j(sec, tdp)
            rated_modeled = board_energy_j(model_s, tdp)
            prop_measured = board_energy_j(sec, proportional_power(tdp, work_s / sec))
            prop_modeled = board_energy_j(model_s, proportional_power(tdp, work_s / model_s))
            rows.append({
                "config_id": config["id"], "split": config["split"], "device": config["device"],
                "hardware": plan["devices"][config["device"]]["registry"], "workload": workload["registry"],
                "phase": phase, "unit": UNITS[phase], "batch": config["batch"],
                "prompt_tokens": config.get("prompt_tokens"), "output_tokens": config.get("output_tokens"),
                "efficiency": protocol.efficiency_of(plan, config),
                "runs": len(sources[config["id"]]), "gpus": len({r["device"]["uuid"] for r in trials}),
                "power_limits_w": power_limits,
                "trials_used": n, "trials_failed": failed.get(config["id"], 0), "units_per_trial": units,
                "measured_s_per_unit": sec, "measured_s_per_unit_ci95": sec_ci,
                "measured_j_per_unit": joule, "measured_j_per_unit_ci95": joule_ci,
                "measured_power_w": power, "measured_power_w_ci95": power_ci,
                "rated_power_w": rated_w, "measured_power_over_rated": power / rated_w,
                "idle_power_w": idle_w, "idle_over_rated": idle_w / rated_w,
                "sampled_over_counter": sampled, "counter_refresh_ms": refresh_ms,
                "modeled_s_per_unit": model_s, "modeled_over_measured_time": model_s / sec,
                "utilization_at_measured_time": work_s / sec, "utilization_at_modeled_time": work_s / model_s,
                "rated_measured_time_j_per_unit": rated_measured,
                "rated_measured_time_over_measured": rated_measured / joule,
                "rated_modeled_time_j_per_unit": rated_modeled,
                "rated_modeled_time_over_measured": rated_modeled / joule,
                "proportional_measured_time_j_per_unit": prop_measured,
                "proportional_measured_time_over_measured": prop_measured / joule,
                "proportional_modeled_time_j_per_unit": prop_modeled,
                "proportional_modeled_time_over_measured": prop_modeled / joule,
                "_j_per_unit_by_run": {run: sum(v) / len(v) for run, v in per_run.items()},
            })
    text = _csv(COLUMNS, rows)
    choice_columns, choices = device_choices(plan, rows)
    choice_text = _csv(choice_columns, choices)
    provenance = {
        "source": "GPU board-energy measurements made with paper/scripts/measure_gpu_energy.py",
        "plan_sha256": plan_sha,
        "analysis_sha256": protocol.sha256_file(Path(__file__)),
        "protocol_sha256": protocol.sha256_file(Path(protocol.__file__)),
        "summary_sha256": protocol.sha256_bytes(text.encode()),
        "device_choice_sha256": protocol.sha256_bytes(choice_text.encode()),
        "device_choice": "lowest energy per unit of work among devices measured at the same workload, phase, "
                         "batch, and token counts. measured_clear: its 95% interval lies below the others'. "
                         "measured_every_run_pair: it also wins for every pairing of one run per device "
                         "(added after measurement, once the two A100 boards differed by more than the trial "
                         "intervals)",
        "boundary": plan["boundary"],
        "estimates": "RunFootprintModel accelerator energy, it_ratio=1, pue=1, at two device powers: registry "
                     "TDP, and Engine.solve's proportional rule TDP x (idle + dynamic x utilization). Modeled "
                     "time from ServingModel (prefill: ttft at seq_len=prompt_tokens, decode: itl at the plan's "
                     "model_seq_len rule) or SingleNodeModel (batch / training throughput). Utilization is the "
                     "package's FLOP count over time x peak FLOP/s.",
        "package_power_rule": {"idle_fraction": calibration.ENERGY_IDLE_FRACTION,
                               "dynamic_fraction": calibration.ENERGY_DYNAMIC_FRACTION,
                               "source": "mlsysim/engine/calibration.py"},
        "interval": "two-sided 95% Student-t interval over trials, warm-up trials excluded",
        "runs_used": used,
        "runs_excluded": excluded,
        "rows": len(rows),
        "device_choice_rows": len(choices),
    }
    return text, provenance, choice_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the committed summary is stale")
    parser.add_argument("--root", type=Path, default=DATA)
    args = parser.parse_args(argv)
    text, provenance, choice_text = summarize(args.root)
    csv_path, prov_path = args.root / "summary.csv", args.root / "summary.provenance.json"
    choice_path = args.root / "device_choice.csv"
    prov_text = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    for run in provenance["runs_excluded"]:
        print(f"excluded {run['run']}: " + "; ".join(run["problems"]), file=sys.stderr)
    if args.check:
        if not provenance["runs_used"] and not csv_path.exists():
            print("no runs and no committed summary: nothing to check")
            return 0
        stale = [p.name for p, new in ((csv_path, text), (prov_path, prov_text), (choice_path, choice_text))
                 if not p.exists() or p.read_text() != new]
        if stale:
            print("stale: " + ", ".join(stale) + " (rerun without --check)", file=sys.stderr)
            return 1
        print(f"summary matches {len(provenance['runs_used'])} verified runs ({provenance['rows']} rows)")
        return 0
    csv_path.write_text(text)
    prov_path.write_text(prov_text)
    choice_path.write_text(choice_text)
    print(f"wrote {provenance['rows']} rows from {len(provenance['runs_used'])} runs to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
