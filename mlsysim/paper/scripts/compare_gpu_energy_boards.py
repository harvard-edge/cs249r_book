"""Compare GPU board energy between boards and between the three declared tests.

Every verified run of ``paper/data/gpu-energy/``, ``gpu-energy-workloads/`` and
``gpu-energy-boards/`` is one board. For each configuration of the boards plan,
this collects each run's mean energy per unit of work from its trials, the same
way ``analyze_gpu_energy.py`` does for ``measured_every_run_pair``, and writes:

- ``board_energy.csv``: one row per test, run and reported phase.
- ``board_summary.csv``: one row per configuration. For each device it gives the
  spread between runs measured with the boards plan's seed, and how many runs
  measured with another seed fall inside that range. For the H100 and the A100
  it counts the pairings of one H100 run and one A100 run where the H100 uses
  less energy, over the boards test alone and over all three tests.

Configurations of different tests match on device, phase, batch, token counts
and the whole workload entry of their plan (model, revision, precision, engine
settings), apart from its name. Each run is one board in the committed data, so
the counts are counted over runs. ``power_w`` is the mean energy per unit over
the mean time per unit. Run from ``mlsysim/``; ``--check`` fails if the
committed files are stale.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_energy_protocol as protocol  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
BOARDS = "gpu-energy-boards"
TESTS = ("gpu-energy", "gpu-energy-workloads", BOARDS)
PAIR = ("H100", "A100")

ENERGY_COLUMNS = ["test", "seed", "run", "gpu_uuid", "power_limit_w", "driver_version", "device", "workload",
                  "phase", "batch", "prompt_tokens", "output_tokens", "trials", "j_per_unit", "s_per_unit",
                  "power_w"]


def _key(plan: dict, config: dict, phase: str) -> tuple:
    workload = plan["workloads"][config["workload"]]
    return (config["device"], workload["registry"], phase, config["batch"], config.get("prompt_tokens"),
            config.get("output_tokens"), json.dumps(workload, sort_keys=True))


def run_rows(root: Path, test: str, keys: set[tuple]) -> list[dict]:
    """One row per verified run and reported phase whose configuration matches ``keys``."""
    plan_path = root / "plan.json"
    plan = protocol.load_plan(plan_path)
    plan_sha = protocol.sha256_file(plan_path)
    configs = {c["id"]: c for c in plan["configs"]}
    rows = []
    for run_dir in sorted(p for p in (root / "runs").glob("*") if p.is_dir()):
        problems = protocol.manifest_problems(run_dir, plan_sha, plan)
        if problems:
            print(f"excluded {test}/{run_dir.name}: " + "; ".join(problems), file=sys.stderr)
            continue
        device = json.loads((run_dir / "manifest.json").read_text())["device"]
        limit_mw = device.get("enforced_power_limit_mw")
        trials: dict[str, list[dict]] = {}
        for record in protocol.read_trials(run_dir / protocol.TRIALS_FILE):
            if not record["warmup"] and record["status"] == "ok":
                trials.setdefault(record["config_id"], []).append(record)
        for config_id, records in trials.items():
            config = configs[config_id]
            kind = plan["workloads"][config["workload"]]["kind"]
            for phase in protocol.REPORTED[kind]:
                key = _key(plan, config, phase)
                if key not in keys:
                    continue
                summaries = [protocol.reported_summary(r, phase, config) for r in records]
                joules = sum(s["energy_j"] / s["units"] for s in summaries) / len(summaries)
                seconds = sum(s["seconds"] / s["units"] for s in summaries) / len(summaries)
                rows.append({
                    "test": test, "seed": plan["seed"], "run": run_dir.name, "gpu_uuid": device.get("uuid"),
                    "power_limit_w": None if limit_mw is None else limit_mw / 1000.0,
                    "driver_version": device.get("driver_version"),
                    "device": key[0], "workload": key[1], "phase": phase, "batch": key[3],
                    "prompt_tokens": key[4], "output_tokens": key[5], "trials": len(summaries),
                    "j_per_unit": joules, "s_per_unit": seconds, "power_w": joules / seconds,
                })
    return rows


def summary_rows(rows: list[dict], seed: int, devices: list[str]) -> tuple[list[str], list[dict]]:
    """Per configuration: spread at ``seed``, other-seed runs inside it, and H100-A100 pairings."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["workload"], row["phase"], row["batch"], row["prompt_tokens"], row["output_tokens"])
        groups.setdefault(key, []).append(row)
    columns = ["workload", "phase", "batch", "prompt_tokens", "output_tokens"]
    for device in devices:
        columns += [f"{device}_runs_same_seed", f"{device}_boards_same_seed", f"{device}_min_j_same_seed",
                    f"{device}_max_j_same_seed", f"{device}_spread_same_seed", f"{device}_runs_other_seed",
                    f"{device}_other_seed_inside"]
    first, second = PAIR
    columns += [f"{first}_vs_{second}_pairs_boards_test", f"{first}_lower_boards_test",
                f"{first}_vs_{second}_pairs_all_tests", f"{first}_lower_all_tests"]
    summaries = []
    for key, members in groups.items():
        out = dict(zip(columns[:5], key))
        for device in devices:
            mine = [r for r in members if r["device"] == device]
            same = [r["j_per_unit"] for r in mine if r["seed"] == seed]
            other = [r["j_per_unit"] for r in mine if r["seed"] != seed]
            low, high = (min(same), max(same)) if same else (None, None)
            out.update({
                f"{device}_runs_same_seed": len(same),
                f"{device}_boards_same_seed": len({r["gpu_uuid"] for r in mine if r["seed"] == seed}),
                f"{device}_min_j_same_seed": low, f"{device}_max_j_same_seed": high,
                f"{device}_spread_same_seed": high / low - 1 if same else None,
                f"{device}_runs_other_seed": len(other),
                f"{device}_other_seed_inside": sum(low <= j <= high for j in other) if same else None,
            })
        for scope, tests in (("boards_test", {BOARDS}), ("all_tests", set(TESTS))):
            a = [r["j_per_unit"] for r in members if r["device"] == first and r["test"] in tests]
            b = [r["j_per_unit"] for r in members if r["device"] == second and r["test"] in tests]
            out[f"{first}_vs_{second}_pairs_{scope}"] = len(a) * len(b)
            out[f"{first}_lower_{scope}"] = sum(x < y for x in a for y in b)
        summaries.append(out)
    return columns, summaries


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _csv(columns: list[str], rows: list[dict]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _fmt(row[k]) for k in columns})
    return stream.getvalue()


def compare(data: Path = DATA) -> tuple[str, str]:
    """``board_energy.csv`` and ``board_summary.csv`` text for the three tests under ``data``."""
    boards_plan = protocol.load_plan(data / BOARDS / "plan.json")
    keys = {_key(boards_plan, c, phase) for c in boards_plan["configs"]
            for phase in protocol.REPORTED[boards_plan["workloads"][c["workload"]]["kind"]]}
    rows = [row for test in TESTS for row in run_rows(data / test, test, keys)]
    columns, summaries = summary_rows(rows, boards_plan["seed"], list(boards_plan["devices"]))
    return _csv(ENERGY_COLUMNS, rows), _csv(columns, summaries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the committed files are stale")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args(argv)
    energy, summary = compare(args.data)
    paths = {args.data / BOARDS / "board_energy.csv": energy, args.data / BOARDS / "board_summary.csv": summary}
    if args.check:
        stale = [p.name for p, text in paths.items() if not p.exists() or p.read_text() != text]
        if stale:
            print("stale: " + ", ".join(stale) + " (rerun without --check)", file=sys.stderr)
            return 1
        print(f"board comparison matches {energy.count(chr(10)) - 1} run rows")
        return 0
    for path, text in paths.items():
        path.write_text(text)
    print(f"wrote {energy.count(chr(10)) - 1} run rows to {args.data / BOARDS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
