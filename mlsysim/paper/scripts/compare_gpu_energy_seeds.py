"""Compare GPU board energy between two seeds measured on the same board.

Each container of ``paper/data/gpu-energy-seeds/`` measured the plan in
``seed-2193/`` and the plan in ``seed-2198/`` on one GPU, one after the other,
and wrote both runs under the same folder name. This pairs those runs. For each
configuration of the plans it writes, into ``gpu-energy-seeds/``:

- ``seed_pairs.csv``: one row per pair and reported phase, with each seed's
  mean energy per unit of work and the ratio of seed 2198's to seed 2193's.
  ``first_seed_run`` names the folder whose run the container measured first.
- ``seed_summary.csv``: one row per device and configuration, with the mean,
  lowest and highest ratio, how many pairs have a ratio below 1, the mean ratio
  by which seed ran first, the spread between boards within each seed, and
  whether the boards have the same order by energy with both seeds. It also
  gives the ratio of the mean energy of ``gpu-energy-workloads/`` to that of
  ``gpu-energy-boards/`` from ``gpu-energy-boards/board_energy.csv``.

A pair whose two runs report different GPUs stops the script. Runs without a
partner are listed and left out. Run from ``mlsysim/``; ``--check`` fails if
the committed files are stale.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_gpu_energy_boards as boards  # noqa: E402
import gpu_energy_protocol as protocol  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
SEEDS = "gpu-energy-seeds"
FIRST, SECOND = "seed-2193", "seed-2198"

PAIR_COLUMNS = ["device", "workload", "phase", "batch", "prompt_tokens", "output_tokens", "run", "gpu_uuid",
                "power_limit_w", "driver_version", "first_seed_run", "seed_a", "seed_b", "j_per_unit_a",
                "j_per_unit_b", "ratio_b_over_a"]
SUMMARY_COLUMNS = ["device", "workload", "phase", "batch", "prompt_tokens", "output_tokens", "pairs",
                   "mean_ratio", "min_ratio", "max_ratio", "pairs_below_1", "pairs_a_first", "mean_ratio_a_first",
                   "pairs_b_first", "mean_ratio_b_first", "spread_a", "spread_b", "same_board_order",
                   "workloads_over_boards"]
CONFIG = ("device", "workload", "phase", "batch", "prompt_tokens", "output_tokens")


def _started(root: Path) -> dict[str, str]:
    return {run.name: json.loads((run / "manifest.json").read_text()).get("started_utc", "")
            for run in (root / "runs").glob("*") if (run / "manifest.json").exists()}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _test_gap(board_energy: Path) -> dict[tuple, float]:
    """Mean energy of gpu-energy-workloads/ runs over that of gpu-energy-boards/ runs, per configuration."""
    if not board_energy.exists():
        return {}
    groups: dict[tuple, dict[str, list[float]]] = {}
    with board_energy.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["device"], row["workload"], row["phase"], int(row["batch"]),
                   int(row["prompt_tokens"]) if row["prompt_tokens"] else None,
                   int(row["output_tokens"]) if row["output_tokens"] else None)
            groups.setdefault(key, {}).setdefault(row["test"], []).append(float(row["j_per_unit"]))
    return {key: _mean(tests["gpu-energy-workloads"]) / _mean(tests["gpu-energy-boards"])
            for key, tests in groups.items() if "gpu-energy-workloads" in tests and "gpu-energy-boards" in tests}


def compare(data: Path = DATA) -> tuple[str, str]:
    """``seed_pairs.csv`` and ``seed_summary.csv`` text for the paired runs under ``data``."""
    roots = {name: data / SEEDS / name for name in (FIRST, SECOND)}
    plans = {name: protocol.load_plan(root / "plan.json") for name, root in roots.items()}
    keys = {boards._key(plans[FIRST], c, phase) for c in plans[FIRST]["configs"]
            for phase in protocol.REPORTED[plans[FIRST]["workloads"][c["workload"]]["kind"]]}
    rows = {name: {(r["run"],) + tuple(r[c] for c in CONFIG): r for r in boards.run_rows(root, name, keys)}
            for name, root in roots.items()}
    started = {name: _started(root) for name, root in roots.items()}
    for name, other in ((FIRST, SECOND), (SECOND, FIRST)):
        for run in sorted({k[0] for k in rows[name]} - {k[0] for k in rows[other]}):
            print(f"{name}/{run} has no run in {other}/ and is left out", file=sys.stderr)
    pairs = []
    for key in sorted(rows[FIRST].keys() & rows[SECOND].keys(), key=str):
        a, b = rows[FIRST][key], rows[SECOND][key]
        if a["gpu_uuid"] != b["gpu_uuid"]:
            raise SystemExit(f"{key[0]}: {FIRST} ran on {a['gpu_uuid']} and {SECOND} on {b['gpu_uuid']}")
        first = FIRST if started[FIRST][key[0]] < started[SECOND][key[0]] else SECOND
        pairs.append({**{c: a[c] for c in CONFIG}, "run": key[0], "gpu_uuid": a["gpu_uuid"],
                      "power_limit_w": a["power_limit_w"], "driver_version": a["driver_version"],
                      "first_seed_run": first, "seed_a": a["seed"], "seed_b": b["seed"],
                      "j_per_unit_a": a["j_per_unit"], "j_per_unit_b": b["j_per_unit"],
                      "ratio_b_over_a": b["j_per_unit"] / a["j_per_unit"]})
    gap = _test_gap(data / boards.BOARDS / "board_energy.csv")
    groups: dict[tuple, list[dict]] = {}
    for pair in pairs:
        groups.setdefault(tuple(pair[c] for c in CONFIG), []).append(pair)
    summary = []
    for key, members in groups.items():
        ratios = [p["ratio_b_over_a"] for p in members]
        a_first = [p["ratio_b_over_a"] for p in members if p["first_seed_run"] == FIRST]
        b_first = [p["ratio_b_over_a"] for p in members if p["first_seed_run"] == SECOND]
        energy_a = [p["j_per_unit_a"] for p in members]
        energy_b = [p["j_per_unit_b"] for p in members]
        order_a = [p["run"] for p in sorted(members, key=lambda p: p["j_per_unit_a"])]
        order_b = [p["run"] for p in sorted(members, key=lambda p: p["j_per_unit_b"])]
        summary.append({
            **dict(zip(CONFIG, key)), "pairs": len(members), "mean_ratio": _mean(ratios),
            "min_ratio": min(ratios), "max_ratio": max(ratios), "pairs_below_1": sum(r < 1 for r in ratios),
            "pairs_a_first": len(a_first), "mean_ratio_a_first": _mean(a_first) if a_first else None,
            "pairs_b_first": len(b_first), "mean_ratio_b_first": _mean(b_first) if b_first else None,
            "spread_a": max(energy_a) / min(energy_a) - 1, "spread_b": max(energy_b) / min(energy_b) - 1,
            "same_board_order": order_a == order_b, "workloads_over_boards": gap.get(key),
        })
    return boards._csv(PAIR_COLUMNS, pairs), boards._csv(SUMMARY_COLUMNS, summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the committed files are stale")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args(argv)
    pairs, summary = compare(args.data)
    paths = {args.data / SEEDS / "seed_pairs.csv": pairs, args.data / SEEDS / "seed_summary.csv": summary}
    count = pairs.count("\n") - 1
    if args.check:
        stale = [p.name for p, text in paths.items() if not p.exists() or p.read_text() != text]
        if stale:
            print("stale: " + ", ".join(stale) + " (rerun without --check)", file=sys.stderr)
            return 1
        print(f"seed comparison matches {count} pair rows")
        return 0
    for path, text in paths.items():
        path.write_text(text)
    print(f"wrote {count} pair rows to {args.data / SEEDS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
