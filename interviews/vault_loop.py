#!/usr/bin/env python3
"""
Vault Loop — Overnight autonomous generation for balanced question coverage.

Unlike vault_fill.py (which fills deficit cells once), this script:
1. Fills deficit cells (coverage gaps)
2. Balances over-represented cells by boosting under-represented ones
3. Loops until the cube is both FULL and BALANCED
4. Logs every round to vault_loop.log

Balance metric: coefficient of variation (CV) per track.
Target: CV < 0.30 across all competency areas within each track,
        AND total deficit ≤ 10.

Usage:
    # Run overnight (stops when saturated)
    python3 vault_loop.py

    # Dry run — show balance report only
    python3 vault_loop.py --dry-run

    # Set max rounds (default: 20)
    python3 vault_loop.py --max-rounds 50

    # Set parallelism (default: 8)
    python3 vault_loop.py --parallel 10
"""

import json
import logging
import math
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

from engine.taxonomy import (
    normalize_tag, get_area_for_tag, ALL_TAGS, TAXONOMY,
    get_cell_target, LEVELS_LIST,
)

console = Console()

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_FILE = Path(__file__).parent / "vault_loop.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("vault_loop")


# ── Cube Analysis ────────────────────────────────────────────────────────────

TRACKS = ["cloud", "edge", "mobile", "tinyml"]
AREAS = list(TAXONOMY.keys()) + ["cross-cutting"]


def load_cube():
    """Load corpus and build 3D coverage cube."""
    corpus_path = Path(__file__).parent / "corpus.json"
    with open(corpus_path) as f:
        corpus = json.load(f)

    cube = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for q in corpus:
        t = q.get("track", "?")
        l = q.get("level", "?")
        if l in ("L6", "L6%2B"):
            l = "L6+"
        tag = normalize_tag(q.get("topic", ""))
        area = get_area_for_tag(tag) if tag in ALL_TAGS else "unmapped"
        cube[t][area][l] += 1

    return cube, len(corpus)


def compute_balance(cube):
    """
    Compute balance metrics per track.

    Returns:
        balance_report: dict with per-track stats
        imbalanced_cells: list of cells that need boosting for balance
    """
    balance_report = {}
    imbalanced_cells = []

    for track in TRACKS:
        area_totals = {}
        for area in AREAS:
            total = sum(cube[track][area].get(l, 0) for l in LEVELS_LIST)
            target = sum(get_cell_target(track, area, l) for l in LEVELS_LIST)
            area_totals[area] = {"count": total, "target": target}

        # Compute stats across areas (excluding zero-target areas)
        counts = [v["count"] for v in area_totals.values() if v["target"] > 0]
        targets = [v["target"] for v in area_totals.values() if v["target"] > 0]

        if not counts:
            continue

        mean_count = sum(counts) / len(counts)
        mean_target = sum(targets) / len(targets)

        # Ratio: actual / target for each area
        ratios = []
        for area in AREAS:
            info = area_totals[area]
            if info["target"] > 0:
                ratio = info["count"] / info["target"]
                ratios.append(ratio)

        mean_ratio = sum(ratios) / len(ratios) if ratios else 1.0
        variance = sum((r - mean_ratio) ** 2 for r in ratios) / len(ratios) if ratios else 0
        cv = math.sqrt(variance) / mean_ratio if mean_ratio > 0 else 0

        balance_report[track] = {
            "mean_count": mean_count,
            "mean_target": mean_target,
            "mean_ratio": mean_ratio,
            "cv": cv,
            "area_totals": area_totals,
        }

        # Find under-filled areas relative to the track's mean ratio
        # If an area's fill ratio is < 70% of the track's mean, boost it
        for area in AREAS:
            info = area_totals[area]
            if info["target"] == 0:
                continue
            area_ratio = info["count"] / info["target"]
            if area_ratio < mean_ratio * 0.70 and area_ratio < 1.5:
                # Find which levels in this area are lowest
                for level in LEVELS_LIST:
                    cell_target = get_cell_target(track, area, level)
                    cell_count = cube[track][area].get(level, 0)
                    if cell_target > 0 and cell_count < cell_target * 1.2:
                        boost = max(1, cell_target - cell_count)
                        imbalanced_cells.append({
                            "track": track,
                            "area": area,
                            "level": level,
                            "current": cell_count,
                            "target": cell_target,
                            "deficit": boost,
                            "reason": f"balance (ratio {area_ratio:.1f} vs track mean {mean_ratio:.1f})",
                        })

    return balance_report, imbalanced_cells


def find_deficit_cells(cube):
    """Find all cells below their weighted target (coverage gaps)."""
    cells = []
    from vault_fill import CONCEPT_MAP

    for t in TRACKS:
        for a in AREAS:
            for l in LEVELS_LIST:
                target = get_cell_target(t, a, l)
                current = cube[t][a].get(l, 0)
                deficit = max(0, target - current)
                if deficit > 0:
                    cells.append({
                        "track": t,
                        "area": a,
                        "level": l,
                        "current": current,
                        "target": target,
                        "deficit": deficit,
                        "concept": CONCEPT_MAP.get(a, a),
                        "reason": "coverage",
                    })

    return cells


def print_balance_report(balance_report, round_num=None):
    """Pretty-print the balance report."""
    title = "Balance Report"
    if round_num is not None:
        title += f" (Round {round_num})"

    table = Table(box=box.ROUNDED, title=title)
    table.add_column("Track", width=8)
    table.add_column("Avg Count", justify="center", width=10)
    table.add_column("Avg Target", justify="center", width=10)
    table.add_column("Fill Ratio", justify="center", width=10)
    table.add_column("CV", justify="center", width=8)
    table.add_column("Status", width=10)

    for track in TRACKS:
        if track not in balance_report:
            continue
        info = balance_report[track]
        cv = info["cv"]
        status = "[green]BALANCED[/]" if cv < 0.30 else "[yellow]UNEVEN[/]" if cv < 0.50 else "[red]IMBALANCED[/]"
        table.add_row(
            track,
            f"{info['mean_count']:.1f}",
            f"{info['mean_target']:.1f}",
            f"{info['mean_ratio']:.2f}x",
            f"{cv:.2f}",
            status,
        )

    console.print(table)


def print_area_breakdown(balance_report):
    """Show per-area fill ratios within each track."""
    for track in TRACKS:
        if track not in balance_report:
            continue
        info = balance_report[track]

        table = Table(box=box.SIMPLE, title=f"{track.upper()} — Area Breakdown")
        table.add_column("Area", width=14)
        table.add_column("Count", justify="center", width=7)
        table.add_column("Target", justify="center", width=7)
        table.add_column("Ratio", justify="center", width=7)
        table.add_column("Bar", width=20)

        sorted_areas = sorted(
            info["area_totals"].items(),
            key=lambda x: x[1]["count"] / max(x[1]["target"], 1),
        )

        for area, data in sorted_areas:
            if data["target"] == 0:
                continue
            ratio = data["count"] / data["target"]
            bar_len = min(20, int(ratio * 10))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            color = "green" if ratio >= 1.0 else "yellow" if ratio >= 0.7 else "red"
            table.add_row(
                area,
                str(data["count"]),
                str(data["target"]),
                f"[{color}]{ratio:.1f}x[/{color}]",
                f"[{color}]{bar}[/{color}]",
            )

        console.print(table)


# ── Main Loop ────────────────────────────────────────────────────────────────

def is_saturated(deficit_cells, balance_report):
    """Check if we should stop."""
    total_deficit = sum(c["deficit"] for c in deficit_cells)

    # All tracks balanced?
    all_balanced = all(
        info["cv"] < 0.30
        for info in balance_report.values()
    )

    return total_deficit <= 10 and all_balanced


def run_round(cells_to_fill, parallel: int = 8):
    """Run one generation round: plan → run → merge → rebuild."""
    if not cells_to_fill:
        return 0

    from vault_fill import VAULT_DIR, CONCEPT_MAP

    VAULT_DIR.mkdir(exist_ok=True)

    # Clean old vault files
    for f in VAULT_DIR.glob("*.json"):
        f.unlink()

    worker = Path(__file__).parent / "engine" / "vault_worker.py"
    commands = []
    for i, c in enumerate(cells_to_fill):
        out_file = VAULT_DIR / f"cell_{i:03d}_{c['track']}_{c['area']}_{c['level']}.json"
        concept = c.get("concept", CONCEPT_MAP.get(c["area"], c["area"]))
        cmd = [
            "python3", str(worker),
            c["track"], concept, c["level"],
            "1",  # Generate 1 per cell for even distribution
            str(out_file),
        ]
        commands.append((cmd, out_file, c))

    # Run in parallel batches
    total_generated = 0
    batch_size = parallel

    for batch_start in range(0, len(commands), batch_size):
        batch = commands[batch_start:batch_start + batch_size]
        procs = []

        for cmd, out_file, cell in batch:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=str(Path(__file__).parent),
            )
            procs.append((proc, out_file, cell))

        for proc, out_file, cell in procs:
            proc.wait()
            if out_file.exists():
                try:
                    data = json.loads(out_file.read_text())
                    total_generated += len(data)
                    console.print(
                        f"  [green]✓[/] {cell['track']}/{cell['area']}/{cell['level']} "
                        f"+{len(data)} ({cell.get('reason', 'coverage')})"
                    )
                except Exception:
                    console.print(f"  [red]✗[/] {cell['track']}/{cell['area']}/{cell['level']} parse error")
            else:
                console.print(f"  [red]✗[/] {cell['track']}/{cell['area']}/{cell['level']} no output")

    # Merge
    from vault_fill import cmd_merge
    cmd_merge()

    return total_generated


def main(max_rounds: int = 20, parallel: int = 8, dry_run: bool = False):
    """Main loop: fill + balance until saturated."""
    log.info("=" * 60)
    log.info(f"VAULT LOOP STARTED | max_rounds={max_rounds} parallel={parallel}")
    log.info("=" * 60)

    for round_num in range(1, max_rounds + 1):
        console.print()
        console.print(Rule(f"[bold cyan]Round {round_num} / {max_rounds}[/]", style="cyan"))

        # Rebuild corpus first
        subprocess.run(
            ["python3", "build_corpus.py"],
            capture_output=True,
            cwd=str(Path(__file__).parent),
        )

        # Analyze
        cube, total = load_cube()
        deficit_cells = find_deficit_cells(cube)
        balance_report, imbalanced_cells = compute_balance(cube)

        total_deficit = sum(c["deficit"] for c in deficit_cells)

        log.info(
            f"Round {round_num}: {total} questions | "
            f"deficit={total_deficit} in {len(deficit_cells)} cells | "
            f"imbalanced={len(imbalanced_cells)} cells"
        )

        # Print reports
        console.print(Panel(
            f"[bold]{total}[/] questions\n"
            f"[bold red]{len(deficit_cells)}[/] deficit cells ({total_deficit} questions)\n"
            f"[bold yellow]{len(imbalanced_cells)}[/] imbalanced cells",
            title=f"[bold]Round {round_num} Status",
            border_style="cyan",
        ))

        print_balance_report(balance_report, round_num)
        print_area_breakdown(balance_report)

        # Check saturation
        if is_saturated(deficit_cells, balance_report):
            log.info(f"SATURATED at round {round_num}! Deficit ≤ 10 and all tracks balanced.")
            console.print(Panel(
                f"[bold green]SATURATED![/]\n\n"
                f"Total questions: {total}\n"
                f"Deficit: {total_deficit}\n"
                f"All tracks have CV < 0.30",
                title="[bold green]Cube Complete",
                border_style="green",
            ))
            break

        if dry_run:
            console.print("[dim]Dry run — not generating.[/]")
            break

        # Combine deficit + imbalanced cells, dedup by (track, area, level)
        seen = set()
        cells_to_fill = []

        # Deficit cells first (coverage is priority)
        for c in deficit_cells:
            key = (c["track"], c["area"], c["level"])
            if key not in seen:
                seen.add(key)
                cells_to_fill.append(c)

        # Then balance cells
        for c in imbalanced_cells:
            key = (c["track"], c["area"], c["level"])
            if key not in seen:
                seen.add(key)
                cells_to_fill.append(c)

        if not cells_to_fill:
            log.info("Nothing to generate. Stopping.")
            break

        log.info(f"Generating {len(cells_to_fill)} cells ({sum(c['deficit'] for c in cells_to_fill)} questions)")

        generated = run_round(cells_to_fill, parallel=parallel)
        log.info(f"Round {round_num} complete: +{generated} questions")

        # Cooldown between rounds (respect Gemini rate limits)
        if round_num < max_rounds:
            console.print("[dim]Cooling down 10s before next round...[/]")
            time.sleep(10)

    # Final summary
    cube, total = load_cube()
    deficit_cells = find_deficit_cells(cube)
    balance_report, _ = compute_balance(cube)
    total_deficit = sum(c["deficit"] for c in deficit_cells)

    log.info(f"FINAL: {total} questions | deficit={total_deficit} | rounds={round_num}")

    console.print()
    console.print(Rule("[bold green]Final State[/]", style="green"))
    print_balance_report(balance_report)
    print_area_breakdown(balance_report)

    console.print(Panel(
        f"[bold]{total}[/] total questions\n"
        f"[bold]{total_deficit}[/] remaining deficit\n"
        f"[bold]{round_num}[/] rounds completed\n"
        f"Log: {LOG_FILE}",
        title="[bold green]Vault Loop Complete",
        border_style="green",
    ))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vault Loop — overnight balanced generation")
    parser.add_argument("--max-rounds", type=int, default=20, help="Max generation rounds (default: 20)")
    parser.add_argument("--parallel", type=int, default=8, help="Max parallel Gemini calls (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="Show balance report without generating")
    args = parser.parse_args()

    main(max_rounds=args.max_rounds, parallel=args.parallel, dry_run=args.dry_run)
