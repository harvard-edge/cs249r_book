#!/usr/bin/env python3
"""
Vault Fill — Parallel question generation for every empty cube cell.

Architecture:
1. Analyze the 3D coverage cube
2. Generate one shell command per deficit cell
3. Each command writes to a temp JSON file (no file conflicts)
4. Run ALL commands in parallel (via xargs or background processes)
5. Merge step: read all temp JSONs, validate, insert into markdown files

Usage:
    # Generate the commands (dry run)
    python3 vault_fill.py plan

    # Run everything in parallel
    python3 vault_fill.py run

    # Merge temp files into markdown
    python3 vault_fill.py merge

    # All three steps
    python3 vault_fill.py all
"""

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.rule import Rule
from rich import box

from engine.taxonomy import (
    normalize_tag, get_area_for_tag, ALL_TAGS, TAXONOMY,
    get_cell_target, LEVELS_LIST,
)
from engine.schemas import Question
from engine.render import render_markdown, append_to_markdown_file

console = Console()

VAULT_DIR = Path(__file__).parent / "_vault"
CONCEPT_MAP = {
    "compute": "GPU roofline, arithmetic intensity, compute-bound vs memory-bound, TOPS/W",
    "memory": "memory hierarchy, KV-cache, VRAM accounting, SRAM tensor arena, DMA",
    "precision": "quantization INT8/FP16, mixed precision, calibration, overflow",
    "architecture": "scaling laws, CNN vs transformer, depthwise separable, MoE, NAS",
    "latency": "TTFT/TPOT, continuous batching, real-time deadlines, queueing theory",
    "power": "thermal throttling, duty cycling, battery drain, energy harvesting, cooling",
    "optimization": "pruning, distillation, operator fusion, flash-attention, speculative decoding",
    "parallelism": "data/tensor/pipeline parallelism, AllReduce, FSDP/ZeRO",
    "networking": "NVLink vs InfiniBand, PCIe, network topology, RDMA, bus protocols",
    "deployment": "model serving, OTA updates, container orchestration, rollout, RAG, guardrails",
    "reliability": "monitoring, data drift, fault tolerance, checkpointing, watchdog timers",
    "data": "data pipelines, feature stores, training-serving skew, sensor pipeline, data quality",
    "cross-cutting": "security, privacy, federated learning, economics, TCO, A/B testing",
}


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


def find_deficit_cells(cube):
    """Find all cells below their weighted target."""
    tracks = ["cloud", "edge", "mobile", "tinyml"]
    areas = list(TAXONOMY.keys()) + ["cross-cutting"]
    cells = []

    for t in tracks:
        for a in areas:
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
                    })

    return cells


def cmd_plan():
    """Show the generation plan."""
    cube, total = load_cube()
    cells = find_deficit_cells(cube)

    console.print(Panel(
        f"[bold]{total}[/] questions in corpus\n"
        f"[bold red]{len(cells)}[/] cells below target\n"
        f"[bold]{sum(c['deficit'] for c in cells)}[/] questions to generate\n"
        f"[bold cyan]{len(cells)}[/] parallel Gemini calls (~30s wall time)",
        title="[bold]Vault Fill Plan",
        border_style="cyan",
    ))

    table = Table(box=box.SIMPLE, title="Deficit Cells")
    table.add_column("Track", width=8)
    table.add_column("Area", width=14)
    table.add_column("Level", width=6)
    table.add_column("Have", width=5, justify="center")
    table.add_column("Need", width=5, justify="center")
    table.add_column("+Gen", width=5, justify="center")

    for c in sorted(cells, key=lambda x: (-x["deficit"], x["track"], x["area"])):
        table.add_row(
            c["track"], c["area"], c["level"],
            str(c["current"]), str(c["target"]),
            f"[bold cyan]+{c['deficit']}[/]",
        )

    console.print(table)


def cmd_run(max_parallel: int = 8):
    """Generate questions for all deficit cells in parallel."""
    cube, total = load_cube()
    cells = find_deficit_cells(cube)

    if not cells:
        console.print("[bold green]Vault is full! No deficit cells.[/]")
        return

    VAULT_DIR.mkdir(exist_ok=True)

    # Clean old vault files
    for f in VAULT_DIR.glob("*.json"):
        f.unlink()

    console.print(f"[bold]Launching {len(cells)} parallel generation jobs...[/]")
    console.print(f"[dim]Max parallel: {max_parallel} | Output: {VAULT_DIR}/[/]")
    console.print()

    # Build commands using the vault worker script (avoids shell quoting issues)
    worker = Path(__file__).parent / "engine" / "vault_worker.py"
    commands = []
    for i, c in enumerate(cells):
        out_file = VAULT_DIR / f"cell_{i:03d}_{c['track']}_{c['area']}_{c['level']}.json"
        cmd = [
            "python3", str(worker),
            c["track"], c["concept"], c["level"],
            str(c["deficit"]), str(out_file),
        ]
        commands.append((cmd, out_file, c))

    # Run in parallel batches
    t0 = time.time()
    total_generated = 0
    batch_size = max_parallel

    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold]{task.description}[/]"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating...", total=len(commands))

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

            # Wait for batch
            for proc, out_file, cell in procs:
                proc.wait()
                if out_file.exists():
                    try:
                        data = json.loads(out_file.read_text())
                        count = len(data)
                        total_generated += count
                        console.print(
                            f"  [green]✓[/] {cell['track']}/{cell['area']}/{cell['level']} "
                            f"+{count} questions"
                        )
                    except Exception:
                        console.print(
                            f"  [red]✗[/] {cell['track']}/{cell['area']}/{cell['level']} "
                            f"parse error"
                        )
                else:
                    console.print(
                        f"  [red]✗[/] {cell['track']}/{cell['area']}/{cell['level']} "
                        f"no output"
                    )

                progress.advance(task)

    elapsed = time.time() - t0
    vault_files = list(VAULT_DIR.glob("*.json"))
    console.print(Panel(
        f"[bold green]{total_generated}[/] questions generated\n"
        f"[bold]{len(vault_files)}[/] vault files\n"
        f"[bold]{elapsed:.0f}s[/] wall time\n"
        f"Run [cyan]python3 vault_fill.py merge[/] to insert into markdown",
        title="[bold green]Generation Complete",
        border_style="green",
    ))


def cmd_merge():
    """Merge vault JSON files into markdown question files."""
    if not VAULT_DIR.exists():
        console.print("[red]No vault directory. Run 'vault_fill.py run' first.[/]")
        return

    vault_files = sorted(VAULT_DIR.glob("*.json"))
    if not vault_files:
        console.print("[red]No vault files to merge.[/]")
        return

    console.print(Rule("[bold cyan]Merging Vault[/]", style="cyan"))

    total_merged = 0
    total_failed = 0

    for vf in vault_files:
        try:
            data = json.loads(vf.read_text())
        except Exception:
            console.print(f"  [red]✗[/] {vf.name}: invalid JSON")
            total_failed += 1
            continue

        for raw in data:
            try:
                # Handle options format
                if "options" in raw and raw["options"]:
                    for opt in raw["options"]:
                        if isinstance(opt, str):
                            raw["options"] = None
                            break

                q = Question(**raw)
                target = append_to_markdown_file(q)
                total_merged += 1
                console.print(
                    f"  [green]✓[/] {q.level} {q.title[:40]} → {target.name}"
                )
            except Exception as e:
                total_failed += 1
                title = raw.get("title", "?")[:30]
                console.print(f"  [red]✗[/] {title}: {str(e)[:50]}")

    # Rebuild corpus
    console.print()
    console.print("[dim]Rebuilding corpus.json...[/]")
    subprocess.run(["python3", "build_corpus.py"], capture_output=True)

    console.print(Panel(
        f"[bold green]{total_merged}[/] questions merged into markdown\n"
        f"[bold red]{total_failed}[/] failed\n"
        f"corpus.json rebuilt",
        title="[bold green]Merge Complete",
        border_style="green",
    ))

    # Clean vault
    for vf in vault_files:
        vf.unlink()
    console.print("[dim]Vault cleaned.[/]")


def cmd_all(max_parallel: int = 8):
    """Plan + Run + Merge in one shot."""
    cmd_plan()
    console.print()
    cmd_run(max_parallel=max_parallel)
    console.print()
    cmd_merge()

    # Final coverage
    cube, total = load_cube()
    cells = find_deficit_cells(cube)
    filled = 312 - len(cells) - 21  # 21 are SKIP cells
    console.print(Panel(
        f"[bold]{total}[/] total questions\n"
        f"[bold green]{filled + 21}/312[/] cells filled ({(filled + 21)/312:.0%})\n"
        f"[bold]{len(cells)}[/] cells still below target",
        title="[bold]Final Coverage",
        border_style="cyan",
    ))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vault Fill — parallel cube generation")
    parser.add_argument("command", choices=["plan", "run", "merge", "all"], default="plan", nargs="?")
    parser.add_argument("--parallel", type=int, default=6, help="Max parallel Gemini calls (default: 6)")
    args = parser.parse_args()

    {"plan": cmd_plan, "run": cmd_run, "merge": cmd_merge, "all": cmd_all}[args.command](
        *([args.parallel] if args.command in ("run", "all") else [])
    )
