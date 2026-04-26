#!/usr/bin/env python3
"""
StaffML Question Generation Engine — One Command

Usage:
    python3 generate.py                          # Survey + auto-generate to fill gaps
    python3 generate.py --budget 50              # Generate up to 50 questions
    python3 generate.py --target 5               # Target 5 questions per concept×level cell
    python3 generate.py --track cloud            # Only fill gaps in cloud track
    python3 generate.py --level L1,L2            # Only fill L1 and L2 gaps
    python3 generate.py --dry-run                # Show plan without generating
    python3 generate.py --model gemini-3.1-pro-preview  # Use a different model

The engine:
1. Surveys the existing corpus coverage matrix
2. Identifies gaps (concept×level cells below target)
3. Generates questions to fill gaps, prioritizing emptiest cells
4. Validates each question (solver + arithmetic + dedup)
5. Writes approved questions to markdown files
6. Stops when coverage saturates (all cells ≥ target)
7. Prints a before/after coverage report

Grounded in:
- Automatic Item Generation (Gierl & Haladyna, 2013)
- Bloom's Revised Taxonomy (Anderson & Krathwohl, 2001)
- Evidence-Centered Design (Mislevy, Almond & Lukas, 2003)
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich import box

from engine.schemas import GenerationRequest
from engine.bloom import COMPETENCY_AREAS

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANNER = r"""[bold cyan]
  ╔═╗╔╦╗╔═╗╔═╗╔═╗╔╦╗╦    [bold white]StaffML Generation Engine[/bold white]
  ╚═╗ ║ ╠═╣╠╣ ╠╣ ║║║║    [dim]Automated gap-filling with saturation detection[/dim]
  ╚═╝ ╩ ╩ ╩╚  ╚  ╩ ╩╩═╝  [dim]v0.1.0[/dim]
[/bold cyan]"""

ALL_TRACKS = ["cloud", "edge", "mobile", "tinyml"]
ALL_LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6+"]

LEVEL_COLORS = {
    "L1": "bright_green", "L2": "blue", "L3": "bright_green",
    "L4": "blue", "L5": "yellow", "L6+": "red",
}

TRACK_ICONS = {"cloud": "☁️ ", "edge": "🤖", "mobile": "📱", "tinyml": "🔬"}

# Concept targets per track — what the engine should generate questions about.
# Extracted from TOPIC_MAP.md competency areas × track manifestations.
GENERATION_TARGETS: dict[str, list[dict]] = {
    "cloud": [
        {"concept": "GPU roofline model and compute vs memory bound analysis", "competency": "compute-analysis"},
        {"concept": "VRAM accounting: weights + optimizer + activations + KV-cache", "competency": "memory-systems"},
        {"concept": "FP16/BF16 mixed precision training and loss scaling", "competency": "numerical-representation"},
        {"concept": "Transformer scaling laws and attention complexity O(n²)", "competency": "model-architecture-cost"},
        {"concept": "TTFT/TPOT latency decomposition and continuous batching", "competency": "latency-throughput"},
        {"concept": "GPU TDP, PUE, and liquid cooling economics", "competency": "power-thermal"},
        {"concept": "KV-cache memory management and PagedAttention", "competency": "model-optimization"},
        {"concept": "Kubernetes autoscaling and canary deployment for ML", "competency": "deployment-serving"},
        {"concept": "Data drift detection and training-serving skew", "competency": "monitoring-reliability"},
        {"concept": "Prompt injection defense and model theft prevention", "competency": "security-privacy-fairness"},
        # Production ML competencies (per Chip Huyen review — "production-shaped hole")
        {"concept": "Feature store consistency: online vs offline path divergence and time semantics", "competency": "data-engineering"},
        {"concept": "Training data pipeline debugging: silent schema changes, late-arriving data, label quality", "competency": "data-engineering"},
        {"concept": "Model accuracy regression investigation: 'accuracy dropped 2% last Tuesday, walk me through triage'", "competency": "operational-judgment"},
        {"concept": "GPU cluster cost optimization: idle time analysis, spot vs reserved, right-sizing", "competency": "operational-judgment"},
    ],
    "edge": [
        {"concept": "TOPS/W efficiency vs raw TOPS under thermal envelope", "competency": "compute-analysis"},
        {"concept": "DRAM budget shared with OS and DMA transfers", "competency": "memory-systems"},
        {"concept": "INT8 quantization-aware training and calibration", "competency": "numerical-representation"},
        {"concept": "CNN vs Transformer for real-time edge vision", "competency": "model-architecture-cost"},
        {"concept": "Worst-case execution time and 30 FPS frame deadlines", "competency": "latency-throughput"},
        {"concept": "15-75W thermal envelope and DVFS P-states", "competency": "power-thermal"},
        {"concept": "TensorRT optimization and structured pruning", "competency": "model-optimization"},
        {"concept": "OTA firmware updates and A/B partitioning", "competency": "deployment-serving"},
        {"concept": "Degradation ladders and watchdog timers for edge", "competency": "monitoring-reliability"},
        {"concept": "Physical tampering and adversarial patch defense", "competency": "security-privacy-fairness"},
        {"concept": "Edge model accuracy regression: sensor drift, environmental change, calibration decay", "competency": "operational-judgment"},
        {"concept": "Fleet-wide firmware update strategy and version convergence", "competency": "data-engineering"},
    ],
    "mobile": [
        {"concept": "NPU delegation and heterogeneous CPU/GPU/NPU scheduling", "competency": "compute-analysis"},
        {"concept": "Shared RAM with OS, no dedicated VRAM, app eviction", "competency": "memory-systems"},
        {"concept": "Float16 on NPU with quantized CPU fallback", "competency": "numerical-representation"},
        {"concept": "MobileNet/EfficientNet design and on-device LLM feasibility", "competency": "model-architecture-cost"},
        {"concept": "UI jank budget 16ms at 60 FPS and ANR timeouts", "competency": "latency-throughput"},
        {"concept": "3-5W total SoC power and thermal throttling", "competency": "power-thermal"},
        {"concept": "Core ML and TFLite operator fusion optimization", "competency": "model-optimization"},
        {"concept": "App store delivery and on-demand model download", "competency": "deployment-serving"},
        {"concept": "Silent accuracy loss and federated analytics", "competency": "monitoring-reliability"},
        {"concept": "On-device differential privacy and federated learning", "competency": "security-privacy-fairness"},
        {"concept": "A/B testing on-device models: cohort management, metric collection, rollback", "competency": "operational-judgment"},
        {"concept": "Model download pipeline: CDN caching, delta updates, size budgets for app stores", "competency": "data-engineering"},
    ],
    "tinyml": [
        {"concept": "CMSIS-NN SIMD utilization on Cortex-M without FPU", "competency": "compute-analysis"},
        {"concept": "SRAM partitioning, flat tensor arena, flash vs SRAM", "competency": "memory-systems"},
        {"concept": "INT8 zero-point arithmetic and requantization between layers", "competency": "numerical-representation"},
        {"concept": "Depthwise separable convolutions and NAS for MCUs", "competency": "model-architecture-cost"},
        {"concept": "Microsecond inference and interrupt-driven pipelines", "competency": "latency-throughput"},
        {"concept": "Milliwatt power budgets and energy harvesting duty cycles", "competency": "power-thermal"},
        {"concept": "Mixed-precision quantization and operator scheduling for peak RAM", "competency": "model-optimization"},
        {"concept": "Flash programming and FOTA firmware updates", "competency": "deployment-serving"},
        {"concept": "Watchdog timers and hard real-time self-test routines", "competency": "monitoring-reliability"},
        {"concept": "Side-channel attacks and model extraction from flash", "competency": "security-privacy-fairness"},
        {"concept": "Field failure diagnosis: sensor noise, power brown-out, flash corruption, memory leak", "competency": "operational-judgment"},
        {"concept": "Over-the-air model update validation: checksum, rollback, A/B partition integrity", "competency": "data-engineering"},
    ],
}


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def load_corpus_coverage() -> dict[str, dict[str, int]]:
    """Load the existing corpus and build a track×level count matrix."""
    corpus_path = Path(__file__).parent / "corpus.json"
    if not corpus_path.exists():
        console.print("[bold red]corpus.json not found.[/] Run [cyan]python3 build_corpus.py[/] first.")
        return {}

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    # Build matrix: {track: {level: count}}
    matrix: dict[str, dict[str, int]] = {}
    for q in corpus:
        track = q.get("track", "unknown")
        level = q.get("level", "unknown")
        # Normalize
        if level == "L6" or level == "L6%2B":
            level = "L6+"
        if track not in matrix:
            matrix[track] = {}
        matrix[track][level] = matrix[track].get(level, 0) + 1

    return matrix


def identify_gaps(
    matrix: dict[str, dict[str, int]],
    target: int,
    tracks: list[str],
    levels: list[str],
) -> list[dict]:
    """Identify concept×level cells below the target threshold.

    Returns a list of generation jobs sorted by priority (emptiest first).
    """
    jobs = []

    for track in tracks:
        track_counts = matrix.get(track, {})
        targets = GENERATION_TARGETS.get(track, [])

        for level in levels:
            current = track_counts.get(level, 0)
            deficit = target - current

            if deficit <= 0:
                continue  # This cell is saturated

            # How many to generate for this cell?
            # Spread across concepts proportionally
            per_concept = max(1, deficit // max(len(targets), 1))

            for t in targets:
                jobs.append({
                    "track": track,
                    "level": level,
                    "concept": t["concept"],
                    "competency": t["competency"],
                    "current": current,
                    "target": target,
                    "deficit": deficit,
                    "count": min(per_concept, 2),  # Max 2 per concept per batch
                    "priority": deficit,  # Higher deficit = higher priority
                })

    # Sort: biggest gaps first
    jobs.sort(key=lambda j: (-j["priority"], j["track"], j["level"]))
    return jobs


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_coverage_matrix(
    matrix: dict[str, dict[str, int]],
    target: int,
    title: str = "Coverage Matrix",
) -> None:
    """Print a rich coverage matrix with color-coded saturation."""
    table = Table(title=title, box=box.ROUNDED, title_style="bold", show_lines=True)
    table.add_column("Track", style="bold", width=12)

    for level in ALL_LEVELS:
        color = LEVEL_COLORS.get(level, "white")
        table.add_column(level, width=8, justify="center", header_style=f"bold {color}")
    table.add_column("Total", width=8, justify="center", style="bold")

    for track in ALL_TRACKS:
        row = [f"{TRACK_ICONS.get(track, '')} {track}"]
        track_counts = matrix.get(track, {})
        total = 0

        for level in ALL_LEVELS:
            count = track_counts.get(level, 0)
            total += count

            if count == 0:
                cell = "[bold red]  0  [/]"
            elif count < target // 2:
                cell = f"[red]{count:3d}[/]"
            elif count < target:
                cell = f"[yellow]{count:3d}[/]"
            else:
                cell = f"[green]{count:3d} ✓[/]"

            row.append(cell)
        row.append(f"[bold]{total}[/]")
        table.add_row(*row)

    # Totals row
    totals_row = ["[bold]Total[/]"]
    grand_total = 0
    for level in ALL_LEVELS:
        level_total = sum(matrix.get(t, {}).get(level, 0) for t in ALL_TRACKS)
        grand_total += level_total
        totals_row.append(f"[bold]{level_total}[/]")
    totals_row.append(f"[bold cyan]{grand_total}[/]")
    table.add_row(*totals_row)

    console.print(table)

    # Saturation summary
    total_cells = len(ALL_TRACKS) * len(ALL_LEVELS)
    saturated = sum(
        1 for t in ALL_TRACKS for l in ALL_LEVELS
        if matrix.get(t, {}).get(l, 0) >= target
    )
    pct = saturated / total_cells * 100

    if pct >= 100:
        console.print(f"\n[bold green]Corpus is SATURATED[/] — all {total_cells} cells at ≥{target} questions")
    elif pct >= 75:
        console.print(f"\n[bold yellow]Coverage: {pct:.0f}%[/] — {saturated}/{total_cells} cells at ≥{target} questions")
    else:
        console.print(f"\n[bold red]Coverage: {pct:.0f}%[/] — {saturated}/{total_cells} cells at ≥{target} questions")


def print_generation_plan(jobs: list[dict], budget: int) -> None:
    """Print the planned generation jobs."""
    table = Table(title="Generation Plan", box=box.SIMPLE, title_style="bold")
    table.add_column("#", width=4)
    table.add_column("Track", width=8)
    table.add_column("Level", width=6)
    table.add_column("Current", width=8, justify="center")
    table.add_column("Target", width=8, justify="center")
    table.add_column("Generate", width=9, justify="center")
    table.add_column("Concept", ratio=2)

    total_planned = 0
    for i, job in enumerate(jobs):
        if total_planned >= budget:
            break

        count = min(job["count"], budget - total_planned)
        total_planned += count

        icon = TRACK_ICONS.get(job["track"], "")
        level_str = f"[{LEVEL_COLORS.get(job['level'], 'white')}]{job['level']}[/]"
        current_str = f"[red]{job['current']}[/]" if job["current"] < job["target"] else f"[green]{job['current']}[/]"

        table.add_row(
            str(i + 1),
            f"{icon}{job['track']}",
            level_str,
            current_str,
            str(job["target"]),
            f"[bold cyan]+{count}[/]",
            job["concept"][:60],
        )

    console.print(table)
    console.print(f"\n  [bold]Total planned:[/] {total_planned} questions (budget: {budget})")


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = Path(__file__).parent / "_generation_checkpoint.json"


def _save_checkpoint(stats: dict, completed_jobs: list[int]) -> None:
    """Save generation progress for crash recovery (per Dean review)."""
    checkpoint = {
        "stats": stats,
        "completed_jobs": completed_jobs,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def _load_checkpoint() -> tuple[dict, list[int]] | None:
    """Load checkpoint from previous interrupted run."""
    if not CHECKPOINT_FILE.exists():
        return None
    try:
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        return data["stats"], data["completed_jobs"]
    except (json.JSONDecodeError, KeyError):
        return None


def _clear_checkpoint() -> None:
    """Remove checkpoint after successful completion."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


def run_generation(
    jobs: list[dict],
    budget: int,
    model: str,
    dry_run: bool,
    skip_solver: bool,
) -> dict:
    """Run the generation loop across all planned jobs.

    Includes crash recovery: progress is checkpointed after each successful
    write so interrupted runs can resume (per Jeff Dean review).

    Returns stats dict.
    """
    import datetime as dt
    from engine.generate import generate_questions
    from engine.validate import validate_question
    from engine.render import render_markdown, append_to_markdown_file

    numbers_path = Path(__file__).parent / "NUMBERS.md"
    hardware_context = numbers_path.read_text(encoding="utf-8") if numbers_path.exists() else ""

    # Load ChromaDB for dedup against existing corpus
    chroma_collection = None
    try:
        from engine.embed import QuestionEmbedder
        embedder = QuestionEmbedder()
        if embedder.collection.count() > 0:
            chroma_collection = embedder.collection
            console.print(f"  [dim]Dedup enabled: {embedder.collection.count()} questions in vector store[/]")
    except Exception:
        console.print("  [dim]Dedup disabled (run 'python3 -m engine.cli embed' first)[/]")

    # Track all questions generated in this batch for within-batch dedup
    batch_texts: list[str] = []

    # Check for checkpoint from previous interrupted run
    checkpoint = _load_checkpoint()
    completed_jobs: list[int] = []

    if checkpoint is not None:
        prev_stats, completed_jobs = checkpoint
        console.print(Panel(
            f"[yellow]Resuming from checkpoint:[/] {len(completed_jobs)} jobs already completed.\n"
            f"Previous stats: {prev_stats['written']} written, {prev_stats['rejected']} rejected.",
            title="[bold yellow]Crash Recovery",
            border_style="yellow",
        ))
        stats = prev_stats
    else:
        stats = {
            "attempted": 0, "generated": 0, "validated": 0,
            "written": 0, "rejected": 0, "api_errors": 0,
        }

    total_to_generate = min(sum(j["count"] for j in jobs), budget)
    generated_so_far = 0

    console.print()
    console.print(Rule("[bold cyan]Generating[/]", style="cyan"))

    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold]{task.description}[/]"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task(
            "Filling gaps...", total=total_to_generate
        )

        for job_idx, job in enumerate(jobs):
            if generated_so_far >= budget:
                break

            # Skip jobs completed in a previous (interrupted) run
            if job_idx in completed_jobs:
                continue

            count = min(job["count"], budget - generated_so_far)
            stats["attempted"] += count

            track = job["track"]
            level = job["level"]
            concept = job["concept"]

            progress.update(
                main_task,
                description=f"{TRACK_ICONS.get(track, '')} {track}/{level}: {concept[:40]}..."
            )

            if dry_run:
                generated_so_far += count
                stats["generated"] += count
                stats["validated"] += count
                progress.advance(main_task, count)
                continue

            # Generate
            request = GenerationRequest(
                track=track,
                concept=concept,
                target_level=level,
                competency_area=job["competency"],
                count=count,
            )

            try:
                questions = generate_questions(request, model_name=model)
            except Exception as e:
                stats["api_errors"] += 1
                console.print(f"  [red]API error for {track}/{level}:[/] {str(e)[:80]}")
                # Checkpoint on error so we can resume
                _save_checkpoint(stats, completed_jobs)
                continue

            stats["generated"] += len(questions)

            # Stamp provenance on each question
            for q in questions:
                q.source = "generated"
                q.model_used = model
                q.generation_timestamp = dt.datetime.now().isoformat()

            # Validate + write
            for q in questions:
                # Step A: Check against existing corpus (ChromaDB)
                result = validate_question(
                    q,
                    hardware_context=hardware_context,
                    chroma_collection=chroma_collection,
                    skip_solver=skip_solver,
                )

                # Step B: Within-batch dedup using sentence embeddings
                # Same nomic-embed model as ChromaDB for consistency
                if result.passed and batch_texts:
                    q_text = f"{q.title} {q.scenario}"
                    try:
                        from engine.embed import embed_texts
                        import numpy as np

                        # Embed the candidate + all previous batch questions
                        all_texts = batch_texts + [q_text]
                        embeddings = embed_texts(all_texts)

                        # Cosine similarity between candidate and each previous
                        candidate = embeddings[-1:]
                        previous = embeddings[:-1]
                        # Normalize for cosine similarity
                        candidate_norm = candidate / np.linalg.norm(candidate, axis=1, keepdims=True)
                        previous_norm = previous / np.linalg.norm(previous, axis=1, keepdims=True)
                        sims = (candidate_norm @ previous_norm.T).flatten()

                        max_sim = float(sims.max()) if len(sims) > 0 else 0
                        if max_sim > 0.80:  # Embedding similarity threshold
                            result.passed = False
                            result.is_duplicate = True
                            result.duplicate_similarity = max_sim
                            result.issues.append(
                                f"Within-batch duplicate: {max_sim:.0%} similar "
                                f"to another generated question (embedding)"
                            )
                    except Exception:
                        pass  # Skip batch dedup on error

                if result.passed:
                    stats["validated"] += 1
                    target_file = append_to_markdown_file(q)
                    stats["written"] += 1
                    batch_texts.append(f"{q.title} {q.scenario}")
                    console.print(
                        f"  [green]✓[/] {TRACK_ICONS.get(track, '')} "
                        f"[{LEVEL_COLORS.get(level, 'white')}]{level}[/] "
                        f"[bold]{q.title}[/] → [dim]{target_file.name}[/]"
                    )
                else:
                    stats["rejected"] += 1
                    issues = "; ".join(result.issues[:2])
                    console.print(
                        f"  [red]✗[/] {TRACK_ICONS.get(track, '')} "
                        f"[{LEVEL_COLORS.get(level, 'white')}]{level}[/] "
                        f"[dim]{q.title}: {issues[:60]}[/]"
                    )

                generated_so_far += 1
                progress.advance(main_task)

            # Mark job complete and checkpoint
            completed_jobs.append(job_idx)
            _save_checkpoint(stats, completed_jobs)

    # Clean up checkpoint on successful completion
    _clear_checkpoint()
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="StaffML — One-command question generation with saturation detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 generate.py                        # Auto-fill all gaps
  python3 generate.py --budget 20            # Generate up to 20 questions
  python3 generate.py --target 8             # 8 questions per cell before saturating
  python3 generate.py --track cloud --level L1,L2   # Focus on cloud L1/L2
  python3 generate.py --dry-run              # Show plan without generating
  python3 generate.py --model gemini-3.1-pro-preview      # Use Flash for speed
        """,
    )
    parser.add_argument("--budget", type=int, default=30, help="Max questions to generate this run (default: 30)")
    parser.add_argument("--target", type=int, default=5, help="Target questions per track×level cell (default: 5)")
    parser.add_argument("--track", type=str, default=None, help="Comma-separated tracks to focus on")
    parser.add_argument("--level", type=str, default=None, help="Comma-separated levels to focus on (e.g. L1,L2)")
    parser.add_argument("--model", type=str, default="gemini-3.1-pro-preview", help="Gemini model to use")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without generating")
    parser.add_argument("--skip-solver", action="store_true", help="Skip solver validation (faster)")

    args = parser.parse_args()

    # Parse track/level filters
    tracks = args.track.split(",") if args.track else ALL_TRACKS
    levels = args.level.split(",") if args.level else ALL_LEVELS

    console.print(BANNER)

    # ── Phase 1: Survey ────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Phase 1: Survey[/]", style="cyan"))
    console.print()

    t0 = time.time()
    matrix = load_corpus_coverage()
    if not matrix:
        return

    print_coverage_matrix(matrix, args.target, title="Current Coverage (BEFORE)")

    # ── Phase 2: Plan ──────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Phase 2: Plan[/]", style="cyan"))
    console.print()

    jobs = identify_gaps(matrix, args.target, tracks, levels)

    if not jobs:
        console.print(Panel(
            "[bold green]All cells at or above target![/]\n"
            f"Every track×level cell has ≥{args.target} questions.\n"
            "The corpus is saturated — diminishing returns from here.",
            title="[bold green]Corpus Saturated",
            border_style="green",
        ))
        return

    print_generation_plan(jobs, args.budget)

    if args.dry_run:
        console.print("\n[dim]Dry run — no questions generated. Remove --dry-run to execute.[/]")
        return

    # Confirm
    console.print()
    total_planned = min(sum(j["count"] for j in jobs), args.budget)
    console.print(
        f"[bold]Will generate up to {total_planned} questions using {args.model}.[/]"
    )
    console.print("[dim]Press Ctrl+C to cancel.[/]")
    console.print()

    # ── Phase 3: Generate ──────────────────────────────────────────────────
    gen_t0 = time.time()
    stats = run_generation(
        jobs, args.budget, args.model,
        dry_run=False, skip_solver=args.skip_solver,
    )
    gen_time = time.time() - gen_t0

    # ── Phase 4: Report ────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Phase 4: Report[/]", style="cyan"))
    console.print()

    # Rebuild coverage
    if stats["written"] > 0:
        console.print("[dim]Rebuilding corpus.json...[/]")
        import subprocess
        subprocess.run(
            ["python3", str(Path(__file__).parent / "build_corpus.py")],
            capture_output=True,
        )
        new_matrix = load_corpus_coverage()
        print_coverage_matrix(new_matrix, args.target, title="Updated Coverage (AFTER)")

    # Final stats panel
    stats_table = Table(box=box.DOUBLE_EDGE, border_style="cyan", title="Run Summary", title_style="bold cyan")
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value", justify="right")

    stats_table.add_row("Attempted", str(stats["attempted"]))
    stats_table.add_row("Generated", f"[bold]{stats['generated']}[/]")
    stats_table.add_row("Validated", f"[bold green]{stats['validated']}[/]")
    stats_table.add_row("Written", f"[bold green]{stats['written']}[/]")
    if stats["rejected"]:
        stats_table.add_row("Rejected", f"[bold red]{stats['rejected']}[/]")
    if stats["api_errors"]:
        stats_table.add_row("API Errors", f"[bold red]{stats['api_errors']}[/]")
    stats_table.add_row("Time", f"[bold]{gen_time:.0f}s[/]")
    stats_table.add_row("Model", f"[cyan]{args.model}[/]")

    pass_rate = stats["validated"] / max(stats["generated"], 1)
    rate_color = "green" if pass_rate >= 0.8 else "yellow" if pass_rate >= 0.5 else "red"
    stats_table.add_row("Pass Rate", f"[bold {rate_color}]{pass_rate:.0%}[/]")

    console.print(stats_table)

    total_time = time.time() - t0
    console.print(f"\n[dim]Total wall time: {total_time:.0f}s[/]")


if __name__ == "__main__":
    main()
