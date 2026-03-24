#!/usr/bin/env python3
"""
StaffML Question Generation Engine — CLI

Usage:
    python3 -m engine.cli embed
    python3 -m engine.cli gaps
    python3 -m engine.cli generate --track cloud --concept "KV-cache memory" --level L3
    python3 -m engine.cli pipeline --track cloud --concept "KV-cache memory" --level L4 --count 3
    python3 -m engine.cli report
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.columns import Columns
from rich.text import Text
from rich.rule import Rule
from rich import box

from .schemas import GenerationRequest

console = Console()

# ---------------------------------------------------------------------------
# Branding
# ---------------------------------------------------------------------------

BANNER = r"""[bold cyan]
  ╔═╗╔╦╗╔═╗╔═╗╔═╗╔╦╗╦    [bold white]Question Generation Engine[/bold white]
  ╚═╗ ║ ╠═╣╠╣ ╠╣ ║║║║    [dim]Bloom's × mlsysim × Gemini Pro[/dim]
  ╚═╝ ╩ ╩ ╩╚  ╚  ╩ ╩╩═╝  [dim]v0.1.0[/dim]
[/bold cyan]"""

LEVEL_COLORS = {
    "L1": "bright_green", "L2": "blue", "L3": "bright_green",
    "L4": "blue", "L5": "yellow", "L6+": "red",
}

TRACK_ICONS = {
    "cloud": "☁️ ", "edge": "🤖", "mobile": "📱",
    "tinyml": "🔬", "foundations": "📐",
}


def _level_badge(level: str) -> str:
    color = LEVEL_COLORS.get(level, "white")
    return f"[bold {color}]{level}[/bold {color}]"


def _track_badge(track: str) -> str:
    icon = TRACK_ICONS.get(track, "❓")
    return f"{icon} {track}"


# ---------------------------------------------------------------------------
# cmd: embed
# ---------------------------------------------------------------------------

def cmd_embed(args: argparse.Namespace) -> None:
    """Embed the existing corpus into ChromaDB."""
    from .embed import QuestionEmbedder

    console.print(BANNER)

    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding corpus into ChromaDB...", total=None)
        embedder = QuestionEmbedder()
        count = embedder.embed_corpus()
        progress.update(task, description=f"[bold green]Embedded {count} questions")

    report = embedder.coverage_report()
    _print_coverage(report)


# ---------------------------------------------------------------------------
# cmd: gaps
# ---------------------------------------------------------------------------

def cmd_gaps(args: argparse.Namespace) -> None:
    """Find semantic gaps in question coverage."""
    from .embed import QuestionEmbedder, extract_concepts_from_topic_map

    console.print(BANNER)

    embedder = QuestionEmbedder()
    if embedder.collection.count() == 0:
        console.print("[bold red]No embeddings found.[/] Run [cyan]python3 -m engine.cli embed[/] first.")
        return

    concepts = extract_concepts_from_topic_map()
    if not concepts:
        console.print("[bold red]Could not extract concepts from TOPIC_MAP.md[/]")
        return

    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Analyzing {len(concepts)} concepts...", total=None)
        gaps = embedder.find_gaps(concepts, threshold=args.threshold)
        progress.update(task, description="Analysis complete")

    gap_count = sum(1 for g in gaps if g["is_gap"])
    covered_count = len(gaps) - gap_count

    # Summary panel
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column(justify="right")
    summary.add_column()
    summary.add_row("[bold red]Gaps", f"[bold]{gap_count}[/] concepts need questions")
    summary.add_row("[bold green]Covered", f"[bold]{covered_count}[/] concepts have adequate coverage")
    summary.add_row("[dim]Threshold", f"cosine similarity < {args.threshold}")
    console.print(Panel(summary, title="[bold]Gap Analysis", border_style="cyan"))

    # Gap table
    table = Table(
        title="Concept Coverage",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold",
    )
    table.add_column("Status", width=6, justify="center")
    table.add_column("Coverage", width=9, justify="center")
    table.add_column("Concept", ratio=3)
    table.add_column("Nearest Question", ratio=2)

    for g in gaps:
        if g["is_gap"]:
            status = "[bold red]GAP[/]"
            cov_style = "red"
        else:
            status = "[bold green]OK[/]"
            cov_style = "green"

        cov = f"[{cov_style}]{g['coverage']:.3f}[/]"
        concept = g["concept"][:70]

        nearest = ""
        if g["nearest_questions"]:
            n = g["nearest_questions"][0]
            nearest = f"[dim]{n['title'][:40]}[/] ({n['track']}/{n['level']})"

        table.add_row(status, cov, concept, nearest)

    console.print(table)


# ---------------------------------------------------------------------------
# cmd: generate
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    """Generate questions using Gemini Pro."""
    from .generate import generate_questions
    from .render import render_markdown

    console.print(BANNER)
    _print_job_config(args)

    request = GenerationRequest(
        track=args.track,
        concept=args.concept,
        target_level=args.level,
        competency_area=args.competency or "compute-analysis",
        count=args.count,
    )

    t0 = time.time()
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Calling Gemini Pro...", total=None)
        questions = generate_questions(request, model_name=args.model, temperature=args.temperature)
        elapsed = time.time() - t0
        progress.update(task, description=f"[bold green]Generated {len(questions)} question(s) in {elapsed:.1f}s")

    for i, q in enumerate(questions):
        _print_question_card(q, i + 1)


# ---------------------------------------------------------------------------
# cmd: pipeline
# ---------------------------------------------------------------------------

def cmd_pipeline(args: argparse.Namespace) -> None:
    """Full pipeline: generate → validate → render."""
    from .generate import generate_questions
    from .validate import validate_question
    from .render import render_markdown, render_corpus_entry, append_to_markdown_file

    console.print(BANNER)
    _print_job_config(args)

    # Load hardware context
    numbers_path = Path(__file__).parent.parent / "NUMBERS.md"
    hardware_context = numbers_path.read_text(encoding="utf-8") if numbers_path.exists() else ""

    # Optional: ChromaDB for dedup
    chroma_collection = None
    if not args.no_dedup:
        try:
            from .embed import QuestionEmbedder
            embedder = QuestionEmbedder()
            if embedder.collection.count() > 0:
                chroma_collection = embedder.collection
                console.print(f"  [dim]Dedup:[/] {embedder.collection.count()} questions in vector store")
        except ImportError:
            pass

    request = GenerationRequest(
        track=args.track,
        concept=args.concept,
        target_level=args.level,
        competency_area=args.competency or "compute-analysis",
        count=args.count,
    )

    pipeline_t0 = time.time()
    stats = {"generated": 0, "validated": 0, "written": 0, "rejected": 0}

    # ── Stage 1: Generate ─────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Stage 1: Generate[/]", style="cyan"))

    t0 = time.time()
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Calling {args.model} for {args.count} {args.level} question(s)...", total=None
        )
        questions = generate_questions(request, model_name=args.model, temperature=args.temperature)
        gen_time = time.time() - t0
        stats["generated"] = len(questions)
        progress.update(
            task,
            description=f"[bold green]{len(questions)} question(s) generated ({gen_time:.1f}s)"
        )

    if not questions:
        console.print("[bold red]No questions generated. Check model output.[/]")
        return

    # Show generated titles
    for i, q in enumerate(questions):
        console.print(f"  {_level_badge(q.level)} [bold]{q.title}[/] · [dim]{q.topic}[/]")

    # ── Stage 2: Validate ─────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Stage 2: Validate[/]", style="cyan"))

    approved = []
    val_results = []

    for i, q in enumerate(questions):
        with Progress(
            SpinnerColumn("dots"),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            solver_label = "solver + " if not args.skip_solver else ""
            task = progress.add_task(
                f"Q{i+1}: {solver_label}arithmetic + dedup...", total=None
            )

            t0 = time.time()
            result = validate_question(
                q,
                hardware_context=hardware_context,
                chroma_collection=chroma_collection,
                model_name=args.model,
                skip_solver=args.skip_solver,
            )
            val_time = time.time() - t0

            if result.passed:
                progress.update(task, description=f"[bold green]Q{i+1}: PASSED ({val_time:.1f}s)")
                approved.append(q)
                stats["validated"] += 1
            else:
                progress.update(task, description=f"[bold red]Q{i+1}: FAILED ({val_time:.1f}s)")
                stats["rejected"] += 1

            val_results.append((q, result))

    # Validation summary table
    val_table = Table(box=box.SIMPLE, padding=(0, 1))
    val_table.add_column("#", width=3)
    val_table.add_column("Title", ratio=2)
    val_table.add_column("Solver", width=8, justify="center")
    val_table.add_column("Arithmetic", width=10, justify="center")
    val_table.add_column("Dedup", width=8, justify="center")
    val_table.add_column("Result", width=8, justify="center")

    for i, (q, r) in enumerate(val_results):
        solver_ok = "[green]PASS[/]" if r.solver_agrees else "[red]FAIL[/]"
        arith_ok = "[green]PASS[/]" if r.arithmetic_correct else "[red]FAIL[/]"
        dup_ok = "[green]PASS[/]" if not r.is_duplicate else f"[red]{r.duplicate_similarity:.0%}[/]"
        overall = "[bold green]PASS[/]" if r.passed else "[bold red]FAIL[/]"

        if args.skip_solver:
            solver_ok = "[dim]skip[/]"

        val_table.add_row(str(i + 1), q.title[:40], solver_ok, arith_ok, dup_ok, overall)

        if r.issues:
            for issue in r.issues:
                val_table.add_row("", f"  [dim yellow]⚠ {issue[:60]}[/]", "", "", "", "")

    console.print(val_table)

    if not approved:
        console.print("[bold red]No questions passed validation.[/]")
        return

    # ── Stage 3: Render ───────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Stage 3: Render[/]", style="cyan"))

    corpus_entries = []
    for i, q in enumerate(approved):
        _print_question_card(q, i + 1)
        corpus_entries.append(render_corpus_entry(q))

    # ── Stage 4: Write ────────────────────────────────────────────────────
    if args.write:
        console.print()
        console.print(Rule("[bold cyan]Stage 4: Write[/]", style="cyan"))
        for q in approved:
            target = append_to_markdown_file(q)
            stats["written"] += 1
            console.print(f"  [green]→[/] Appended to [cyan]{target.relative_to(Path(__file__).parent.parent)}[/]")
        console.print()
        console.print("[dim]Run[/] [cyan]python3 build_corpus.py[/] [dim]to rebuild corpus.json[/]")

    # ── Final Stats ───────────────────────────────────────────────────────
    pipeline_time = time.time() - pipeline_t0
    _print_pipeline_stats(stats, pipeline_time, args)


# ---------------------------------------------------------------------------
# cmd: report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> None:
    """Show coverage report — terminal or interactive HTML."""
    console.print(BANNER)

    if getattr(args, "html", False):
        from .report import generate_html_report
        with Progress(
            SpinnerColumn("dots"),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating interactive HTML report...", total=None)
            path = generate_html_report(open_browser=not getattr(args, "no_open", False))
            progress.update(task, description=f"[bold green]Report saved")

        console.print(Panel(
            f"[bold]Report:[/] [cyan]{path}[/]\n"
            f"Contains: UMAP embedding map, coverage heatmap, topic treemap, quality stats",
            title="[bold green]HTML Report Generated",
            border_style="green",
        ))
        return

    # Terminal report — use ChromaDB if available, else corpus.json
    try:
        from .embed import QuestionEmbedder
        embedder = QuestionEmbedder()
        if embedder.collection.count() > 0:
            report = embedder.coverage_report()
            _print_coverage(report)
            return
    except ImportError:
        pass

    # Fallback: load from corpus.json directly
    import json
    corpus_path = Path(__file__).parent.parent / "corpus.json"
    if not corpus_path.exists():
        console.print("[bold red]No data found.[/] Run [cyan]python3 build_corpus.py[/] first.")
        return

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    by_track: dict[str, int] = {}
    by_level: dict[str, int] = {}
    by_track_level: dict[str, dict[str, int]] = {}
    for q in corpus:
        t = q.get("track", "unknown")
        l = q.get("level", "?")
        if l in ("L6", "L6%2B"):
            l = "L6+"
        by_track[t] = by_track.get(t, 0) + 1
        by_level[l] = by_level.get(l, 0) + 1
        if t not in by_track_level:
            by_track_level[t] = {}
        by_track_level[t][l] = by_track_level[t].get(l, 0) + 1

    _print_coverage({
        "total": len(corpus),
        "by_track": by_track,
        "by_level": by_level,
        "by_track_level": by_track_level,
    })


# ---------------------------------------------------------------------------
# Rich output helpers
# ---------------------------------------------------------------------------

def _print_job_config(args: argparse.Namespace) -> None:
    """Print the job configuration as a compact panel."""
    config = Table(show_header=False, box=None, padding=(0, 2))
    config.add_column(style="bold", width=12)
    config.add_column()

    config.add_row("Track", _track_badge(args.track))
    config.add_row("Level", _level_badge(args.level))
    config.add_row("Concept", f"[italic]{args.concept}[/]")
    config.add_row("Model", f"[cyan]{args.model}[/]")
    config.add_row("Count", str(args.count))

    if hasattr(args, "write") and args.write:
        config.add_row("Mode", "[bold green]WRITE[/] (will modify files)")
    elif hasattr(args, "write"):
        config.add_row("Mode", "[dim]DRY RUN[/] (add --write to save)")

    if hasattr(args, "skip_solver") and args.skip_solver:
        config.add_row("Solver", "[dim]skipped[/]")

    console.print(Panel(config, title="[bold]Job Configuration", border_style="blue"))


def _print_question_card(q, index: int) -> None:
    """Print a rich question card."""
    from .render import render_markdown

    # Header
    header = Text()
    header.append(f"Q{index} ", style="bold dim")
    header.append(f"{q.level} ", style=f"bold {LEVEL_COLORS.get(q.level, 'white')}")
    header.append(q.title, style="bold")
    header.append(f"  {q.topic}", style="dim")

    # Content sections
    content_parts = []

    # Scenario (truncated for display)
    scenario_display = q.scenario[:300]
    if len(q.scenario) > 300:
        scenario_display += "..."
    content_parts.append(f"[bold]Scenario:[/]\n{scenario_display}")

    content_parts.append(f"\n[bold red]Common Mistake:[/]\n{q.common_mistake[:200]}")

    content_parts.append(f"\n[bold green]Solution:[/]\n{q.realistic_solution[:200]}")

    if q.napkin_math:
        nm_display = q.napkin_math[:300]
        if len(q.napkin_math) > 300:
            nm_display += "..."
        content_parts.append(f"\n[bold yellow]Napkin Math:[/]\n{nm_display}")

    if q.options:
        opts = []
        for opt in q.options:
            marker = "[bold green]✓[/]" if opt.is_correct else "[dim]✗[/]"
            opts.append(f"  {marker} {opt.text[:80]}")
        content_parts.append("\n[bold]Options:[/]\n" + "\n".join(opts))

    if q.deep_dive_title:
        content_parts.append(f"\n[dim]📖 {q.deep_dive_title}[/]")

    body = "\n".join(content_parts)
    console.print(Panel(body, title=header, border_style="green", padding=(1, 2)))


def _print_pipeline_stats(stats: dict, total_time: float, args) -> None:
    """Print final pipeline statistics."""
    console.print()

    table = Table(
        title="Pipeline Summary",
        box=box.DOUBLE_EDGE,
        title_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Generated", f"[bold]{stats['generated']}[/] questions")
    table.add_row("Validated", f"[bold green]{stats['validated']}[/] passed")
    if stats["rejected"] > 0:
        table.add_row("Rejected", f"[bold red]{stats['rejected']}[/] failed")
    table.add_row("Written", f"[bold]{stats['written']}[/] to files" if stats["written"] > 0 else "[dim]0 (dry run)[/]")
    table.add_row("Total Time", f"[bold]{total_time:.1f}s[/]")
    table.add_row("Model", f"[cyan]{args.model}[/]")

    pass_rate = stats["validated"] / max(stats["generated"], 1)
    if pass_rate >= 0.8:
        rate_style = "bold green"
    elif pass_rate >= 0.5:
        rate_style = "bold yellow"
    else:
        rate_style = "bold red"
    table.add_row("Pass Rate", f"[{rate_style}]{pass_rate:.0%}[/]")

    console.print(table)

    if stats["written"] == 0 and stats["validated"] > 0 and not getattr(args, "write", False):
        console.print(
            "\n[dim]Tip: Add [cyan]--write[/cyan] to append approved questions to the markdown files.[/]"
        )


def _print_coverage(report: dict) -> None:
    """Rich coverage report with tables and bar charts."""
    console.print()
    console.print(
        Panel(
            f"[bold]{report['total']}[/] questions across "
            f"[bold]{len(report.get('by_track', {}))}[/] tracks and "
            f"[bold]{len(report.get('by_level', {}))}[/] levels",
            title="[bold]Corpus Coverage Report",
            border_style="cyan",
        )
    )

    # Track distribution
    track_table = Table(title="By Track", box=box.SIMPLE, title_style="bold")
    track_table.add_column("Track", width=14)
    track_table.add_column("Count", width=6, justify="right")
    track_table.add_column("Distribution", ratio=1)

    max_count = max(report.get("by_track", {}).values(), default=1)
    for track, count in sorted(report.get("by_track", {}).items()):
        bar_len = int(40 * count / max_count)
        bar = f"[cyan]{'█' * bar_len}[/] {count}"
        track_table.add_row(_track_badge(track), str(count), bar)

    # Level distribution
    level_table = Table(title="By Level", box=box.SIMPLE, title_style="bold")
    level_table.add_column("Level", width=8)
    level_table.add_column("Count", width=6, justify="right")
    level_table.add_column("Distribution", ratio=1)

    max_level = max(report.get("by_level", {}).values(), default=1)
    for level, count in sorted(report.get("by_level", {}).items()):
        color = LEVEL_COLORS.get(level, "white")
        bar_len = int(40 * count / max_level)
        bar = f"[{color}]{'█' * bar_len}[/] {count}"
        level_table.add_row(_level_badge(level), str(count), bar)

    console.print(Columns([track_table, level_table], padding=4))

    # Cross-tabulation matrix
    if "by_track_level" in report and report["by_track_level"]:
        matrix = Table(
            title="Track × Level Matrix",
            box=box.ROUNDED,
            title_style="bold",
            show_lines=True,
        )
        levels = sorted(set(
            l for tl in report["by_track_level"].values() for l in tl
        ))

        matrix.add_column("Track", style="bold", width=14)
        for level in levels:
            color = LEVEL_COLORS.get(level, "white")
            matrix.add_column(level, width=7, justify="center", style=color)
        matrix.add_column("Total", width=7, justify="center", style="bold")

        for track in sorted(report["by_track_level"].keys()):
            row = [_track_badge(track)]
            total = 0
            for level in levels:
                count = report["by_track_level"][track].get(level, 0)
                total += count
                if count == 0:
                    row.append("[dim]—[/]")
                elif count < 10:
                    row.append(f"[bold red]{count}[/]")
                else:
                    row.append(str(count))
            row.append(f"[bold]{total}[/]")
            matrix.add_row(*row)

        console.print(matrix)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="StaffML Question Generation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- embed ---
    subparsers.add_parser("embed", help="Embed corpus into ChromaDB")

    # --- gaps ---
    sub_gaps = subparsers.add_parser("gaps", help="Find gaps in question coverage")
    sub_gaps.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold for gaps")

    # --- generate ---
    sub_gen = subparsers.add_parser("generate", help="Generate questions (no validation)")
    sub_gen.add_argument("--track", required=True, choices=["cloud", "edge", "mobile", "tinyml", "foundations"])
    sub_gen.add_argument("--concept", required=True, help="The concept to generate questions about")
    sub_gen.add_argument("--level", required=True, choices=["L1", "L2", "L3", "L4", "L5", "L6+"])
    sub_gen.add_argument("--competency", default=None, help="Competency area (from TOPIC_MAP)")
    sub_gen.add_argument("--count", type=int, default=1, help="Number of questions to generate")
    sub_gen.add_argument("--model", default="gemini-2.5-pro", help="Gemini model (e.g. gemini-2.5-pro, gemini-2.5-flash)")
    sub_gen.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")

    # --- pipeline ---
    sub_pipe = subparsers.add_parser("pipeline", help="Full pipeline: generate → validate → render")
    sub_pipe.add_argument("--track", required=True, choices=["cloud", "edge", "mobile", "tinyml", "foundations"])
    sub_pipe.add_argument("--concept", required=True, help="The concept to generate questions about")
    sub_pipe.add_argument("--level", required=True, choices=["L1", "L2", "L3", "L4", "L5", "L6+"])
    sub_pipe.add_argument("--competency", default=None, help="Competency area")
    sub_pipe.add_argument("--count", type=int, default=1, help="Number of questions to generate")
    sub_pipe.add_argument("--model", default="gemini-2.5-pro", help="Gemini model (e.g. gemini-2.5-pro, gemini-2.5-flash)")
    sub_pipe.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    sub_pipe.add_argument("--write", action="store_true", help="Actually write to markdown files")
    sub_pipe.add_argument("--skip-solver", action="store_true", help="Skip the LLM solver validation")
    sub_pipe.add_argument("--no-dedup", action="store_true", help="Skip ChromaDB dedup check")

    # --- report ---
    sub_report = subparsers.add_parser("report", help="Show coverage report")
    sub_report.add_argument("--html", action="store_true", help="Generate interactive HTML report with UMAP")
    sub_report.add_argument("--no-open", action="store_true", help="Don't auto-open the HTML report in browser")

    args = parser.parse_args()

    commands = {
        "embed": cmd_embed,
        "gaps": cmd_gaps,
        "generate": cmd_generate,
        "pipeline": cmd_pipeline,
        "report": cmd_report,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
