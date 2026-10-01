"""
Tiny🔥Torch Benchmark Commands

Run the baseline environment speed check and the capstone benchmark.
Results are saved locally under .tito/benchmarks/. There is no community
upload yet, so nothing here claims to submit anything.
"""

import json
import math
import os
import platform
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .base import BaseCommand
from ..core.exceptions import TinyTorchCLIError

def _get_rng():
    """Lazily import numpy and get standard random generator."""
    try:
        import numpy as np
        return np, np.random.default_rng(7)
    except ImportError:
        raise TinyTorchCLIError(
            "NumPy is required to run benchmarks.\n"
            "Please ensure your virtual environment is activated: source .venv/bin/activate"
        )


from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console



def argparse_suppress() -> str:
    """Hide a no-op compatibility flag from --help."""
    from argparse import SUPPRESS
    return SUPPRESS


def geometric_mean(values: List[float]) -> float:
    """Geometric mean of positive ratios: exp(mean(log(v)))."""
    if not values or any(v <= 0 for v in values):
        raise ValueError("geometric mean needs positive values")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def summarize_times(times_ms: Dict[str, float]) -> Dict[str, float]:
    """
    Summarize the environment check from measured times only.

    Reports each time plus their geometric mean, which weights every check
    equally no matter how many milliseconds it takes.
    """
    # 2026-09-28: this once claimed a geometric mean while summing times, and
    # scored against hand-typed "reference laptop" times that had no source.
    # Those references were 219-3,392x slower than an Apple-silicon laptop, and
    # min(100, ...) hid it by giving everyone 100/100. Raw times are the honest
    # output until a reference machine is actually measured.
    summary = {f"{name}_ms": t for name, t in times_ms.items()}
    summary["geometric_mean_ms"] = geometric_mean(list(times_ms.values()))
    return summary


class BenchmarkCommand(BaseCommand):
    """Benchmark commands - baseline and capstone performance evaluation."""

    @property
    def name(self) -> str:
        return "benchmark"

    @property
    def description(self) -> str:
        return "Run benchmarks - baseline (NumPy environment speed check) and capstone (Module 20)"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add benchmark subcommands."""
        subparsers = parser.add_subparsers(
            dest='benchmark_command',
            help='Benchmark operations',
            metavar='COMMAND'
        )

        # Baseline benchmark
        baseline_parser = subparsers.add_parser(
            'baseline',
            help='Run baseline environment speed check (times NumPy, not TinyTorch)'
        )
        # Kept so existing scripts keep working; there is no submission step.
        baseline_parser.add_argument(
            '--skip-submit',
            action='store_true',
            help=argparse_suppress()
        )

        # Capstone benchmark
        capstone_parser = subparsers.add_parser(
            'capstone',
            help='Run capstone benchmark (full Module 20 performance evaluation)'
        )
        capstone_parser.add_argument(
            '--track',
            choices=['speed', 'compression', 'accuracy', 'efficiency', 'all'],
            default='all',
            help='Which track to benchmark (default: all)'
        )
        capstone_parser.add_argument(
            '--skip-submit',
            action='store_true',
            help=argparse_suppress()
        )

    def run(self, args: Namespace) -> int:
        """Execute benchmark command."""
        if not args.benchmark_command:
            self.console.print("[yellow]Please specify a benchmark command: baseline or capstone[/yellow]")
            return 1

        if args.benchmark_command == 'baseline':
            return self._run_baseline(args)
        elif args.benchmark_command == 'capstone':
            return self._run_capstone(args)
        else:
            self.console.print(f"[red]Unknown benchmark command: {args.benchmark_command}[/red]")
            return 1

    def _run_baseline(self, args: Namespace) -> int:
        """Run baseline benchmark - lightweight setup validation."""
        console = self.console

        console.print(Panel(
            "[bold cyan]🎯 Baseline Environment Speed Check[/bold cyan]\n\n"
            "Times plain NumPy operations (elementwise ops, matmul, a two-layer\n"
            "forward pass) to check how fast this Python/NumPy install is.\n"
            "[dim]It does not run your TinyTorch code.[/dim]",
            title="Baseline Benchmark",
            border_style="cyan"
        ))

        # Run baseline benchmarks
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running baseline benchmarks...", total=None)

            # Benchmark 1: Tensor operations
            progress.update(task, description="[cyan]Testing tensor operations...")
            tensor_time = self._benchmark_tensor_ops()

            # Benchmark 2: Matrix multiply
            progress.update(task, description="[cyan]Testing matrix multiplication...")
            matmul_time = self._benchmark_matmul()

            # Benchmark 3: Simple forward pass
            progress.update(task, description="[cyan]Testing forward pass...")
            forward_time = self._benchmark_forward_pass()

            progress.update(task, completed=True)

        raw_metrics = summarize_times({
            "tensor_ops": tensor_time,
            "matmul": matmul_time,
            "forward_pass": forward_time,
        })

        results_table = Table(title="Baseline Environment Check (NumPy)", show_header=True, header_style="bold cyan")
        results_table.add_column("Check", style="cyan")
        results_table.add_column("Measured time", justify="right", style="green")
        results_table.add_row("Elementwise ops (100×100 add, mul, sum)", f"{tensor_time:.4f} ms")
        results_table.add_row("Matrix multiply (100×100)", f"{matmul_time:.4f} ms")
        results_table.add_row("Two-layer forward pass (784→128→10)", f"{forward_time:.4f} ms")
        results_table.add_row("", "")
        results_table.add_row("[bold]Geometric mean[/bold]", f"[bold]{raw_metrics['geometric_mean_ms']:.4f} ms[/bold]")

        console.print("\n")
        console.print(results_table)
        console.print("[dim]Lower is faster. This times NumPy on this machine, not your TinyTorch modules.[/dim]")

        results = {
            "benchmark_type": "baseline",
            "measures": "NumPy environment speed (does not run TinyTorch code)",
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "raw_metrics": raw_metrics,
            "metrics": raw_metrics,
        }

        # Save results
        benchmark_dir = Path(".tito") / "benchmarks"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = benchmark_dir / f"baseline_{timestamp_str}.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        console.print(f"\n[green]✅ Results saved to: {results_file}[/green]")

        # Success message
        console.print(Panel(
            f"[bold green]Baseline environment check complete[/bold green]\n\n"
            f"📊 Geometric-mean time: [bold]{raw_metrics['geometric_mean_ms']:.4f} ms[/bold]\n"
            f"NumPy runs on this machine; this does not test your TinyTorch code.\n\n"
            f"💡 Run [cyan]tito benchmark capstone[/cyan] after Module 20 to benchmark your model",
            title="Done",
            border_style="green"
        ))
        self._report_local_only(results_file)

        return 0

    def _run_capstone(self, args: Namespace) -> int:
        """Run capstone benchmark - full Module 20 performance evaluation."""
        console = self.console

        console.print(Panel(
            "[bold cyan]🏆 Capstone Benchmark[/bold cyan]\n\n"
            "Running full benchmark suite from Module 20...",
            title="Capstone Benchmark",
            border_style="cyan"
        ))

        # Check if Module 20 is available
        try:
            from tinytorch.perf.benchmarking import Benchmark
        except ImportError:
            console.print(Panel(
                "[red]❌ Module 19 (Benchmarking) not available[/red]\n\n"
                "Please complete Module 19 first:\n"
                "  [cyan]tito module complete 19[/cyan]",
                title="Error",
                border_style="red"
            ))
            return 1

        # Check if Module 20 competition code is available. `tito module
        # complete 20` exports the capstone notebook to tinytorch/olympics.py
        # (see tito/commands/module/workflow.py's export path mapping), not
        # tinytorch/competition/submit.py -- that module has never existed,
        # so this check previously always failed and reported "Module 20 not
        # complete" even for students who genuinely finished it.
        try:
            from tinytorch.olympics import generate_submission
        except ImportError:
            # 2026-09-28: this used to sleep for a second and save a fixed
            # "basic_score": 75 as if something had been measured.
            console.print(Panel(
                "[red]❌ Module 20 (Capstone) is required[/red]\n\n"
                "The capstone benchmark measures your Module 20 model, which is not\n"
                "exported yet, so there is nothing to measure. No results were saved.\n\n"
                "Complete Module 20 first:\n"
                "  [cyan]tito module complete 20[/cyan]",
                title="Error",
                border_style="red"
            ))
            return 1

        # 2026-09-28: this path used to display and save fixed placeholder
        # metrics (latency 45.2 ms, 87.5% accuracy, overall 90/100) that were
        # the same for every student. Nothing measures the Module 20 model
        # yet, so say so and fail instead of printing numbers.
        console.print(Panel(
            "[yellow]Capstone measurement is not implemented yet.[/yellow]\n\n"
            "Module 20 is exported, but this command does not yet run Module 19's\n"
            "Benchmark against your model, so it has no numbers to report.\n"
            "No results were saved.\n\n"
            "Measure your model directly with the Benchmark class from Module 19\n"
            "(see the Module 20 notebook).",
            title="Not implemented",
            border_style="yellow"
        ))
        return 1

    def _benchmark_tensor_ops(self) -> float:
        """Benchmark basic tensor operations."""
        import time
        np, rng = _get_rng()

        # Create tensors
        a = rng.standard_normal((100, 100)).astype(np.float32)
        b = rng.standard_normal((100, 100)).astype(np.float32)

        # Warmup
        for _ in range(5):
            _ = a + b
            _ = a * b

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = a + b
            _ = a * b
            _ = np.sum(a)
        end = time.perf_counter()

        return (end - start) * 1000 / 100  # Convert to milliseconds per operation

    def _benchmark_matmul(self) -> float:
        """Benchmark matrix multiplication."""
        import time
        np, rng = _get_rng()

        a = rng.standard_normal((100, 100)).astype(np.float32)
        b = rng.standard_normal((100, 100)).astype(np.float32)

        # Warmup
        for _ in range(5):
            _ = np.dot(a, b)

        # Benchmark
        start = time.perf_counter()
        for _ in range(50):
            _ = np.dot(a, b)
        end = time.perf_counter()

        return (end - start) * 1000 / 50  # milliseconds per matmul

    def _benchmark_forward_pass(self) -> float:
        """Benchmark simple forward pass simulation."""
        import time
        np, rng = _get_rng()

        # Simulate a simple forward pass
        x = rng.standard_normal((1, 784)).astype(np.float32)
        w1 = rng.standard_normal((784, 128)).astype(np.float32)
        w2 = rng.standard_normal((128, 10)).astype(np.float32)

        # Warmup
        for _ in range(5):
            h = np.maximum(0, np.dot(x, w1))  # ReLU
            _ = np.dot(h, w2)

        # Benchmark
        start = time.perf_counter()
        for _ in range(20):
            h = np.maximum(0, np.dot(x, w1))
            _ = np.dot(h, w2)
        end = time.perf_counter()

        return (end - start) * 1000 / 20  # milliseconds per forward pass

    def _get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": str(os.cpu_count() or "unknown")
        }

    def _report_local_only(self, results_file: Path) -> None:
        """Say plainly where results live. There is no upload step."""
        # 2026-09-28: this once asked "submit to the community?" (default yes)
        # and then only wrote a local file via a stub uploader.
        self.console.print(
            f"[dim]Results are stored locally only ({results_file}). "
            "TinyTorch has no community upload for benchmarks yet.[/dim]"
        )
