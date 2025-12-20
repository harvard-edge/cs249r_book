#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔬 MILESTONE: Profile KV Cache                            ║
║                  Measure Optimization Impact Scientifically                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 HISTORICAL CONTEXT:
Production ML systems require rigorous performance measurement. This milestone
teaches scientific profiling - the foundation of all optimization work.

🎯 WHAT YOU'RE BUILDING:
Using YOUR Tiny🔥Torch implementations, you'll profile KV caching to see how it
transforms O(n²) generation to O(n) - measuring the 6-10× speedup scientifically!

✅ REQUIRED MODULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01-13 (Core)         : YOUR complete transformer stack
  Module 14 (Profiling)       : YOUR profiler measures parameters & memory
  Module 18 (Memoization)     : YOUR KV-cache for generation speedup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ PROFILING WORKFLOW:
    ┌──────────────────┐
    │ Profile Model    │ ← YOUR profiler counts parameters
    │ Architecture     │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Baseline Latency │ ← O(n²) - recomputes all positions
    │ (No Cache)       │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Cached Latency   │ ← O(n) - reuses K/V values
    │ (KV Cache)       │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Comparison       │ ← Quantify the optimization impact
    │ Dashboard        │
    └──────────────────┘

# =============================================================================
# 📊 YOUR MODULES IN ACTION
# =============================================================================
#
# ┌─────────────────────┬────────────────────────────────┬─────────────────────────────┐
# │ What You Built      │ How It's Used Here             │ Systems Impact              │
# ├─────────────────────┼────────────────────────────────┼─────────────────────────────┤
# │ Module 14: Profiler │ Counts parameters, measures    │ Scientific measurement      │
# │                     │ memory and latency             │ enables data-driven opt     │
# │                     │                                │                             │
# │ Module 18: KV Cache │ Caches keys/values across      │ Transforms O(n²) → O(n)     │
# │                     │ generation steps               │ 6-10× speedup!              │
# │                     │                                │                             │
# │ Module 13: GPT      │ Transformer model being        │ Production models need      │
# │                     │ profiled and optimized         │ this exact optimization     │
# └─────────────────────┴────────────────────────────────┴─────────────────────────────┘
#
# =============================================================================

📊 EXPECTED RESULTS:
- Baseline: 2-5 tokens/sec (O(n²) complexity)
- With KV Cache: 20-50 tokens/sec (O(n) complexity)
- Speedup: 6-10× faster generation

💡 KEY INSIGHT:
Profiling turns "it feels faster" into "we measured 8.3× speedup with p<0.001"
This is how production ML systems are optimized!
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

from tinytorch.core.transformer import GPT
from tinytorch.core.tokenization import CharTokenizer
from tinytorch.core.tensor import Tensor
from tinytorch.perf.profiling import Profiler
from tinytorch.perf.memoization import enable_kv_cache, disable_kv_cache

console = Console()

def show_welcome():
    """Display welcome panel."""
    welcome = Panel(
        "[bold cyan]🔬 Profiling KV Cache Performance[/bold cyan]\n\n"
        "You've implemented KV caching to speed up generation.\n"
        "Now let's measure its impact scientifically!\n\n"
        "[dim]This demo shows how profiling guides optimization.[/dim]",
        title="[bold]Milestone 15: Performance Profiling[/bold]",
        border_style="cyan",
        box=box.DOUBLE
    )
    console.print(welcome)
    console.print()

def profile_model_architecture(model, profiler):
    """Profile the model architecture."""
    console.print(Panel(
        "[bold yellow]Step 1: Profile Model Architecture[/bold yellow]\n"
        "Understanding model complexity",
        border_style="yellow"
    ))

    param_count = profiler.count_parameters(model)
    memory = profiler.measure_memory(model, (1, 10))

    # Create architecture table
    table = Table(title="Model Architecture Profile", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Insight", style="dim")

    table.add_row(
        "Total Parameters",
        f"{param_count:,}",
        "Model size indicator"
    )
    table.add_row(
        "Parameter Memory",
        f"{memory['parameter_memory_mb']:.2f} MB",
        "Storage requirement"
    )
    table.add_row(
        "Peak Memory",
        f"{memory['peak_memory_mb']:.2f} MB",
        "Runtime memory usage"
    )

    console.print(table)
    console.print()

    return param_count, memory

def profile_baseline_generation(model, tokenizer, prompt, profiler, max_new_tokens=30):
    """Profile generation WITHOUT KV caching."""
    console.print(Panel(
        "[bold red]Step 2: Profile Baseline (No Cache)[/bold red]\n"
        "O(n²) complexity - recomputes all positions",
        border_style="red"
    ))

    # Disable cache if enabled
    disable_kv_cache(model)

    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    input_tensor = Tensor(np.array([tokens]))

    # Measure latency for multiple token generations
    console.print("[dim]Measuring latency across 30 tokens...[/dim]")

    import time
    times = []
    for i in range(max_new_tokens):
        # Measure single token generation
        start = time.perf_counter()
        _ = model.forward(input_tensor)
        end = time.perf_counter()
        times.append(end - start)

        # Expand context for next token (simulating autoregressive)
        if i < max_new_tokens - 1:
            next_token = np.random.randint(0, tokenizer.vocab_size)
            # Maintain 2D shape: (batch_size, seq_len)
            new_seq = np.append(input_tensor.data[0], next_token)
            input_tensor = Tensor(new_seq.reshape(1, -1))

    avg_latency = np.mean(times) * 1000  # Convert to ms
    total_time = sum(times)
    tokens_per_sec = max_new_tokens / total_time

    # Create baseline table
    table = Table(title="Baseline Performance", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="red")
    table.add_column("Notes", style="dim")

    table.add_row(
        "Avg Token Latency",
        f"{avg_latency:.3f} ms",
        "Increases with sequence length"
    )
    table.add_row(
        "Tokens per Second",
        f"{tokens_per_sec:.2f} tok/s",
        "Baseline generation speed"
    )
    table.add_row(
        "Total Time",
        f"{total_time:.3f} s",
        f"For {max_new_tokens} tokens"
    )
    table.add_row(
        "Complexity",
        "O(n²)",
        "Recomputes all positions"
    )

    console.print(table)
    console.print()

    return {
        'avg_latency': avg_latency,
        'tokens_per_sec': tokens_per_sec,
        'total_time': total_time
    }

def profile_cached_generation(model, tokenizer, prompt, profiler, max_new_tokens=30):
    """Profile generation WITH KV caching."""
    console.print(Panel(
        "[bold green]Step 3: Profile Cached Generation[/bold green]\n"
        "O(n) complexity - caches previous computations",
        border_style="green"
    ))

    # Enable cache
    enable_kv_cache(model)

    # Tokenize prompt
    tokens = tokenizer.encode(prompt)

    console.print("[dim]Measuring cached latency across 30 tokens...[/dim]")

    import time
    times = []

    # Initialize with prompt
    input_tensor = Tensor(np.array([tokens]))
    _ = model.forward(input_tensor)

    # Generate tokens one at a time (cached path)
    for i in range(max_new_tokens):
        # Measure single token generation (seq_len=1, cache enabled)
        next_token = np.random.randint(0, tokenizer.vocab_size)
        single_token_input = Tensor(np.array([[next_token]]))

        start = time.perf_counter()
        _ = model.forward(single_token_input)
        end = time.perf_counter()
        times.append(end - start)

    avg_latency = np.mean(times) * 1000  # Convert to ms
    total_time = sum(times)
    tokens_per_sec = max_new_tokens / total_time

    # Create cached table
    table = Table(title="Cached Performance", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Notes", style="dim")

    table.add_row(
        "Avg Token Latency",
        f"{avg_latency:.3f} ms",
        "Constant regardless of length"
    )
    table.add_row(
        "Tokens per Second",
        f"{tokens_per_sec:.2f} tok/s",
        "Optimized generation speed"
    )
    table.add_row(
        "Total Time",
        f"{total_time:.3f} s",
        f"For {max_new_tokens} tokens"
    )
    table.add_row(
        "Complexity",
        "O(n)",
        "Reuses cached K/V"
    )

    console.print(table)
    console.print()

    return {
        'avg_latency': avg_latency,
        'tokens_per_sec': tokens_per_sec,
        'total_time': total_time
    }

def show_comparison(baseline, cached):
    """Show side-by-side comparison."""
    console.print(Panel(
        "[bold magenta]Step 4: Performance Comparison[/bold magenta]\n"
        "Quantifying the optimization impact",
        border_style="magenta"
    ))

    speedup = cached['tokens_per_sec'] / baseline['tokens_per_sec']
    latency_reduction = (1 - cached['avg_latency'] / baseline['avg_latency']) * 100
    time_saved = baseline['total_time'] - cached['total_time']

    # Create comparison table
    table = Table(title="🏆 KV Cache Impact", box=box.DOUBLE)
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Baseline", style="red", justify="right")
    table.add_column("Cached", style="green", justify="right")
    table.add_column("Improvement", style="bold yellow", justify="right")

    table.add_row(
        "Tokens/Second",
        f"{baseline['tokens_per_sec']:.2f}",
        f"{cached['tokens_per_sec']:.2f}",
        f"[bold green]{speedup:.2f}× faster[/bold green]"
    )
    table.add_row(
        "Avg Latency (ms)",
        f"{baseline['avg_latency']:.3f}",
        f"{cached['avg_latency']:.3f}",
        f"[bold green]↓{latency_reduction:.1f}%[/bold green]"
    )
    table.add_row(
        "Total Time (s)",
        f"{baseline['total_time']:.3f}",
        f"{cached['total_time']:.3f}",
        f"[bold green]Saved {time_saved:.3f}s[/bold green]"
    )

    console.print(table)
    console.print()

    # Show insights
    insights = Panel(
        f"[bold green]✅ KV Caching achieves {speedup:.2f}× speedup![/bold green]\n\n"
        f"[cyan]Why it works:[/cyan]\n"
        f"• Baseline: O(n²) - recomputes attention for all positions\n"
        f"• Cached: O(n) - reuses previous keys/values\n\n"
        f"[yellow]Real-world impact:[/yellow]\n"
        f"• 100 tokens: saves {time_saved * 3.33:.2f}s\n"
        f"• 1000 tokens: saves {time_saved * 33.3:.2f}s\n\n"
        f"[dim]This is how production LLMs achieve fast generation![/dim]",
        title="[bold]🎓 Learning Insight[/bold]",
        border_style="yellow",
        box=box.ROUNDED
    )
    console.print(insights)

def main():
    """Run profiling demo."""
    show_welcome()

    # Initialize model and profiler
    console.print("[bold]Initializing model...[/bold]")

    vocab = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?;:'\"-()[]0123456789")
    tokenizer = CharTokenizer(vocab)

    # Use tokenizer.vocab_size to account for special tokens (UNK, etc.)
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=64
    )

    profiler = Profiler()
    console.print("[green]✅ Model initialized[/green]\n")

    # Profile architecture
    profile_model_architecture(model, profiler)

    # Profile baseline
    prompt = "Hello"
    baseline = profile_baseline_generation(model, tokenizer, prompt, profiler)

    # Profile cached
    cached = profile_cached_generation(model, tokenizer, prompt, profiler)

    # Show comparison
    show_comparison(baseline, cached)

    # Final summary
    console.print(Panel(
        "[bold cyan]🎯 Profiling Complete![/bold cyan]\n\n"
        "You've learned how to:\n"
        "✅ Profile model architecture (parameters, memory)\n"
        "✅ Measure baseline performance\n"
        "✅ Measure optimized performance\n"
        "✅ Quantify optimization impact\n\n"
        "[yellow]Next steps:[/yellow]\n"
        "• Use profiling to guide other optimizations\n"
        "• Profile different model sizes\n"
        "• Compare different architectures\n\n"
        "[dim]Data-driven optimization > guesswork![/dim]",
        title="[bold]Module 18 Complete[/bold]",
        border_style="green",
        box=box.DOUBLE
    ))

if __name__ == "__main__":
    main()
