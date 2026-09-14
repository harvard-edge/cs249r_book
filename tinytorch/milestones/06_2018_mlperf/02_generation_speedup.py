#!/usr/bin/env python3
"""Milestone 06.2: measure KV caching without changing the computation.

Replay one fixed token sequence through the GPT built in Module 13. The baseline
recomputes each causal prefix; Module 18's cache processes only the new token.
Both paths run embedding, attention, feed-forward layers, and output projection.
Check their logits before comparing repeated timings. This is an inference
microbenchmark on an untrained model, not a language-quality evaluation.

Required modules: 01-08, 11-13, and 18 (including their prerequisites).
The attention work across N prefixes is cubic without a cache and quadratic
with a cache. Actual speedup depends on sequence length and machine overhead;
there is no required timing ratio.
"""
from pathlib import Path
import sys
import time

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def replay_prefixes(model, tokens, cache=None):
    """Return the next-token logits at every position of one fixed sequence.

    The cache cursor advances once after all layers process a token. Resetting
    at entry makes each replay an independent request, including warmup runs.
    """
    from tinytorch.core.tensor import Tensor

    if cache is not None:
        cache.reset()
    logits = []
    for position in range(tokens.shape[1]):
        if cache is None:
            output = model(Tensor(tokens[:, :position + 1]))
        else:
            output = model(Tensor(tokens[:, position:position + 1]),
                           start_pos=cache.seq_pos)
            cache.advance()
        logits.append(output.data[:, -1, :].copy())
    return np.stack(logits, axis=1)


def measure_replay(model, tokens, cache=None, repeats=7):
    """Warm up once, then measure independent complete replays in milliseconds."""
    replay_prefixes(model, tokens, cache)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        replay_prefixes(model, tokens, cache)
        elapsed.append((time.perf_counter() - start) * 1000)
    return float(np.median(elapsed))


def compare_cached_replay(model, tokens, repeats=7):
    """Verify equivalent outputs and time the same model with and without caching."""
    from tinytorch.core.autograd import no_grad
    from tinytorch.perf.memoization import enable_kv_cache, disable_kv_cache

    if getattr(model, '_cache_enabled', False):
        raise ValueError('Pass an uncached model; this comparison manages its cache.')
    if tokens.ndim != 2 or tokens.shape[0] != 1 or not 0 < tokens.shape[1] <= model.max_seq_len:
        raise ValueError('Use one nonempty token sequence within the model context length.')
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')

    with no_grad():
        expected = replay_prefixes(model, tokens)
        baseline_ms = measure_replay(model, tokens, repeats=repeats)
        cache = enable_kv_cache(model)
        try:
            actual = replay_prefixes(model, tokens, cache)
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            cached_ms = measure_replay(model, tokens, cache, repeats)
            memory = cache.get_memory_usage()
            return {
                'baseline_ms': baseline_ms,
                'cached_ms': cached_ms,
                'speedup': baseline_ms / cached_ms,
                'max_logit_error': float(np.max(np.abs(actual - expected))),
                'cache_bytes': int(round(memory['total_mb'] * 1024 * 1024)),
                'tokens_processed': cache.seq_pos,
            }
        finally:
            disable_kv_cache(model)


def main():
    console = Console()
    try:
        from tinytorch.core.transformers import GPT
        from tinytorch.perf.memoization import enable_kv_cache
    except ImportError as error:
        console.print(f"Missing implementation: {error}")
        console.print("Export modules 01-08, 11-13, and 18, including their prerequisites.")
        return 1

    console.print('[bold cyan]Milestone 06.2: KV Cache — Same Computation, Less Repetition[/bold cyan]')
    model = GPT(vocab_size=27, embed_dim=32, num_layers=2, num_heads=2, max_seq_len=64)
    tokens = np.random.default_rng(7).integers(0, model.vocab_size, (1, 64))
    results = compare_cached_replay(model, tokens)

    table = Table(title='Fixed-sequence inference replay (median of 7 runs)')
    table.add_column('Path')
    table.add_column('Total (ms)', justify='right')
    table.add_column('Per token (ms)', justify='right')
    for label, key in [('Full causal prefixes', 'baseline_ms'), ('Cached single tokens', 'cached_ms')]:
        table.add_row(label, f"{results[key]:.3f}", f"{results[key] / tokens.shape[1]:.3f}")
    console.print(table)
    console.print(f"Logit equivalence passed; maximum difference: {results['max_logit_error']:.2e}")
    console.print(f"Measured speed ratio (baseline/cached): {results['speedup']:.2f}×")
    console.print(f"Preallocated K+V storage: {results['cache_bytes']:,} bytes")
    console.print('The cache reuses earlier projections; each new query still reads the growing prefix.')
    console.print('Small workloads can be slower with caching. Correctness is required; speedup is measured.')
    console.print('[bold green]MILESTONE 06.2 COMPLETE[/bold green]')
    return 0


if __name__ == '__main__':
    sys.exit(main())
