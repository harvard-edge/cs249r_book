#!/usr/bin/env python3
"""
Generative Serving (2022-Present): KV Cache Generation Speedup
=============================================================================

📚 HISTORICAL CONTEXT:
When OpenAI deployed ChatGPT in late 2022, naive autoregressive generation was
catastrophically slow: generating token N required recomputing the key and value
vectors for all previous N-1 tokens through every transformer layer. The computational
cost per sequence grew cubically: O(S^3) total attention operations.

The introduction of Key-Value (KV) Caching transformed generative AI serving:
- Prefill Phase: Compute and cache keys and values for all prompt tokens once.
- Decode Phase: Compute only the single query vector for the new token, retrieve
  cached keys and values from DRAM/SRAM, and append the new token in O(1) time.
This slashes decoding cost to O(S^2) and delivers interactive token streaming!

🎯 MILESTONE 06 PART 2: MEASURE YOUR KV CACHE SPEEDUP
Replay a token sequence through YOUR TinyGPT model with and without YOUR KV Cache.
Verify numerical output invariance (identical logits) and measure the real wall-clock
speedup delivered by YOUR Module 18 implementation!

✅ REQUIRED MODULES (Run after Module 18):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR strided tensor data structure
  Module 06 (Autograd)      : YOUR no_grad inference scope
  Module 11 (Embeddings)    : YOUR Token + Learned Positional Embeddings
  Module 12 (Attention)     : YOUR Causal Multi-Head Attention
  Module 13 (Transformers)  : YOUR Stacked Transformer Blocks & TinyGPT
  Module 18 (Memoization)   : YOUR KV Cache state reuse engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Naive vs. KV Cached Autoregressive Serving):
    NAIVE (O(S^2) per step):
    Step 1: [Tok 1] ──────────────▶ Compute Q, K, V
    Step 2: [Tok 1, Tok 2] ───────▶ Recompute ALL Q, K, V  (Redundant!)
    Step 3: [Tok 1, Tok 2, Tok 3] ─▶ Recompute ALL Q, K, V  (Wasteful!)

    WITH YOUR KV CACHE (O(1) per step):
    Step 1: [Tok 1] ──────────────▶ Store K1, V1 in YOUR Cache
    Step 2: [Tok 2] ──────────────▶ Compute only Q2, K2, V2; reuse K1, V1!
    Step 3: [Tok 3] ──────────────▶ Compute only Q3, K3, V3; reuse K1, K2, V1, V2!
"""

from contextlib import nullcontext, redirect_stdout
import io
from pathlib import Path
import sys
import time

import numpy as np
from rich.console import Console
from rich.table import Table

# Add project root, and this folder for the shared gate references
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# =============================================================================
# 🎓 ZONE 1: STUDENT CORE LEGO BRICKS (YOUR Modules 01, 06, 13, 18)
# =============================================================================

try:
    from tinytorch.core.tensor import Tensor  # noqa: E402
    from tinytorch.core.autograd import no_grad  # noqa: E402
    from tinytorch.core.transformers import GPT  # noqa: E402
    from tinytorch.perf.memoization import enable_kv_cache, disable_kv_cache  # noqa: E402
except ImportError as error:
    Console().print(f"Missing implementation: {error}")
    Console().print("Export modules 01-08, 11-13, and 18, including their prerequisites.")
    sys.exit(1)

from milestones.try_it import try_it  # noqa: E402
from mlperf_gates import (  # noqa: E402
    CAUSAL_LEAK_MAX, CONTEXT_DEPENDENCE_MIN, LOGIT_SPREAD_MIN, POSITION_SPREAD_MIN,
    context_probe, logit_spread, logits_nondegenerate, position_spread,
)


def replay_prefixes(model, tokens, cache=None):
    """Return the next-token logits at every position of one fixed sequence.

    The cache cursor advances once after all layers process a token. Resetting
    at entry makes each replay an independent request, including warmup runs.
    Cached forwards run inside an explicit generation scope; ordinary forwards
    resume when the scope exits, including after an exception.
    """
    from tinytorch.core.tensor import Tensor

    if cache is not None:
        cache.reset()
    logits = []
    with cache.generation() if cache is not None else nullcontext():
        for position in range(tokens.shape[1]):
            if cache is None:
                output = model(Tensor(tokens[:, :position + 1]))
            else:
                output = model(Tensor(tokens[:, position:position + 1]),
                               start_pos=cache.seq_pos)
                cache.advance()
            logits.append(output.data[:, -1, :].copy())
    return np.stack(logits, axis=1)


# =============================================================================
# 📊 ZONE 2: MILESTONE HARNESS & BENCHMARK UX
# =============================================================================

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


def check_model_logits(model, tokens):
    """
    Check that the model under test computes something before timing it.

    2026-09-29: the cache equivalence below is the only check this part used
    to make, and an all-zero model satisfies it (0 == 0). A knockout study
    passed this milestone with YOUR Tensor matmul, Linear, embedding lookup,
    or LayerNorm returning zeros. So first: the logits must vary, no position
    may read the future, and every position must read its past. Returns a list
    of (failed check, measured, required, lesson); empty when all pass.
    """
    with no_grad():
        spread = logit_spread(model(Tensor(tokens)).data)
        probe = context_probe(model, tokens[0], model.vocab_size, Tensor)
        positions = position_spread(model, int(tokens[0][0]), len(tokens[0]), Tensor)
    failures = []
    if not logits_nondegenerate(spread):
        failures.append((
            'Logits vary across tokens and positions',
            f"std across vocabulary {spread['vocab']:.2e}, across positions {spread['positions']:.2e}",
            f">= {LOGIT_SPREAD_MIN:.0e} each",
            'YOUR GPT predicts the same (often all-zero) scores everywhere. Check '
            'that Tensor matmul and add (Module 01), Linear (Module 03), and LayerNorm '
            '(Module 13) return their computed values rather than zeros.'))
    if probe['leak'] > CAUSAL_LEAK_MAX:
        failures.append((
            'No position reads the future',
            f"changing later tokens moved earlier logits by {probe['leak']:.2e}",
            f"<= {CAUSAL_LEAK_MAX:.0e}",
            'A position predicts the next token from tokens up to itself only. Check '
            'the causal mask (Modules 12 and 13).'))
    if probe['dependence'] < CONTEXT_DEPENDENCE_MIN:
        failures.append((
            'Every position reads its past',
            f"changing earlier tokens moved the logits by at most {probe['dependence']:.2e}",
            f">= {CONTEXT_DEPENDENCE_MIN:.0e}",
            'The prediction at a position ignores the tokens before it, so there is '
            'nothing for a KV cache to reuse. Check that the embedding lookup returns '
            'the token rows (Module 11) and that attention mixes positions (Module 12).'))
    if positions < POSITION_SPREAD_MIN:
        failures.append((
            'Positions are distinguishable',
            f"one token repeated {len(tokens[0])} times gave logits within {positions:.2e} at every position",
            f">= {POSITION_SPREAD_MIN:.0e}",
            'Without positional encoding, attention sees a set, not a sequence. Check that '
            'YOUR PositionalEncoding (Module 11) adds the position table to the token '
            'embeddings instead of returning its input.'))
    return failures


TRY_IT_ALPHABET = " abcdefghijklmnopqrstuvwxyz"  # token 0 is space, 1-26 are letters


def text_to_tokens(text, max_len):
    """Map lowercase letters and spaces to the 27 token ids this GPT reads."""
    kept = [TRY_IT_ALPHABET.index(ch) for ch in text.lower() if ch in TRY_IT_ALPHABET]
    return np.array([kept[:max_len]], dtype=np.int64)


def try_cache_lengths(model, console, read=None):
    """After a pass, time YOUR cache on sequences the student types."""

    def measure(text):
        tokens = text_to_tokens(text, model.max_seq_len)
        if tokens.shape[1] == 0:
            console.print('[yellow]Type some letters; other characters are skipped.[/yellow]')
            return
        # YOUR Module 18 announces every enable/disable; the verdict above
        # already showed that, so each try-it line keeps only its measurement.
        with redirect_stdout(io.StringIO()):
            results = compare_cached_replay(model, tokens, repeats=3)
        n = tokens.shape[1]
        faster = results['speedup'] >= 1
        console.print(
            f"  {n:>2} tokens: full prefixes {results['baseline_ms']:8.2f} ms, "
            f"YOUR cache {results['cached_ms']:8.2f} ms  ->  "
            f"[{'green' if faster else 'yellow'}]{results['speedup']:.2f}x[/]"
            f"  [dim](outputs identical, max diff {results['max_logit_error']:.1e})[/dim]")

    return try_it(
        console,
        'Type any sentence and YOUR GPT processes it one token at a time, '
        'with and without YOUR KV cache.\n'
        'Try a short word, then a long sentence: the uncached path recomputes '
        'every earlier token at every step, so its cost grows faster with length.\n'
        f'[dim]Letters and spaces only, up to {model.max_seq_len} tokens. This GPT is '
        'untrained, so what you are measuring is cost, not the words it predicts.[/dim]',
        '[yellow]Sentence > [/yellow]',
        measure,
        read=read,
    )


def main():
    console = Console()
    console.print('[bold cyan]Milestone 06.2: KV Cache: Same Computation, Less Repetition[/bold cyan]')
    model = GPT(vocab_size=27, embed_dim=32, num_layers=2, num_heads=2, max_seq_len=64)
    tokens = np.random.default_rng(7).integers(0, model.vocab_size, (1, 64))
    failures = check_model_logits(model, tokens)
    if failures:
        console.print('[bold red]✗ MILESTONE 06.2 NOT PASSED: YOUR GPT does not compute '
                       'a working language model, so there is no computation to cache.[/bold red]')
        for name, measured, required, lesson in failures:
            console.print(f'  [red]✗ {name}[/red]: measured {measured}; required {required}')
            console.print(f'    [yellow]{lesson}[/yellow]')
        return 1
    console.print('Model check passed: logits vary, no position reads the future, '
                  'every position reads its past.')
    try:
        results = compare_cached_replay(model, tokens)
    except AssertionError as mismatch:
        # 2026-09-29: a cache mismatch used to surface as a raw traceback.
        detail = str(mismatch).strip().splitlines()
        console.print('[bold red]✗ MILESTONE 06.2 NOT PASSED: cached logits differ from '
                      'recomputed logits.[/bold red]')
        for line in detail[:6]:
            if line.strip():
                console.print(f'  [dim]{line.strip()}[/dim]')
        console.print(
            '[yellow]A KV cache is only an optimization if it changes nothing but the cost. '
            'Decoding one token at a time with YOUR cache must reproduce the logits of '
            'running the full prefix (tolerance rtol=1e-4, atol=1e-5). Check that '
            'KVCache.update writes K and V at the current position, get() returns every '
            'position written so far, and advance() runs once per token (Module 18).[/yellow]')
        return 1

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
    try_cache_lengths(model, console)
    return 0


if __name__ == '__main__':
    sys.exit(main())
