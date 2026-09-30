"""
Independent references for the Milestone 06 pass gates.
=======================================================

A gate is only as strong as the reference it compares against. Every check
here computes its expected answer WITHOUT calling the module under test:

* Module 14 (Profiler): parameter and FLOP counts derived from the Linear
  layers' shapes, by hand arithmetic.
* Module 19 (Benchmarking): statistics and Pareto frontiers of small fixtures
  whose answers are worked out by hand in the comments.
* Modules 11 and 13 (a GPT built from YOUR embeddings and transformer blocks):
  black-box probes of what any working language model must do. Its logits vary
  (it is not a constant function), a position never reads the future
  (causality), and a position does read its past (context dependence).

2026-09-29 (release audit): a knockout study found both Part 1 and Part 2 of
this milestone passed with these modules sabotaged. An all-zero GPT satisfies
"cached logits == recomputed logits" trivially (0 == 0), and the Profiler and
Pareto outputs were printed but never checked. These helpers close those gaps.
They are plain NumPy so tests can drive them with fixtures.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np


# =============================================================================
# Module 14: Profiler references
# =============================================================================

def linear_reference_counts(linears) -> Tuple[int, int]:
    """
    Parameters and per-sample forward FLOPs of a chain of Linear layers.

    Computed from each layer's in_features, out_features, and whether it has
    a bias. Module 14's convention: a Linear layer costs 2 * in * out FLOPs
    per sample (one multiply and one add per weight); the bias add and the
    activations between layers are not counted.

    DigitMLP, 64 -> 32 -> 10 with biases:
        parameters = (64*32 + 32) + (32*10 + 10) = 2080 + 330 = 2,410
        FLOPs      = 2 * (64*32 + 32*10)         = 2 * 2368  = 4,736
    """
    params = flops = 0
    for layer in linears:
        n_in, n_out = int(layer.in_features), int(layer.out_features)
        has_bias = getattr(layer, 'bias', None) is not None
        params += n_in * n_out + (n_out if has_bias else 0)
        flops += 2 * n_in * n_out
    return params, flops


# =============================================================================
# Module 04: loss reference
# =============================================================================

# 2026-09-29: with YOUR CrossEntropyLoss forward returning 0, the baseline still
# trained (backward, Module 06, carries the gradient on its own), so the
# accuracy gate passed while the reported loss was wrong. One batch is now
# compared against this reference. float32 logits against a float64 reference
# agree to ~1e-7 relative; the tolerance leaves room for accumulation order.
LOSS_RTOL, LOSS_ATOL = 1e-4, 1e-5


def reference_cross_entropy(logits, targets) -> float:
    """
    Mean negative log-likelihood of integer targets under softmax(logits).

    Numerically stable log-softmax: subtract each row's max before exp, so
    log_softmax(z) = (z - max) - log(sum(exp(z - max))). Averaged over every
    target (Module 04's reduction: -mean of the selected log-probabilities).
    """
    z = np.asarray(logits, dtype=np.float64)
    z = z.reshape(-1, z.shape[-1])
    t = np.asarray(targets).reshape(-1).astype(np.int64)
    shifted = z - z.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(t.size), t].mean())


def loss_matches_reference(student_loss: float, reference: float) -> bool:
    return bool(np.isfinite(student_loss)
                and abs(student_loss - reference) <= LOSS_ATOL + LOSS_RTOL * abs(reference))


# =============================================================================
# Module 19: Benchmarking references
# =============================================================================

# BenchmarkResult fixture. Worked by hand:
#   mean   = (2 + 4 + 4 + 5 + 10) / 5 = 25 / 5 = 5
#   median = middle of [2, 4, 4, 5, 10] = 4
#   sample std: deviations -3, -1, -1, 0, 5 -> squares sum to 36 -> sqrt(36/4) = 3
STATS_FIXTURE = [2.0, 4.0, 4.0, 5.0, 10.0]
STATS_EXPECTED = {'mean': 5.0, 'median': 4.0, 'std': 3.0,
                  'min_val': 2.0, 'max_val': 10.0, 'count': 5}

# pareto_frontier fixtures: (points, lower_is_better, expected frontier).
PARETO_FIXTURES = [
    # Latency (lower better) vs accuracy (higher better). c is beaten by a on
    # both; d ties b on accuracy but is slower, so b dominates it.
    ({'a': (5.0, 0.80), 'b': (40.0, 0.94), 'c': (40.0, 0.70), 'd': (60.0, 0.94)},
     (True, False), ['a', 'b']),
    # Identical points do not dominate each other (no strict improvement).
    ({'x': (1.0, 1.0), 'y': (1.0, 1.0), 'z': (2.0, 2.0)},
     (True, True), ['x', 'y']),
    # Three minimized objectives. s beats p (ties twice, better on the third)
    # and r; q is the only point with a 1 in the middle objective.
    ({'p': (1.0, 3.0, 2.0), 'q': (2.0, 1.0, 3.0), 'r': (3.0, 3.0, 3.0), 's': (1.0, 3.0, 1.0)},
     (True, True, True), ['q', 's']),
]


def reference_pareto(points: Dict[str, Sequence[float]], lower_is_better) -> List[str]:
    """Brute-force frontier, used only to cross-check the fixtures in tests."""
    def better_or_equal(x, y, low):
        return x <= y if low else x >= y

    def strictly_better(x, y, low):
        return x < y if low else x > y

    frontier = []
    for name, v in points.items():
        dominated = False
        for other, w in points.items():
            if other == name:
                continue
            if all(better_or_equal(a, b, lo) for a, b, lo in zip(w, v, lower_is_better)) and \
                    any(strictly_better(a, b, lo) for a, b, lo in zip(w, v, lower_is_better)):
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    return frontier


def check_benchmark_stats(BenchmarkResult) -> List[str]:
    """Problems with YOUR BenchmarkResult on the fixture (empty list = correct)."""
    result = BenchmarkResult('fixture', list(STATS_FIXTURE))
    problems = []
    for attr, want in STATS_EXPECTED.items():
        got = getattr(result, attr, None)
        if got is None or not np.isfinite(got) or abs(float(got) - want) > 1e-9:
            problems.append(f"{attr}={got!r} (expected {want:g})")
    return problems


def check_pareto(pareto_frontier) -> List[str]:
    """Problems with YOUR pareto_frontier on the fixtures (empty list = correct)."""
    problems = []
    for points, lower, expected in PARETO_FIXTURES:
        got = list(pareto_frontier(dict(points), lower))
        if got != expected:
            problems.append(f"{sorted(points)} -> {got} (expected {expected})")
    return problems


def latency_stats_consistent(result) -> bool:
    """A measured latency BenchmarkResult: positive, finite, min <= mean/median <= max."""
    vals = [result.mean, result.median, result.min_val, result.max_val]
    if not all(np.isfinite(v) for v in vals):
        return False
    return (result.min_val > 0 and result.min_val <= result.median <= result.max_val
            and result.min_val - 1e-12 <= result.mean <= result.max_val + 1e-12)


# =============================================================================
# Modules 11 and 13: black-box probes of a GPT's logits
# =============================================================================
#
# Calibration (2026-09-29): 10 fresh untrained GPTs (embed 32, 2 layers,
# 2 heads) for each workload Parts 1 and 2 use (vocab 28 / 16 tokens and
# vocab 27 / 64 tokens), correct modules:
#   logit spread across the vocabulary    0.83-1.16
#   logit spread across positions         0.53-0.81
#   future leak (causality)               0.0 exactly, every run
#   context dependence, weakest cut       0.45-2.49
# Knockouts (provenance harness): Linear zeroed (03), MatMul/Add zeroed (01),
# LayerNorm + MLP zeroed (13) -> logits exactly 0, spread 0.0. Embedding
# lookup zeroed (11) or attention bypassed (12) -> spread normal (the
# positional embedding still varies) but context dependence exactly 0.0.
# The floors sit 40x or more under the weakest correct value.
LOGIT_SPREAD_MIN = 1e-2
CAUSAL_LEAK_MAX = 1e-4
CONTEXT_DEPENDENCE_MIN = 1e-3
# 2026-09-29: a final knockout matrix found Part 2 passing with the positional
# encoding replaced by an identity (knockout 11b): attention still mixes
# content, so the three checks above hold. One token repeated along the
# sequence separates the cases: 8 fresh GPT(27, 32, 2, 2, 64) x 4 tokens gave
# a spread of 1.15-1.95 with a working position table and 1.1e-6 to 1.6e-6
# without one. The floor sits ~1000x from both.
POSITION_SPREAD_MIN = 1e-3


def logit_spread(logits: np.ndarray) -> Dict[str, float]:
    """
    How much [S, V] logits vary across the vocabulary and across positions.

    A constant (for example all-zero) model has 0 on both: every position
    predicts the same distribution over the same tokens.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 3:
        logits = logits[0]
    if logits.ndim != 2 or not np.all(np.isfinite(logits)):
        return {'vocab': 0.0, 'positions': 0.0, 'finite': False}
    return {'vocab': float(np.std(logits, axis=1).mean()),
            'positions': float(np.std(logits, axis=0).mean()),
            'finite': True}


def logits_nondegenerate(spread: Dict[str, float]) -> bool:
    return bool(spread['finite'] and spread['vocab'] >= LOGIT_SPREAD_MIN
                and spread['positions'] >= LOGIT_SPREAD_MIN)


def position_spread(model, token_id: int, seq_len: int, Tensor) -> float:
    """
    Largest logit difference between positions when one token repeats.

    Without positional information every position holds the same vector and
    attends to a set of identical vectors, so all positions give the same
    logits. A working positional encoding makes them differ.
    """
    ids = np.full((1, seq_len), int(token_id), dtype=np.int64)
    logits = np.asarray(model(Tensor(ids)).data, dtype=np.float64)
    if logits.ndim == 3:
        logits = logits[0]
    if not np.all(np.isfinite(logits)):
        return 0.0
    return float(np.abs(logits - logits[0:1]).max())


def context_probe(model, tokens: Sequence[int], vocab_size: int, Tensor) -> Dict[str, float]:
    """
    Two properties every autoregressive language model must have.

    * Causality: change every token AFTER position k; logits at 0..k must not
      move (they never read those tokens). Reported as the largest change.
    * Context dependence: change every token BEFORE position k (keeping token
      k); logits at k must move, or the model ignores its context (at best a
      bigram model). Reported as the smallest, over the cuts, of the largest
      change at k.

    Adapted from Milestone 05's causality probe (transformer_gates.py); kept
    as its own copy so the two milestones evolve independently.
    """
    base_ids = np.asarray(tokens, dtype=np.int64).reshape(-1)
    seq_len = int(base_ids.shape[0])
    if seq_len < 4:
        raise ValueError('context probe needs at least 4 tokens')

    def run(ids):
        return np.asarray(model(Tensor(ids[np.newaxis, :])).data, dtype=np.float64)[0]

    def largest(change):
        # A non-finite change is never a pass: inf for a leak, 0 for dependence.
        return float(change.max()) if np.all(np.isfinite(change)) else float('nan')

    base = run(base_ids)
    leak, dependence = 0.0, float('inf')
    for cut in sorted({1, seq_len // 4, seq_len // 2, seq_len - 2}):
        future = base_ids.copy()
        future[cut + 1:] = (future[cut + 1:] + 1) % vocab_size
        moved = largest(np.abs(run(future)[:cut + 1] - base[:cut + 1]))
        leak = float('inf') if np.isnan(moved) else max(leak, moved)

        past = base_ids.copy()
        past[:cut] = (past[:cut] + 1) % vocab_size
        change = largest(np.abs(run(past)[cut] - base[cut]))
        dependence = 0.0 if np.isnan(change) else min(dependence, change)
    return {'leak': leak, 'dependence': dependence, 'seq_len': seq_len}


def context_probe_passed(probe: Dict[str, float]) -> bool:
    return bool(probe['leak'] <= CAUSAL_LEAK_MAX
                and probe['dependence'] >= CONTEXT_DEPENDENCE_MIN)


def main() -> int:
    """
    Check YOUR Modules 14, 19, and the Module 11-13 GPT in about a second.

    `python mlperf_gates.py` runs every reference check the two parts of this
    milestone use, without training anything: exit 0 when all pass.
    """
    import contextlib
    import io
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from networks import DigitMLP
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.transformers import GPT
    from tinytorch.perf.benchmarking import BenchmarkResult, pareto_frontier
    from tinytorch.perf.profiling import Profiler

    problems = []
    mlp = DigitMLP()
    ref_params, ref_flops = linear_reference_counts(mlp.layers)
    if Profiler().count_parameters(mlp) != ref_params:
        problems.append(f"count_parameters != {ref_params}")
    if Profiler().count_flops(mlp, (1, 64)) != ref_flops:
        problems.append(f"count_flops != {ref_flops}")
    problems += check_benchmark_stats(BenchmarkResult) + check_pareto(pareto_frontier)
    with contextlib.redirect_stdout(io.StringIO()):
        gpt = GPT(vocab_size=28, embed_dim=32, num_layers=2, num_heads=2, max_seq_len=32)
        tokens = np.random.default_rng(11).integers(0, 28, (1, 16))
        spread = logit_spread(gpt(Tensor(tokens)).data)
        probe = context_probe(gpt, tokens[0], 28, Tensor)
    if not logits_nondegenerate(spread):
        problems.append(f"GPT logits degenerate: {spread}")
    if not context_probe_passed(probe):
        problems.append(f"GPT context probe failed: {probe}")
    for problem in problems:
        print(f"✗ {problem}")
    print("✓ all Milestone 06 reference checks pass" if not problems else
          f"{len(problems)} reference check(s) failed")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
