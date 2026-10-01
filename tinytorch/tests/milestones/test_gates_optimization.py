"""
Milestone 06 pass gates: the scripts pass only when the student's modules work.

2026-09-29: Part 1 (01_optimization_olympics.py) used to print COMPLETE and exit
0 whatever the modules did. A sabotage audit passed it with a no-op optimizer,
a dequantizer returning zeros, a no-op pruner, and a KV cache returning zero
keys. Part 2 failed a KV mismatch with a raw traceback. Each test below breaks
one student module at import time (through a sitecustomize hook) and requires
the milestone to exit 1 with its teaching message, not a traceback.

Each run takes a few seconds (Part 1 trains a small MLP for 40 epochs).

2026-09-29 (knockout study): both parts still passed with the Profiler
(Module 14), BenchmarkResult/pareto_frontier (Module 19), the embedding lookup
(Module 11), LayerNorm (Module 13), or Linear (Module 03) returning zeros. The
references now live in milestones/06_2018_mlperf/mlperf_gates.py; the fast
tests below pin those references on fixtures, and the sabotage tests prove
each gate fires.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

TINYTORCH_ROOT = Path(__file__).resolve().parents[2]
# Root holding the milestone scripts under test. Overridable so the same tests
# can be pointed at an older copy of the scripts (for example HEAD's).
SCRIPT_ROOT = Path(os.environ.get("MILESTONE_GATE_ROOT", TINYTORCH_ROOT))
PART1 = SCRIPT_ROOT / "milestones" / "06_2018_mlperf" / "01_optimization_olympics.py"
PART2 = SCRIPT_ROOT / "milestones" / "06_2018_mlperf" / "02_generation_speedup.py"

sys.path.insert(0, str(TINYTORCH_ROOT / "milestones" / "06_2018_mlperf"))
import mlperf_gates as G  # noqa: E402

SABOTAGE_HOOK = textwrap.dedent('''
    import os
    _S = os.environ.get("TT_GATE_SABOTAGE", "")
    if _S:
        import numpy as np
        from tinytorch.core.tensor import Tensor
        if _S == "opt_noop":
            import tinytorch.core.optimizers as o
            for cls in (o.SGD, o.Adam, o.AdamW):
                cls.step = lambda self: None
        elif _S == "quant_zero":
            import tinytorch.perf.quantization as Q
            _od = Q.Quantizer.dequantize_tensor
            def _deq(*a, **k):
                t = _od(*a, **k)
                return Tensor(np.zeros_like(t.data))
            Q.Quantizer.dequantize_tensor = staticmethod(_deq)
        elif _S == "prune_noop":
            import tinytorch.perf.compression as C
            C.Compressor.magnitude_prune = staticmethod(lambda model, sparsity=0.5, **k: model)
        elif _S == "kv_zero_keys":
            import tinytorch.perf.memoization as M
            _og = M.KVCache.get
            def _get(self, layer_idx):
                k, v = _og(self, layer_idx)
                return Tensor(np.zeros_like(k.data)), v
            M.KVCache.get = _get
        elif _S == "kv_drift":
            # Storage round-trips exactly for one token; keys drift once a
            # prefix exists, so only the cached-vs-recomputed logit gate sees it.
            import tinytorch.perf.memoization as M
            _og = M.KVCache.get
            def _get(self, layer_idx):
                k, v = _og(self, layer_idx)
                if self.seq_pos >= 2:
                    k = Tensor(k.data * 1.05)
                return k, v
            M.KVCache.get = _get
        elif _S == "profiler_params_zero":
            import tinytorch.perf.profiling as P
            P.Profiler.count_parameters = lambda self, model: 0
        elif _S == "profiler_flops_zero":
            import tinytorch.perf.profiling as P
            P.Profiler.count_flops = lambda self, model, input_shape: 0
        elif _S == "bench_stats_zero":
            import tinytorch.perf.benchmarking as B
            _opi = B.BenchmarkResult.__post_init__
            def _pi(self):
                _opi(self)
                self.mean = self.median = 0.0
            B.BenchmarkResult.__post_init__ = _pi
        elif _S == "pareto_admit_all":
            import tinytorch.perf.benchmarking as B
            B.pareto_frontier = lambda points, lower_is_better: list(points)
        elif _S in ("embedding_zero", "layernorm_zero"):
            if _S == "embedding_zero":
                from tinytorch.core.embeddings import EmbeddingFunction as _F
            else:
                from tinytorch.core.transformers import LayerNormFunction as _F
            _of = _F.forward
            _F.forward = lambda self, *a, **k: np.zeros_like(_of(self, *a, **k))
        elif _S == "loss_zero":
            from tinytorch.core.losses import CrossEntropyFunction as _CF
            _ocf = _CF.forward
            _CF.forward = lambda self, *a, **k: np.zeros_like(_ocf(self, *a, **k))
        elif _S == "pe_identity":
            # PositionalEncoding returns its input: attention sees a set.
            import tinytorch.core.embeddings as _E
            for _n in dir(_E):
                _c = getattr(_E, _n)
                if isinstance(_c, type) and "Positional" in _n and hasattr(_c, "forward"):
                    _c.forward = lambda self, x, *a, **k: x
        elif _S == "linear_zero":
            import tinytorch.core.layers as L
            _olf = L.Linear.forward
            L.Linear.forward = lambda self, *a, **k: _olf(self, *a, **k) * 0.0
        else:
            raise SystemExit("unknown sabotage " + _S)
''')


@pytest.fixture(scope="module")
def hook_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("gate_hook")
    (d / "sitecustomize.py").write_text(SABOTAGE_HOOK)
    return d


def run_script(script, hook_dir, sabotage=""):
    env = dict(os.environ)
    env.update(TINYTORCH_NON_INTERACTIVE="1", CI="true", COLUMNS="200",
               PYTHONPATH=os.pathsep.join([str(hook_dir), str(TINYTORCH_ROOT)]),
               TT_GATE_SABOTAGE=sabotage)
    proc = subprocess.run([sys.executable, str(script)], cwd=TINYTORCH_ROOT, env=env,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


def test_part1_correct_code_passes(hook_dir):
    rc, out = run_script(PART1, hook_dir)
    assert rc == 0, out[-3000:]
    assert "MILESTONE 06 COMPLETE" in out
    assert "Pass gates" in out


@pytest.mark.parametrize("sabotage, gate", [
    ("opt_noop", "Baseline accuracy clears the floor"),
    ("quant_zero", "dequantize(quantize(w)) recovers w"),
    ("prune_noop", "Pruning reaches the requested sparsity"),
    ("kv_zero_keys", "KV cache returns what was stored"),
    ("kv_drift", "Cached logits match recomputed logits"),
    ("profiler_params_zero", "Profiler counts every parameter"),
    ("profiler_flops_zero", "Profiler FLOPs match the layer shapes"),
    ("bench_stats_zero", "BenchmarkResult statistics match hand-computed values"),
    ("pareto_admit_all", "pareto_frontier keeps exactly the non-dominated points"),
    ("embedding_zero", "GPT reads its context"),
    ("layernorm_zero", "GPT logits vary across tokens and positions"),
    ("loss_zero", "CrossEntropyLoss matches a NumPy reference"),
    ("pe_identity", "GPT tells positions apart"),
])
def test_part1_sabotage_fails(hook_dir, sabotage, gate):
    rc, out = run_script(PART1, hook_dir, sabotage)
    assert rc == 1, f"{sabotage} passed Milestone 06:\n{out[-3000:]}"
    assert "Traceback" not in out, out[-3000:]
    assert "MILESTONE 06 NOT PASSED" in out
    assert f"Pass gate failed: {gate}" in out, out[-3000:]
    assert "MILESTONE 06 COMPLETE" not in out


def test_part2_correct_code_passes(hook_dir):
    rc, out = run_script(PART2, hook_dir)
    assert rc == 0, out[-3000:]
    assert "MILESTONE 06.2 COMPLETE" in out


@pytest.mark.parametrize("sabotage", ["kv_zero_keys", "kv_drift"])
def test_part2_broken_cache_fails_without_traceback(hook_dir, sabotage):
    rc, out = run_script(PART2, hook_dir, sabotage)
    assert rc == 1, out[-3000:]
    assert "Traceback" not in out, out[-3000:]
    assert "MILESTONE 06.2 NOT PASSED" in out


@pytest.mark.parametrize("sabotage, check", [
    ("embedding_zero", "Every position reads its past"),
    ("layernorm_zero", "Logits vary across tokens and positions"),
    ("linear_zero", "Logits vary across tokens and positions"),
    # 2026-09-29: passed the final knockout matrix (knockout 11b).
    ("pe_identity", "Positions are distinguishable"),
])
def test_part2_dead_model_fails_before_cache_check(hook_dir, sabotage, check):
    """An all-zero or context-blind GPT used to pass: cached 0 == recomputed 0."""
    rc, out = run_script(PART2, hook_dir, sabotage)
    assert rc == 1, f"{sabotage} passed Milestone 06.2:\n{out[-3000:]}"
    assert "Traceback" not in out, out[-3000:]
    assert "MILESTONE 06.2 NOT PASSED" in out
    assert f"✗ {check}" in out, out[-3000:]
    assert "MILESTONE 06.2 COMPLETE" not in out


# =============================================================================
# Fast tests: the gate references themselves, on fixtures
# =============================================================================

class _FakeLinear:
    def __init__(self, n_in, n_out, bias=True):
        self.in_features, self.out_features = n_in, n_out
        self.bias = object() if bias else None


def test_reference_counts_digit_mlp_by_hand():
    # 64 -> 32 -> 10 with biases: (2048 + 32) + (320 + 10) params, 2 * 2368 FLOPs.
    assert G.linear_reference_counts([_FakeLinear(64, 32), _FakeLinear(32, 10)]) == (2410, 4736)
    assert G.linear_reference_counts([_FakeLinear(8, 4, bias=False)]) == (32, 64)


def test_reference_counts_agree_with_real_digit_mlp_and_profiler():
    from networks import DigitMLP
    from tinytorch.perf.profiling import Profiler
    model = DigitMLP()
    params, flops = G.linear_reference_counts(model.layers)
    assert params == sum(p.data.size for p in model.parameters()) == 2410
    profiler = Profiler()
    assert profiler.count_parameters(model) == params
    assert profiler.count_flops(model, (1, 64)) == flops == 4736


@pytest.mark.parametrize("points, lower, expected", G.PARETO_FIXTURES)
def test_pareto_fixture_answers_are_right(points, lower, expected):
    assert G.reference_pareto(points, lower) == expected


def test_pareto_check_accepts_correct_and_rejects_admit_all():
    from tinytorch.perf.benchmarking import pareto_frontier
    assert G.check_pareto(pareto_frontier) == []
    assert G.check_pareto(G.reference_pareto) == []
    assert len(G.check_pareto(lambda points, lower: list(points))) == len(G.PARETO_FIXTURES)


def test_benchmark_stats_check_accepts_correct_and_rejects_zeroed():
    from tinytorch.perf.benchmarking import BenchmarkResult
    assert G.check_benchmark_stats(BenchmarkResult) == []

    class Zeroed(BenchmarkResult):
        def __post_init__(self):
            super().__post_init__()
            self.mean = self.median = 0.0

    problems = G.check_benchmark_stats(Zeroed)
    assert any(p.startswith("mean=") for p in problems)
    assert any(p.startswith("median=") for p in problems)


def test_latency_stats_consistency():
    from tinytorch.perf.benchmarking import BenchmarkResult
    good = BenchmarkResult("lat", [0.03, 0.031, 0.05])
    assert G.latency_stats_consistent(good)
    good.mean = 0.0
    assert not G.latency_stats_consistent(good)
    assert not G.latency_stats_consistent(BenchmarkResult("lat", [0.0, 0.0]))


class _Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)


class _ToyLM:
    """Token table + a mixing rule over positions; returns [1, S, V] logits."""

    def __init__(self, mode, vocab=11, dim=6, seed=0):
        r = np.random.default_rng(seed)
        self.emb = r.standard_normal((vocab, dim))
        self.head = r.standard_normal((dim, vocab))
        self.mode = mode

    def __call__(self, tokens):
        x = self.emb[tokens.data[0]]                       # [S, D]
        s = np.arange(1, len(x) + 1)[:, None]
        if self.mode == "causal":                          # mean of the prefix
            h = np.cumsum(x, axis=0) / s
        elif self.mode == "leaky":                         # mean of the suffix
            h = np.cumsum(x[::-1], axis=0)[::-1] / s[::-1]
        elif self.mode == "bigram":                        # current token only
            h = x
        else:                                              # "zero"
            h = np.zeros_like(x)
        return _Tensor((h @ self.head)[None])


def _probe(mode):
    tokens = np.random.default_rng(3).integers(0, 11, 16)
    model = _ToyLM(mode)
    return G.logit_spread(model(_Tensor(tokens[None])).data), \
        G.context_probe(model, tokens, 11, _Tensor)


def test_probe_passes_a_causal_model():
    spread, probe = _probe("causal")
    assert G.logits_nondegenerate(spread)
    assert probe["leak"] == 0.0
    assert probe["dependence"] > 1e-2
    assert G.context_probe_passed(probe)


def test_probe_catches_future_leak():
    _, probe = _probe("leaky")
    assert probe["leak"] > G.CAUSAL_LEAK_MAX
    assert not G.context_probe_passed(probe)


def test_probe_catches_context_blind_model():
    spread, probe = _probe("bigram")
    assert G.logits_nondegenerate(spread)       # varied logits alone are not enough
    assert probe["dependence"] == 0.0
    assert not G.context_probe_passed(probe)


def test_spread_catches_constant_model():
    spread, probe = _probe("zero")
    assert spread["vocab"] == spread["positions"] == 0.0
    assert not G.logits_nondegenerate(spread)
    assert not G.logits_nondegenerate(G.logit_spread(np.full((4, 5), np.nan)))


def test_reference_cross_entropy_by_hand_and_against_student():
    # Uniform logits over 4 classes: loss = log 4 whatever the target.
    assert abs(G.reference_cross_entropy(np.zeros((3, 4)), [0, 1, 3]) - np.log(4)) < 1e-12
    # Stable for huge logits: [1000, 0] with target 0 -> log(1 + e^-1000) ~ 0.
    assert G.reference_cross_entropy([[1000.0, 0.0]], [0]) < 1e-12
    assert abs(G.reference_cross_entropy([[1000.0, 0.0]], [1]) - 1000.0) < 1e-9
    from tinytorch.core.losses import CrossEntropyLoss
    from tinytorch.core.tensor import Tensor
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((32, 10)).astype(np.float32) * 3
    targets = rng.integers(0, 10, 32)
    student = float(np.asarray(CrossEntropyLoss()(Tensor(logits), Tensor(targets)).data))
    ref = G.reference_cross_entropy(logits, targets)
    assert G.loss_matches_reference(student, ref)
    assert not G.loss_matches_reference(0.0, ref)
    assert not G.loss_matches_reference(float("nan"), ref)
