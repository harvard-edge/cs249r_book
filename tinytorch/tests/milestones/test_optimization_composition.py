"""Optimizations must execute the same work and report their actual candidates."""
import numpy as np
import pytest

from test_milestones_smoke import MILESTONES_DIR, _import_milestone


def test_generation_replays_every_token_and_restores_uncached_model():
    from tinytorch.core.transformers import GPT
    from tinytorch.core.tensor import Tensor
    generation = _import_milestone(MILESTONES_DIR / '06_2018_mlperf/02_generation_speedup.py')
    model = GPT(vocab_size=11, embed_dim=8, num_layers=2, num_heads=2, max_seq_len=8)
    tokens = np.array([[1, 3, 2, 7, 4, 1, 5, 2]])
    before = model(Tensor(tokens)).data.copy()
    result = generation.compare_cached_replay(model, tokens, repeats=1)
    assert result['tokens_processed'] == tokens.shape[1]
    assert result['max_logit_error'] < 1e-4
    assert result['cache_bytes'] == 2 * 2 * 8 * 8 * 4
    assert result['baseline_ms'] > 0 and result['cached_ms'] > 0
    assert not model._cache_enabled
    np.testing.assert_allclose(model(Tensor(tokens)).data, before)



def test_cached_replay_scope_preserves_ordinary_single_token_forward():
    from tinytorch.core.transformers import GPT
    from tinytorch.core.tensor import Tensor
    from tinytorch.perf.memoization import enable_kv_cache, disable_kv_cache

    generation = _import_milestone(MILESTONES_DIR / '06_2018_mlperf/02_generation_speedup.py')
    model = GPT(vocab_size=11, embed_dim=8, num_layers=2, num_heads=2, max_seq_len=4)
    tokens = np.array([[1, 3, 2, 7]])
    expected = generation.replay_prefixes(model, tokens)
    ordinary_token = Tensor([[5]])
    expected_single = model(ordinary_token).data.copy()
    cache = enable_kv_cache(model)
    try:
        # Repeated timing runs reset the cache, use the prefix, then leave scope.
        for _ in range(2):
            actual = generation.replay_prefixes(model, tokens, cache)
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            assert cache.seq_pos == tokens.shape[1]
            np.testing.assert_allclose(model(ordinary_token).data, expected_single, atol=1e-6)
        # A failed replay must also release the cached execution scope.
        with pytest.raises((ValueError, IndexError)):
            generation.replay_prefixes(model, np.array([[1, 99]]), cache)
        np.testing.assert_allclose(model(ordinary_token).data, expected_single, atol=1e-6)
    finally:
        disable_kv_cache(model)

def test_generation_rejects_different_cached_computation(monkeypatch):
    from tinytorch.core.transformers import GPT
    generation = _import_milestone(MILESTONES_DIR / '06_2018_mlperf/02_generation_speedup.py')
    model = GPT(vocab_size=11, embed_dim=8, num_layers=1, num_heads=2, max_seq_len=4)
    replay = generation.replay_prefixes

    def wrong_cached_logits(model, tokens, cache=None):
        result = replay(model, tokens, cache)
        return result + 1 if cache is not None else result

    monkeypatch.setattr(generation, 'replay_prefixes', wrong_cached_logits)
    with pytest.raises(AssertionError):
        generation.compare_cached_replay(model, np.array([[1, 3]]), repeats=1)
    assert not model._cache_enabled


def test_optimization_candidates_are_independent_and_benchmarked():
    from tinytorch.core.tensor import Tensor
    from tinytorch.perf.quantization import Quantizer
    from tinytorch.perf.compression import Compressor
    from tinytorch.perf.benchmarking import Benchmark
    networks = _import_milestone(MILESTONES_DIR / '06_2018_mlperf/networks.py')
    olympics = _import_milestone(MILESTONES_DIR / '06_2018_mlperf/01_optimization_olympics.py')
    model = networks.DigitMLP()
    x = Tensor(np.random.default_rng(3).normal(size=(5, 64)))
    labels = np.arange(5)
    before = [p.data.copy() for p in model.parameters()]
    size = sum(p.data.nbytes for p in model.parameters())
    quant = olympics.step_2_quantize(model, size, 0, x, labels, Quantizer, networks.DigitMLP)
    prune = olympics.step_3_prune(model, 0, x, labels, Compressor, networks.DigitMLP)
    for candidate in [quant, prune]:
        assert candidate['model'] is not model
        assert candidate['actual_bytes'] == size
        assert all(not np.shares_memory(a.data, b.data)
                   for a, b in zip(model.parameters(), candidate['model'].parameters()))
        measured = olympics.step_6_benchmark(candidate['model'], x, labels, -999, Benchmark)
        expected = np.mean(np.argmax(candidate['model'](x).data, axis=1) == labels) * 100
        assert measured['accuracy'] == expected
        assert measured['mean_latency'] > 0
    for actual, original in zip(model.parameters(), before):
        np.testing.assert_array_equal(actual.data, original)
    assert any(not np.array_equal(a.data, b.data)
               for a, b in zip(model.parameters(), quant['model'].parameters()))
    assert any(not np.array_equal(a.data, b.data)
               for a, b in zip(model.parameters(), prune['model'].parameters()))


def test_reusable_transformer_residual_reaches_both_branches():
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.optimizers import SGD
    networks = _import_milestone(MILESTONES_DIR / '06_2018_mlperf/networks.py')
    model = networks.MinimalTransformer(vocab_size=11, embed_dim=8, num_heads=2, seq_len=4)
    optimizer = SGD(model.parameters(), lr=0.01)
    output = model(Tensor([[1, 3, 2]]))
    optimizer.zero_grad()
    (output * output).sum().backward()
    for layer in [model.token_embed, model.attention, model.ff1, model.ff2]:
        assert all(p.grad is not None and np.all(np.isfinite(p.grad)) for p in layer.parameters())
        assert any(np.any(p.grad != 0) for p in layer.parameters())


def test_dataset_manager_handles_nested_paths_and_odd_sample_counts(tmp_path):
    data = _import_milestone(MILESTONES_DIR / 'data_manager.py')
    manager = data.DatasetManager(tmp_path / 'nested' / 'datasets')
    for count in [1, 5, 6]:
        x, y = manager.get_perceptron_data(count)
        assert x.shape == (count, 2)
        assert y.shape == (count,)
        assert np.isfinite(x).all()
