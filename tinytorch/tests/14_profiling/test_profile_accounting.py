"""Hand counts catch per-sample/batch and shape-propagation errors."""
import numpy as np
import pytest
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear, Sequential
from tinytorch.core.activations import ReLU
from tinytorch.core.spatial import Conv2d
from tinytorch.perf.profiling import Profiler


def test_sequential_per_sample_count_does_not_include_batch():
    model = Sequential(Linear(4, 3), ReLU(), Linear(3, 2))
    for batch in [1, 7]:
        assert Profiler().count_flops(model, (batch, 4)) == 24 + 3 + 12


def test_linear_counts_each_sequence_position():
    assert Profiler().count_flops(Linear(4, 3), (7, 5, 4)) == 5 * 24


def test_convolution_shape_propagates_to_second_layer():
    model = Sequential(Conv2d(1, 2, 3), Conv2d(2, 3, 3))
    expected = 6 * 6 * 3 * 3 * 1 * 2 * 2 + 4 * 4 * 3 * 3 * 2 * 3 * 2
    assert Profiler().count_flops(model, (5, 1, 8, 8)) == expected


def test_layer_and_training_throughput_use_whole_batch(monkeypatch):
    profiler = Profiler()
    monkeypatch.setattr(profiler, 'measure_latency', lambda *args, **kwargs: 2.)
    model, x = Linear(4, 3), Tensor(np.ones((7, 4)))
    report = profiler.profile_layer(model, x.shape)
    assert report['gflops_per_second'] == pytest.approx(7 * 24 / 1e9 / .002)
    report = profiler.profile_backward_pass(model, x)
    assert report['total_gflops_per_second'] == pytest.approx(7 * 72 / 1e9 / .006)


def test_empty_timing_sample_is_rejected():
    with pytest.raises(ValueError):
        Profiler().measure_latency(Linear(4, 3), Tensor(np.ones((1, 4))), iterations=0)


def test_memory_trace_lifecycle_on_forward_failure():
    import tracemalloc
    class FailingModel:
        def forward(self, x):
            raise ValueError('bad input')
    assert not tracemalloc.is_tracing()
    with pytest.raises(ValueError):
        Profiler().measure_memory(FailingModel(), (1, 2))
    assert not tracemalloc.is_tracing()
    tracemalloc.start()
    try:
        with pytest.raises(ValueError):
            Profiler().measure_memory(FailingModel(), (1, 2))
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()
