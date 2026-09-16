"""Hand counts catch per-sample/batch and shape-propagation errors."""
from pathlib import Path
import runpy
import tracemalloc

import numpy as np
import pytest
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear, Sequential
from tinytorch.core.activations import ReLU
from tinytorch.core.spatial import Conv2d, MaxPool2d, AvgPool2d
# Exercise the source of truth without regenerating the shared package.
_source = runpy.run_path(str(Path(__file__).resolve().parents[2] /
                            'src/14_profiling/14_profiling.py'))
Profiler = _source['Profiler']
analyze_weight_distribution = _source['analyze_weight_distribution']


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


@pytest.mark.parametrize('pool_type', [MaxPool2d, AvgPool2d])
@pytest.mark.parametrize('stride', [2, (2, 2)])
def test_pooling_shape_propagates_to_following_convolution(pool_type, stride):
    pool = pool_type(2)
    # Accept scalar strides and Module 09's normalized pair representation.
    pool.stride = stride
    model = Sequential(Conv2d(1, 2, 3), pool, Conv2d(2, 3, 3))
    # 10 -> 8 after conv, 8 -> 4 after pool, 4 -> 2 after conv.
    conv_flops = 8 * 8 * 9 * 1 * 2 * 2 + 2 * 2 * 9 * 2 * 3 * 2
    pool_estimate = 2 * 8 * 8  # documented one operation per input element
    assert Profiler().count_flops(model, (5, 1, 10, 10)) == conv_flops + pool_estimate


def test_rectangular_pooling_shape_propagates():
    pool = MaxPool2d((2, 3))
    pool.stride = (2, 3)
    model = Sequential(pool, Conv2d(2, 3, 3))
    # Pool maps (8, 12) to (4, 4), then convolution produces (2, 2).
    assert Profiler().count_flops(model, (5, 2, 8, 12)) == 2 * 8 * 12 + 2 * 2 * 9 * 2 * 3 * 2


@pytest.mark.parametrize('caller_tracing', [False, True])
def test_memory_includes_preexisting_parameters_and_actual_output(caller_tracing):
    class WideOutput:
        def __init__(self):
            self.weight = Tensor(np.zeros((2048, 2048), dtype=np.float32))

        def forward(self, x):
            return Tensor(np.zeros((1024, 1024), dtype=np.float32))

    assert not tracemalloc.is_tracing()
    if caller_tracing:
        tracemalloc.start()
    try:
        model = WideOutput()
        # A caller can reset its historical peak without removing tracked
        # parameters: this catches counting those parameters twice.
        if caller_tracing:
            tracemalloc.reset_peak()
        report = Profiler().measure_memory(model, (1, 1))
        assert report['parameter_memory_mb'] == 16
        assert report['activation_memory_mb'] == pytest.approx(4 + 4 / 1024**2)
        assert report['peak_memory_mb'] >= 20
        # Tensor construction can briefly hold both its input array and copy.
        # That allows ~8 MiB new allocation, but not another 16 MiB of weights.
        assert report['peak_memory_mb'] < 26
        assert tracemalloc.is_tracing() == caller_tracing
    finally:
        if caller_tracing:
            tracemalloc.stop()


def test_identity_output_is_not_counted_twice():
    class Identity:
        def forward(self, x):
            return x
    report = Profiler().measure_memory(Identity(), (1, 1024))
    assert report['activation_memory_mb'] == 1024 * 4 / 1024**2


@pytest.mark.parametrize('model', [ReLU(), Sequential()])
def test_weight_distribution_handles_empty_parameter_lists(model):
    assert analyze_weight_distribution(model) == {'error': 'No weights found'}
