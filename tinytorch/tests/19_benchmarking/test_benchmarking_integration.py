#!/usr/bin/env python3
"""
Integration tests for Module 19: Benchmarking.

A benchmark harness is only worth trusting if its timer measures real elapsed
time and its statistics describe the samples it collected. These tests pin both.
"""

import time

import numpy as np

from tinytorch.perf.benchmarking import precise_timer, BenchmarkResult


def test_precise_timer_measures_real_elapsed_time():
    """The timer must track a known sleep, not return a constant."""
    with precise_timer() as t:
        time.sleep(0.05)
    elapsed = t.elapsed if hasattr(t, "elapsed") else float(t)

    assert elapsed >= 0.045, (
        f"precise_timer reported {elapsed:.4f}s for a 50ms sleep; it is not measuring"
    )
    assert elapsed < 1.0, f"precise_timer reported an implausible {elapsed:.4f}s"


def test_precise_timer_distinguishes_two_durations():
    """A constant-returning timer would pass a single-duration check."""
    with precise_timer() as short:
        time.sleep(0.01)
    with precise_timer() as long:
        time.sleep(0.06)

    s = short.elapsed if hasattr(short, "elapsed") else float(short)
    l = long.elapsed if hasattr(long, "elapsed") else float(long)
    assert l > s, f"Timer did not distinguish a 60ms sleep ({l:.4f}s) from a 10ms one ({s:.4f}s)"


def test_benchmark_result_statistics_describe_their_samples():
    """Mean, min and max must come from the values, not from thin air."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = BenchmarkResult(metric_name="latency_ms", values=list(values))

    assert abs(result.mean - 3.0) < 1e-9, f"mean was {result.mean}, expected 3.0"
    assert abs(result.min_val - 1.0) < 1e-9, f"min was {result.min_val}, expected 1.0"
    assert abs(result.max_val - 5.0) < 1e-9, f"max was {result.max_val}, expected 5.0"
    assert result.count == len(values), f"count was {result.count}, expected {len(values)}"

    # A single-sample run has no spread; the interval must collapse, not error.
    single = BenchmarkResult(metric_name="latency_ms", values=[7.0])
    assert abs(single.mean - 7.0) < 1e-9
    assert single.count == 1


def test_confidence_interval_brackets_the_mean_and_narrows_with_samples():
    rng = np.random.default_rng(0)
    few = BenchmarkResult(metric_name="latency_ms",
                          values=list(rng.normal(10.0, 1.0, 8)))
    many = BenchmarkResult(metric_name="latency_ms",
                           values=list(rng.normal(10.0, 1.0, 800)))

    for r in (few, many):
        assert r.ci_lower <= r.mean <= r.ci_upper, (
            f"Confidence interval [{r.ci_lower}, {r.ci_upper}] does not contain the mean {r.mean}"
        )

    assert (many.ci_upper - many.ci_lower) < (few.ci_upper - few.ci_lower), (
        "The confidence interval did not narrow with 100x more samples; "
        "it is probably not computed from the sample count"
    )


if __name__ == "__main__":
    test_precise_timer_measures_real_elapsed_time()
    test_precise_timer_distinguishes_two_durations()
    test_benchmark_result_statistics_describe_their_samples()
    test_confidence_interval_brackets_the_mean_and_narrows_with_samples()
    print("✅ Benchmarking integration tests passed")


def test_small_allocator_peak_keeps_its_measurement_identity():
    from tinytorch.perf.benchmarking import Benchmark
    bench = Benchmark([object()], [], measurement_runs=2)
    bench.profiler.measure_memory = lambda model, shape: {'peak_memory_mb': 0.125}
    bench.profiler.count_parameters = lambda model: 1000000
    result = bench.run_memory_benchmark()['model_0']
    assert result.values == [0.125, 0.125]


def test_failed_probe_cannot_return_chance_accuracy():
    import pytest
    from tinytorch.perf.benchmarking import _simulated_accuracy
    class Broken:
        def forward(self, x):
            raise ValueError('broken forward')
    with pytest.raises(RuntimeError, match='no score was measured'):
        _simulated_accuracy(Broken(), {})


def test_mlperf_rejects_label_misalignment_and_empty_input():
    import pytest
    from tinytorch.perf.benchmarking import MLPerf
    from tinytorch.core.tensor import Tensor
    perf = MLPerf()
    with pytest.raises(ValueError, match='at least one'):
        perf.run_standard_benchmark(object(), 'keyword_spotting', test_inputs=[])
    with pytest.raises(ValueError, match='one class index'):
        perf.run_standard_benchmark(object(), 'keyword_spotting', test_inputs=[Tensor([1])], labels=[])


def test_failed_benchmark_prevents_overall_compliance():
    from tinytorch.perf.benchmarking import MLPerf
    perf = MLPerf()
    success = dict(accuracy=1.0, mean_latency_ms=1, p99_latency_ms=1,
                   throughput_fps=1000, target_accuracy=0.9, target_latency_ms=10,
                   accuracy_met=True, latency_met=True, compliant=True)
    report = perf._compile_report_data({'ok': success, 'failed': {'error': 'shape'}})
    assert report['summary']['total_benchmarks'] == 2
    assert report['summary']['compliance_rate'] == 0.5
    assert not report['summary']['overall_compliant']


def test_real_dataset_cannot_silently_become_synthetic_accuracy():
    import pytest
    from tinytorch.core.tensor import Tensor
    from tinytorch.perf.benchmarking import Benchmark
    class Identity:
        def forward(self, x):
            return x
    dataset = [(Tensor([[1, 2, 3, 4]]), np.array([0]))]
    benchmark = Benchmark([Identity()], [dataset])
    with pytest.raises(ValueError, match='evaluate'):
        benchmark.run_accuracy_benchmark()
    result = benchmark.run_accuracy_benchmark(simulate=True)['model_0']
    assert result.metadata['simulated'] is True


def test_synthetic_labels_cannot_claim_classroom_compliance():
    from tinytorch.core.tensor import Tensor
    from tinytorch.perf.benchmarking import MLPerf
    class Classifier:
        def forward(self, x):
            return Tensor([[0.0, 1.0]])
    perf = MLPerf()
    perf.benchmarks['keyword_spotting']['target_accuracy'] = 0
    perf.benchmarks['keyword_spotting']['max_latency_ms'] = 10000
    result = perf.run_standard_benchmark(Classifier(), 'keyword_spotting', num_runs=1)
    assert result['synthetic_labels'] is True
    assert result['compliant'] is False
