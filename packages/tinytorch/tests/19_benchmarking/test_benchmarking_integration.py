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
