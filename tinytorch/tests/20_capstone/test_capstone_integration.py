#!/usr/bin/env python3
"""
Integration tests for Module 20: Capstone.

The capstone's product is a submission: a JSON-serialisable record whose numbers
must come from the benchmark that produced them. These tests check the pipeline
end to end and pin the improvement arithmetic.
"""

import json

import numpy as np

from tinytorch.core.tensor import Tensor
from tinytorch.olympics import SimpleMLP, BenchmarkReport, generate_submission


def _data(n=16, seed=0):
    rng = np.random.default_rng(seed)
    X = Tensor(rng.standard_normal((n, 10)).astype(np.float32))
    y = rng.integers(0, 3, size=n)
    return X, y


def test_benchmark_report_measures_the_model_it_is_given():
    X, y = _data()
    small = BenchmarkReport("small").benchmark_model(SimpleMLP(hidden_size=8), X, y)
    large = BenchmarkReport("large").benchmark_model(SimpleMLP(hidden_size=128), X, y)

    for key in ("accuracy", "latency_ms_mean", "latency_ms_median",
                "model_size_mb", "parameter_count", "throughput_samples_per_sec"):
        assert key in small, f"Benchmark result is missing '{key}'"

    # A report that ignored its model would give both networks the same size.
    assert large["parameter_count"] > small["parameter_count"], (
        "A 128-unit network did not report more parameters than an 8-unit one; "
        f"{large['parameter_count']} vs {small['parameter_count']}"
    )
    assert 0.0 <= small["accuracy"] <= 1.0 or 0.0 <= small["accuracy"] <= 100.0
    assert small["latency_ms_mean"] > 0, "Latency measured as zero or negative"


def test_submission_is_json_serialisable_and_carries_provenance():
    X, y = _data()
    report = BenchmarkReport("baseline")
    report.benchmark_model(SimpleMLP(), X, y)

    submission = generate_submission(report)

    for key in ("baseline", "submission_type", "timestamp", "tinytorch_version"):
        assert key in submission, f"Submission is missing '{key}'"

    # It must survive a round trip through JSON, which is how it is delivered.
    restored = json.loads(json.dumps(submission))
    assert restored["baseline"] == submission["baseline"], (
        "Submission did not survive a JSON round trip unchanged"
    )


def test_improvements_are_computed_from_the_two_reports():
    """An optimized submission must report ratios that match its own numbers."""
    X, y = _data()
    baseline = BenchmarkReport("baseline")
    baseline.benchmark_model(SimpleMLP(hidden_size=128), X, y)

    optimized = BenchmarkReport("optimized")
    optimized.benchmark_model(SimpleMLP(hidden_size=8), X, y)

    submission = generate_submission(
        baseline, optimized, student_name="test", techniques_applied=["smaller_hidden"]
    )

    assert "optimized" in submission, "Optimized submission lacks an 'optimized' section"
    assert "improvements" in submission, "Optimized submission lacks 'improvements'"

    improvements = submission["improvements"]
    assert "compression_ratio" in improvements, "No compression_ratio reported"

    # The smaller network must be reported as smaller, and by exactly the factor
    # its own recorded metrics imply. This is the claim the whole submission rests
    # on, so it is checked against the numbers in the submission itself.
    b_params = submission["baseline"]["metrics"]["parameter_count"]
    o_params = submission["optimized"]["metrics"]["parameter_count"]
    assert o_params < b_params, (
        f"The optimized model reported more parameters ({o_params}) than the baseline ({b_params})"
    )
    expected = b_params / o_params
    assert abs(improvements["compression_ratio"] - expected) < 0.01 * expected, (
        f"compression_ratio {improvements['compression_ratio']:.3f} does not match the "
        f"submission's own parameter counts ({b_params} / {o_params} = {expected:.3f})"
    )

    # Likewise the speedup must follow from the recorded latencies.
    b_lat = submission["baseline"]["metrics"]["latency_ms_median"]
    o_lat = submission["optimized"]["metrics"]["latency_ms_median"]
    assert abs(improvements["speedup"] - b_lat / o_lat) < 0.01 * (b_lat / o_lat), (
        f"speedup {improvements['speedup']:.3f} does not match the recorded latencies "
        f"({b_lat:.4f} / {o_lat:.4f} = {b_lat / o_lat:.3f})"
    )

    assert submission["optimized"]["techniques_applied"] == ["smaller_hidden"], (
        "techniques_applied did not round-trip into the optimized section"
    )
    assert submission["student_name"] == "test", "student_name did not round-trip"


if __name__ == "__main__":
    test_benchmark_report_measures_the_model_it_is_given()
    test_submission_is_json_serialisable_and_carries_provenance()
    test_improvements_are_computed_from_the_two_reports()
    print("✅ Capstone integration tests passed")


def test_report_rejects_broadcast_labels_and_empty_runs():
    import pytest
    X, y = _data(3)
    report = BenchmarkReport()
    with pytest.raises(ValueError, match='one class index'):
        report.benchmark_model(SimpleMLP(), X, y[:, None], num_runs=1)
    with pytest.raises(ValueError, match='positive'):
        report.benchmark_model(SimpleMLP(), X, y, num_runs=0)


def test_optimization_demo_reports_actual_dense_storage(tmp_path, monkeypatch):
    import runpy
    from pathlib import Path
    # This demonstration intentionally stays in the notebook, not the package.
    source = Path(__file__).resolve().parents[2] / "src/20_capstone/20_capstone.py"
    notebook = runpy.run_path(str(source))
    monkeypatch.chdir(tmp_path)
    submission = notebook["run_optimization_workflow_example"]()
    # The teaching quantizer retains reference and rounded arrays, both FP32.
    baseline = submission['baseline']['metrics']['model_size_mb']
    optimized = submission['optimized']['metrics']['model_size_mb']
    assert optimized == 2 * baseline
    assert submission['improvements']['compression_ratio'] == 0.5


def test_schema_checks_optimized_metrics_and_finite_values():
    import pytest
    from tinytorch.olympics import validate_submission_schema
    X, y = _data(2)
    baseline = BenchmarkReport('baseline')
    baseline.benchmark_model(SimpleMLP(), X, y, num_runs=1)
    optimized = BenchmarkReport('optimized')
    optimized.benchmark_model(SimpleMLP(), X, y, num_runs=1)
    submission = generate_submission(baseline, optimized)
    submission['optimized']['metrics']['latency_ms_mean'] = float('inf')
    with pytest.raises(AssertionError, match='finite'):
        validate_submission_schema(submission)
