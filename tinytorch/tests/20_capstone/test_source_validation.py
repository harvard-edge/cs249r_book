"""Source-direct checks for invalid capstone measurements and comparisons."""
from pathlib import Path
import runpy

import numpy as np
import pytest


@pytest.fixture(scope="module")
def capstone():
    return runpy.run_path(str(Path(__file__).resolve().parents[2] /
                              "src/20_capstone/20_capstone.py"))


@pytest.mark.parametrize("scores,labels", [
    (np.full((3, 2), np.nan), [0, 0, 0]),
    (np.full((3, 2), np.inf), [0, 0, 0]),
    (np.zeros(3), [0, 0, 0]),
    (np.zeros((1, 2)), [0, 0, 0]),
    (np.zeros((3, 0)), [0, 0, 0]),
    (np.zeros((3, 2)), [0, -1, 0]),
    (np.zeros((3, 2)), [0, 2, 0]),
    (np.zeros((3, 2)), [0.0, 1.0, 0.0]),
    (np.zeros((3, 2)), [True, False, True]),
    (np.zeros((3, 2)), [0, 1]),
])
def test_invalid_classifier_cannot_produce_report(capstone, scores, labels):
    Tensor = capstone['Tensor']

    class Classifier:
        def parameters(self):
            return [Tensor([1.0])]

        def forward(self, x):
            return Tensor(scores)

    report = capstone['BenchmarkReport']('invalid')
    with pytest.raises(ValueError):
        report.benchmark_model(Classifier(), Tensor(np.zeros((3, 2))), labels, num_runs=1)
    assert report.metrics == {}


def submission(capstone, median=True):
    baseline = capstone['BenchmarkReport']('baseline')
    baseline.metrics = dict(parameter_count=10, model_size_mb=2.0, accuracy=0.8,
                            latency_ms_mean=4.0, latency_ms_std=0.0,
                            throughput_samples_per_sec=100.0)
    if median:
        baseline.metrics['latency_ms_median'] = 3.0
    optimized = capstone['BenchmarkReport']('optimized')
    optimized.metrics = dict(baseline.metrics, model_size_mb=1.0,
                             latency_ms_mean=2.0, accuracy=0.75)
    if median:
        optimized.metrics['latency_ms_median'] = 1.0
    return capstone['generate_submission'](baseline, optimized)


@pytest.mark.parametrize('median', [False, True])
def test_valid_comparison_accepts_mean_fallback_and_median(capstone, median):
    value = submission(capstone, median)
    assert capstone['validate_submission_schema'](value)
    del value['improvements']  # Comparisons need not supply derived metrics.
    assert capstone['validate_submission_schema'](value)


@pytest.mark.parametrize('section', ['baseline', 'optimized'])
@pytest.mark.parametrize('metric,value', [
    ('latency_ms_median', np.nan), ('latency_ms_median', 0.0),
    ('latency_ms_std', -1.0), ('latency_ms_std', np.inf),
    ('throughput_samples_per_sec', 0.0), ('throughput_samples_per_sec', 'fast'),
    ('accuracy', True), ('parameter_count', 1.5),
])
def test_invalid_optional_or_mistyped_metrics_rejected(capstone, section, metric, value):
    data = submission(capstone)
    data[section]['metrics'][metric] = value
    with pytest.raises(AssertionError):
        capstone['validate_submission_schema'](data)


@pytest.mark.parametrize('metric', ['speedup', 'compression_ratio', 'accuracy_delta'])
@pytest.mark.parametrize('value', ['fabricated', float('nan'), True, 123.0, None])
def test_invalid_or_inconsistent_improvements_rejected(capstone, metric, value):
    data = submission(capstone)
    data['improvements'][metric] = value
    with pytest.raises(AssertionError):
        capstone['validate_submission_schema'](data)


def test_comparison_requires_underlying_report(capstone):
    data = submission(capstone)
    del data['optimized']
    with pytest.raises(AssertionError):
        capstone['validate_submission_schema'](data)


def test_finite_classifier_reports_real_accuracy(capstone):
    Tensor = capstone['Tensor']

    class Classifier:
        def parameters(self):
            return [Tensor([1.0])]

        def forward(self, x):
            return Tensor(np.tile([1.0, 0.0], (x.shape[0], 1)))

    report = capstone['BenchmarkReport']('finite')
    metrics = report.benchmark_model(Classifier(), Tensor(np.zeros((3, 2))),
                                     [0, 1, 0], num_runs=2)
    assert metrics['accuracy'] == pytest.approx(2 / 3)
    assert capstone['validate_submission_schema'](capstone['generate_submission'](report))
