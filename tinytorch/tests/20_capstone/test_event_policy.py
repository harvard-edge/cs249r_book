"""Classroom event policy belongs to the capstone, after measurement."""
import copy

import numpy as np
import pytest

from tinytorch import olympics
from tinytorch.core.tensor import Tensor
from tinytorch.perf import benchmarking


@pytest.mark.parametrize('event,accuracy,latency,size,expected', [
    ('latency_sprint', 0.85, 99, 9, True),
    ('latency_sprint', 0.849, 1, 1, False),
    ('memory_challenge', 0.85, 99, 9, True),
    ('memory_challenge', 0.849, 1, 1, False),
    ('accuracy_contest', 0.5, 99.999, 9.999, True),
    ('accuracy_contest', 0.99, 100, 9, False),
    ('accuracy_contest', 0.99, 99, 10, False),
    ('extreme_push', 0.80, 99, 9, True),
    ('extreme_push', 0.799, 1, 1, False),
    ('all_around', 0.0, 1000, 1000, True),
])
def test_event_boundaries(event, accuracy, latency, size, expected):
    metrics = dict(accuracy=accuracy, latency_ms_median=latency,
                   latency_ms_mean=500, model_size_mb=size)
    original = copy.deepcopy(metrics)
    assert olympics.qualifies_event(metrics, olympics.OlympicEvent(event)) is expected
    assert metrics == original  # Eligibility never rewrites the measurements.


@pytest.mark.parametrize('key,value', [
    ('accuracy', -0.1), ('accuracy', 1.1), ('accuracy', np.nan),
    ('latency_ms_median', 0), ('latency_ms_median', -1),
    ('latency_ms_median', np.inf), ('model_size_mb', 0),
    ('model_size_mb', -1), ('model_size_mb', np.nan),
])
def test_invalid_measurements_cannot_qualify(key, value):
    metrics = dict(accuracy=0.9, latency_ms_median=1, model_size_mb=1)
    metrics[key] = value
    with pytest.raises(ValueError):
        olympics.qualifies_event(metrics, olympics.OlympicEvent.ALL_AROUND)


def test_missing_measurement_and_unknown_event_fail():
    metrics = dict(accuracy=0.9, latency_ms_median=1, model_size_mb=1)
    for key in metrics:
        with pytest.raises(KeyError):
            olympics.qualifies_event({k: v for k, v in metrics.items() if k != key},
                                     olympics.OlympicEvent.LATENCY_SPRINT)
    with pytest.raises(ValueError):
        olympics.qualifies_event(metrics, 'misspelled_event')


def test_benchmarking_exports_measurements_not_capstone_events():
    assert callable(benchmarking.precise_timer)
    assert not hasattr(benchmarking, 'OlympicEvent')
    assert not hasattr(benchmarking, 'qualifies_event')
    assert len(olympics.OlympicEvent) == 5


def test_real_report_can_be_valid_without_qualifying():
    class WrongClassifier:
        def parameters(self):
            return [Tensor([1.0])]

        def count_parameters(self):
            return 1

        def forward(self, x):
            return Tensor(np.tile([1.0, 0.0], (x.shape[0], 1)))

    report = olympics.BenchmarkReport('always_wrong')
    report.benchmark_model(WrongClassifier(), Tensor(np.zeros((4, 2))),
                           np.ones(4, dtype=int), num_runs=2)
    submission = olympics.generate_submission(report)
    assert olympics.validate_submission_schema(submission)
    assert not olympics.qualifies_event(submission['baseline']['metrics'],
                                       olympics.OlympicEvent.LATENCY_SPRINT)
    assert 'event' not in submission  # Existing submission format stays intact.
