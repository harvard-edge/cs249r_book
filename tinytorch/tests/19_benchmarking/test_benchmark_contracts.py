"""Regression checks against Module 19's source, independent of package export age."""
import runpy
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def source():
    root = Path(__file__).resolve().parents[2]
    return runpy.run_path(str(root / 'src/19_benchmarking/19_benchmarking.py'))


@pytest.mark.parametrize('prediction', [
    np.full((1, 10), np.nan), np.full(10, np.inf), np.ones(20),
    np.ones((2, 5)), np.ones((10, 1)), np.array([]),
])
def test_invalid_multiclass_output_cannot_receive_compliance(source, prediction):
    class Classifier:
        def forward(self, x):
            return prediction
    with pytest.raises(ValueError, match='Prediction'):
        source['MLPerf']().run_standard_benchmark(
            Classifier(), 'image_classification', test_inputs=[source['Tensor']([[1.]])],
            labels=np.array([0]))


@pytest.mark.parametrize('prediction', [np.array([np.nan]), np.ones(3), np.array([2.])])
def test_invalid_binary_output_is_rejected(source, prediction):
    with pytest.raises(ValueError):
        source['MLPerf']()._run_accuracy_test(None, [prediction], 'keyword_spotting', 1, np.array([0]))


@pytest.mark.parametrize('labels', [np.array([0.5]), np.array([-1]), np.array([10]), np.array([np.nan])])
def test_labels_must_be_integer_class_indices(source, labels):
    with pytest.raises(ValueError, match='integer class index'):
        source['MLPerf']()._run_accuracy_test(None, [np.arange(10)], 'image_classification', 1, labels)


def test_valid_scores_preserve_class_identity(source):
    perf = source['MLPerf']()
    assert perf._run_accuracy_test(None, [np.arange(10)], 'image_classification', 1, np.array([9])) == 1
    assert perf._run_accuracy_test(None, [np.array([[0., 1.]])], 'keyword_spotting', 1, np.array([1])) == 1
    assert perf._run_accuracy_test(None, [0.75], 'keyword_spotting', 1, np.array([1])) == 1


def test_optimization_comparison_propagates_shape_and_keeps_unnamed_models(source, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    class Model:
        def __init__(self):
            self.layer = source['Linear'](10, 2)
            self.seen = []
        def forward(self, x):
            self.seen.append(x.shape)
            return self.layer(x)
        def parameters(self):
            return self.layer.parameters()
        def evaluate(self, dataset):
            return 0.9
    base, optimized = Model(), Model()
    result = source['analyze_optimization_techniques'](
        base, [optimized], [{}], input_shape=(2, 10))
    assert result['base_model'] == 'model_0'
    assert set(result['optimized_results']['model_1']) == {'latency', 'accuracy', 'memory', 'energy'}
    assert result['improvements']['model_1']['accuracy_retention'] == 1
    assert set(base.seen + optimized.seen) == {(2, 10)}


def test_copied_and_colliding_model_names_preserve_all_measurements(source, tmp_path):
    class Model:
        def __init__(self, name, score):
            self.name, self.score = name, score
        def forward(self, x):
            return x
        def evaluate(self, dataset):
            return self.score
    suite = source['BenchmarkSuite'](
        [Model('same', .1), Model('same', .2), Model('same_1', .3)], [{}], str(tmp_path))
    suite.benchmark.warmup_runs = 0
    suite.benchmark.measurement_runs = 1
    results = suite.run_full_benchmark(input_shape=(1, 4))
    names = suite.benchmark.model_names
    assert len(set(names)) == 3
    assert all(set(metric) == set(names) for metric in results.values())
    assert [results['accuracy'][name].mean for name in names] == [.1, .2, .3]
    assert '(estimated)' in suite.generate_report()


def test_pruning_analysis_reports_dense_storage_and_models_int8_explicitly(source, capsys):
    source['analyze_optimization_tradeoffs']()
    output = capsys.readouterr().out
    lines = output.splitlines()
    baseline = next(line for line in lines if line.startswith('Baseline'))
    pruned = next(line for line in lines if line.startswith('Pruning (70%)'))
    assert baseline.split()[-2] == pruned.split()[-2]
    assert 'modeled packed representation' in output
    assert 'zeros still occupy storage' in output
