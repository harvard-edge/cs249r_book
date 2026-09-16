"""Graded dropout cells must be repeatable without changing a learner's RNG."""

import copy
from pathlib import Path
import runpy

import numpy as np
import pytest


@pytest.fixture
def source_module():
    source = Path(__file__).resolve().parents[2] / 'src/03_layers/03_layers.py'
    return runpy.run_path(str(source))


@pytest.mark.parametrize('test_name', [
    'test_unit_generate_dropout_mask',
    'test_unit_dropout_layer',
])
def test_dropout_graded_cell_repeats_without_consuming_rng(source_module, test_name):
    graded_test = source_module[test_name]
    namespace = graded_test.__globals__
    learner_rng = np.random.default_rng(19)
    namespace['rng'] = learner_rng

    # Simulate arbitrary prior notebook activity between repeated cell executions.
    for _ in range(200):
        learner_rng.random(1000)
        before = copy.deepcopy(learner_rng.bit_generator.state)
        graded_test()
        assert namespace['rng'] is learner_rng
        assert learner_rng.bit_generator.state == before


@pytest.mark.parametrize('test_name', [
    'test_unit_generate_dropout_mask',
    'test_unit_dropout_layer',
])
def test_dropout_graded_cell_restores_rng_after_failure(source_module, test_name, monkeypatch):
    graded_test = source_module[test_name]
    namespace = graded_test.__globals__
    learner_rng = namespace['rng']
    before = copy.deepcopy(learner_rng.bit_generator.state)

    def broken_mask(self, shape):
        raise AssertionError('student implementation failed')

    monkeypatch.setattr(source_module['Dropout'], '_generate_dropout_mask', broken_mask)
    with pytest.raises(AssertionError, match='student implementation failed'):
        graded_test()
    assert namespace['rng'] is learner_rng
    assert learner_rng.bit_generator.state == before
