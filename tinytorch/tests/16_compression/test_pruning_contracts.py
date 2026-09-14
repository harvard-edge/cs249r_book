"""Pruning boundaries and pipeline reports must match actual mutations."""
import numpy as np
import pytest
from tinytorch.core.layers import Linear, Sequential
from tinytorch.perf.compression import magnitude_prune, structured_prune, compress_model

@pytest.mark.parametrize('ratio,count', [(0, 0), (.5, 4), (1, 8)])
def test_equal_magnitudes_and_endpoints(ratio, count):
    layer = Linear(2, 4, bias=False)
    layer.weight.data[:] = 1
    magnitude_prune(Sequential(layer), ratio)
    assert np.count_nonzero(layer.weight.data == 0) == count


def test_structured_full_pruning_keeps_classifier_head():
    hidden, head = Linear(2, 4), Linear(4, 2)
    before = head.weight.data.copy()
    structured_prune(Sequential(hidden, head), 1)
    assert not np.any(hidden.weight.data) and not np.any(hidden.bias.data)
    np.testing.assert_array_equal(head.weight.data, before)


@pytest.mark.parametrize('config', [{'low_rank': .5}, {'magnitude_prune': .5, 'structured_prune': 2}, {'typo': .2}])
def test_invalid_pipeline_rejects_before_mutation(config):
    layer = Linear(2, 4)
    before = layer.weight.data.copy()
    with pytest.raises(ValueError):
        compress_model(Sequential(layer), config)
    np.testing.assert_array_equal(layer.weight.data, before)
