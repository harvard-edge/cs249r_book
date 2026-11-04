"""Update-equivalence regressions, independent of performance estimates."""
import pytest
from mlsysim.labs.experiments.v2_05_experiments import data_parallel_update, update_equivalence_fixture


def test_count_weighted_reduction_matches_direct_global_gradient_under_repartition():
    items = ((1., 1.), (2., 1.), (3., 2.), (4., 3.))
    direct_gradient = sum((0.5 * x - y) * x for x, y in items) / len(items)
    for shards in ((items,), (items[:1], items[1:]), ((), items), tuple((p,) for p in items)):
        result = data_parallel_update(shards)
        assert result["reduced_gradient"] == pytest.approx(direct_gradient)
        assert result["next_weight"] == pytest.approx(0.5 - 0.1 * direct_gradient)
        assert result["equivalent"]


def test_equal_worker_mean_breaks_only_after_unequal_partition():
    equal = update_equivalence_fixture("equal", "worker-mean")
    unequal = update_equivalence_fixture("unequal", "worker-mean")
    assert equal["items"] == unequal["items"]
    assert equal["reference_gradient"] == unequal["reference_gradient"]
    assert equal["equivalent"]
    assert not unequal["equivalent"]
    assert unequal["update_error"] > 0
    assert update_equivalence_fixture("unequal", "valid-count")["equivalent"]


def test_allreduce_sum_is_not_already_a_mean():
    result = update_equivalence_fixture("equal", "unnormalized-sum")
    assert result["reduced_gradient"] == pytest.approx(result["reference_gradient"] * result["global_valid_count"])
    assert not result["equivalent"]


@pytest.mark.parametrize("shards,kwargs", [([], {}), ([[]], {}), ([[(float('nan'), 1)]], {}), ([[(1, 1)]], {"reduction": "magic"}), ([[], [(1, 1)]], {"reduction": "worker-mean"})])
def test_invalid_contract_inputs_are_rejected(shards, kwargs):
    with pytest.raises(ValueError):
        data_parallel_update(shards, **kwargs)
