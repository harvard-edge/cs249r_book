"""Constraint grammar fails closed before design evaluation begins."""

import pytest

from mlsysim.engine.dse import DSE


@pytest.mark.parametrize(
    "expression",
    [
        "cost != 1",
        "cost = 1",
        "cost <> 1",
        "cost <= nonsense",
        "cost <= nan",
        "cost <= inf",
        "cost >= -inf",
        "cost <= 1e999",
        "cost<=1",
        "cost <",
        "cost < 1 extra",
        "",
    ],
)
@pytest.mark.parametrize("values", [[1], []])
def test_invalid_constraint_fails_before_evaluation(expression, values):
    calls = []
    search = DSE({"x": values}, "minimize: cost", ["cost >= 0", expression])
    with pytest.raises(ValueError, match="[Cc]onstraint"):
        search.search(lambda p: calls.append(p) or {"cost": p["x"]})
    assert calls == []


@pytest.mark.parametrize(
    "expression, accepted",
    [
        ("cost < 2", [1]),
        ("cost <= 2", [1, 2]),
        ("cost > 2", [3]),
        ("cost >= 2", [2, 3]),
        ("cost == 2", [2]),
    ],
)
@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_supported_operators_preserve_boundary_and_ranking(expression, accepted, direction):
    search = DSE({"x": [1, 2, 3]}, f"{direction}: cost", [expression])
    result = search.search(lambda p: {"cost": p["x"]})
    expected = accepted if direction == "minimize" else accepted[::-1]
    assert [row["params"]["x"] for row in result["top_candidates"]] == expected
    assert result["best_params"] == {"x": expected[0]}


def test_multiple_constraints_nested_metrics_and_scientific_thresholds():
    result = DSE(
        {"x": [-3, -2, -1, 0, 1]},
        "maximize: metrics.cost",
        ["metrics.cost >= -2e0", "metrics.cost < +.0"],
    ).search(lambda p: {"metrics": {"cost": p["x"]}})
    assert [row["params"]["x"] for row in result["top_candidates"]] == [-1, -2]


def test_valid_constraints_with_no_match_keep_no_configuration_error():
    with pytest.raises(ValueError, match="No valid configurations"):
        DSE({"x": [1]}, "minimize: cost", ["cost > 1"]).search(lambda p: {"cost": p["x"]})


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("path", ["typo", "metrics.typo", "typo.cost"])
@pytest.mark.parametrize("role", ["objective", "constraint"])
def test_missing_metric_raises_on_first_feasible_result(direction, path, role):
    calls = []
    objective = path if role == "objective" else "metrics.cost"
    constraints = [f"{path} >= 0"] if role == "constraint" else []

    def evaluate(params):
        calls.append(params)
        return {"metrics": {"cost": params["x"]}}

    with pytest.raises(ValueError, match=path):
        DSE({"x": [1, 2]}, f"{direction}: {objective}", constraints).search(evaluate)
    assert calls == [{"x": 1}]


def test_missing_object_attribute_is_not_an_infinite_metric():
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="metrics.typo"):
        DSE({"x": [1]}, "minimize: metrics.typo").search(
            lambda p: SimpleNamespace(metrics=SimpleNamespace(cost=p["x"]))
        )


def test_evaluator_failures_and_known_infeasibility_are_still_skipped():
    from types import SimpleNamespace

    def evaluate(params):
        if params["x"] == 1:
            raise ValueError("candidate cannot be evaluated")
        if params["x"] == 2:
            return SimpleNamespace(feasibility=SimpleNamespace(status="FAIL"))
        return SimpleNamespace(feasibility=SimpleNamespace(status="PASS"), cost=3)

    result = DSE({"x": [1, 2, 3]}, "minimize: cost", ["cost >= 0"]).search(evaluate)
    assert result["best_params"] == {"x": 3}
