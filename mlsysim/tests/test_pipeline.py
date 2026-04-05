"""
Unit tests for mlsysim.core.pipeline — the Pipeline composer.

Tests construction, validation, explain(), run(), and repr.
"""

import pytest

from mlsysim.core.pipeline import Pipeline, CompositionError
from mlsysim.core.solver import SingleNodeModel
from mlsysim.hardware.registry import Hardware
from mlsysim.models.registry import Models


class TestPipelineConstruction:
    """Verify Pipeline construction and validation."""

    def test_empty_pipeline_raises(self):
        with pytest.raises(ValueError):
            Pipeline([])

    def test_single_resolver_has_len_1(self):
        pipe = Pipeline([SingleNodeModel()])
        assert len(pipe) == 1

    def test_multiple_resolvers(self, single_node_solver, serving_solver):
        pipe = Pipeline([single_node_solver, serving_solver])
        assert len(pipe) == 2


class TestPipelineExplain:
    """Verify explain() output."""

    def test_explain_returns_nonempty_string(self, single_node_solver):
        pipe = Pipeline([single_node_solver])
        result = pipe.explain()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_explain_contains_resolver_name(self, single_node_solver):
        pipe = Pipeline([single_node_solver])
        result = pipe.explain()
        assert "SingleNodeModel" in result


class TestPipelineRun:
    """Verify run() execution."""

    def test_run_single_node_returns_dict_with_key(self):
        pipe = Pipeline([SingleNodeModel()])
        resnet = Models.ResNet50
        a100 = Hardware.A100
        result = pipe.run(model=resnet, hardware=a100, batch_size=32)
        assert "SingleNodeModel" in result

    def test_run_result_contains_performance(self):
        pipe = Pipeline([SingleNodeModel()])
        resnet = Models.ResNet50
        a100 = Hardware.A100
        result = pipe.run(model=resnet, hardware=a100, batch_size=32)
        perf = result["SingleNodeModel"]
        assert perf.feasible is True


class TestPipelineRepr:
    """Verify __repr__ output."""

    def test_repr_contains_resolver_names(self, single_node_solver, serving_solver):
        pipe = Pipeline([single_node_solver, serving_solver])
        r = repr(pipe)
        assert "SingleNodeModel" in r
        assert "ServingModel" in r

    def test_repr_starts_with_pipeline(self, single_node_solver):
        pipe = Pipeline([single_node_solver])
        assert repr(pipe).startswith("Pipeline(")
