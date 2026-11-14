"""Integration check against the real public pipeline API."""

import pytest

from mlsysbook_forge import ForgeSession, PipelineCallback, ResultKind, RunContext


def test_callback_does_not_change_solver_outputs(tmp_path):
    mlsysim = pytest.importorskip("mlsysim")
    from mlsysim.engine.pipeline import Pipeline
    from mlsysim.solvers import FailureIntervalModel

    inputs = dict(component_mtbf=mlsysim.Q_(100, "h"), n_components=10, run_duration=mlsysim.Q_(20, "h"))
    plain = Pipeline([FailureIntervalModel()]).run(**inputs)
    context = RunContext("pipeline", "a" * 40, ResultKind.ANALYTICAL)
    with ForgeSession(context, mode="offline", directory=tmp_path) as session:
        instrumented = Pipeline([FailureIntervalModel()])
        instrumented.register_callback(PipelineCallback(session, experiment_id="baseline"))
        observed = instrumented.run(**inputs)
        assert observed == plain
        assert len(session.receipts()) == 1
