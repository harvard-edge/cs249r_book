import json
from dataclasses import dataclass
import sqlite3
import sys

import pytest

from mlsysbook_forge import ForgeSession, PipelineCallback, ResultKind, RunContext
from mlsysbook_forge.adapters import publish_capture


def context(**kwargs):
    return RunContext("v1-training", "a" * 40, ResultKind.ANALYTICAL, **kwargs)


def test_disabled_has_no_files_or_sdk(tmp_path):
    before = set(sys.modules)
    with ForgeSession(context(), directory=tmp_path) as session:
        assert session.capture({"x": 1}, action_id="a")["status"] == "disabled"
        session.log_artifact(tmp_path / "missing.json", action_id="b")
    assert list(tmp_path.iterdir()) == []
    assert not ({"wandb", "weave"} & (set(sys.modules) - before))


def test_capture_survives_restart_and_deduplicates(tmp_path):
    kwargs = dict(mode="offline", directory=tmp_path, session_id="lesson")
    with ForgeSession(context(), **kwargs) as session:
        first = publish_capture(session, {"latency": {"value": 4, "unit": "ms"}})
    with ForgeSession(context(), **kwargs) as session:
        second = publish_capture(session, {"latency": {"value": 4, "unit": "ms"}})
        assert first["id"] == second["id"]
        assert second["duplicate"]
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_private_fields_filtered_recursively_without_mutating_input(tmp_path):
    data = {"student_id": "alice", "nested": {"interpretation": "private", "WANDB_API_KEY": "secret", "latency": 4}}
    with ForgeSession(context(config={"password": "secret", "batch_size": 8}), mode="offline", directory=tmp_path) as session:
        session.capture(data, action_id="a")
    output = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert output["payload"] == {"nested": {"latency": 4}}
    assert output["context"]["config"] == {"batch_size": 8}
    assert data["student_id"] == "alice"


def test_json_backed_capture_uses_public_export_before_filtering(tmp_path):
    @dataclass(frozen=True)
    class Capture:
        _snapshot_json: str

        def to_dict(self):
            return json.loads(self._snapshot_json)

    capture = Capture('{"result":{"latency":4},"decision":"private reflection"}')
    with ForgeSession(context(), mode="offline", directory=tmp_path) as session:
        publish_capture(session, capture)
    output = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert output["payload"] == {"result": {"latency": 4}}


def test_unconfirmed_delivery_is_not_retried(tmp_path):
    calls = []

    class Backend:
        def __init__(self, **kwargs):
            pass

        def publish(self, *args):
            calls.append(args)
            raise TimeoutError("Bearer secret-should-not-be-logged")

        def finish(self):
            pass

    kwargs = dict(mode="online", directory=tmp_path, session_id="stable", entity="owner", project="private", backend_factory=Backend)
    with ForgeSession(context(), **kwargs) as session:
        with pytest.warns(RuntimeWarning, match="TimeoutError"):
            result = session.capture({"value": 5}, action_id="same")
        assert result["status"] == "unconfirmed"
    with ForgeSession(context(), **kwargs) as session:
        assert session.capture({"value": 5}, action_id="same")["duplicate"]
    assert len(calls) == 1
    assert "secret-should" not in next(tmp_path.rglob("*.json")).read_text()


def test_changed_action_or_context_is_rejected(tmp_path):
    kwargs = dict(mode="offline", directory=tmp_path, session_id="stable")
    with ForgeSession(context(), **kwargs) as session:
        session.capture({"x": 1}, action_id="a")
        with pytest.raises(ValueError, match="different evidence"):
            session.capture({"x": 2}, action_id="a")
    with ForgeSession(context(seed=2), **kwargs) as session:
        with pytest.raises(ValueError, match="different experiment"):
            session.capture({"x": 1}, action_id="a")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_invalid_evidence_rejected_before_delivery(tmp_path, value):
    with ForgeSession(context(), mode="offline", directory=tmp_path) as session:
        with pytest.raises(ValueError):
            session.capture({"value": value}, action_id="a")


def test_pipeline_callback_preserves_result_and_stage_identity(tmp_path):
    result = {"value": 3, "unit": "ms"}
    with ForgeSession(context(), mode="offline", directory=tmp_path) as session:
        callback = PipelineCallback(session, experiment_id="baseline")
        callback.on_stage_end("Solver", result)
        callback.on_stage_end("Solver", result)
        assert len(session.receipts()) == 2
    assert result == {"value": 3, "unit": "ms"}


def test_pipeline_survives_local_evidence_storage_failure():
    class BrokenStorage:
        def capture(self, *args, **kwargs):
            raise OSError("disk unavailable")

    with pytest.warns(RuntimeWarning, match="lesson continues"):
        PipelineCallback(BrokenStorage(), experiment_id="x").on_stage_end("solver", {"x": 1})


def test_metrics_and_artifacts(tmp_path):
    artifact = tmp_path / "result.json"
    artifact.write_text('{"accuracy": 0.8, "email": "private@example.com"}')
    with ForgeSession(context(), mode="offline", directory=tmp_path / "runs") as session:
        session.log_metrics({"accuracy": 0.8}, step=0, action_id="epoch-0")
        session.log_artifact(artifact, action_id="report")
        with pytest.raises(ValueError):
            session.log_metrics({"bad": True}, step=0, action_id="bad")
        assert all(r["status"] == "saved" for r in session.receipts())


def test_interrupted_action_stays_visible_without_resending(tmp_path):
    with ForgeSession(context(), mode="offline", directory=tmp_path, session_id="crash") as session:
        receipt = session.capture({"x": 1}, action_id="a")
    with sqlite3.connect(tmp_path / "crash" / "journal.sqlite3") as db:
        db.execute("UPDATE events SET status='sending'")
    with ForgeSession(context(), mode="offline", directory=tmp_path, session_id="crash") as session:
        assert session.capture({"x": 1}, action_id="a")["status"] == "sending"
        assert session.receipts()[0]["id"] == receipt["id"]
