"""Badge truthfulness regressions: absence of failure is not evidence of success."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("volume_status", Path(__file__).with_name("update_volume_status.py"))
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)
VOLUME = status.VOLUMES[0]


def job(state="completed", conclusion="success", format_name="HTML"):
    return {"name": f"Build Linux · Vol I ({format_name})", "status": state, "conclusion": conclusion,
            "steps": [{"name": f"🔨 Build {format_name} (Linux)", "status": state, "conclusion": conclusion}]}


def built_volume():
    return [job(format_name=name) for name in ("HTML", "PDF", "EPUB")]


def shared_gates():
    return [dict(job(), name=name) for name in ("Pre-commit Checks", "Collect Results", "EPUB validate")]


@pytest.mark.parametrize("jobs,message", [
    ([], "not validated"),
    ([job("in_progress", None)], "building"),
    ([job(), job("queued", None)], "building"),
    ([job(conclusion="skipped")], "not validated"),
    ([job(), job(conclusion="skipped")], "not validated"),
    ([job(conclusion="cancelled")], "cancelled"),
    ([job(conclusion="failure")], "failing"),
    ([job(), job(conclusion="timed_out")], "failing"),
    ([job()], "not validated"),
    ([job(), job()], "not validated"),
    (built_volume(), "passing"),
])
def test_green_requires_every_matched_job_to_complete_successfully(jobs, message):
    badge = status.evaluate_volume_status(VOLUME, jobs + shared_gates())
    assert badge["message"] == message
    assert (badge["color"] == "brightgreen") == (message == "passing")


def test_old_green_does_not_hide_missing_current_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(status, "STATUS_DIR", tmp_path)
    (tmp_path / "vol1.json").write_text('{"message":"passing","color":"brightgreen"}')
    assert status.evaluate_volume_status(VOLUME, [dict(job(), name="Build Vol II (HTML)")])["color"] != "brightgreen"


def test_api_failure_is_not_partial_success(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("network unavailable")
    monkeypatch.setattr(status.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="complete workflow job evidence"):
        status.fetch_workflow_jobs("owner/repo", "123", "placeholder")


@pytest.mark.parametrize("gate", ["Pre-commit Checks", "Collect Results", "EPUB validate"])
def test_failed_shared_validation_cannot_leave_green_volume(gate):
    jobs = built_volume() + shared_gates() + [dict(job(conclusion="failure"), name=gate)]
    assert status.evaluate_volume_status(VOLUME, jobs)["message"] == "failing"


@pytest.mark.parametrize("state,conclusion,message", [
    ("queued", None, "building"),
    ("completed", "skipped", "not validated"),
    ("completed", "cancelled", "cancelled"),
])
def test_shared_validation_must_finish_successfully(state, conclusion, message):
    gates = shared_gates()
    gates[-1].update(status=state, conclusion=conclusion)
    assert status.evaluate_volume_status(VOLUME, built_volume() + gates)["message"] == message


def test_missing_shared_gate_is_not_success():
    assert status.evaluate_volume_status(VOLUME, built_volume() + shared_gates()[:-1])["message"] == "not validated"


def test_disabled_matrix_jobs_are_not_build_evidence():
    jobs = built_volume()
    for build in jobs:
        build["steps"][0]["conclusion"] = "skipped"
    assert status.evaluate_volume_status(VOLUME, jobs + shared_gates())["message"] == "not validated"
