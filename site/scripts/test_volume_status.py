"""Badge truthfulness regressions: absence of failure is not evidence of success."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("volume_status", Path(__file__).with_name("update_volume_status.py"))
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)
VOLUME = status.VOLUMES[0]


def job(state="completed", conclusion="success"):
    return {"name": "Build Linux · Vol I (HTML)", "status": state, "conclusion": conclusion}


@pytest.mark.parametrize("jobs,message", [
    ([], "not validated"),
    ([job("in_progress", None)], "building"),
    ([job(), job("queued", None)], "building"),
    ([job(conclusion="skipped")], "not validated"),
    ([job(), job(conclusion="skipped")], "not validated"),
    ([job(conclusion="cancelled")], "cancelled"),
    ([job(conclusion="failure")], "failing"),
    ([job(), job(conclusion="timed_out")], "failing"),
    ([job()], "passing"),
    ([job(), job()], "passing"),
])
def test_green_requires_every_matched_job_to_complete_successfully(jobs, message):
    badge = status.evaluate_volume_status(VOLUME, jobs)
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
    jobs = [job(), dict(job(conclusion="failure"), name=gate)]
    assert status.evaluate_volume_status(VOLUME, jobs)["message"] == "failing"
