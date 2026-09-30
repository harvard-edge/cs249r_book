"""
Milestone completion is recorded per part, never on a partial run.

2026-09-29 release audit (B1): `tito milestone run 03` recorded Milestone 03
after only the XOR part, `run 06 --part 2` recorded 06 without Part 1 ever
running, `--skip-checks` still recorded completion, and `tito milestone test`
exited 0 with requirements missing. These tests pin the fixed contract:

- a milestone completes only when every required part has a recorded pass;
- the non-interactive default runs every required part, in order;
- `--part N` runs and records only Part N;
- a failed part is recorded as failed, never as a pass;
- `--skip-checks` records nothing;
- legacy milestones.json files load without crashing or inventing passes;
- progress sync reports only milestones complete under that rule.

Every test runs in a throwaway project (tmp_path) with stub milestone
scripts, so the real .tito/ ledger of the checkout is never touched.
Assertions read the raw milestones.json so they judge the file students
and the sync payload actually see.
"""

import io
import json
from argparse import Namespace
from pathlib import Path

import pytest
from rich.console import Console

import tito.commands.milestone as milestone_mod
from tito.commands.milestone import MILESTONE_SCRIPTS, MilestoneCommand
from tito.core.config import CLIConfig

STUB = """\
import json, sys
from pathlib import Path
me = {path!r}
with open("ran.log", "a", encoding="utf-8") as f:
    f.write(me + "\\n")
codes = json.loads(Path("outcomes.json").read_text()) if Path("outcomes.json").exists() else {{}}
print("STUB RAN " + me)
sys.exit(codes.get(me, 0))
"""


# The contract, stated independently of the registry under test.
REQUIRED_PARTS = {"01": [1], "02": [1], "03": [1, 2], "04": [1], "05": [1, 2], "06": [1, 2], "07": [1]}


def _script_paths(milestone_id):
    m = MILESTONE_SCRIPTS[milestone_id]
    return [s["script"] for s in m["scripts"]] if "scripts" in m else [m["script"]]


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Isolated project: all modules complete, stub scripts, no real .tito."""
    (tmp_path / "pyproject.toml").write_text("[tool.tinytorch]\n", encoding="utf-8")
    tito = tmp_path / ".tito"
    tito.mkdir()
    (tito / "progress.json").write_text(
        json.dumps({"completed_modules": [f"{i:02d}" for i in range(1, 21)]}), encoding="utf-8"
    )
    for mid in MILESTONE_SCRIPTS:
        for rel in _script_paths(mid):
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(STUB.format(path=rel), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TINYTORCH_NON_INTERACTIVE", "1")
    monkeypatch.setenv("TITO_NO_SYNC", "1")
    # Stub scripts stand in for the student's package; there is nothing to import.
    monkeypatch.setattr(milestone_mod, "_validate_required_exports", lambda mods: [])
    monkeypatch.setattr(MilestoneCommand, "_offer_progress_sync", lambda self, mid, name: None)
    return tmp_path


def _cmd(root):
    cmd = MilestoneCommand(CLIConfig.from_project_root(root))
    out = io.StringIO()
    cmd.console = Console(file=out, force_terminal=False, color_system=None, width=200)
    return cmd, out


def run(root, milestone_id, part=None, run_all=False, skip_checks=False):
    cmd, out = _cmd(root)
    args = Namespace(milestone_command="run", milestone_id=milestone_id, part=part,
                     all=run_all, skip_checks=skip_checks, non_interactive=True)
    rc = cmd._handle_run_command(args)
    return rc, out.getvalue()


def ledger(root):
    path = root / ".tito" / "milestones.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def completed(root):
    return ledger(root).get("completed_milestones", [])


def ran(root):
    log = root / "ran.log"
    return log.read_text(encoding="utf-8").split() if log.exists() else []


def fail(root, script):
    path = root / "outcomes.json"
    codes = json.loads(path.read_text()) if path.exists() else {}
    codes[script] = 1
    path.write_text(json.dumps(codes))


def test_part1_alone_does_not_complete_03(project):
    rc, out = run(project, "03", part=1)
    assert rc == 0
    assert ran(project) == _script_paths("03")[:1]
    assert "03" not in completed(project)
    assert "Part 2" in out and "NOT complete" in out


def test_both_parts_complete_03_with_timestamps(project):
    run(project, "03", part=1)
    assert "03" not in completed(project)
    rc, out = run(project, "03", part=2)
    assert rc == 0
    assert "03" in completed(project)
    parts = ledger(project)["part_results"]["03"]
    assert parts["1"]["status"] == parts["2"]["status"] == "passed"
    assert parts["1"]["timestamp"] and parts["2"]["passed_at"]
    assert "MILESTONE ACHIEVED" in out


def test_part2_alone_does_not_complete_06(project):
    rc, _ = run(project, "06", part=2)
    assert rc == 0
    assert ran(project) == _script_paths("06")[1:2]
    assert "06" not in completed(project)


def test_optional_part_never_completes_04(project):
    run(project, "04", part=2)  # CIFAR-10 is an optional extension
    assert "04" not in completed(project)
    assert ledger(project)["part_results"]["04"]["2"]["status"] == "passed"


def test_failed_part_is_never_recorded_as_pass(project):
    fail(project, _script_paths("03")[1])
    rc, out = run(project, "03", part=2)
    assert rc != 0
    record = ledger(project)["part_results"]["03"]["2"]
    assert record["status"] == "failed" and "passed_at" not in record
    assert "03" not in completed(project)

    fail(project, _script_paths("01")[0])
    rc, _ = run(project, "01")
    assert rc != 0
    assert "01" not in completed(project)


def test_default_stops_at_failed_required_part(project):
    fail(project, _script_paths("03")[0])
    rc, _ = run(project, "03")
    assert rc != 0
    assert ran(project) == _script_paths("03")[:1]
    assert "03" not in completed(project)
    assert ledger(project)["part_results"]["03"]["1"]["status"] == "failed"


def test_skip_checks_records_nothing(project):
    rc, out = run(project, "01", skip_checks=True)
    assert rc == 0
    assert ran(project) == _script_paths("01")
    data = ledger(project)
    assert data.get("completed_milestones", []) == []
    assert not data.get("part_results")
    assert "NOT be recorded" in out or "Nothing was recorded" in out


@pytest.mark.parametrize("milestone_id", ["03", "05", "06"])
def test_non_interactive_default_runs_all_required_parts(project, milestone_id):
    rc, _ = run(project, milestone_id)
    assert rc == 0
    required = REQUIRED_PARTS[milestone_id]
    scripts = _script_paths(milestone_id)
    assert ran(project) == [scripts[p - 1] for p in required]
    assert milestone_id in completed(project)


def test_registry_required_parts():
    assert {m: MILESTONE_SCRIPTS[m].get("required_parts") for m in REQUIRED_PARTS} == REQUIRED_PARTS


def _legacy(root, completed_ids):
    (root / ".tito" / "milestones.json").write_text(json.dumps({
        "unlocked_milestones": list(completed_ids),
        "completed_milestones": list(completed_ids),
        "completion_dates": {m: "2026-09-01T00:00:00" for m in completed_ids},
        "unlock_dates": {m: "2026-09-01T00:00:00" for m in completed_ids},
        "total_unlocked": len(completed_ids),
    }), encoding="utf-8")


def test_old_format_json_loads_without_inventing_passes(project):
    _legacy(project, ["01", "03"])
    cmd, out = _cmd(project)
    assert cmd._handle_list_command(Namespace(simple=True)) == 0
    lines = out.getvalue()
    # Single-script 01 could only have been recorded after its one script passed.
    assert "✅ 01 -" in lines
    # Legacy 03 may have come from the XOR part alone: not trusted.
    assert "✅ 03 -" not in lines

    run(project, "03", part=1)
    data = ledger(project)
    assert "03" not in data["completed_milestones"]
    assert "2" not in data["part_results"].get("03", {})
    assert "01" in data["completed_milestones"]


@pytest.mark.parametrize("raw", ["{}", "[]", '{"completed_milestones": "01"}', "not json"])
def test_malformed_ledger_does_not_crash(project, raw):
    (project / ".tito" / "milestones.json").write_text(raw, encoding="utf-8")
    cmd, _ = _cmd(project)
    assert cmd._handle_list_command(Namespace(simple=True)) == 0
    assert run(project, "01")[0] == 0
    assert "01" in completed(project)


def test_sync_reports_only_complete_milestones(project, monkeypatch):
    from tito.core import auth
    from tito.core.submission import SubmissionHandler

    monkeypatch.setattr(auth, "get_user_email", lambda: "student@example.com")
    _legacy(project, ["01", "03"])
    run(project, "06", part=2)  # partial new-style run: must not report 06

    handler = SubmissionHandler(CLIConfig.from_project_root(project), Console(file=io.StringIO()))
    reported = {m["id"]: m["completed"] for m in
                handler.assemble_payload()["milestone_progress"]["unlocked_milestones"]}
    assert reported.get("01") is True
    assert reported.get("03") is False
    assert reported.get("06", False) is False


def test_milestone_test_exits_nonzero_when_requirements_missing(project):
    (project / ".tito" / "progress.json").write_text(json.dumps({"completed_modules": []}))
    cmd, _ = _cmd(project)
    assert cmd._handle_test_command(Namespace(milestone_id="01")) != 0
