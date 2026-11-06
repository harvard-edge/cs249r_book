"""Placement notebook consumes immutable saved conditions and honest holds."""
import importlib.util
import re
from pathlib import Path

import pytest
from marimo._ast.errors import IncompleteRefsError
from mlsysbook_labs.experiment_evidence import capture_evidence
from mlsysim.labs.experiments.v1_02_experiments import (
    STUDY_VERSION, locality_study, resource_study, operation_study, placement_study, study_profile,
)
from mlsysim.labs.experiments.v1_16_experiments import audit_ledger, replay_capture

LAB = Path(__file__).resolve().parents[1] / "vol1/lab_02_system_paradigms.py"


class MemoryLedger:
    def __init__(self): self.records = {}
    async def load_async(self): return self.records
    async def asave(self, *, chapter, design, track): self.records[chapter] = design


def app_and_defaults():
    spec = importlib.util.spec_from_file_location("placement_reference", LAB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outputs, defaults = module.app.run()
    assert any("Part A" in getattr(output, "text", "") for output in outputs)
    return module.app, defaults


def rerun(app, defaults, overrides):
    overrides = dict(overrides)
    while True:
        try: return app.run(defs=overrides)[1]
        except IncompleteRefsError as exc:
            match = re.search(r"Missing:\s*\[(.*?)\]", str(exc))
            if not match: raise
            for name in re.findall(r"'([^']+)'", match.group(1)):
                overrides[name] = defaults.get(name)


def study_captures(track):
    p=study_profile(track);version={"scenario_version":STUDY_VERSION}
    def cap(part,base,result,**kwargs):
        return capture_evidence(track=track,part=part,prediction="Intentionally wrong original prediction",
            inputs=result["inputs"],baseline=base,result=result,model_key="v1_02_experiments",upstream_inputs=version,**kwargs)
    records={
        "A":cap("A",locality_study(track,payload_mb=p["payload_options_mb"][0],connection_mbps=p["connection_mbps"]),
            locality_study(track,payload_mb=p["payload_options_mb"][-1],connection_mbps=p["connection_mbps"])),
        "B":cap("B",resource_study(track,workload_case="large_working_set",intervention="compute"),
            resource_study(track,workload_case="large_working_set",intervention="capacity")),
        "C":cap("C",operation_study(track,duty_cycle=1.0),operation_study(track,duty_cycle=0.0,battery_multiplier=2.0)),
    }
    saved={part:record.to_dict()["result"]["inputs"] for part,record in records.items()}
    args=dict(payload_mb=saved["A"]["payload_mb"],connection_mbps=saved["A"]["connection_mbps"],
        workload_case=saved["B"]["workload_case"],intervention=saved["B"]["intervention"],
        duty_cycle=saved["C"]["duty_cycle"],battery_multiplier=saved["C"]["battery_multiplier"])
    base=placement_study(track,**args);stress=placement_study(track,**args,raw_data_allowed=False)
    records["D"]=capture_evidence(track=track,part="D",prediction="Cheapest always wins",
        inputs=stress["inputs"],baseline=base,result=stress,alternatives=tuple(stress["candidates"].values()),
        model_key="v1_02_experiments",upstream_inputs={**version,"saved_conditions":saved},
        decision={"chosen":"hold","rejected":"raw"})
    return records


@pytest.mark.parametrize("track",["tinyml","mobile","edge","cloud"])
def test_saved_conditions_drive_placement_and_failed_runs_support_hold_report(track):
    app,defaults=app_and_defaults();records=study_captures(track);ledger=MemoryLedger()
    changes={"track":track,"a_prediction":"Remote arithmetic can repair the deadline",
        "b_prediction":"More arithmetic fixes a memory failure","c_prediction":"Always off is the best policy",
        "d_prediction":"Cheapest always wins","rationale":"Hold: observation service fails, despite lower endpoint energy.",
        "uncertainty":"Whole-device radio energy and lifecycle labor remain unknown; revisit after supplied service requirements change."}
    overrides={name:defaults[name] for name in changes}
    for name,value in changes.items(): overrides[name]._value=value
    notes = {part:f"Saved {part}: held the task fixed, changed one condition; reverse the decision after the binding requirement changes." for part in "ABCD"}
    overrides["investigation_notes"] = defaults["investigation_notes"]
    overrides["investigation_notes"]._value = notes
    interpretations = {part:{"capture":record.to_dict(),"response":notes[part]} for part,record in records.items()}
    overrides.update(get_evidence=lambda:records,get_interpretations=lambda:interpretations,
                     get_trials=lambda:{part:[record.to_dict()] for part,record in records.items()},ledger=ledger)
    result=rerun(app,defaults,overrides)
    assert result["complete"]
    assert result["d_result"]["inputs"]["duty_cycle"] == 0
    assert result["c_result"]["inputs"]["duty_cycle"] != 0 # live control is independent
    assert result["d_result"]["inputs"]["intervention"] == "capacity"
    assert not result["d_result"]["feasible"]
    brief = result["decision_brief"]
    assert brief["state"] == "hold"
    assert brief["rejected_alternative"] == "Raw remote"
    assert "service" in brief["decisive_obligation"]
    assert brief["next_chapter_question"] == result["story"].episode(3).question
    # A complete discussion is downloadable before a deliberate persistent commit.
    assert "report" in result
    assert not ledger.records
    result["save_discussion"]._value = True
    saved = rerun(app, result, {**overrides, "save_discussion": result["save_discussion"]})
    assert "report" in saved
    assert ledger.records[2]["recommendation"] == "hold"
    assert ledger.records[2]["team_decision_brief"] == brief
    assert result["report"].snapshot["result_snapshot"]["team_decision_brief"] == brief
    assert ledger.records[2]["interpretations"] == interpretations
    assert ledger.records[2]["investigation_history"]["C"][0] == records["C"].to_dict()
    assert ledger.records[2]["journey"]["scenario_id"] == result["story"].scenario_id
    assert all(note in result["report"].markdown for note in notes.values())
    # Editing prose reconstructs the action and cannot silently overwrite saved work.
    previous = ledger.records[2]["rationale"]
    edited = dict(overrides)
    edited["rationale"]._value = "Changed discussion: gather additional energy evidence before deployment."
    pending = rerun(app, defaults, edited)
    assert pending["complete"] and "report" in pending
    assert not pending["save_discussion"].value
    assert ledger.records[2]["rationale"] == previous
    for record in records.values():
        snapshot=record.to_dict()
        assert replay_capture("v1_02_experiments",snapshot)["result"] == snapshot["result"]
    consumer = audit_ledger(ledger.records, track, required_chapters=(2,))
    assert consumer.records[0].status.value == "failed"  # A completed investigation, not missing/incompatible evidence.
    assert "report" in result


def test_recapturing_upstream_conditions_invalidates_saved_placement_without_rewriting_it():
    app,defaults=app_and_defaults();records=study_captures("tinyml")
    old=records["D"].to_dict()
    base=operation_study("tinyml",duty_cycle=0.0)
    changed=operation_study("tinyml",duty_cycle=0.01,battery_multiplier=2.0)
    records["C"]=capture_evidence(track="tinyml",part="C",prediction="Original guess",inputs=changed["inputs"],
        baseline=base,result=changed,upstream_inputs={"scenario_version":STUDY_VERSION},model_key="v1_02_experiments")
    result=rerun(app,defaults,{"get_evidence":lambda:records})
    assert result["audit"].stale == ("D",)
    assert not result["complete"]
    assert result["decision_brief"]["state"] == "needs_update"
    assert result["decision_brief"]["rejected_alternative"] == "Not current"
    assert "hold" not in result["decision_brief"]["decision"].lower()
    assert "report" not in result
    assert records["D"].to_dict() == old


def test_interpretation_cannot_silently_follow_a_different_saved_pair_or_edited_response():
    app,defaults=app_and_defaults();records=study_captures("tinyml")
    notes={part:f"{part}: changed one condition; this requirement would reverse the decision." for part in "ABCD"}
    defaults["investigation_notes"]._value=notes
    interpretations={part:{"capture":record.to_dict(),"response":notes[part]} for part,record in records.items()}
    overrides={"get_evidence":lambda:records,"get_interpretations":lambda:interpretations,
        "investigation_notes":defaults["investigation_notes"]}
    initial=rerun(app,defaults,overrides)
    assert all(initial["notes_current"].values())
    # Same text cannot be passed off as an interpretation of recaptured conditions.
    changed=locality_study("tinyml",payload_mb=study_profile("tinyml")["payload_options_mb"][0],
        connection_mbps=study_profile("tinyml")["connection_options_mbps"][-1])
    records["A"]=capture_evidence(track="tinyml",part="A",prediction="Original prediction",
        inputs=changed["inputs"],baseline=records["A"].to_dict()["baseline"],result=changed,
        upstream_inputs={"scenario_version":STUDY_VERSION},model_key="v1_02_experiments")
    recaptured=rerun(app,defaults,overrides)
    assert not recaptured["notes_current"]["A"]
    assert recaptured["notes_current"]["B"]
    notes["B"]="Edited interpretation awaiting attachment."
    edited=rerun(app,defaults,overrides)
    assert not edited["notes_current"]["B"]
    assert interpretations["B"]["response"] != notes["B"]


def test_conditional_brief_uses_saved_placement_and_ignores_moving_live_review():
    app, defaults = app_and_defaults()
    records = study_captures("tinyml")
    profile = study_profile("tinyml")
    version = {"scenario_version": STUDY_VERSION}
    operation = operation_study("tinyml", duty_cycle=profile["minimum_observation_fraction"],
        battery_multiplier=profile["battery_multipliers"][-1])
    records["C"] = capture_evidence(track="tinyml", part="C", prediction="Try reserve first",
        inputs=operation["inputs"], baseline=records["C"].to_dict()["baseline"], result=operation,
        upstream_inputs=version, model_key="v1_02_experiments")
    saved = {part: records[part].to_dict()["result"]["inputs"] for part in "ABC"}
    args = dict(payload_mb=saved["A"]["payload_mb"], connection_mbps=saved["A"]["connection_mbps"],
        workload_case=saved["B"]["workload_case"], intervention=saved["B"]["intervention"],
        duty_cycle=saved["C"]["duty_cycle"], battery_multiplier=saved["C"]["battery_multiplier"])
    base = placement_study("tinyml", **args)
    stressed = placement_study("tinyml", **args, raw_data_allowed=False)
    assert stressed["candidates"]["local"]["feasible"]
    records["D"] = capture_evidence(track="tinyml", part="D", prediction="Investigate local",
        inputs=stressed["inputs"], baseline=base, result=stressed,
        upstream_inputs={**version, "saved_conditions": saved},
        decision={"chosen":"local", "rejected":"raw"},
        chosen_result=stressed["candidates"]["local"], model_key="v1_02_experiments")
    initial = rerun(app, defaults, {"get_evidence":lambda:records})
    brief = initial["decision_brief"]
    assert brief["state"] == "conditional"
    assert brief["decision"] == "Local retained for target validation"
    assert "privacy" in brief["decisive_obligation"]
    assert "unresolved" in brief["scope"]
    # A different unsaved review can fail, without rewriting the team's saved claim.
    defaults["d_connection"]._value = False
    changed = rerun(app, defaults, {"get_evidence":lambda:records,
        "d_connection":defaults["d_connection"]})
    assert changed["d_result"]["inputs"]["connectivity_available"] is False
    assert changed["decision_brief"] == brief
