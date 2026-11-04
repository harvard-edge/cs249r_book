"""Chapter-focused reference lab: saved population, meaningful contrasts, holds."""
import importlib.util
import re
from pathlib import Path

import pytest
from marimo._ast.errors import IncompleteRefsError
from mlsysbook_labs.experiment_evidence import capture_evidence
from mlsysim.engine.constraint_explorer import INTRO_SCENARIO_VERSION, evaluate, task_factors
from mlsysim.labs.experiments.v1_16_experiments import audit_ledger, replay_capture

LAB = Path(__file__).resolve().parents[1] / "vol1/lab_01_ml_intro.py"


class MemoryLedger:
    def __init__(self):
        self.records = {}
        self.load_count = 0
        self.saved_tracks = []
    async def load_async(self):
        self.load_count += 1
        return self.records
    async def asave(self, *, track, chapter, design):
        self.saved_tracks.append(track)
        self.save(chapter=chapter, design=design)
    def save(self, *, chapter, design):
        self.records[chapter] = design
    async def flush(self):
        pass


def app_and_defaults():
    spec = importlib.util.spec_from_file_location("intro_reference", LAB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outputs, defaults = module.app.run()
    assert any("Part A" in getattr(output, "text", "") for output in outputs), "The tabs must be displayed, not merely created"
    return module.app, defaults


def rerun(app, defaults, overrides):
    # A cell's exported variables must be overridden together in Marimo.
    overrides = dict(overrides)
    while True:
        try:
            return app.run(defs=overrides)[1]
        except IncompleteRefsError as exc:
            match = re.search(r"Missing:\s*\[(.*?)\]", str(exc))
            if not match:
                raise
            for name in re.findall(r"'([^']+)'", match.group(1)):
                overrides[name] = defaults[name] if name in defaults else None


def captured_study(track):
    version = {"version": INTRO_SCENARIO_VERSION}
    def cap(part, base, result, **kwargs):
        return capture_evidence(track=track, part=part, prediction="An intentionally wrong guess",
            inputs={"tested": part}, baseline=base, result=result,
            upstream_inputs=version, model_key="constraint_explorer.evaluate", **kwargs)
    base = evaluate(track)
    population = evaluate(track, shift_pct=25)
    c_result = evaluate(track, candidate="balanced", shift_pct=25)
    records = {
        "A": cap("A", base, population),
        "B": cap("B", base, evaluate(track, compute_scale=2)),
        "C": capture_evidence(track=track, part="C", prediction="Highest quality always wins",
            inputs={"candidate": "balanced"}, baseline=evaluate(track, candidate="large", shift_pct=25),
            result=c_result, chosen_result=c_result, decision="Hold: requirements or evidence unresolved",
            upstream_inputs={**version, "saved_population": 25.0}, model_key="constraint_explorer.evaluate"),
        "D": capture_evidence(track=track, part="D", prediction="Less work proves quality",
            inputs={"choice": "baseline"}, baseline=task_factors(track), result=task_factors(track, "unverified"),
            alternatives=(task_factors(track, "algorithm"),), chosen_result=task_factors(track), decision="baseline",
            upstream_inputs=version, model_key="constraint_explorer.task_factors.v1"),
    }
    return records


@pytest.mark.parametrize("track", ["tinyml", "mobile", "edge", "cloud"])
def test_reference_renders_all_parts_and_completes_an_honest_hold(track):
    app, default = app_and_defaults()
    evidence = captured_study(track)
    ledger = MemoryLedger()
    changes = {"track": track, "a_prediction": "Neither", "b_prediction": "Arithmetic",
               "c_prediction": "Highest quality is deployable", "d_prediction": "Fewer samples",
               "a_labels": True, "a_shift": 5, "c_decision": "Hold: requirements or evidence unresolved",
               "final_trigger": "Population or outcome evidence changes", "rationale": "Hold: the saved population and fixed resource limits leave a requirement unresolved; keep the host baseline until quality evidence arrives."}
    overrides = {name: default[name] for name in changes}
    for name, value in changes.items():
        overrides[name]._value = value
    overrides.update(get_evidence=lambda: evidence, ledger=ledger)
    result = rerun(app, default, overrides)
    assert result["track_id"] == track
    project = result["project"]
    assert result["report"].scenario == project.title
    assert result["final_decision"]["deployment_scenario_id"] == project.scenario_id
    # Ready evidence enables the separate save action; it must not write a draft.
    assert result["ready"]
    assert ledger.records == {}
    assert ledger.load_count == 0
    assert not result["save_discussion"].value
    result["save_discussion"]._value = True
    saved = rerun(app, default, {**overrides, "save_discussion": result["save_discussion"]})
    assert saved["ready"]
    assert ledger.load_count == 1
    assert ledger.saved_tracks == [track]
    assert ledger.records[1]["deployment_scenario_version"] == project.version
    assert ledger.records[1]["scenario_version"] == INTRO_SCENARIO_VERSION
    assert result["saved_population"] == 25
    assert result["c_result"]["inputs"]["shift_pct"] == 25
    assert result["a_result"]["inputs"]["shift_pct"] == 5  # Live is not saved.
    assert result["ready"]
    assert result["c_result"]["quality_pct"] > 0
    assert result["final_decision"]["deployment"].startswith("Hold")
    assert ledger.records[1]["evidence"]["D"]["result"]["quality_pct"] is None
    assert ledger.records[1]["evidence"]["D"]["chosen_result"]["card_id"] == "baseline"
    # Editing after saving recreates an unpressed action and preserves the saved record.
    _saved_rationale = ledger.records[1]["rationale"]
    overrides["rationale"]._value += " Revisit after independently collected task outcomes."
    edited = rerun(app, default, overrides)
    assert edited["ready"]
    assert not edited["save_discussion"].value
    assert ledger.records[1]["rationale"] == _saved_rationale
    assert ledger.saved_tracks == [track]
    edited["save_discussion"]._value = True
    rerun(app, default, {**overrides, "save_discussion": edited["save_discussion"]})
    assert ledger.records[1]["rationale"] == overrides["rationale"].value
    assert ledger.load_count == 2
    assert ledger.saved_tracks == [track, track]
    audited = audit_ledger(ledger.records, track, required_chapters=(1,))
    assert audited.records[0].status.value == "failed"  # Investigated failure, not missing.
    for capture in evidence.values():
        snapshot = capture.to_dict()
        replayed = replay_capture("constraint_explorer", snapshot)
        assert replayed["result"] == snapshot["result"]


def test_recapturing_population_invalidates_candidate_decision_without_rewriting_it():
    app, default = app_and_defaults()
    evidence = captured_study("tinyml")
    evidence["A"] = capture_evidence(track="tinyml", part="A", prediction="Original guess",
        inputs={"shift_pct": 5}, baseline=evaluate("tinyml"), result=evaluate("tinyml", shift_pct=5),
        upstream_inputs={"version": INTRO_SCENARIO_VERSION}, model_key="constraint_explorer.evaluate")
    result = rerun(app, default, {"get_evidence": lambda: evidence})
    assert result["saved_population"] == 5
    assert result["audit"].stale == ("C",)
    assert not result["ready"]
    assert evidence["C"].to_dict()["prediction"] == "Highest quality always wins"
