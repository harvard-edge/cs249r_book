"""Replay and dependency checks for the actual operations notebook cells."""
import ast
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace

from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence
from mlsysim.labs.experiments import v2_12_experiments as engine

SOURCE = Path(__file__).resolve().parents[1] / "vol2/lab_12_ops_scale.py"


def cell(definition, environment):
    tree = ast.parse(SOURCE.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
              and any((isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == definition)
                      or (isinstance(n, ast.FunctionDef) and n.name == definition)
                      for n in ast.walk(node)))
    fn.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), namespace)
    values = namespace[fn.name](**{arg.arg: environment[arg.arg] for arg in fn.args.args})
    names = [n.id for n in fn.body[-1].value.elts]
    return dict(zip(names, values))


def environment():
    env = {name: getattr(engine, name) for name in dir(engine) if name.startswith("evaluate_")}
    settings = dict(adoption=.75, canary_fraction=.05, canary_hours=24,
                    coupling="chain", alert_rate=.05, monitor_incident="stale features",
                    monitor_hours=6, join_rule="availability time", useful_utilization=.5,
                    response_choice="none", response_rejected="detection", release_changed=True, release_refresh=["offline-evaluation"])
    env.update({name: SimpleNamespace(value=value) for name, value in settings.items()})
    env.update({f"{part}_prediction": SimpleNamespace(value="prediction") for part in "abcde"})
    env.update(track_id="cloud", asdict=asdict, is_dataclass=is_dataclass)
    env.update(cell("serialize", env))
    env.update(cell("a_base", env))
    return env


def test_actual_notebook_capture_replays_all_new_primary_instruments_and_stales():
    env = environment()
    captures = {}
    def set_evidence(update):
        captures.update(update(captures))
    env.update(set_evidence=set_evidence, capture_evidence=capture_evidence,
               mo=SimpleNamespace(ui=SimpleNamespace(button=lambda **kwargs: SimpleNamespace(**kwargs))))
    widgets = cell("a_capture", env)
    for name in ("a_capture", "b_capture", "d_capture"):
        widgets[name].on_click(None)
    a, b, d = [captures[part].to_dict() for part in "ABD"]
    assert a["result"]["output"]["coupling"] == env["serialize"](engine.evaluate_coupling("cloud", a["inputs"]["topology"]))
    assert a["result"]["output"]["ownership"] == env["serialize"](engine.evaluate_ownership("cloud", a["inputs"]["useful_utilization"]))
    assert b["result"]["output"]["feature"] == env["serialize"](engine.evaluate_feature_join("cloud", b["inputs"]["join_rule"]))
    args = {key: d["inputs"][key] for key in ("incident", "observation_hours", "per_check_false_positive_rate")}
    expected = env["serialize"](engine.evaluate_monitoring("cloud", **args))
    assert d["result"]["output"]["monitoring"] == expected
    assert d["chosen_result"]["output"]["monitoring"] == expected
    upstream = {"A": {"track_id": "cloud", "topology": "chain", "useful_utilization": .5},
                "B": {"track_id": "cloud", "join_rule": "availability time"},
                "D": {"track_id": "cloud", **args}}
    audit = audit_evidence(captures, track="cloud", required_parts=tuple("ABD"), per_part_upstream_inputs=upstream)
    assert not audit.stale
    upstream["B"]["join_rule"] = "event time only"
    upstream["D"]["incident"] = "healthy"
    changed = audit_evidence(captures, track="cloud", required_parts=tuple("ABD"), per_part_upstream_inputs=upstream)
    assert set(changed.stale) == {"B", "D"}
    assert captures["B"].to_dict() == b


def test_actual_release_capture_replays_selective_refresh_and_invalidates_candidate_edit():
    env = environment()
    captures = {}
    def set_evidence(update):
        captures.update(update(captures))
    env.update(set_evidence=set_evidence, capture_evidence=capture_evidence,
               mo=SimpleNamespace(ui=SimpleNamespace(button=lambda **kwargs: SimpleNamespace(**kwargs))))
    cell("a_capture", env)["e_capture"].on_click(None)
    snapshot = captures["E"].to_dict()
    args = snapshot["result"]["inputs"]
    replay = engine.evaluate_release_change(**args)
    assert snapshot["result"]["output"] == env["serialize"](replay)
    assert replay.stale_evidence == ("canary-observation",)
    upstream = {"E": {"track_id": "cloud", "changed": True, "refresh_evidence": ["offline-evaluation"]}}
    assert not audit_evidence(captures, track="cloud", required_parts=("E",), per_part_upstream_inputs=upstream).stale
    upstream["E"]["refresh_evidence"].append("canary-observation")
    assert audit_evidence(captures, track="cloud", required_parts=("E",), per_part_upstream_inputs=upstream).stale == ("E",)
    assert captures["E"].to_dict() == snapshot
