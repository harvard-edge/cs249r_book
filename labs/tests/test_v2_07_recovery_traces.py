"""Regression checks for captured recovery subtraces and dependency freshness."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence
from mlsysim.labs.experiments import v2_07_experiments as engine


class Button:
    def __init__(self, **kwargs):
        self.disabled = kwargs["disabled"]
        self.callback = kwargs["on_click"]

    def click(self):
        assert not self.disabled
        self.callback(None)


def capture_cell():
    source = Path(__file__).parents[1] / "vol2/lab_07_fault_tolerance.py"
    tree = ast.parse(source.read_text())
    cell = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and any(isinstance(child, ast.FunctionDef) and child.name == "store" for child in node.body))
    cell.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[cell], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["_"]


@pytest.mark.parametrize("track", engine.TRACKS)
def test_subtrace_captures_replay_and_go_stale_after_any_probe_control_changes(track):
    widget = lambda value: SimpleNamespace(value=value)
    state = {}
    def set_evidence(update):
        state.update(update(dict(state)))
    a_base = engine.training_exposure(track, workers=2)
    a_result = engine.training_exposure(track, workers=4)
    b_base = engine.checkpoint_tradeoff(track, interval_min=20)
    b_result = engine.checkpoint_tradeoff(track, interval_min=40)
    fixture = engine.checkpoint_fault_fixture(track)
    replicas = dict(replicas=3, component_availability=.99, domain_mttf_h=1000, domain_repair_min=5)
    e_base = engine.checkpoint_tradeoff(track, state_scale=20)
    e_write = engine.checkpoint_tradeoff(track, state_scale=20, bandwidth_scale=12, bandwidth_allocation="write")
    e_restore = engine.checkpoint_tradeoff(track, state_scale=20, bandwidth_scale=12, bandwidth_allocation="restore")
    f_base = engine.fault_verification_trace(verification="none")
    f_result = engine.fault_verification_trace()
    g_base = engine.gray_failure_trace(detector="heartbeat")
    g_result = engine.gray_failure_trace()
    h_base = engine.elastic_recovery_trace(preserve_batch=False, data_cursor="rewind", reconstruct_state=False, fence_epoch=False)
    h_result = engine.elastic_recovery_trace()
    values = dict(a_base=a_base, a_result=a_result, a_workers=widget(4), a_prediction=widget("under_25"),
                  b_base=b_base, b_result=b_result, b_interval=widget(40), b_prediction=widget("lost_work"),
                  c_base=fixture["baseline"], c_result=fixture["result"], c_fault=widget("corrupt"), c_prediction=widget(300),
                  d_colocated=engine.replica_availability(**replicas, placement_domains=1),
                  d_spread=engine.replica_availability(**replicas, placement_domains=3), d_replicas=widget(3), d_decision=widget("spread"), d_prediction=widget("spread_higher"),
                  e_base=e_base, e_write=e_write, e_restore=e_restore, e_state_scale=widget(20), e_bandwidth_scale=widget(12), e_decision=widget("restore"), e_rejected=widget("write"), e_prediction=widget("restore"),
                  f_base=f_base, f_result=f_result, f_upstream=f_result["inputs"], f_prediction=widget("escape"),
                  g_base=g_base, g_result=g_result, g_upstream=g_result["inputs"], g_prediction=widget("progress"),
                  h_base=h_base, h_result=h_result, h_upstream=h_result["inputs"], h_prediction=widget("invariants"),
                  mo=SimpleNamespace(ui=SimpleNamespace(button=Button)), capture_evidence=capture_evidence, set_evidence=set_evidence, track_id=track)
    cell = capture_cell()
    result = cell(**{name: values[name] for name in inspect.signature(cell).parameters})
    buttons = dict(zip("ABCDE", result[::2]))
    upstream = dict(zip("ABCDE", result[1::2]))
    for button in buttons.values():
        button.click()
    assert audit_evidence(state, track=track, required_parts="ABCDE", contrast_required_parts="ABCDE", per_part_upstream_inputs=upstream).complete
    for part, key, evaluate in [("A", "fault_trace", engine.fault_verification_trace), ("C", "elastic_trace", engine.elastic_recovery_trace), ("E", "gray_trace", engine.gray_failure_trace)]:
        snapshot = state[part].to_dict()
        for role in ("baseline", "result"):
            trace = snapshot[role][key]
            assert evaluate(**trace["inputs"]) == trace
        changed = {**upstream, part: {**upstream[part], key: {"changed": True}}}
        assert audit_evidence(state, track=track, required_parts="ABCDE", per_part_upstream_inputs=changed).stale == (part,)
        assert isinstance(snapshot["prediction"], dict)
