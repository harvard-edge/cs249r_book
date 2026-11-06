"""Orientation is an explicit, replayable setup action, not deployment evidence."""
import asyncio
from copy import deepcopy

import pytest

from mlsysbook_labs.contexts import preferred_context, restore_orientation
from mlsysbook_labs.onboarding import briefing_comparison, capture_briefing, setup_record, save_setup
from mlsysim.labs.experiments.v2_01_experiments import TRACK_PROFILES
from mlsysim.labs.state import DesignLedger


@pytest.mark.parametrize("track", TRACK_PROFILES)
def test_control_capture_and_reload_keep_prior_chapters(tmp_path, track):
    ledger = DesignLedger(volume="vol2", directory=tmp_path)
    ledger.save(track=track, chapter=1, design={"previous": "evidence"})
    baseline, live = briefing_comparison(track, TRACK_PROFILES[track], "alternative")
    snapshot = capture_briefing(track, "control", baseline, live)
    original = deepcopy(snapshot)
    live["system"] = "changed"
    assert snapshot == original
    payload = setup_record(2, track, snapshot)
    asyncio.run(save_setup(ledger, payload))
    fresh = DesignLedger(volume="vol2", directory=tmp_path)
    assert preferred_context(fresh) == track
    assert restore_orientation(fresh) == payload
    assert fresh.get_design(1) == {"previous": "evidence"}
    assert payload["comparison"]["result_role"] == "interface rehearsal"
    payload["comparison"]["result"]["system"] = "mutated"
    assert restore_orientation(fresh)["comparison"] == snapshot


def test_capture_requires_prediction_and_changed_view():
    baseline, live = briefing_comparison("cloud", TRACK_PROFILES["cloud"], "baseline")
    with pytest.raises(ValueError):
        capture_briefing("cloud", None, baseline, live)
    with pytest.raises(ValueError):
        capture_briefing("cloud", "control", baseline, live)


def test_full_route_uses_packaged_contract_identity():
    from mlsysbook_labs.onboarding import volume_route
    from mlsysbook_labs.learning_paths import get_learning_path
    route = volume_route(2)
    assert route[0] == get_learning_path(2, 1)
    assert route[-1]["title"] == "Fleet Synthesis"
    assert all(row["chapter"] > 0 for row in route)
    assert len({row["chapter"] for row in route}) == len(route)
