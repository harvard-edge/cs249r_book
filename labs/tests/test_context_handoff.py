"""The orientation preference must survive real ledger persistence and aliases."""
from pathlib import Path
from mlsysbook_labs.contexts import CONTEXT_OPTIONS, get_context, preferred_context, restore_orientation
from mlsysbook_labs.experiment_evidence import capture_evidence
from mlsysim.labs.state import DesignLedger


def test_preference_survives_reload_and_later_experiment_override(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    ledger = DesignLedger(volume="vol1")
    ledger.save(track="mobile", chapter=0, design={"preferred_track": "mobile"})
    ledger.save(track="cloud", chapter=1, design={"choice": "standalone comparison"})
    fresh = DesignLedger(volume="vol1")
    assert fresh.get_track() == "cloud"
    assert preferred_context(fresh) == "mobile"
    assert fresh.get_design(1) == {"choice": "standalone comparison"}


def test_legacy_ids_restore_without_modifying_history(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    ledger = DesignLedger(volume="vol1")
    legacy = {"track_id": "oura_ring", "prediction": "original"}
    ledger.save(chapter=0, design=legacy)
    assert preferred_context(ledger) == "tinyml"
    assert ledger.get_design(0) == legacy
    assert get_context("cloud_fleet").track_id == "cloud"
    assert get_context("robotaxi").track_id == "edge"


def test_absent_unknown_and_other_volume_preferences_are_standalone(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    other = DesignLedger(volume="vol2")
    other.save(track="mobile", chapter=0, design={"preferred_track": "mobile"})
    own = DesignLedger(volume="vol1")
    assert preferred_context(own) == "tinyml"
    own.save(track="edge", chapter=0, design={"preferred_track": "unknown"})
    assert preferred_context(own) == "edge"
    assert set(CONTEXT_OPTIONS.values()) == {"tinyml", "mobile", "edge", "cloud"}


def test_restore_rejects_incompatible_or_malformed_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    ledger = DesignLedger(volume="vol1")
    capture = capture_evidence(
        track="edge", part="orientation", prediction="save",
        inputs={"case": "alternative", "context": "edge"},
        baseline={"inputs": {"case": "baseline", "context": "edge"}},
        result={"inputs": {"case": "alternative", "context": "edge"}},
    ).to_dict()
    saved = {"orientation_version": 1, "preferred_track": "edge", "comparison": capture}
    ledger.save(chapter=0, design=saved)
    restored = restore_orientation(ledger)
    restored["comparison"]["prediction"] = "changed"
    assert restore_orientation(ledger)["comparison"]["prediction"] == "save"
    saved["preferred_track"] = "mobile"
    ledger.save(chapter=0, design=saved)
    assert restore_orientation(ledger) == {}
    ledger.save(chapter=0, design={"orientation_version": 1, "comparison": {}})
    assert restore_orientation(ledger) == {}
    ledger.save(chapter=0, design=["legacy malformed"])
    assert restore_orientation(ledger) == {}
    assert preferred_context(ledger) == "tinyml"
