"""The capstone must accept the discussion format students actually save."""
import asyncio

from mlsysbook_labs.fleet_evidence import discussion_fingerprint, save_completed_discussion
from mlsysim.labs.experiments.v2_17_experiments import audit_discussion_dossier
from mlsysim.labs.state import DesignLedger


def test_saved_discussion_is_carried_with_conditions_without_deployment_clearance():
    ledger = DesignLedger(volume="vol2")
    discussion = {
        "schema_version": 1, "lab_id": "v2_13", "track_id": "cloud",
        "model_id": "v2_13_experiments",
        "evidence": {"A": {"track": "cloud", "result": {"scope": "finite fixture"}}},
        "recommendation": "defer", "reevaluation_trigger": "matched adversary evidence",
        "residual_risk": "authorized output leakage remains unmeasured",
    }
    outcome = asyncio.run(save_completed_discussion(
        ledger, chapter=13, track="cloud", design=discussion,
        fingerprint=discussion_fingerprint(discussion), ready=True, requested=True,
    ))
    assert outcome.status == "Saved locally"
    dossier = audit_discussion_dossier(ledger._state.history, track_id="cloud")
    row = next(r for r in dossier["rows"] if r["chapter"] == 13)
    assert row["status"] == "saved_scoped_discussion"
    assert row["recommendation"] == "defer"
    assert row["reversal_trigger"] == discussion["reevaluation_trigger"]
    assert dossier["saved_discussions"]["13"]["evidence"] == discussion["evidence"]
    assert not dossier["deployment_clearance"]
