"""Interface rehearsal records, separate from physical experiment evidence."""
from copy import deepcopy

from .contexts import restore_orientation, get_context
from .experiment_evidence import capture_evidence


def briefing_comparison(track, profile, case):
    """Switch the inspected system boundary; introduce no numerical model."""
    if case not in {"baseline", "alternative"}:
        raise ValueError("Unknown briefing view")
    baseline = {"inputs": {"context": track, "case": "baseline"},
                "system": get_context(track).scenario.title,
                "question": "Which project will your fleet support?"}
    result = baseline if case == "baseline" else {
        "inputs": {"context": track, "case": "alternative"},
        "system": profile.scaling_system,
        "question": "Which system prepares the next model?"}
    return deepcopy(baseline), deepcopy(result)


def capture_briefing(track, prediction, baseline, result):
    if prediction not in {"prediction", "control", "save"}:
        raise ValueError("Lock a structured prediction first")
    if result["inputs"] != {"context": track, "case": "alternative"}:
        raise ValueError("Compare the alternative briefing before capturing")
    return capture_evidence(
        track=track, part="orientation", prediction=prediction,
        inputs=result["inputs"], baseline=baseline, result=result,
        decision="Distinguish deployed service from model preparation",
        model_key="orientation.interface.v1", result_role="interface rehearsal",
    ).to_dict()


def setup_record(volume, track, comparison):
    payload = {"schema_version": 1, "orientation_version": 1,
               "lab_id": f"v{volume}_00", "preferred_track": track,
               "comparison": deepcopy(comparison)}
    class Reader:
        def get_design(self, chapter):
            return payload
    if not restore_orientation(Reader()):
        raise ValueError("A compatible captured comparison is required")
    return payload


async def save_setup(ledger, payload):
    """Explicit save, preserve chapter records, and verify a fresh read."""
    from mlsysim.labs.state import DesignLedger
    await ledger.load_async()
    if ledger.last_load_error:
        raise RuntimeError("Earlier progress could not be loaded; keep a download and retry.")
    await ledger.asave(track=payload["preferred_track"], chapter=0, design=payload)
    fresh = DesignLedger(volume=ledger.volume, directory=ledger.config_dir)
    await fresh.load_async()
    if fresh.last_load_error or fresh.get_design(0) != payload:
        raise RuntimeError("The saved setup could not be read back.")
    return payload


def volume_route(volume):
    """Read the packaged route without maintaining a second chapter inventory."""
    import json
    from importlib import resources
    from .learning_paths import normalize_volume, get_learning_path
    number = normalize_volume(volume)
    source = resources.files("mlsysbook_labs").joinpath("data", "learning_paths", f"vol{number}.json")
    catalog = json.loads(source.read_text(encoding="utf-8"))
    return [get_learning_path(number, row["chapter"]) for row in catalog["chapters"] if row["chapter"] > 0]
