"""A completed draft is not a persisted discussion until a deliberate save."""
import asyncio

from mlsysbook_labs.fleet_evidence import discussion_fingerprint, save_completed_discussion


class Ledger:
    def __init__(self, *, fail=False, mutate=None):
        self.events = []
        self.fail = fail
        self.mutate = mutate

    async def load_async(self):
        self.events.append("load")
        if self.mutate:
            self.mutate()

    async def asave(self, **kwargs):
        self.events.append(("write", kwargs))
        if self.fail:
            raise OSError("Browser storage denied")


def save(ledger, design, **overrides):
    options = dict(chapter=2, track="tinyml", design=design,
                   fingerprint=discussion_fingerprint(design), ready=True, requested=True)
    options.update(overrides)
    return asyncio.run(save_completed_discussion(ledger, **options))


def test_ready_initial_render_never_saves():
    ledger = Ledger()
    outcome = save(ledger, {"rationale": "Draft"}, requested=False)
    assert outcome.status == "Ready to save"
    assert ledger.events == []


def test_incomplete_discussion_never_saves_even_with_old_click():
    ledger = Ledger()
    outcome = save(ledger, None, ready=False)
    assert outcome.status == "Evidence in progress"
    assert ledger.events == []


def test_deliberate_save_awaits_storage_and_records_exact_fingerprint():
    ledger = Ledger()
    design = {"evidence": {"A": {"result": 7}}, "rationale": "A saved comparison"}
    outcome = save(ledger, design)
    assert outcome.status == "Saved locally"
    assert ledger.events[0] == "load"
    record = ledger.events[1][1]
    assert record["track"] == "tinyml" and record["chapter"] == 2
    assert record["design"]["discussion_fingerprint"] == discussion_fingerprint(design)
    assert "discussion_fingerprint" not in design


def test_changed_discussion_cannot_reuse_old_save_action():
    ledger = Ledger()
    previous = {"rationale": "Before", "evidence": {"A": {"result": 7}}}
    current = {**previous, "rationale": "After"}
    outcome = save(ledger, current, fingerprint=discussion_fingerprint(previous))
    assert outcome.status == "Discussion changed"
    assert ledger.events == []


def test_failed_storage_never_reports_saved_and_keeps_download_guidance():
    outcome = save(Ledger(fail=True), {"rationale": "Complete"})
    assert outcome.status == "Storage unavailable"
    assert "Download" in outcome.detail


def test_snapshot_does_not_alias_live_data_during_async_load():
    design = {"evidence": {"A": {"result": 7}}, "rationale": "Before"}
    ledger = Ledger(mutate=lambda: design["evidence"]["A"].update(result=9))
    outcome = save(ledger, design)
    assert outcome.status == "Saved locally"
    assert ledger.events[1][1]["design"]["evidence"]["A"]["result"] == 7
