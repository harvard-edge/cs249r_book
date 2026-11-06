"""Adapters consume public results; they do not change solver calculations."""

from .evidence import digest, sanitize


class PipelineCallback:
    """Attach to Pipeline.register_callback after explicitly opening a session."""

    def __init__(self, session, *, experiment_id: str):
        self.session = session
        self.experiment_id = experiment_id
        self.stage = 0

    def on_stage_end(self, stage_name: str, result):
        self.stage += 1
        self.session.capture(
            {"stage": stage_name, "result": result},
            action_id=f"{self.experiment_id}:{self.stage}:{stage_name}",
        )


def publish_capture(session, capture):
    """Use an immutable comparison's digest to deduplicate a publish button."""
    clean = sanitize(capture)
    return session.capture(clean, action_id="capture:" + digest(clean))
