"""Adapters consume public results; they do not change solver calculations."""

import warnings

from .evidence import digest, sanitize


class PipelineCallback:
    """Attach to Pipeline.register_callback after explicitly opening a session."""

    def __init__(self, session, *, experiment_id: str):
        self.session = session
        self.experiment_id = experiment_id
        self.stage = 0

    def on_stage_end(self, stage_name: str, result):
        self.stage += 1
        try:
            self.session.capture(
                {"stage": stage_name, "result": result},
                action_id=f"{self.experiment_id}:{self.stage}:{stage_name}",
            )
        except Exception as error:  # optional observers must not change solver behavior
            warnings.warn(f"Forge evidence capture failed ({type(error).__name__}); the lesson continues.",
                          RuntimeWarning, stacklevel=2)


def publish_capture(session, capture):
    """Use an immutable comparison's digest to deduplicate a publish button."""
    clean = sanitize(capture)
    return session.capture(clean, action_id="capture:" + digest(clean))
