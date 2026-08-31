"""Progress reporting must be informative, throttled, silenceable, and inert.

The reporter exists so a long run is distinguishable from a hung one. It must
never become a second measurement channel, so these tests pin the properties
that keep it safe: it writes only to the stream it is given, it can be turned
off entirely, and it never raises into a training loop.
"""

from __future__ import annotations

import io

from mlperf.runners.common import TrainingProgress


def emit(total, *, unit="epoch", interval=0.0, steps=None, **kwargs):
    stream = io.StringIO()
    progress = TrainingProgress(
        "workload", total, unit=unit, min_interval_seconds=interval, stream=stream
    )
    for step in steps or range(1, total + 1):
        progress.update(step, **kwargs)
    progress.close("note")
    return stream.getvalue()


def test_reports_each_step_and_a_closing_line():
    out = emit(3, loss=0.5)
    assert "workload  epoch 1/3" in out
    assert "workload  epoch 3/3" in out
    assert "done in" in out and "note" in out


def test_unit_label_is_configurable():
    assert "iter 1/2" in emit(2, unit="iter")


def test_metrics_are_rendered_with_their_names():
    out = emit(1, loss=0.1234567, hr10=0.5)
    assert "loss 0.1235" in out
    assert "hr10 0.5" in out


def test_throttling_suppresses_middle_steps_but_never_the_edges():
    """A 500-epoch run must not print 500 lines, but must still show both ends."""
    out = emit(500, interval=3600.0)
    lines = [line for line in out.splitlines() if "epoch" in line]
    assert len(lines) == 2, f"expected only first and last, got {len(lines)}"
    assert "epoch 1/500" in lines[0]
    assert "epoch 500/500" in lines[1]


def test_eta_is_offered_while_running_and_dropped_at_the_end():
    out = emit(4, interval=0.0)
    first, last = out.splitlines()[0], out.splitlines()[-2]
    assert "eta" in first
    assert "eta" not in last, "a finished run has no remaining time to report"


def test_quiet_environment_silences_everything(monkeypatch):
    monkeypatch.setenv("MLPERF_EDU_QUIET", "1")
    assert emit(3, loss=0.5) == ""


def test_close_is_silent_when_nothing_was_reported():
    stream = io.StringIO()
    TrainingProgress("workload", 5, stream=stream).close("note")
    assert stream.getvalue() == ""


def test_reporting_never_raises_into_the_training_loop():
    """A progress bug must not be able to fail a benchmark run."""
    stream = io.StringIO()
    progress = TrainingProgress("w", 0, min_interval_seconds=0.0, stream=stream)
    progress.update(1, value=float("nan"))
    progress.update(7, value=float("inf"))
    progress.close()
