"""Smoke tests for the Vol IV timing-strip builders (mlsysim.viz.timing_strip)."""
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

from mlsysim.core.units import ureg
from mlsysim.viz import timing_strip as ts


def _ms(x):
    return x * ureg.millisecond


def test_decoder_budget_strip_reads_budget_values():
    b = SimpleNamespace(t_step=_ms(25), t_renew=_ms(50), t_act=_ms(40), t_ddim=_ms(48),
                        t_jitter=_ms(5), ddim_stretch_over=_ms(3))
    strip = ts.decoder_budget_strip(b, _ms(60))
    labels = [s.label for s in strip.spans]
    assert "10 ms slack" in labels
    assert "ends 3 ms past renewal" in labels
    assert "lease 60 ms, expires unless renewed" in labels
    over = [s for s in strip.spans if s.role == "consequence"]
    assert len(over) == 1 and over[0].start == 50 and over[0].end == 53
    fig, ax, axd = ts.plot_timing_strip(strip)
    assert axd is None
    assert len(ax.get_yticklabels()) == 5
