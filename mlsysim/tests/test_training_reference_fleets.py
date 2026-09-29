"""Physical-consistency checks for model reference training runs.

Added 2026-09-29. Two registry defects motivated these tests:

* GPT-3 stored a 1,024-accelerator, 25-day run that implied ~142 TFLOP/s per
  V100 (114% of peak). Patterson et al. (2021), Table 4, reports 10,000 V100s
  at an OpenAI-measured 24.6 TFLOP/s, ~14.8 days, 330 W per GPU, PUE 1.10.
* GPT-4 stored 25,000 accelerators x 90 days beside a 2.5M accelerator-day
  anchor, a 10% self-contradiction. SemiAnalysis (2023) gives "~25,000 A100s
  for 90 to 100 days"; the registry takes the 100-day upper end.
"""

import pytest

from mlsysim import Hardware, Literature, Models
from mlsysim.core.units import Q_, ureg
from mlsysim.models.types import TransformerWorkload


def test_gpt3_reference_run_is_physically_consistent():
    g = Models.Language.GPT3
    n = g.training_accelerators_ref.to("count").magnitude
    days = g.training_days_ref.to("day")
    rate = (g.training_ops / (n * days)).to("TFLOP/s").magnitude
    # Patterson 2021 Table 4: OpenAI-measured 24.6 TFLOP/s per V100.
    assert rate == pytest.approx(24.6, rel=0.01)
    # Guards the >100% MFU bug of the old 1,024 / 25-day pairing.
    peak = Hardware.Cloud.V100.compute.peak_flops.to("TFLOP/s").magnitude
    assert rate < peak
    assert g.training_hardware_label == "V100"
    # 330 W per GPU and PUE 1.10 from the same Table 4 row reproduce the energy.
    energy = (n * days * 330 * ureg.watt * 1.10).to("MWh").magnitude
    assert energy == pytest.approx(g.training_energy_mwh.to("MWh").magnitude, rel=0.01)
    # Independent anchor: PaLM Table 3 reports 21.3% MFU for GPT-3.
    assert abs(rate / peak - float(Literature.TrainingRuns.Gpt3MfuPalmTable)) < 0.03


def test_gpt4_reference_fleet_matches_gpu_day_anchor():
    """SemiAnalysis (2023): ~25,000 A100s for 90 to 100 days, ~2.5M A100-days."""
    g = Models.Language.GPT4
    product = g.training_accelerators_ref.to("count").magnitude * g.training_days_ref.to("day").magnitude
    assert product == pytest.approx(g.training_gpu_days, rel=0.01)


def test_inconsistent_training_fleet_is_rejected():
    with pytest.raises(ValueError, match="training_gpu_days"):
        TransformerWorkload(
            name="inconsistent-fleet",
            architecture="Transformer",
            parameters=Q_("1e9 param"),
            layers=2,
            training_accelerators_ref=Q_("25000 count"),
            training_days_ref=Q_("90 d"),
            training_gpu_days=2.5e6,
        )


@pytest.mark.parametrize(
    "name",
    [n for n in dir(Models.Language) if not n.startswith("_")],
)
def test_reference_run_rate_below_device_peak(name):
    """Every reference run with a resolvable device must sit below that device's peak."""
    m = getattr(Models.Language, name)
    if not isinstance(m, TransformerWorkload):
        pytest.skip("not a transformer workload")
    if None in (m.training_ops, m.training_accelerators_ref, m.training_days_ref, m.training_hardware_label):
        pytest.skip("no complete reference run")
    device = getattr(Hardware.Cloud, m.training_hardware_label, None)
    if device is None:
        pytest.skip(f"hardware label {m.training_hardware_label!r} does not resolve")
    n = m.training_accelerators_ref.to("count").magnitude
    rate = (m.training_ops / (n * m.training_days_ref)).to("TFLOP/s").magnitude
    assert rate < device.compute.peak_flops.to("TFLOP/s").magnitude
