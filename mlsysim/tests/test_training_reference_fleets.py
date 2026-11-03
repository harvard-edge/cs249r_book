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


def test_dense_llm_pretraining_step_is_the_decided_definition():
    """Added 2026-11-03. Pins the 175B reference job's one shared step definition.

    Author decision: TP8 x PP8 x DP16 on the 1,024-GPU reference cluster,
    16-bit mixed precision with unsharded mixed-precision Adam state, and one
    step = one optimizer update over the 1,536-sequence global batch.
    """
    from mlsysim import Scenarios

    s = Scenarios.DenseLLMPretrainingStep
    assert (s.tensor_parallel, s.pipeline_depth, s.data_parallel) == (8, 8, 16)
    assert s.tensor_parallel * s.pipeline_depth * s.data_parallel == s.fleet.total_accelerators
    assert s.global_batch_sequences == 1536
    assert s.microbatches_per_replica == 96
    assert s.precision == "fp16"
    hbm = s.fleet.node.accelerator.memory.capacity
    assert s.state_per_gpu.to("GiB") < hbm  # fits without optimizer sharding
    step = s.step_time.to("s").magnitude
    assert 5 < step < 15  # modeled optimizer step, seconds not milliseconds
    assert "NDR" in s.fleet.fabric.name  # the reference cluster as Volume II describes it
    # NDR value pinned on its own (modeled 0.8322). prov:scaling-efficiency-reference-job-modeled
    # (0.8293) is the HDR sweep; the two differ only in the data-parallel exchange time.
    assert s.result.scaling_efficiency == pytest.approx(0.8322, abs=0.0005)
    assert s.result.dp_communication_latency.to("s").magnitude == pytest.approx(0.205, abs=0.002)


def test_dense_llm_pretraining_step_2k_doubles_the_replicas():
    """Added 2026-11-03. The storage chapter's 2,048-GPU running cluster: same step, 32 replicas."""
    from mlsysim import Scenarios

    ref, big = Scenarios.DenseLLMPretrainingStep, Scenarios.DenseLLMPretrainingStep2K
    assert big.fleet.total_accelerators == 2 * ref.fleet.total_accelerators
    assert (big.tensor_parallel, big.pipeline_depth, big.data_parallel) == (8, 8, 32)
    assert big.global_batch_sequences == ref.global_batch_sequences == 1536
    assert big.state_per_gpu == ref.state_per_gpu
    assert big.step_time < ref.step_time  # more replicas share the same batch


def test_dense_llm_pretraining_step_8k_shows_coordination_cost_growing():
    """Added 2026-11-03. Same job at 8x the reference cluster: faster steps, lower efficiency."""
    from mlsysim import Scenarios

    ref, big = Scenarios.DenseLLMPretrainingStep, Scenarios.DenseLLMPretrainingStep8K
    assert big.fleet.total_accelerators == 8 * ref.fleet.total_accelerators
    assert big.data_parallel == 128 and big.global_batch_sequences == 1536
    assert big.step_time < ref.step_time
    assert big.result.scaling_efficiency < ref.result.scaling_efficiency
