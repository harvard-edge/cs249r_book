"""The 256-, 1,024-, and 8,192-GPU scaling-efficiency tiers are rounded outputs
of the same DistributedModel sweep that drives the vol2 fleet-orchestration
scaling-efficiency figure. If the solver changes, this test fails, and the
editor must decide whether to update the tiers or the solver."""

import pytest

from mlsysim import Literature, ReferenceStats, Scenarios
from mlsysim.core.units import BYTES_ADAM_STATE, BYTES_FP16, BYTES_FP32, GiB
from mlsysim.core.provenance import ProvenanceKind
from mlsysim.solvers import DistributedModel

T = ReferenceStats.TrainingScaleProfiles

MODELED_TIERS = {256: T.Eff256Gpu, 1024: T.Eff1024Gpu, 8192: T.Eff8192Gpu}


def _reference_sweep_efficiency(n_gpus: int) -> float:
    scenario = Scenarios.DenseLLMPretrainingReference
    model = scenario.workload
    fleet = scenario.system
    gpus_per_node = fleet.node.accelerators_per_node
    hbm = fleet.node.accelerator.memory.capacity
    state_bytes = 2 * BYTES_FP16 + BYTES_FP32 + BYTES_ADAM_STATE
    tp, pp = gpus_per_node, 1
    while (model.parameters.magnitude * state_bytes / (tp * pp)).to(GiB) > hbm:
        pp *= 2
    replica = tp * pp
    assert replica == 64
    batch_seqs = int(float(Literature.TrainingRuns.Gpt3BatchTokens) / model.context_length)
    batch_seqs = batch_seqs // 64 * 64  # divisible by every DP degree up to 4,096 GPUs
    assert batch_seqs == 1536
    dp = n_gpus // replica
    result = DistributedModel().solve(
        model,
        fleet.model_copy(update={"count": n_gpus // gpus_per_node}),
        batch_size=batch_seqs,
        precision="fp16",
        efficiency=float(T.LocalComputeEfficiency),
        tp_size=tp,
        pp_size=pp,
        microbatch_count=batch_seqs // dp,
        overlap_comm=True,
        seq_len=model.context_length,
    )
    return float(result.scaling_efficiency)


@pytest.mark.parametrize("n_gpus", sorted(MODELED_TIERS))
def test_modeled_tier_matches_solver(n_gpus):
    tier = float(MODELED_TIERS[n_gpus])
    assert round(_reference_sweep_efficiency(n_gpus), 2) == pytest.approx(tier)


def test_tiers_are_monotone():
    assert float(T.Eff32Gpu) >= float(T.Eff256Gpu) >= float(T.Eff1024Gpu) >= float(T.Eff8192Gpu)


def test_tier_provenance_kinds():
    assert T.Eff8192Gpu.provenance.id != "prov:jiang-megascale-2024"
    for tier in MODELED_TIERS.values():
        assert tier.provenance.kind == ProvenanceKind.DERIVED
        assert tier.provenance.id == "prov:scaling-efficiency-reference-job-modeled"
    assert T.Eff32Gpu.provenance.kind == ProvenanceKind.CONVENTION
