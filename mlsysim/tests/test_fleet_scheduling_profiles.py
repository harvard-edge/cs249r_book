"""Shared fleet-scheduling scenarios and the cold-restart helper (added 2026-10-15).

These guard the reconciliation of the recovery budget, the preemption tax, the
elastic-resize decision, and the utilization examples, which were hand-typed in
several LEGO cells and had drifted apart (a forced preemption reloaded the same
checkpoint ~50x slower than a failure recovery). Every reader now loads these
entries, so a change here propagates instead of diverging.
"""

import pytest

from mlsysim import Literature, Models, ReferenceStats, Systems
from mlsysim.core.units import GB, minute, second
from mlsysim.physics import calc_checkpoint_size

R = Systems.Reliability.Recovery
CKPT_175B = calc_checkpoint_size(Models.Language.GPT3.parameters)


def test_cold_restart_time_is_the_four_term_budget():
    load = CKPT_175B / (float(R.checkpoint_write_bw_gbs) * GB / second)
    expected = (
        float(R.detection_time_s) * second
        + float(R.restart_time_s) * second
        + load
        + float(R.warmup_time_s) * second
    )
    got = R.cold_restart_time(CKPT_175B)
    assert got.to(second).magnitude == pytest.approx(expected.to(second).magnitude)
    # The vol2 recovery-budget example quotes about 6.4 minutes for the 175B checkpoint.
    assert 6.0 < got.to(minute).magnitude < 7.0


def test_scheduler_initiated_restart_skips_detection_only():
    full = R.cold_restart_time(CKPT_175B)
    no_detect = R.cold_restart_time(CKPT_175B, include_detection=False)
    assert (full - no_detect).to(second).magnitude == pytest.approx(float(R.detection_time_s))


def test_checkpoint_bandwidth_is_per_job_share_of_reference_pfs():
    """The recovery profile's 100 GB/s is a tenth of the 1 TB/s reference PFS
    shared by ten concurrent jobs (vol2 storage chapter); keep them in step."""
    per_job = Systems.Storage.PfsOneTbPerSecond.bandwidth.to(GB / second).magnitude / 10
    assert float(R.checkpoint_write_bw_gbs) == pytest.approx(per_job)


def test_forced_preemption_tax_reuses_the_recovery_steps():
    P = ReferenceStats.ForcedPreemptionProfile
    tax = P.LostWork + R.cold_restart_time(CKPT_175B, include_detection=False)
    # Lost work dominates; the relaunch/reload/warmup must stay the cluster's
    # recovery steps, not a separate slower reload.
    assert 20 < tax.to(minute).magnitude < 21
    assert P.UnbudgetedRate.to("1/day").magnitude == pytest.approx(12)


def test_spot_interruption_is_slower_than_cluster_recovery():
    S = ReferenceStats.SpotInterruptionProfile
    assert S.RestartOverhead > R.cold_restart_time(CKPT_175B, include_detection=False)
    assert S.RetainedInterval.to(minute).magnitude == pytest.approx(120)


def test_elastic_resize_profile_efficiencies():
    E = ReferenceStats.ElasticResizeProfile
    assert int(E.SmallGpus) < int(E.StepUpGpus) < int(E.FullGpus)
    eff_mid = float(E.StepUpThroughput) / (float(E.StepUpGpus) / float(E.SmallGpus) * float(E.SmallThroughput))
    eff_full = float(E.FullThroughput) / (float(E.FullGpus) / float(E.SmallGpus) * float(E.SmallThroughput))
    assert eff_mid == pytest.approx(0.90)
    assert eff_full == pytest.approx(0.80)
    assert E.QueueWait < E.Window
    assert E.RescaleOverhead < E.QueueWait


def test_utilization_tiers_narrow():
    T = ReferenceStats.UtilizationTierProfile
    assert float(T.ProductiveEquivalents) < float(T.ActiveEquivalents) < float(T.AllocatedGpus)


def test_fleet_snapshot_shares_are_consistent():
    S = ReferenceStats.FleetUtilizationSnapshot
    assigned, gang, wait = float(S.AssignedShare), float(S.IncompleteGangShare), float(S.InputWaitShare)
    assert 0 < gang < assigned <= 1
    assert wait < assigned - gang  # input waits sit inside complete allocations


def test_palm_mfu_description_matches_value():
    """2026-10-15: the description said 0.462 was 'without self-attention';
    PaLM reports 46.2% counting self-attention and 45.7% without."""
    m = Literature.TrainingRuns.PalmMfu
    assert float(m) == pytest.approx(0.462)
    assert "counting self-attention" in m.description
    assert "45.7%" in m.description


def test_replica_cold_start_is_shared_by_serving_and_goodput_examples():
    """vol2 inference (ColdStartTimeline) and performance engineering (serving
    twin in RuntimeGoodputComposition) both read ReplicaColdStartProfile; the
    serving twin's ready time is the cold start without provisioning."""
    from mlsysim.core.units import BYTES_FP16

    C = ReferenceStats.ReplicaColdStartProfile
    artifact = Models.Language.Llama2_70B.size_in_bytes(BYTES_FP16)
    fetch = (artifact / C.ArtifactFetchBandwidth).to(minute)
    cold = (C.ProvisioningTime + fetch + C.LoadWarmupTime).to(minute)
    ready = (fetch + C.LoadWarmupTime).to(minute)
    assert C.ArtifactFetchBandwidth.to(GB / second).magnitude == pytest.approx(0.5)
    # The artifact fetch dominates the cold start (inference chapter guard).
    assert fetch > C.ProvisioningTime + C.LoadWarmupTime
    assert 6.0 < cold.magnitude < 7.5      # ~6.7 min cold start
    assert 5.5 < ready.magnitude < 6.0     # ~5.7 min to ready after placement
