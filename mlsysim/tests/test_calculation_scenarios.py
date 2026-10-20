"""Calculation scenarios: named inputs plus the solver that owns the arithmetic.

Chapters and lecture slides quote these values; pinning them here means a
registry or solver change that moves one shows up as a failing test instead
of a silent change in print. Values were checked against the hand-derived
chapter results when the scenarios were added (2026-10-20).
"""

import pytest

from mlsysim import Scenarios


def approx(x, rel=1e-3):
    return pytest.approx(x, rel=rel)


def test_decode_bound_llama70b_h100():
    r = Scenarios.Llama3_70B_DecodeH100.result
    assert r.weight_bytes.to("GB").magnitude == approx(141.2)
    assert r.flops_per_step.to("flop").magnitude == approx(1.412e11)
    assert r.compute_time.to("ms").magnitude == approx(0.14277)
    assert r.memory_time.to("ms").magnitude == approx(42.149)
    assert r.memory_share == approx(0.99662)
    assert r.bound_ratio == approx(295.22)
    assert r.intensity.to("flop/byte").magnitude == approx(1.0)
    assert r.peak_fraction == approx(0.0033873)
    assert r.bottleneck == "Memory"


def test_decode_variants_follow_bandwidth_and_precision():
    from mlsysim import Hardware
    base = Scenarios.Llama3_70B_DecodeH100
    assert base.variant(hardware=Hardware.Cloud.A100).result.memory_time.to("ms").magnitude == approx(69.25)
    assert base.variant(hardware=Hardware.Cloud.B200).result.memory_time.to("ms").magnitude == approx(17.65)
    assert base.variant(precision="int8").result.memory_time.to("ms").magnitude == approx(21.07)
    assert base.variant(batch_size=64).result.peak_fraction == approx(0.21678)


def test_transfer_ladder():
    t = Scenarios.BufferTransfer10GB.result.times
    assert t["HBM"].to("ms").magnitude == approx(2.9851)
    assert t["NVLink"].to("ms").magnitude == approx(22.222)
    assert t["InfiniBand"].to("ms").magnitude == approx(200.0)
    assert t["PCIe"].to("ms").magnitude == approx(156.25)


def test_tensor_parallel_step():
    s = Scenarios.Gpt3TensorParallelStepDgxH100
    r = s.result
    assert r.payload.to("MB").magnitude == approx(201.33)
    assert r.ring_factor == approx(1.75)
    assert r.collectives_per_step == 192
    assert r.per_collective_time.to("ms").magnitude == approx(0.78294)
    assert r.communication_time.to("ms").magnitude == approx(150.32)
    assert r.compute_fraction == approx(0.57090)
    pcie = s.variant(link_bandwidth=Scenarios.BufferTransfer10GB.links["PCIe"]).result
    assert pcie.communication_time.to("ms").magnitude == approx(1056.96)
    assert pcie.communication_fraction == approx(0.84089)


def test_training_run_chain():
    r = Scenarios.Gpt3PretrainingFrontier8K.result
    assert r.total_flops.to("flop").magnitude == approx(3.15e23)
    assert r.peak_time.to("hour").magnitude == approx(5.3972)
    assert r.compute_time.to("hour").magnitude == approx(11.994)
    assert r.scaled_time.to("hour").magnitude == approx(21.418)
    assert r.stage_times["checkpoint"].to("hour").magnitude == approx(22.060)
    assert r.stage_times["failures"].to("hour").magnitude == approx(24.266)
    assert r.total_time.to("hour").magnitude == approx(25.480)
    assert r.delivered_fraction == approx(0.21183)


def test_training_state_shards():
    r = Scenarios.Gpt3TrainingStateTp8.result
    assert r.persistent_state.to("GB").magnitude == approx(2800)
    assert r.tp_moments_per_device.to("GB").magnitude == approx(175)
    assert r.tp_master_per_device.to("GB").magnitude == approx(87.5)
    assert r.tp_total_per_device.to("GB").magnitude == approx(375)
    assert r.tp_offloaded_per_device.to("GB").magnitude == approx(112.5)
    assert r.zero_weights_and_moments_per_device.to("GB").magnitude == approx(45.117)


def test_power_transient():
    r = Scenarios.H100BarrierLoadStep.result
    assert r.current_step.to("A").magnitude == approx(375)
    assert r.droop["slow"].to("mV").magnitude == approx(18.75)
    assert r.droop["fast"].to("mV").magnitude == approx(187.5)
    assert r.cluster_power_step.to("kW").magnitude == approx(307.2)
    assert r.ramp_rate.to("GW/s").magnitude == approx(3.072)


def test_cooling():
    air = Scenarios.AiRackAirCooling.result
    assert air.mass_flow.to("kg/s").magnitude == approx(6.6335)
    assert air.volume_flow.to("m**3/s").magnitude == approx(5.5279)
    water = Scenarios.AiRackAirCooling.variant(coolant="water").result
    assert water.volume_flow.to("L/s").magnitude == approx(1.5949)
    assert Scenarios.Nvl72WaterCooling.result.volume_flow.to("L/min").magnitude == approx(172.25)


def test_outage_and_failures():
    r = Scenarios.CduOutageTraining10K.result
    assert r.direct_cost == approx(2048)
    assert r.job_cost == approx(80000)
    assert r.job_cost_with_rollback == {"low": approx(90000), "high": approx(100000)}
    g = Scenarios.GpuFailuresTraining10K.result
    assert g.interval.to("hour").magnitude == approx(5.0)
    assert g.expected_failures == approx(67.2)
    n = Scenarios.NodeFailuresTraining1K.result
    assert n.failures_per_day == approx(3.072)


def test_accelerator_comparisons():
    assert Scenarios.V100ToB200.peak_ratio == approx(18.0)
    assert Scenarios.V100ToB200.tdp_ratio == approx(3.3333)
    assert Scenarios.V100ToB200.efficiency_ratio == approx(5.4)
    assert Scenarios.V100ToH100.peak_ratio == approx(7.912)
    assert Scenarios.V100ToH100.equivalent_hours(10000) == approx(1263.9)


def test_sustained_throughput_budget():
    s = Scenarios.PeakPlannedTrainingBudget
    assert s.required_device_hours == approx(2500)
    assert s.stretch == approx(2.5)


def test_memory_interface_widths():
    from mlsysim import Hardware
    m = Hardware.Tech.Memory
    assert m.HBM3.interface_width.to("bit").magnitude == 1024
    assert m.DDR5.interface_width.to("bit").magnitude == 64


def test_series_lookups():
    from mlsysim import Literature
    s = Literature.Series.AISupercomputers
    assert s.find("name", "Frontier")["power_mw"] == approx(40)
    last = s.largest("power_mw", "first_operational_date", "2025-12-31")
    assert last["first_operational_date"] <= "2025-12-31"
    assert last["power_mw"] == max(r["power_mw"] for r in s.entries
                                   if r["power_mw"] and r["first_operational_date"] <= "2025-12-31")


# --- Data-parallel training at fleet scale (added 2026-10-20) ---------------

def test_gradient_sync():
    r = Scenarios.Gpt3GradientSync.result
    assert r.sync_bytes.to("GB").magnitude == approx(700)
    assert r.transfer_times["InfiniBand"].to("s").magnitude == approx(28)
    assert r.step_times["Ethernet"].to("s").magnitude == approx(57.2)
    assert r.compute_fractions["InfiniBand"] == approx(0.041096)
    assert r.energy_per_step.to("J").magnitude == approx(196)
    assert r.energy_total.to("MJ").magnitude == approx(196)
    fast = Scenarios.Gpt3GradientSyncFasterAccelerator.result
    assert fast.step_times["InfiniBand"].to("s").magnitude == approx(28.6)


def test_communication_intensity():
    r = Scenarios.Gpt3DataParallelIntensity.result
    assert r.local_tokens == 8192
    assert r.intensity == approx(8.138e-5)
    assert r.balance == approx(8.0128e-5)
    assert Scenarios.WorkloadBetaIntensity.intensity / Scenarios.WorkloadAlphaIntensity.intensity == approx(1000)


def test_availability_and_compute_optimal():
    assert Scenarios.NodeAvailability1K.all_up == approx(0.36770)
    assert Scenarios.NodeAvailability10K.all_up == approx(4.5173e-5)
    assert Scenarios.FourNinesAvailability10K.downtime_per_year.to("minute").magnitude == approx(52.56)
    n, d = Scenarios.Gpt3ComputeOptimal.optimal
    assert n == approx(5.1235e10)
    assert d == approx(1.0247e12)


def test_illustrative_fleet_law_scenarios():
    p = Scenarios.FleetLawProfileExample
    assert p.step_time.to("s").magnitude == approx(2.0)
    assert p.compute_fraction == approx(0.40)
    assert Scenarios.FleetLawDoublingExample.efficiency == approx(0.625)
    assert Scenarios.CoordinationOverheadExample.realized_speedup == approx(1.8)
    assert Scenarios.ChinchillaParameterPowerLaw.halving_multiple == approx(7.6804)


def test_growth_and_runs():
    g = Scenarios.NotableRunGrowth2012to2024.result
    assert g.run_count == 146
    assert g.compute_factor == approx(3.92995)
    assert g.fleet_share == approx(0.51151)
    assert Scenarios.GpuFailuresLlama3Run.result.interval.to("hour").magnitude == approx(3.0518)
    assert Scenarios.Gpt3PublishedRun.peak_fraction == approx(0.2, rel=0.05)


# --- Network fabrics (added 2026-10-20) --------------------------------------

def test_exposed_allreduce_step():
    intra, inter = Scenarios.ExposedAllReduceIntraNode, Scenarios.ExposedAllReduceInterNode
    assert sum(q.to("ms").magnitude for q in intra.compute_phases.values()) == approx(intra.compute_time.to("ms").magnitude)
    assert inter.step_time.to("ms").magnitude == approx(230)
    assert inter.exposed_communication_fraction == approx(0.13043)
    assert inter.compute_fraction == approx(0.86957)
    assert intra.compute_fraction == approx(0.99502)


def test_ecmp_hash_collision():
    s = Scenarios.EcmpHashCollision
    assert (s.flows, s.paths, s.worst_load, s.idle_paths) == (8, 4, 4, 1)
    assert s.even_load == approx(2.0)
    assert s.slowdown == approx(2.0)


def test_spine_optics_buildout():
    s = Scenarios.SpineOpticsTraining10K
    assert s.n_links == 20_000
    assert s.total_cost.to("USD").magnitude == approx(10e6)
    assert s.total_power.to("kW").magnitude == approx(400)


def test_go_back_n_window():
    assert Scenarios.GoBackNResendNdr.product.to("KB").magnitude == approx(250)


def test_alpha_beta_ndr_roce():
    r = Scenarios.AlphaBetaNdrRoce.result
    t = {p: {m: v.to("microsecond").magnitude for m, v in d.items()} for p, d in r.times.items()}
    assert t["NDR"]["control"] == approx(1.58)
    assert t["RoCE 100G"]["control"] == approx(5.32)
    assert t["RoCE 400G"]["control"] == approx(5.08)
    assert t["NDR"]["shard"] == approx(7001.5)
    assert t["RoCE 100G"]["shard"] == approx(28005)
    assert t["RoCE 400G"]["shard"] == approx(7005)
    assert r.crossovers["NDR"].to("KB").magnitude == approx(75)
    assert r.crossovers["RoCE 100G"].to("KB").magnitude == approx(62.5)


def test_alpha_beta_hdr_ndr_small_message():
    r = Scenarios.AlphaBetaHdrNdrSmallMessage.result
    hdr, ndr = (r.times[p]["small"].to("microsecond").magnitude for p in ("HDR", "NDR"))
    assert (hdr, ndr) == (approx(1.9), approx(1.7))
    assert 1 - ndr / hdr == approx(0.10526)


def test_nccl_allreduce_fit():
    r = Scenarios.NcclAllReduceH100NodeFit.result
    assert r.row_count == 32
    assert r.floor_time.to("microsecond").magnitude == approx(17.04)
    assert r.bandwidth.to("GB/s").magnitude == approx(274.23)
    assert r.knee.to("MB").magnitude == approx(4.6729)
    assert r.worst_mid_ratio == approx(1.7435)
    assert r.peak_bus_bandwidth.to("GB/s").magnitude == approx(480.16)
    assert r.mid_bus_bandwidth_low.to("GB/s").magnitude == approx(176.77)
    assert r.mid_bus_bandwidth_high.to("GB/s").magnitude == approx(400.53)
    assert r.model_time(r.knee).to("microsecond").magnitude == approx(2 * 17.04)


def test_ring_allreduce_across_oversubscribed_spine():
    r = Scenarios.RingAllReduceAcross4to1Spine.result
    assert r.devices_per_side == 512
    assert r.bisection_full.to("TB/s").magnitude == approx(25.6)
    assert r.bisection_oversubscribed.to("TB/s").magnitude == approx(6.4)
    assert r.ring_factor == approx(1.998047)
    assert r.crossing_bytes.to("GB").magnitude == approx(5115)
    assert r.time_full.to("second").magnitude == approx(0.19980)
    assert r.time_oversubscribed.to("second").magnitude == approx(0.79922)
    assert r.slowdown == approx(4.0)
    assert r.relative_throughput == approx(0.52632)
    assert r.idle_cost.to("USD").magnitude == approx(142.105e6)
    assert Scenarios.RingAllReduceAcross2to1Spine.result.throughput_loss == approx(0.23077)


def test_fat_tree_reach():
    k64, k256 = Scenarios.FatTreeNdrSwitch, Scenarios.FatTree102T4Switch
    assert (k64.radix, k64.two_tier_hosts, k64.three_tier_hosts, k64.three_tier_switches, k64.pod_hosts) == (
        64, 2048, 65536, 5120, 1024)
    assert (k256.radix, k256.two_tier_hosts, k256.three_tier_hosts) == (256, 32768, 4194304)


def test_torus_against_fat_tree():
    s = Scenarios.TorusVsFatTree4K
    assert (s.n_endpoints, s.side, s.torus_cut_links, s.fat_tree_cut_links) == (4096, 16, 512, 2048)
    assert s.torus_bisection.to("TB/s").magnitude == approx(25.6)
    assert s.fat_tree_bisection.to("TB/s").magnitude == approx(102.4)
    assert (s.torus_diameter, s.fat_tree_path_switches) == (24, 5)


def test_rail_hops():
    lat = Scenarios.RailVsLeafSpineHops.latencies
    assert lat["leaf-spine-leaf"].to("microsecond").magnitude == approx(1.8)
    assert lat["rail"].to("microsecond").magnitude == approx(0.6)


def test_incast_superpod_unit():
    r = Scenarios.IncastSuperPodUnit.result
    assert r.offered_load.to("TB/s").magnitude == approx(12.8)
    assert r.oversubscription == approx(256)
    assert r.net_fill_rate.to("TB/s").magnitude == approx(12.75)
    assert r.overflow_time.to("microsecond").magnitude == approx(2.5098)
    assert r.drain_time.to("microsecond").magnitude == approx(640)
    assert r.bandwidth_delay_product.to("KB").magnitude == approx(250)


def test_component_exposure():
    t = Scenarios.TransceiverDegradationRoce4K
    assert t.n_components == 24576
    assert t.p_any == approx(0.21789)
    links = Scenarios.DegradedLinksReferenceFabric
    assert links.expected_events == approx(3.0)
    assert links.p_any == approx(0.95029)


def test_silent_slowdown_run():
    s = Scenarios.DegradedTransceiverRun1K
    assert s.device_hours == approx(737280)
    assert s.lost_device_hours["low"] == approx(14745.6)
    assert s.lost_device_hours["high"] == approx(22118.4)
    assert s.lost_cost["high"].to("USD").magnitude == approx(44236.8)
    assert s.no_overlap_slowdown == approx(0.033333)


def test_nic_sharing_and_diagnosis():
    n = Scenarios.SriovNdrNic
    assert n.slice_rate.to("Gbps").magnitude == approx(50)
    assert n.burst_time.to("second").magnitude == approx(2)
    d = Scenarios.HalfRateLinkDiagnosis
    assert d.slowdown == approx(0.142857)
    assert d.expected_bandwidth.to("GB/s").magnitude == approx(48)
    assert d.measured_bandwidth.to("GB/s").magnitude == approx(24)
