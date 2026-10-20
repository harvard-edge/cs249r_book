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

# --- Feeding and saving a training fleet (added 2026-10-20) -----------------

def test_storage_footprint_of_the_175b_run():
    r = Scenarios.Gpt3StorageFootprint.result
    assert r.checkpoint_bytes.to("GB").magnitude == approx(2450)
    assert r.saves == 4320
    assert r.checkpoint_bytes_written.to("PB").magnitude == approx(10.584)
    assert r.write_read_ratio == approx(1764)
    assert r.average_read_rate.to("MB/s").magnitude == approx(2.3148)
    assert r.shard_count == 2000
    assert r.archived_checkpoints == 44
    assert r.archive_bytes.to("TB").magnitude == approx(116.8)


def test_tiered_checkpoint_and_storm():
    r = Scenarios.Gpt3TieredCheckpoint2K.result
    assert r.per_node_shard.to("GB").magnitude == approx(9.5703)
    assert r.local_write_time.to("s").magnitude == approx(0.34180)
    assert r.drain_time.to("s").magnitude == approx(2.45)
    assert r.durable_share_per_node.to("GB/s").magnitude == approx(3.90625)
    assert r.pause_fraction == approx(5.6966e-4)
    s = Scenarios.Llama2_70BCheckpointStorm1K.result
    assert s.state_bytes.to("GB").magnitude == approx(980)
    assert s.storm_bytes.to("TB").magnitude == approx(1003.52)
    assert s.storm_time.to("s").magnitude == approx(10035.2)
    assert s.single_copy_time.to("s").magnitude == approx(9.8)
    assert s.slowdown == approx(1024)


def test_input_demand_stall_and_prefetch():
    assert Scenarios.TextInputDemand2K.result.bandwidth.to("MB/s").magnitude == approx(167.77)
    assert Scenarios.ImageInputDemand2K.result.bandwidth.to("GB/s").magnitude == approx(303.19)
    assert Scenarios.VideoInputDemand2K.result.bandwidth.to("GB/s").magnitude == approx(2621.44)
    st = Scenarios.OverlappedInputStall.result
    assert st.sequential_step.to("ms").magnitude == approx(450)
    assert st.sequential_stall == approx(0.55556)
    assert st.overlapped_step.to("ms").magnitude == approx(250)
    assert st.overlapped_stall == approx(0.2)
    wall = Scenarios.QuarterDeliveryIoWall
    assert [wall.relative_throughput(s) for s in (1, 2, 4, 16)] == [approx(1), approx(2), approx(4), approx(4)]
    assert wall.plateau == approx(4)
    p = Scenarios.PrefetchDepthByTier.result
    assert p.spans["object"] == approx(2.5)
    assert p.depths == {"nvme": 1, "pfs": 1, "object": 3}
    assert p.buffer_per_node.to("GB").magnitude == approx(1.6)


def test_tiered_cache():
    h = Scenarios.HotTierCacheExample.result
    assert h.mean_service == approx(5.95)
    assert h.delivered_fraction == approx(0.16807)
    m = Scenarios.MultimodalCorpusNodeCache.result
    assert m.hit_rate == approx(0.87771)
    assert m.miss_penalty == approx(7)
    assert m.delivered_rate.to("GB/s").magnitude == approx(16.150)
    assert m.delivered_fraction == approx(0.57680)


def test_host_cpu_tail_and_metadata():
    c = Scenarios.GpuDirectStorageNodeCpu.result.cpu_seconds_per_second
    assert c["host"] == approx(7.68)
    assert c["gds"] == approx(1.92)
    assert Scenarios.StorageTailFanout100.all_up == approx(0.36603)
    assert Scenarios.ImageFileMetadataMeltdown.files_per_shard == approx(4000)
    assert Scenarios.ParallelFileSystem1TBps.servers == 100


def test_storage_economics():
    b = Scenarios.ObjectStorage100TB.result
    assert b.storage_cost == approx(24000)
    assert b.egress_per_read == approx(9000)
    assert b.egress_cost == approx(90000)
    assert b.request_cost == approx(40)
    assert b.archive_retrieval_cost == approx(2000)
    assert b.archive_restore_cost == approx(11000)
    v = Scenarios.VisionCorpusStreamVsStage.result
    assert v.stream_cost == approx(372000)
    assert v.stage_cost == approx(90000)
    assert v.savings == approx(282000)
    assert v.break_even_epochs == approx(4.3333)
    u = Scenarios.InputStallIdleCost1K.result
    assert u.fleet_cost_per_day == approx(49152)
    assert u.loss_per_day == approx(9830.4)
    assert u.loss_total == approx(294912)
    s = Scenarios.SyntheticDataProvenance.result
    assert s.stored_bytes.to("TB").magnitude == approx(1.4)
    assert s.pipeline_bytes.to("TB").magnitude == approx(4.4)


def test_measured_storage_demand():
    r = Scenarios.MLPerfStorageV1Demand.result
    assert r.result_count == 124
    assert r.medians["unet3d"]["H100"] == approx(2.7530)
    assert r.medians["resnet50"]["A100"] == approx(0.091586)
    assert min(r.generation_ratios.values()) == approx(1.5877)
    assert max(r.generation_ratios.values()) == approx(1.9795)


def test_weight_loads_and_rollout():
    t = Scenarios.Gpt3WeightsColdStart.result.times
    assert t["object store"].to("s").magnitude == approx(350)
    assert t["file system share"].to("s").magnitude == approx(87.5)
    assert t["one Gen5 NVMe"].to("s").magnitude == approx(25)
    assert t["eight Gen4 NVMe"].to("s").magnitude == approx(6.25)
    assert t["eight PCIe links"].to("s").magnitude == approx(0.68359)
    assert t["host DRAM"].to("s").magnitude == approx(1.75)
    r = Scenarios.Gpt3RolloutFanout100
    assert r.total_bytes.to("TB").magnitude == approx(35)
    assert r.result.times["file system"].to("s").magnitude == approx(35)
    assert r.result.times["object store"].to("s").magnitude == approx(350)


def test_codec_decode_cores():
    c = Scenarios.NodeStripeCodecs
    assert c.output_bandwidth("light").to("GB/s").magnitude == approx(84)
    assert c.decode_cores("light") == approx(56)
    assert c.decode_cores("dense") == approx(224)


# --- Embodied stopping, floor traction, and domain compute (added 2026-10-20) --

def test_warehouse_elementary_stop():
    r = Scenarios.WarehouseAisleStopAtDriveLimit.result
    assert r.delay.to("ms").magnitude == approx(133.6)
    assert r.delay_travel.to("mm").magnitude == approx(200.4)
    assert r.braking_travel.to("mm").magnitude == approx(562.5)
    assert r.stop_distance.to("mm").magnitude == approx(762.9)
    assert r.spare.to("mm").magnitude == approx(337.1)
    assert r.fits
    assert r.crossover_speed.to("m/s").magnitude == approx(0.5344)
    assert r.braking_over_delay == approx(2.8069)
    assert r.delay_share_of_braking == approx(0.35627)
    assert r.speed_ceiling.to("m/s").magnitude == approx(1.8474)
    from mlsysim.core.units import Q_
    assert r.travel_over(Q_(1, "ms")).to("mm").magnitude == approx(1.5)
    assert r.travel_over(Q_(22, "ms")).to("mm").magnitude == approx(33)


def test_warehouse_stop_variants():
    h = Scenarios.WarehouseAisleStopHalfSpeed.result
    assert h.delay_travel.to("mm").magnitude == approx(100.2)
    assert h.braking_travel.to("mm").magnitude == approx(140.625)
    assert h.stop_distance.to("mm").magnitude == approx(240.825)
    d = Scenarios.WarehouseAisleStopDoubledLease.result
    assert d.delay.to("ms").magnitude == approx(193.6)
    assert d.stop_distance.to("mm").magnitude == approx(852.9)
    assert d.spare.to("mm").magnitude == approx(247.1)
    assert Scenarios.WarehouseScannerSkippedDelay.result.delay_travel.to("mm").magnitude == approx(123)
    assert Scenarios.WarehouseInferenceHiccup.result.delay_travel.to("mm").magnitude == approx(150)


def test_tempe_through_the_stopping_budget():
    s = Scenarios.TempeActionSuppression.result
    assert s.delay_travel.to("m").magnitude == approx(18.709)
    assert s.braking_travel.to("m").magnitude == approx(20.669)
    assert s.crossover_speed.to("m/s").magnitude == approx(18.044)
    assert Scenarios.TempeActionSuppressionAdsCap.result.braking_travel.to("m").magnitude == approx(26.640)
    f = Scenarios.TempeFirstDetectionStop.result
    assert f.braking_travel.to("m").magnitude == approx(21.442)
    assert f.braking_time.to("s").magnitude == approx(2.1802)
    assert f.clear_over_stop == approx(4.9753)


def test_floor_traction_loss():
    r = Scenarios.HydraulicFluidSlickTraction.result
    assert r.braking_distance_ratio == approx(4.0)
    assert r.traction_ceiling.to("m/s**2").magnitude == approx(1.9613)
    assert r.required_friction == approx(0.20394)
    assert not r.premise_holds


def test_robot_policy_compute_gap():
    v = Scenarios.OpenVlaComputeGap.result
    assert v.reference_model == "Llama 3.1-405B"
    assert v.gap == approx(345.45)
    g = Scenarios.Gr00tN1ComputeGap.result
    assert g.reference_model == "Grok 4"
    assert g.gap == approx(7018.8)
    assert g.domain_models_with_compute == 25


# --- Agent trajectories (added 2026-10-20) -----------------------------------

def test_unchecked_horizon_compounding():
    r = Scenarios.UncheckedRepairHorizon.result
    assert r.success[10] == approx(0.904382)
    assert r.success[100] == approx(0.366032)
    assert r.success[200] == approx(r.success[100] ** 2)
    assert r.required_step_error == approx(0.0010530503)
    assert r.error_ratio == approx(9.49622)
    assert r.required_error_horizon == approx(0.105305)
    assert r.small_error_limit == approx(0.1053605)


def test_cost_per_accepted_task_levers():
    r = Scenarios.RepairAttemptCostLevers.result
    assert r.attempt_cost == approx(2.10)
    assert r.cost_per_accepted == approx(5.25)
    assert r.attempts_per_acceptance == approx(2.5)
    assert r.accepted_only_understatement == approx(2.5)
    assert r.cheaper_attempt_cost == approx(1.45)
    assert r.cost_per_accepted_cheaper_tokens == approx(3.625)
    assert r.cost_per_accepted_improved == approx(3.50)
    assert r.token_share == approx(0.6190476)
    assert r.break_even_acceptance == approx(0.5793103)
    # the break-even acceptance prices the cheaper attempt at the original attempt cost
    assert r.attempt_cost / r.break_even_acceptance == approx(r.cost_per_accepted_cheaper_tokens)
    assert r.cost_per_accepted_cheaper_other == approx(4.25)
    assert r.cost_per_accepted_acceptance_gain == approx(4.20)
    assert r.independent_success == approx(0.92224)


def test_append_only_context():
    r = Scenarios.AppendOnlyRepairContext.result
    assert r.last_call_tokens.to("token").magnitude == 51500
    assert r.window_fraction == approx(0.40234375)
    assert r.cumulative_input_tokens.to("token").magnitude == 892500
    assert r.reread_ratio == approx(17.330097)


def test_task_horizon_trend_reproduces_source():
    s = Scenarios.FrontierTaskHorizonTrend
    r = s.result
    assert r.doubling_recent.to("day").magnitude == pytest.approx(s.published_recent_doubling.to("day").magnitude, abs=1)
    assert r.doubling_all.to("day").magnitude == approx(187.7775, rel=1e-4)
    assert r.growth == approx(19427.65, rel=1e-4)
    assert r.first_date == "2019-02-14"


def test_recorded_repair_agent_run():
    r = Scenarios.MiniSweAgentSonnet4Run.result
    assert (r.tasks, r.accepted, r.rejected, r.submitted) == (500, 324, 176, 500)
    assert r.total_cost == approx(185.726585)
    assert r.cost_per_accepted == approx(r.total_cost / r.accepted)
    assert r.accepted_mean_cost == approx(0.3200875)
    assert r.accepted_only_understatement == approx(1.790854)
    assert (r.calls_median, r.calls_min, r.calls_max) == (33, 9, 123)
    assert r.commands == {"read": 8826, "write": 5284, "run": 4463, "net": 9}
    assert (r.observations, r.nonzero_exits, r.truncated_outputs) == (18076, 1349, 298)
    assert sum(b["tasks"] for b in r.call_bins) == r.tasks
    assert [b["accepted"] for b in r.call_bins] == [54, 112, 80, 57, 21]
    t = Scenarios.AstropyBitmaskRepair
    assert (t.model_calls, t.resolved, t.files_in_submission, t.hidden_tests) == (28, True, 6, 180)
    assert t.cost == approx(0.342415)
# --- Model calls and agent trajectories (added 2026-10-20) -------------------

def test_int8_call_floors():
    f = Scenarios.Llama3_70B_Int8DecodeH100.result
    assert f.weight_bytes.to("GB").magnitude == approx(70.6)
    assert f.compute_time.to("us").magnitude == approx(71.349)
    assert f.memory_time.to("ms").magnitude == approx(21.075)
    assert f.bound_ratio == approx(295.37)


def test_trajectory_resend():
    r = Scenarios.CodingAgentResend30Turns.result
    assert r.turns == 30
    assert r.resent_tokens == 892500
    assert r.first_prompt_term == 240000 and r.appended_term == 652500
    assert r.reused_prefix_tokens == 51500
    assert r.reuse_reduction == approx(17.330)
    assert r.output_tokens == 6000
    assert r.input_share == approx(0.99332)
    assert r.prefill_time.to("s").magnitude == approx(63.679)
    assert r.reused_prefill_time.to("s").magnitude == approx(3.6745)
    assert r.decode_time.to("s").magnitude == approx(126.45)
    r60 = Scenarios.CodingAgentResend60Turns.result
    assert r60.resent_tokens == 3135000 and r60.reused_prefix_tokens == 96500
    assert Scenarios.CodingAgentResend90Turns.result.resent_tokens == 6727500


def test_model_calls():
    c = Scenarios.CodingAgentFirstTurnCall.result
    assert c.prefill_time.to("s").magnitude == approx(0.57079)
    assert c.decode_time.to("s").magnitude == approx(4.2149)
    assert c.output_time_share == approx(0.88073)
    assert c.output_token_share == approx(0.024390)
    w = Scenarios.CodingAgentWholeFileEdit.result
    assert w.total_time.to("s").magnitude == approx(39.584)
    assert w.output_time_share == approx(0.98495)
    e = Scenarios.CodingAgentLocalizedEdit.result
    assert e.total_time.to("s").magnitude == approx(2.2653)
    assert e.deadline_margin == approx(6.6216)
    assert e.acceptance == approx(0.57)
    s = Scenarios.SupportAgentTurnCall.result
    assert s.prefill_time.to("s").magnitude == approx(0.58449)
    assert s.decode_time.to("s").magnitude == approx(3.1612)
    assert s.fresh_prefill_time.to("s").magnitude == approx(0.14612)
    assert s.decode_rate.to("1/s").magnitude == approx(47.450)
    k = Scenarios.CanceledStreamDecodeSavings
    assert k.skipped_steps == 4000
    assert k.decode_time_avoided.to("s").magnitude == approx(84.299)


def test_admission_sampling_checker():
    a = Scenarios.MultimodalContextAdmission.result
    assert (a.input_ceiling, a.headroom, a.total_demand, a.excess) == (7168, 1168, 9072, 880)
    assert not a.admitted
    h = Scenarios.MultimodalContextAdmissionHalfReserve.result
    assert (h.total_demand, h.excess) == (8560, 368)
    t = Scenarios.TemperatureNucleusExample.result
    assert t.base_probabilities[0] == approx(0.78399)
    assert t.sharp_probabilities[0] == approx(0.95032)
    assert t.tail_ratio == approx(4.3483)
    assert t.nucleus_size == 2 and t.nucleus_mass == approx(0.95893)
    assert t.nucleus_probabilities[0] == approx(0.81757)
    p = Scenarios.PatchCheckerPosterior.result
    assert p.posterior == approx(0.14373)
    assert p.defective_pass == approx(0.117)
    assert p.pass_share == approx(0.814)
    assert Scenarios.PatchCheckerHalvedFalseAccept.result.posterior == approx(0.077432)


def test_horizon_cost_and_trace_mix():
    h = Scenarios.HorizonCompounding30Steps
    assert h.result.success[h.short_horizon] == approx(0.73970)
    m = Scenarios.LlmCodeTraceTokenMix.result
    assert m.requests == 8819
    assert (m.median_prompt, m.median_output) == (1469, 13)
    assert m.input_share == approx(0.98657)
    assert m.output_time_share == approx(0.80086)
# --- Collective communication (added 2026-10-20) ---------------------------

def test_critical_size_and_fusion():
    x = Scenarios.CriticalSizeByInterconnect.result.crossovers
    assert x["NVLink"].to("KB").magnitude == approx(675)
    assert x["PCIe Gen5"].to("KB").magnitude == approx(224)
    assert x["RoCE"].to("KB").magnitude == approx(93.75)
    assert x["HDR"].to("KB").magnitude == approx(87.5)
    assert x["NDR"].to("KB").magnitude == approx(75)
    f = Scenarios.RoutingMessageFusionNdr
    assert f.separate_time.to("microsecond").magnitude == approx(150)
    assert f.fused_time.to("microsecond").magnitude == approx(9.5)
    assert f.speedup == approx(15.789)


def test_allreduce_schedules():
    b = Scenarios.DdpBucketRingNdr.result
    assert b.chunk[64].to("KB").magnitude == approx(390.625)
    assert b.chunk[1024].to("KB").magnitude == approx(24.414)
    assert b.ring_rounds[1024] == 2046
    assert b.ring_time[64].to("ms").magnitude == approx(1.1734)
    assert b.ring_time[1024].to("ms").magnitude == approx(4.0680)
    g = Scenarios.GradientBufferAllReduceNdr.result
    assert g.ring_byte_time[64].to("ms").magnitude == approx(39.375)
    assert g.tree_byte_time[64].to("ms").magnitude == approx(240)
    assert g.star_byte_time[64].to("ms").magnitude == approx(1280)
    assert g.star_ring_limit[64] == 32
    assert g.tree_rounds == {8: 6, 64: 12, 1024: 20}
    assert g.crossover_estimate[1024].to("MB").magnitude == approx(7.68)
    r = Scenarios.RingVsTreeSmallBuffer.result
    assert r.ring_startup_time[64].to("microsecond").magnitude == 1260
    assert r.tree_startup_time[64].to("microsecond").magnitude == 120
    assert r.ring_time[64].to("microsecond").magnitude == approx(1456.875)
    assert r.tree_time[64].to("microsecond").magnitude == approx(1320)
    assert r.crossover_estimate[64].to("MB").magnitude == approx(1.0667)
    assert r.crossover_exact[64].to("MB").magnitude == approx(1.1364)
    s = Scenarios.Llama2_70BGradientRing64Ndr
    assert s.result.ring_byte_time[64].to("ms").magnitude == approx(5512.5)
    assert s.result.ring_startup_time[64].to("ms").magnitude == approx(0.189)
    assert s.saved_device_hours(64) == approx(4900.2)


def test_hierarchical_allreduce():
    h = Scenarios.HierarchicalAllReduce8x8DgxH100.result
    assert h.reduce_scatter_time.to("ms").magnitude == approx(1.9444)
    assert h.inter_time.to("ms").magnitude == approx(4.396)
    assert h.total_time.to("ms").magnitude == approx(8.2849)
    assert h.flat_bound_time.to("ms").magnitude == approx(40)
    assert h.speedup_vs_flat_bound == approx(4.8281)
    a = Scenarios.HierarchicalAllReduceReferenceCluster.result
    assert a.flat_ring_time.to("s").magnitude == approx(13.9864, rel=1e-2)
    assert 4 < a.speedup_vs_flat_ring < 5
    assert Scenarios.HierarchicalAllReduceBucket100MB.result.total_time.to("ms").magnitude == approx(0.84748)


def test_moe_dispatch():
    d = Scenarios.MoeDispatch8Gpu64Experts.result
    assert d.tokens_per_pair == 64
    assert d.sent_bytes.to("MB").magnitude == approx(1.835)
    assert d.injection_bound.to("microsecond").magnitude == approx(38.20016)
    assert d.hot_remote_tokens == 672
    assert d.uniform_receive_bound.to("microsecond").magnitude == approx(36.70016)
    assert d.hot_receive_bound.to("microsecond").magnitude == approx(55.05024)


def test_ring_trace_and_error_feedback():
    t = Scenarios.RingTraceFourGpu.result
    assert t.totals == [10, 26, 18, 34]
    assert t.scatter_reduce[0] == [(3, 17), (0, 3), (1, 13), (2, 11)]
    assert t.held == [{0, 1, 2, 3}] * 4
    e = Scenarios.ErrorFeedbackFiveSteps.result
    assert e.sent == [0, 1, 0, 0, 1]
    assert e.final_error == approx(-0.4)
    assert e.naive_total_sent == 0
    assert Scenarios.ErrorFeedbackConstantGradient.result.total_sent == 3


def test_compression_and_overlap():
    p = Scenarios.CompressionPayback8x
    assert p.total_time.to("ms").magnitude == approx(7)
    assert p.speedup == approx(5.7143)
    assert p.pays(p.communication_time) and not p.pays(p.fast_communication_time)
    c = Scenarios.Int8CompressionRun.result
    assert c.compressed_totals["slow"].to("s").magnitude == approx(38500)
    assert c.compressed_totals["fast"].to("s").magnitude == approx(33825)
    assert c.changes["slow"] == approx(-0.23)
    r = Scenarios.GradientEncodingRatios
    assert r.ratio("INT8", "FP32") == approx(4)
    assert r.ratio("Top-k", "FP32") == approx(50)
    assert r.ratio("1-bit", "BF16") == approx(16)
    assert Scenarios.ShardedDataParallelCalls7B.calls("FSDP, reshard after forward") == 96
    o = Scenarios.OverlapBudget7B64Gpu.result
    assert o.buckets_per_layer == 9
    assert o.sequential_time.to("ms").magnitude == approx(1344)
    assert o.overlapped_time.to("ms").magnitude == approx(879)
    assert Scenarios.OverlapBudget7B64GpuIdealBuckets.result.overlapped_time.to("ms").magnitude == approx(487.63)
# --- Distributed training (added 2026-10-20) ---------------------------------

def test_parallel_layouts_175b():
    a = Scenarios.Gpt3DataParallel1K.result
    assert a.n_devices == 1024
    assert a.step_flops.to("flop").magnitude == approx(2.1e18)
    assert a.compute_time.to("s").magnitude == approx(2.1)
    assert a.ring_bytes.to("GB").magnitude == approx(699.32)
    assert a.allreduce_time.to("s").magnitude == approx(14.0)
    assert a.exposed_time.to("s").magnitude == approx(11.9)
    assert a.efficiency == approx(0.15)
    assert a.overlap_share == approx(0.15)
    assert a.state_per_device.to("GB").magnitude == approx(2800)
    assert a.min_devices_for_state == 35
    b = Scenarios.Gpt3TensorDataParallel1K.result
    assert b.dp_shard.to("GB").magnitude == approx(43.75)
    assert b.allreduce_time.to("s").magnitude == approx(1.75)
    assert b.efficiency_no_overlap == approx(0.54545)
    assert b.efficiency == approx(0.82759)
    assert b.fleet_mfu == approx(0.41379)
    assert b.double_counted_mfu == approx(0.34245)
    c = Scenarios.Gpt3ThreeDParallel1K.result
    assert c.dp_shard.to("GB").magnitude == approx(10.9375)
    assert c.bubble == approx(3 / 35)
    assert c.efficiency == approx(0.87273)
    assert c.state_per_device.to("GB").magnitude == approx(87.5)
    assert c.zero1_state_per_device.to("GB").magnitude == approx(23.926)
    d = Scenarios.Gpt3ThreeDParallel16K.result
    assert d.n_devices == 16384
    assert d.bubble == approx(15 / 47)
    assert d.state_per_device.to("GB").magnitude == approx(21.875)


def test_zero_sharding_and_activations():
    z = Scenarios.Llama2_7B_ZeroStages64.result
    assert z.stage_state["DDP"].to("GB").magnitude == approx(112)
    assert z.stage_state["ZeRO-1"].to("GB").magnitude == approx(29.3125)
    assert z.stage_state["ZeRO-2"].to("GB").magnitude == approx(15.53125)
    assert z.stage_state["ZeRO-3"].to("GB").magnitude == approx(1.75)
    assert z.optimizer_state.to("GB").magnitude == approx(84)
    assert z.traffic_multiple["ZeRO-3"] == approx(1.5)
    g = Scenarios.Gpt3Zero3On64.result
    assert g.stage_state["ZeRO-3"].to("GB").magnitude == approx(43.75)
    assert g.activations.to("GB").magnitude == approx(4.8318)
    assert g.zero3_total_per_device.to("GB").magnitude == approx(48.582)
    assert g.gather_traffic.to("TB").magnitude == approx(1.05)
    assert g.gather_time.to("s").magnitude == approx(21)
    act = Scenarios.Gpt3ActivationsMicrobatch4
    assert act.activations("none").to("TB").magnitude == approx(1.1017)
    assert act.activations("full").to("GB").magnitude == approx(19.327)
    assert act.recompute_overhead == approx(1 / 3)


def test_gpt2_data_parallel_scale_out():
    one = Scenarios.Gpt2DataParallelOneNode.result
    assert one.intra_bytes.to("GB").magnitude == approx(5.25)
    assert one.communication_time.to("ms").magnitude == approx(11.667)
    assert one.efficiency == approx(0.99356)
    assert one.training_time.to("hour").magnitude == approx(3.1453)
    four = Scenarios.Gpt2DataParallelFourNodes10GbE.result
    assert four.inter_bytes.to("GB").magnitude == approx(5.8125)
    assert four.inter_time.to("ms").magnitude == approx(4650)
    assert four.update_time.to("ms").magnitude == approx(6461.7)
    assert four.speedup == approx(8.9141)
    assert four.efficiency == approx(0.27857)
    assert four.cost == approx(358.98)
    acc = Scenarios.Gpt2GradientAccumulationOneNode.result
    assert acc.global_batch == 512
    assert acc.communication_overhead == approx(0.0016204)
    assert acc.training_time.to("hour").magnitude == approx(3.1301)
    assert acc.cost == approx(100.16)


def test_pipeline_tensor_parallel_and_ring_attention():
    g = Scenarios.GPipeFourStageWalkthrough
    assert (g.slots, g.idle_slots) == (14, 6)
    assert g.bubble == approx(3 / 7)
    assert Scenarios.Pipeline8Stage32Micro.bubble == approx(7 / 39)
    assert Scenarios.Pipeline16Stage32Micro.microbatches_for_bubble(0.1) == approx(135)
    assert Scenarios.Pipeline8Stage8Micro.bubble == approx(7 / 15)
    t = Scenarios.Gpt3TensorParallelLayerTp8.result
    assert t.compute_time.to("ms").magnitude == approx(1.8874)
    assert t.transfer_times["NVLink"].to("ms").magnitude == approx(0.39147)
    assert t.transfer_times["InfiniBand NDR"].to("ms").magnitude == approx(3.5232)
    r = Scenarios.H100RingAttentionChunk.min_chunk_tokens
    assert r["NVLink"] == approx(1098.9)
    assert r["InfiniBand NDR"] == approx(9890)


def test_sync_strategies_and_critical_batch():
    b8, b64, s64 = (Scenarios.TransformerBsp8Workers, Scenarios.TransformerBsp64Workers,
                    Scenarios.TransformerSsp64Workers)
    assert (b8.speedup, b64.speedup, s64.speedup) == (approx(6.7935), approx(40.096), approx(38.095))
    assert (b8.cost, b64.cost, s64.cost) == (approx(26.169), approx(35.470), approx(37.333))
    assert s64.stale_lr_factor == approx(0.44721)
    c = Scenarios.CriticalBatchPilotTrace7B
    assert c.critical_batch == approx(16000)
    assert c.global_batch == 8192
    assert c.crossover_devices == approx(125)


def test_alignment_memory_and_fleet_facts():
    a = Scenarios.Llama2_70B_PpoAlignment.result
    assert a.policy_state.to("GB").magnitude == approx(1120)
    assert a.ppo_total.to("GB").magnitude == approx(1494)
    assert a.dpo_total.to("GB").magnitude == approx(1260)
    assert a.dpo_reduction == approx(0.15663)
    assert a.policy_per_device.to("GB").magnitude == approx(35)
    assert a.kv_cache.to("GB").magnitude == approx(85.899)
    assert a.min_devices_for_ppo == 19
    m = Scenarios.MLPerfLlmScalingSeries.result
    assert min(m.efficiencies) == approx(0.70786)
    assert max(m.efficiencies) == approx(0.83852)
    assert Scenarios.GpuFailuresTraining1K.result.interval.to("hour").magnitude == approx(48.828)
    assert Scenarios.StragglerFreeStep1K.all_up == approx(4.3171e-5)


def test_memory_wall_residual_share_and_recompute():
    w = Scenarios.NotableModelMemoryWall.result
    assert w.capacity_growth == approx(15)
    assert w.demand_orders == approx(3.9545)
    assert (w.crossover_name, w.crossover_year) == ("GPT-3", 2020)
    assert w.crossover_gap == approx(34.92)
    r = Scenarios.ResidualOnePercent512
    assert r.residual_share == approx(0.83797)
    n = Scenarios.NaiveFourStagePipeline
    assert (n.slots, n.idle_slots, n.fill_slots, n.phase_slots) == (8, 6, 3, 4)
    a = Scenarios.Gpt3ActivationsMicrobatch4
    assert (a.forward_equivalents, a.recompute_forward_equivalents) == (approx(3), approx(4))
# --- Single-machine iron law, Amdahl, and silent degradation (added 2026-10-20) ---

def test_input_space_and_gpt3_cluster_time():
    i = Scenarios.ImageNetInputSpace
    assert i.elements == 150528
    assert i.digits == 362508
    assert i.test_examples == 50000
    assert Scenarios.Gpt3TrainingTime1KA100.training_time.to("day").magnitude == approx(25.278)
    assert Scenarios.Gpt3TrainingTime1KA100HigherUtilization.training_time.to("day").magnitude == approx(18.959)


def test_iron_law_examples():
    r = Scenarios.IronLawMemoryBoundExample.result
    assert r.memory_time.to("ms").magnitude == approx(20)
    assert r.compute_time.to("ms").magnitude == approx(10)
    assert r.sequential_time.to("ms").magnitude == approx(35)
    assert r.overlapped_time.to("ms").magnitude == approx(25)
    assert r.upgraded_sequential_time.to("ms").magnitude == approx(25)
    assert r.upgraded_overlapped_time.to("ms").magnitude == approx(15)
    assert r.naive_upgraded_time.to("ms").magnitude == approx(17.5)
    assert r.sequential_reduction == approx(0.28571)
    c = Scenarios.IronLawComputeHeavyExample.result
    assert c.sequential_time.to("s").magnitude == approx(11)
    assert c.upgraded_compute_time.to("s").magnitude == approx(4)
    assert c.upgraded_sequential_time.to("s").magnitude == approx(7)
    assert c.naive_upgraded_time.to("s").magnitude == approx(5.5)


def test_single_node_iron_law_terms():
    r = Scenarios.ResNet50Batch64StepH100
    assert r.result.latency_memory.to("ms").magnitude == approx(0.11310)
    assert r.result.latency_compute.to("ms").magnitude == approx(1.0613)
    assert r.result.latency_overhead.to("ms").magnitude == approx(0.51)
    assert r.result.latency.to("ms").magnitude == approx(1.5713)
    assert r.memory_traffic.to("GB").magnitude == approx(0.37888)
    assert r.iron_law.sequential_reduction == approx(0.033573)
    g = Scenarios.Gpt2XlDecodeStepH100
    assert g.result.latency_memory.to("ms").magnitude == approx(0.98507)
    assert g.result.latency_compute.to("ms").magnitude == approx(0.0060667)
    assert g.memory_traffic.to("GB").magnitude == approx(3.3)
    assert g.result.arithmetic_intensity.magnitude == approx(0.90909)
    assert g.iron_law.upgraded_sequential_time.to("ms").magnitude == approx(0.98860)
    assert g.iron_law.sequential_reduction == approx(0.33254)


def test_amdahl_request_path():
    a = Scenarios.ModelSpeedupRequestPath.result
    assert a.total_time.to("ms").magnitude == approx(130)
    assert a.upgraded_total_time.to("ms").magnitude == approx(100)
    assert a.untouched_time.to("ms").magnitude == approx(85)
    assert a.overall_speedup == approx(1.3)
    assert a.end_to_end_reduction == approx(0.23077)
    assert a.stage_reduction == approx(0.66667)
    assert a.stage_fraction == approx(0.34615)
    assert a.speedup_bound == approx(1.5294)


def test_recall_shift_and_unit_skew():
    p = Scenarios.PedestrianRecallWinterShift
    assert p.miss_rate_before == approx(0.05)
    assert p.miss_rate_after == approx(0.15)
    assert p.miss_rate_ratio == approx(3.0)
    k = Scenarios.AccountAgeUnitSkew
    assert k.serving_value.to("s").magnitude == approx(35596800)
    assert k.skew_ratio == approx(86400)


def test_measured_silent_degradation():
    r = Scenarios.ImageNetV2Shift.result
    assert r.model_count == 67
    assert r.baseline_count == 3
    assert (r.gap_min, r.gap_max) == (approx(6.2), approx(13.6))
    assert (r.network_gap_min, r.network_gap_max) == (approx(10.3), approx(13.6))
    assert r.mean_gap == approx(11.824)
    assert r.slope == approx(0.96763)
    assert r.intercept == approx(-9.4523)
    g = Scenarios.GoogleFluTrendsDrift.result
    assert g.season_ratios["11-12"] == approx(1.2674)
    assert g.season_ratios["12-13"] == approx(1.6572)
    assert g.stable_max_deviation == approx(0.13247)
    assert g.peak_proxy == approx(10.555)
    assert g.peak_reference == approx(6.0608)
    assert g.peak_ratio == approx(1.7415)
    assert g.max_week_ratio == approx(2.3398)
# --- Deployment paradigms: where a model can run (added 2026-10-20) ---------

def test_fiber_propagation_floors():
    arm = Scenarios.RoboticArmRegionalCloud
    assert arm.round_trip.to("ms").magnitude == approx(15.0)
    assert arm.overrun.to("ms").magnitude == approx(5.0)
    assert Scenarios.CrossCountryFiberPath.round_trip.to("ms").magnitude == approx(36.0)
    r = Scenarios.AwsUsEast1Routes.routes
    assert r["Ohio"].distance.to("km").magnitude == approx(483.9, rel=1e-2)
    assert r["Ohio"].round_trip.to("ms").magnitude == approx(4.84, rel=1e-2)
    assert r["Ohio"].measured_over_floor == approx(2.78, rel=1e-2)
    assert r["Oregon"].round_trip.to("ms").magnitude == approx(35.03, rel=1e-2)
    assert r["Tokyo"].measured_round_trip.to("ms").magnitude == approx(148.291)


def test_factory_cameras_and_drone_offload():
    c = Scenarios.FactoryCameras1080p
    assert c.stream_rate.to("MB/s").magnitude == approx(186.624)
    assert c.total_rate.to("GB/s").magnitude == approx(18.6624)
    assert c.link_shortfall == approx(14.92992)
    assert c.event_reduction == approx(933120)
    d = Scenarios.Drone4KObstacleOffload
    assert d.payload.to("MB").magnitude == approx(24.8832)
    assert d.transmit_time.to("ms").magnitude == approx(1990.656)
    assert d.remote_lower_bound.to("ms").magnitude == approx(2100.656)
    assert d.deadline.to("ms").magnitude == approx(16.6667)
    assert d.deadline_ratio == approx(126.04)
    assert Scenarios.Drone4KObstacleOffloadCompressed.transmit_time.to("ms").magnitude == approx(19.90656)


def test_single_node_rooflines():
    a = Scenarios.ResNet50A100Batch1
    assert a.result.latency_compute.to("ms").magnitude == approx(0.026282)
    assert a.result.latency_memory.to("ms").magnitude == approx(0.027621, rel=2e-3)
    assert a.traffic.to("MB").magnitude == approx(56.32)
    assert a.intensity.to("flop/byte").magnitude == approx(145.60)
    assert a.ridge.to("flop/byte").magnitude == approx(153.02)
    m = Scenarios.ResNet50PhoneNpuBatch1
    assert m.traffic.to("MB").magnitude == approx(28.16)
    assert m.result.latency_memory.to("ms").magnitude == approx(0.55, rel=1e-2)
    p = Scenarios.A100AtTenFlopPerByte
    assert p.attainable.to("TFLOP/s").magnitude == approx(20.39)
    assert p.peak_fraction == approx(0.065353)
    apps = {x["app"]: x for x in Scenarios.TpuV1ProductionRoofline.apps}
    assert apps["MLP0"]["roof_tops"] == approx(13.630)
    assert apps["CNN0"]["roof_tops"] == approx(92.0)
    assert apps["CNN0"]["bound"] == "compute" and apps["LSTM1"]["bound"] == "memory"
    assert Scenarios.TPUv4Pod4096.peak_flops.to("EFLOP/s").magnitude == approx(1.1264)


def test_trends_and_clock_power():
    assert Scenarios.DoubledClockPower.power_ratio == approx(8.0)
    g = Scenarios.IllustrativeMemoryWall
    assert g.compute_annual_factor == approx(1.5874)
    assert g.gap == approx(16.408)
    assert Scenarios.MemoryWallFlopsTrend.growth == approx(3.0235)
    assert Scenarios.MemoryWallDramTrend.growth == approx(1.5910)
    assert Scenarios.MemoryWallInterconnectTrend.growth == approx(1.41, rel=1e-2)
    t = Scenarios.MicroprocessorTrendsAroundDennard
    assert t.growth("transistors", "mid_2000s", "early_2020s") == approx(190.79)
    assert t.peak("frequency", "after_2005") == approx(4655.5)


def test_battery_heat_and_energy():
    b = Scenarios.AlwaysOnObjectDetectorPhone
    assert b.runtime.to("hour").magnitude == approx(7.5)
    assert b.required_energy.to("Wh").magnitude == approx(48)
    assert b.budget_ratio == approx(3.2)
    t = Scenarios.UnoptimizedVisionModelPhone
    assert t.reduced_power.to("W").magnitude == approx(3.0)
    assert t.headroom.to("W").magnitude == pytest.approx(0.0, abs=1e-9)
    e = Scenarios.EnergyPerInferenceTiers
    assert e.span("kws", "llm_query") == approx(1e8)
    assert e.energies["mobilenet_phone"].to("mJ").magnitude == approx(2.5959)
    assert e.energies["resnet_edge"].to("mJ").magnitude == approx(15.91)
    assert e.energies["resnet_cloud"].to("mJ").magnitude == approx(75.467)
    o = Scenarios.RadioUploadVsMobileNetV2
    assert o.transmit_energy.to("mJ").magnitude == approx(100)
    assert o.ratio_accelerator == approx(1000)
    assert o.ratio_device == approx(38.522)


def test_doorbell_braking_voice_and_pipeline():
    f = Scenarios.SmartDoorbellFrame.result
    assert f.latency.to("ms").magnitude == approx(101.01)
    assert Scenarios.SmartDoorbell.evaluate().performance.status == "PASS"
    assert Scenarios.SmartDoorbellResponsiveChime.evaluate().performance.status == "FAIL"
    assert Scenarios.EmergencyBraking100kmh.distance.to("m").magnitude == approx(2.7778)
    v = Scenarios.VoiceAssistantBillionDevices
    assert v.stream_rate.to("KB/s").magnitude == approx(32)
    assert v.fleet_stream_rate.to("TB/s").magnitude == approx(32)
    assert v.upload_reduction == approx(864)
    assert v.fleet_query_rate.to("GB/s").magnitude == approx(37.037)
    assert v.cloud_detection_cost_per_year.to("USD/year").magnitude == approx(5e8)
    assert v.accelerator_hours_per_day == approx(1.1111e6)
    assert v.datacenters_average == approx(4.6296)
    assert v.peak_ratio == approx(4.5)
    assert v.datacenters_peak == approx(20.833)
    p = Scenarios.PhoneCameraPipeline
    assert p.total.to("ms").magnitude == approx(200)
    assert p.accelerated_total.to("ms").magnitude == approx(146)
    assert p.overall_speedup == approx(1.3699)
    assert p.limit == approx(1.4286)
# --- Physical-plant budgets of the warehouse mobile manipulator (added 2026-10-20) ---

def test_stopping_budgets_warehouse_base():
    body = Scenarios.WarehouseBaseStopBodyTerms.result
    assert body.braking_travel.to("mm").magnitude == approx(562.5)
    assert body.overhead.to("mm").magnitude == approx(150)
    assert body.delay_travel.to("mm").magnitude == approx(30)
    assert body.required_distance.to("mm").magnitude == approx(742.5)
    assert body.spare.to("mm").magnitude == approx(357.5)
    assert body.speed_ceiling.to("m/s").magnitude == approx(1.90977)
    assert body.added_delay_travel.to("mm").magnitude == approx(150)
    assert body.braking_time.to("ms").magnitude == approx(750)
    full = Scenarios.WarehouseBaseStopFullDelay.result
    assert full.delay_travel.to("mm").magnitude == approx(200.4)
    assert full.stop_distance.to("mm").magnitude == approx(762.9)
    assert full.required_distance.to("mm").magnitude == approx(912.9)
    assert full.spare.to("mm").magnitude == approx(187.1)
    assert full.speed_ceiling.to("m/s").magnitude == approx(1.70039)
    assert full.crossover_speed.to("m/s").magnitude == approx(0.5344)
    fast = Scenarios.WarehouseBaseStopFastSetpoint.result
    assert fast.required_distance.to("mm").magnitude == approx(1417.2)
    assert fast.spare.to("mm").magnitude == approx(-317.2)
    insp = Scenarios.WarehouseBaseStopInspectedFloor
    assert insp.a_brake.to("m/s**2").magnitude == approx(1.17680)
    assert insp.result.speed_ceiling.to("m/s").magnitude == approx(1.34632)
    step = Scenarios.WarehouseBaseSpeedStep.result
    assert step.braking_growth == approx(0.77778)
    assert step.braking_added.to("mm").magnitude == approx(437.5)
    assert step.equivalent_delay.to("ms").magnitude == approx(291.667)


def test_braking_ceilings_and_hub_drive():
    r = Scenarios.WarehouseBaseBrakingCeilings.result
    assert r.wheel_torque.to("N*m").magnitude == approx(60)
    assert r.force.to("N").magnitude == approx(902.256)
    assert r.m_eff.to("kg").magnitude == approx(387)
    assert r.a_torque.to("m/s**2").magnitude == approx(2.33141)
    assert r.torque_margin == approx(0.16570)
    assert r.a_torque_rated.to("m/s**2").magnitude == approx(0.77714)
    assert r.mu_required == approx(0.20394)
    assert r.mu_required_rated == approx(0.25017)
    assert r.a_traction_dry.to("m/s**2").magnitude == approx(4.79673)
    assert r.a_traction_inspected.to("m/s**2").magnitude == approx(1.17680)
    assert r.cg_ratio_max == approx(4.90333)
    assert r.a_tip is None and r.passes is False
    assert r.wheel_speed.to("rpm").magnitude == approx(107.699)
    assert r.back_emf.to("V").magnitude == approx(28.2171)
    assert r.v_brake.to("V").magnitude == approx(10.7171)
    assert r.v_drive.to("V").magnitude == approx(45.7171)
    e = Scenarios.WarehouseBaseHubDriveElectrical.result
    assert e.time_constant.to("ms").magnitude == approx(2.69286)
    assert e.ramp_time.to("ms").magnitude == approx(1.12202)
    assert e.ramp_share == approx(0.056101)
    assert e.full_step_current.to("A").magnitude == approx(60)
    assert e.short_peak_wheel_speed.to("rpm").magnitude == approx(177.308)


def test_grasp_freshness_profile():
    r = Scenarios.WristCameraGraspAgeProfile.result
    assert r.deadline.to("ms").magnitude == approx(60)
    assert r.stage_sum_p50.to("ms").magnitude == approx(35)
    assert r.stage_sum_p99.to("ms").magnitude == approx(50)
    assert r.stage_sum_max.to("ms").magnitude == approx(80.7)
    assert r.focus_share == approx(0.52857)
    assert r.focus_rate.to("Hz").magnitude == approx(54.054)
    assert r.travel_p50.to("mm").magnitude == approx(7)
    assert r.travel_tail.to("mm").magnitude == approx(13.68)
    assert r.overrun_tail.to("mm").magnitude == approx(1.68)


def test_joint_gearing_heat_and_stall():
    g = Scenarios.GearedRotorFlywheel.result
    assert g.reflected_inertia.to("kg*m**2").magnitude == approx(0.056)
    assert g.flywheel_mass.to("kg").magnitude == approx(11.2)
    assert g.matched_ratio == approx(38.9597)
    assert g.acceleration_share == approx(0.67651)
    c = Scenarios.PandaClassJointGearRatios.result
    assert c.motor_constant.magnitude == approx(0.26579)
    assert c.copper_loss[50].to("W").magnitude == approx(59.3175)
    assert c.copper_loss[100].to("W").magnitude == approx(14.8294)
    assert c.reflected_inertia[100].to("kg*m**2").magnitude == approx(0.734)
    assert c.loss_inertia_product[50].magnitude == approx(10.8848)
    assert c.over_peak[10] == approx(2.52723)
    b = Scenarios.PandaClassJointBurst.result
    assert b.time_constant.to("s").magnitude == approx(300)
    assert b.horizon_cold.to("s").magnitude == approx(78.4162)
    assert b.horizon_hot.to("s").magnitude == approx(43.0813)
    assert b.duty == approx(0.20002)
    assert b.off_time_scheduled.to("s").magnitude == approx(12)
    assert b.periodic_peak.to("degC").magnitude == approx(142.002)
    s = Scenarios.PandaClassElbowStall.result
    assert s.feedback_gain == approx(0.41961)
    assert s.steady_rise.to("K").magnitude == approx(198.425)
    assert s.sustainable_hold.to("N*m").magnitude == approx(72.9822)
    assert s.derating == approx(0.16112)
    d = Scenarios.SealedJointCatalogDerating.result
    assert d.ambient_share == approx(0.93250)
    assert d.combined_share == approx(0.65938)


def test_drive_bus_campaign_and_throttle():
    r = Scenarios.WarehouseDriveBusStop.result
    assert r.sag.to("V").magnitude == approx(4.2)
    assert r.sustained_depleted.to("V").magnitude == approx(37.8)
    assert r.margin_depleted.to("V").magnitude == approx(4.8)
    assert r.crossing_current.to("A").magnitude == approx(150)
    assert r.kinetic_energy.to("J").magnitude == approx(414)
    assert r.capacitor_only_endpoint.to("V").magnitude == approx(176.475)
    assert r.capacitor_to_clamp.to("J").magnitude == approx(7.524)
    assert r.peak_power.to("W").magnitude == approx(828)
    assert r.resistor_max.to("ohm").magnitude == approx(4.34783)
    z = Scenarios.CleanLoadedStopCampaign.result
    assert z.coefficient == approx(2.99573)
    assert z.one_in == approx(1001.42)
    t = Scenarios.ThrottledNavigationBackbone.result
    assert t.added_age.to("ms").magnitude == approx(11)
    assert t.added_travel.to("mm").magnitude == approx(16.5)


# --- Context engineering (added 2026-10-20) -----------------------------------

def test_repair_context_budget():
    r = Scenarios.RepairContextBudget.result
    assert r.input_limit == 120000 and r.fixed_tokens == 39000
    assert r.evidence_tokens == 105000 and r.evidence_share == approx(0.8203125)
    assert r.observation_budget == 81000
    assert r.turns_fit == 20 and r.overflow_turn == 21 and r.overflow_observations == 84000
    assert r.digest_turns == 40 and r.max_observation_per_turn == 2700
    assert r.selected_request == 47000
    assert r.append_bill == 3030000 and r.select_bill == 1410000
    assert r.bill_ratio == approx(2.148936)
    assert r.append_cumulative[-1] == r.append_bill and r.select_cumulative[-1] == r.select_bill
    assert r.final_append_request == 159000 and r.final_append_with_reserve == 167000


def test_toy_head_attention_dilution():
    r = Scenarios.ToyHeadAttentionDilution.result
    assert r.short_weight == approx(0.9678567) and r.long_weight == approx(0.0289470)
    assert r.weight_with_lookalikes is None
    x = Scenarios.ToyHeadLookalikeExcerpt.result
    assert x.weight_without_lookalikes == approx(0.0693553)
    assert x.weight_with_lookalikes == approx(0.0333480)
    assert x.lookalike_equivalent_keys == approx(46408.43)
    assert x.weight_half_unrelated == approx(0.0429599)
    assert x.weight_no_unrelated == approx(0.0603562)


def test_fiction_livebench_through_60k():
    r = Scenarios.FictionLiveBenchThrough60K.result
    assert r.scored_models == 53 and r.held_count == 3
    assert r.median["0"] == approx(1.0) and r.median["8k"] == approx(0.583) and r.median["60k"] == approx(0.556)
    assert r.q25["60k"] == approx(0.417) and r.q75["60k"] == approx(0.694)
    assert r.lowest_last == approx(0.222) and r.highest_last == approx(1.0)


def test_repair_prefix_layout_bill():
    r = Scenarios.RepairPrefixLayoutBill.result
    assert r.request_tokens == 47000 and r.total_input == 1410000 and r.warm_calls == 27
    assert r.uncached_tokens == 327000 and r.cached_tokens == 1083000
    assert r.hit_rate == approx(0.7680851)
    assert r.unstable_cost.to("USD").magnitude == approx(4.23)
    assert r.stable_cost.to("USD").magnitude == approx(1.3059)
    assert r.cost_ratio == approx(3.239145) and r.prefill_ratio == approx(4.311927)
    assert r.rewrite_tokens == 32000 and r.rewrite_in_leaf_calls == approx(4.0)
    e = Scenarios.RepairPrefixLayoutEveryFourthCall.result
    assert Scenarios.RepairPrefixLayoutEveryFourthCall.trunk_rewrites == 7
    assert e.uncached_tokens == 447000 and e.stable_cost.to("USD").magnitude == approx(1.6299)


def test_repair_compaction_watermarks():
    r = Scenarios.RepairCompactionWatermarks.result
    assert r.cycle_turns == 12 and r.trigger_length == 95000 and r.overshoot == 3000
    assert r.length_with_reserve == 103000 and r.fits_window
    assert r.passes == 2 == sum(1 for x in r.lengths if x >= Scenarios.RepairCompactionWatermarks.trigger)
    assert r.alternate_cycle_turns == 3 and r.alternate_passes == 10 and r.frequency_ratio == approx(4.0)


def test_superseded_excerpt_invalidation():
    r = Scenarios.SupersededExcerptInvalidation.result
    assert r.removed_total == 46150 and r.removed_suffix == 31150
    assert r.replaced_total == 46174 and r.replaced_suffix == 31174
    assert r.appended_total == 47128 and r.appended_suffix == 128
    assert r.suffix_ratio == approx(243.546875)


def test_compaction_constraint_retention():
    r = Scenarios.CompactionConstraintRetention.result
    assert r.success[10] == approx(0.5987369) and r.success[40] == approx(0.1285121)
    assert r.first_horizon_below_target == 14
    assert Scenarios.UncheckedRepairHorizon.result.first_horizon_below_target == 11


def test_protected_field_check_payback():
    r = Scenarios.ProtectedFieldCheckPayback.result
    assert r.exposures == 36
    assert r.survival_unchecked == approx(0.3340277) and r.acceptance_unchecked == approx(0.2004166)
    assert r.cost_per_accepted_unchecked == approx(9.979213)
    assert r.undetected_loss == approx(0.003)
    assert r.survival_checked == approx(0.8974819) and r.acceptance_checked == approx(0.5384891)
    assert r.attempt_cost_checked == approx(2.15) and r.cost_per_accepted_checked == approx(3.992652)
    assert r.max_loss_for_target == approx(0.0014238)
    assert r.checked_rounds_below_floor == 20
