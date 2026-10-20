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
    assert r.result.latency_memory.to("ms").magnitude == approx(0.11290868)
    assert r.result.latency_compute.to("ms").magnitude == approx(1.0613)
    assert r.result.latency_overhead.to("ms").magnitude == approx(0.51)
    assert r.result.latency.to("ms").magnitude == approx(1.5713)
    assert r.memory_traffic.to("GB").magnitude == approx(0.37824407)
    assert r.iron_law.sequential_reduction == approx(0.033520318)
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
    assert a.traffic.to("MB").magnitude == approx(56.225470)
    assert a.intensity.to("flop/byte").magnitude == approx(145.84138)
    assert a.ridge.to("flop/byte").magnitude == approx(153.02)
    m = Scenarios.ResNet50PhoneNpuBatch1
    assert m.traffic.to("MB").magnitude == approx(28.112735)
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


# --- Fault tolerance (added 2026-10-20) --------------------------------------
def test_gpu_failure_ladder():
    s = Scenarios.GpuFailureLadder
    assert s.intervals[8].to("hour").magnitude == approx(6250)
    assert s.intervals[10_000].to("hour").magnitude == approx(5)
    assert s.failures_per_day[1000] == approx(0.48)
    assert s.failures_per_day[25_000] == approx(12)


def test_full_system_failure_cascade():
    r = Scenarios.FullSystemFailures10K.result
    assert r.component_rates["gpu"] == approx(2e-5)
    assert r.component_rates["pcie"] == approx(5e-6)
    assert r.component_rates["network"] == approx(8.3333e-7)
    assert r.component_rates["power"] == 0
    assert r.device_rate == approx(2.58333e-5)
    assert r.system_rate == approx(0.258333)
    assert r.mtbf.to("hour").magnitude == approx(3.87097)
    assert r.failures_per_day == approx(6.2)
    big = Scenarios.FullSystemFailures100K.result
    assert big.mtbf.to("hour").magnitude == approx(0.387097)


def test_young_daly_intervals():
    s = Scenarios.Gpt3CheckpointInterval10K
    r = s.result
    assert s.checkpoint_bytes.to("TB").magnitude == approx(2.45)
    assert r.write_time.to("s").magnitude == approx(24.5)
    assert r.mtbf.to("hour").magnitude == approx(3.87)
    assert r.interval.to("s").magnitude == approx(826.237)
    assert r.write_share == approx(r.rework_share)
    assert r.waste == approx(0.059305)
    from mlsysim.core.units import Q_
    assert s.waste_at(Q_(1, "hour")) == approx(0.136005)
    assert Scenarios.Gpt3CheckpointInterval100K.result.waste == approx(0.187515)
    assert Scenarios.FiveMinuteCheckpoint1K.result.interval.to("minute").magnitude == approx(173.205)
    assert Scenarios.FifteenMinuteCheckpoint1K.result.interval.to("minute").magnitude == approx(300)


def test_recovery_budget_and_hot_spare():
    r1 = Scenarios.Gpt3RecoveryBudget1K.result
    assert r1.total.to("s").magnitude == approx(384.5)
    assert r1.reload.to("s").magnitude == approx(24.5)
    assert r1.failures_per_day == approx(0.49152)
    assert r1.lost_per_day.to("minute").magnitude == approx(3.14982)
    r10 = Scenarios.Gpt3RecoveryBudget10K.result
    assert r10.lost_per_day.to("minute").magnitude == approx(30.76)
    assert r10.day_share == approx(0.0213611)
    assert r10.gpu_hours_per_day == approx(5126.67)
    h = Scenarios.HotSpareNode10K
    assert h.spare_cost_per_day.to("USD").magnitude == approx(384)
    assert h.value_per_minute.to("USD").magnitude == approx(2066.67)
    assert h.breakeven.to("s").magnitude == approx(11.1484)


def test_straggler_replicas_probe_sdc():
    s = Scenarios.HalfSpeedStraggler1K
    assert s.slow_step.to("s").magnitude == approx(2)
    assert s.throughput == approx(0.5)
    assert s.idle_cost.to("USD").magnitude == approx(1024)
    assert Scenarios.ServingReplica99.replicated_downtime_per_year.to("day").magnitude == approx(3.65)
    assert Scenarios.ServingReplicas99x2.replicated_downtime_per_year.to("minute").magnitude == approx(52.56)
    assert Scenarios.ServingReplicas99x3.replicated_downtime_per_year.to("s").magnitude == approx(31.536)
    assert Scenarios.GpuAvailability10K.all_up == approx(0.367861)
    early, late = Scenarios.InferenceReadinessProbe.detection_window
    assert (early.to("s").magnitude, late.to("s").magnitude) == (12, 17)
    a = Scenarios.SdcExposure10K
    assert a.interval.to("hour").magnitude == approx(100.0)
    assert a.run_events == approx(21.6)
    b = Scenarios.SdcExposure100K
    assert b.exposure == approx(55.5556)
    assert b.steps_between == approx(18000.5)
# --- Fleet orchestration (added 2026-10-20) ---

def test_mg1_queue_wait():
    assert Scenarios.Mg1ExponentialJobQueueAt80.wait_ratio == approx(4)
    assert Scenarios.Mg1MixedJobQueueAt80.wait_ratio == approx(20)
    assert Scenarios.Mg1MixedJobQueueAt80.variability_factor == approx(5)
    assert Scenarios.Mg1ExponentialJobQueueAt80.variant(utilization=0.98).wait_ratio == approx(49)
    assert Scenarios.Mg1ExponentialJobQueueAt80.variant(utilization=0.60).wait_ratio == approx(1.5)


def test_kalos_job_size_mix():
    r = Scenarios.KalosJobSizeMix.result
    assert r.job_count == 19907
    assert r.classes["single"]["job_share"] == approx(0.86281)
    assert r.classes["single"]["time_share"] == approx(0.0014040)
    assert r.classes["large"]["job_share"] == approx(0.043201)
    assert r.classes["large"]["time_share"] == approx(0.96253)
    assert r.classes["single"]["mean_gpu_hours"] == approx(0.12110)
    assert r.classes["large"]["mean_gpu_hours"] == approx(1658.06)
    assert r.service_cv_lower_bound == approx(5.0175)
    assert r.state_time_shares["completed"] == approx(0.31270)
    assert r.state_time_shares["failed"] == approx(0.062880)
    assert r.state_time_shares["canceled"] == approx(0.60737)


def test_drf_and_packing():
    d = Scenarios.DrfTwoTenantExample.result
    assert d["bundle_counts"] == {"A": 4, "B": 2}
    assert d["tenants"]["A"]["dominant_share"] == approx(0.5)
    assert d["tenants"]["B"]["dominant_share"] == approx(0.5)
    assert d["tenants"]["B"]["shares"]["cpu_cores"] == approx(0.125)
    assert Scenarios.SixGpuJobsOnDgxH100Nodes.usable_share == approx(0.75)
    assert Scenarios.FragmentedEightNodeSnapshot.free_share == approx(0.296875)


def test_idle_capacity_and_policy_gain():
    assert Scenarios.ReservedGpusIdleThreeWeeks.cost == approx(504000)
    assert Scenarios.PartialGangHold.cost == approx(1800)
    assert Scenarios.PartialGangHold.missing_gpus == 124
    assert Scenarios.Training10KIdleHour.cost == approx(20000)
    u = Scenarios.MixedFleetPolicyFix1K
    assert u.productive_gain == approx(245.76)
    assert u.raw_gpu_equivalent == approx(409.6)
    assert u.annual_value == approx(7176192)


def test_preemption_and_spot():
    p = Scenarios.ForcedPreemption175BFineTune64.result
    assert p.reload.to("s").magnitude == approx(24.5)
    assert p.per_event.to("minute").magnitude == approx(20.4083)
    assert p.cost_per_event == approx(43.5378)
    assert p.lost_per_day.to("hour").magnitude == approx(4.08167)
    assert p.gpu_hours_per_day == approx(261.227)
    s = Scenarios.SpotInterruptionEffectiveCost
    assert s.nominal_discount == approx(0.65)
    assert s.effective_cost_ratio == approx(0.48125)
    r = Scenarios.Llama2_70BSpotRun512
    assert r.on_demand_cost == approx(344064)
    assert r.spot_total_hours == approx(450.8)
    assert r.spot_total_cost == approx(161566.72)
    assert r.savings_share == approx(0.53042)
    assert r.longer_share == approx(0.34167)


def test_elastic_sharing_and_lending():
    e = Scenarios.ElasticResizeDecision
    assert e.start_small_epochs == approx(24)
    assert e.wait_epochs == approx(64)
    assert e.step_up_epochs == approx(39.7)
    assert e.gpu_hours == {"start_small": approx(3072), "wait": approx(10240), "step_up": approx(5632)}
    assert e.epochs_per_gpu_hour["step_up"] == approx(0.0070490)
    assert e.rescale_break_even_h == approx(8.8889)
    c = Scenarios.ServingConsolidationA100
    assert c.exclusive_unused == approx(0.675)
    assert (c.exclusive_gpus, c.shared_gpus) == (180, 130)
    assert c.saved_share == approx(0.27778)
    assert c.annual_value == approx(876000)
    assert c.min_devices_large == 5
    d = Scenarios.DiurnalServingLending1K
    assert (d.peak_demand, d.lent, d.static_active, d.dynamic_active, d.headroom) == (283, 133, 768, 901, 6)
    assert d.annual_value == approx(2330160)
    t = Scenarios.TensorParallelExchange1GB.result.times
    assert t["NVLink"].to("ms").magnitude == approx(2.2222)
    assert t["InfiniBand NDR"].to("ms").magnitude == approx(20)


def test_quota_replica_and_saturated_queue():
    q = Scenarios.TeamQuotaLending500
    assert (q.used, q.lendable) == (300, 200)
    r = Scenarios.Gpt3SelfContainedReplicaDgxH100
    assert (r.tensor_parallel, r.pipeline_depth, r.replica_gpus) == (8, 8, 64)
    assert Scenarios.Mg1ExponentialJobQueueAt98.wait_ratio == approx(49)
    assert Scenarios.MixedFleetPolicyFix1K.job_memory_need.to("GB").magnitude == approx(35)
# --- ML lifecycle: the retinopathy screener (added 2026-10-20) ---------------

def test_screener_requirements_and_late_memory_limit():
    r = Scenarios.RetinopathyScreenerRequirements
    assert r.power_low.to("W").magnitude == approx(7)
    assert r.power_high.to("W").magnitude == approx(15)
    assert r.passes("all clinic photos")
    assert r.failing_slices == ["cataract patients"]
    late = Scenarios.RetinopathyScreenerLateMemoryLimit
    assert late.overrun_ratio == approx(8)
    assert late.work_discarded.to("day").magnitude == approx(152)
    assert late.work_discarded.to("month").magnitude == approx(4.99384)
    assert late.repair_elapsed.to("day").magnitude == approx(6)


def test_slow_versus_fast_iteration():
    s = Scenarios.SlowVersusFastIteration
    assert s.slow_runs == approx(26)
    assert s.slow_final == approx(98.9)
    assert s.fast_possible_runs == approx(4368)
    assert s.fast_uncapped == approx(100)
    assert s.fast_final == approx(99)
    assert s.ideas_ratio == approx(3.84615)
    assert s.final_gap == approx(0.1)


def test_screening_counts_and_image_yield():
    a = Scenarios.AlwaysNormalScreener
    assert a.accuracy == approx(0.9)
    assert a.sensitivity == 0
    assert a.missed == 10
    f = Scenarios.IdxDrPivotalTrial
    assert f.sensitivity == approx(0.873737)
    assert f.specificity == approx(0.895330)
    assert Scenarios.IdxDrPivotalAnalyzable.usable_share == approx(0.918161)
    assert Scenarios.IdxDrPivotalImageability.usable_share == approx(0.961268)
    assert Scenarios.ThaiClinicRetinalPhotoYield.rejected_share == approx(0.213819)


def test_fundus_photo_resized_to_224():
    z = Scenarios.FundusPhotoAt224
    assert z.scale == approx(0.261377)
    assert z.resized_features == [approx(2.09102), approx(3.13652)]
    assert z.resized(z.feature_mid) == approx(2.61377)


def test_clinic_upload_and_cloud_versus_edge():
    u = Scenarios.RuralClinicRetinalUpload
    assert u.daily_volume.to("MB").magnitude == approx(2000)
    assert u.upload_time.to("s").magnitude == approx(8000)
    assert u.window_share == approx(0.277778)
    assert u.file_upload_time.to("s").magnitude == approx(20)
    assert u.reduction == approx(2000)
    c = Scenarios.RetinopathyScreeningCloudVersusEdge
    assert c.annual_images == 9_125_000
    assert c.cloud_annual.to("USD/year").magnitude == approx(136_250)
    assert c.edge_capex.to("USD").magnitude == approx(250_000)
    assert c.edge_annual.to("USD/year").magnitude == approx(34_125)
    assert c.annual_savings.to("USD/year").magnitude == approx(102_125)
    assert c.payback.to("year").magnitude == approx(2.44798)
    assert c.cloud_upload_time.to("s").magnitude == approx(20)


def test_monitoring_cadences():
    m = Scenarios.RetinopathyMonitoringCadences
    assert m.minutes["Quarter"] == approx(131_400)
    assert m.span_orders == approx(5.11860)
# --- Test-time deliberation (added 2026-10-20) --------------------------------

def test_probe_and_allocation_axes():
    h = Scenarios.ProbeVsGeneratedExplanation.result
    assert h.token_time.to("ms").magnitude == approx(21.0746)
    assert h.answer_time.to("s").magnitude == approx(21.5804)
    assert h.answer_cost.to("USD").magnitude == approx(0.01536)
    assert h.time_ratio == approx(539.51)
    a = Scenarios.OutputCeilingThreeAllocations.result
    assert (a.depth_input, a.breadth_input, a.feedback_input) == (4096, 16384, 16380)
    assert (a.depth_output, a.breadth_output, a.feedback_output) == (4096, 4096, 3072)
    assert a.depth_time.to("s").magnitude == approx(86.3217)
    assert a.breadth_time.to("s").magnitude == approx(21.5804)
    assert a.feedback_time.to("s").magnitude == approx(69.2413)
    assert a.depth_cost.to("USD").magnitude == approx(0.073728)
    assert a.breadth_cost.to("USD").magnitude == approx(0.110592)
    assert a.feedback_cost.to("USD").magnitude == approx(0.09522)
    assert a.path_ratio == approx(4.0) and a.bill_ratio == approx(1.5)
    assert a.breadth_premium == 12288 and a.grown_premium == 98304
    assert a.grown_premium_cost.to("USD").magnitude == approx(0.294912)
    assert a.serial_breadth_time.to("s").magnitude == approx(86.3217)


def test_pools_estimator_and_coverage():
    p = Scenarios.IndependentCandidatePools.result
    assert p.pass_at[8] == approx(0.942352)
    assert p.last_gain == approx(0.0247063)
    assert p.parallel_per_success == approx(8.4894)
    assert p.sequential_candidates == approx(3.14117)
    assert p.sequential_per_success == approx(1 / 0.30)
    assert p.cost_ratio == approx(2.54682)
    assert p.mixed_plug_in == approx(0.942352) and p.mixed_true == approx(0.499672)
    e = Scenarios.PassAtKSubsetEstimate.result
    assert e.subset_estimate == approx(0.718266) and e.plug_in == approx(0.67232)
    c = Scenarios.RepeatedSamplingCoverageVsVote.result
    assert c.coverage_small == approx(0.984375) and c.vote_small == approx(0.429688)
    assert c.vote_large == approx(0.59375)
    assert c.vote_gain_after_knee < 0.01 < 0.05 < c.coverage_gain_small
    assert c.repair_first == approx(0.15872) and c.repair_max == approx(0.56)


def test_verified_pool_and_argmax_noise():
    v = Scenarios.SparseValidPoolFirstPass.result
    assert v.pass_probability == approx(0.064)
    assert v.precision == approx(0.703125)
    assert v.any_selection == approx(0.985490)
    assert v.wrong_selection == approx(0.292567)
    assert v.any_false_pass == approx(0.707035)
    assert v.task_loss.to("USD").magnitude == approx(146.284)
    assert v.output_bill.to("USD").magnitude == approx(0.98304)
    assert v.loss_to_bill == approx(148.808)
    assert Scenarios.SparseValidPoolHalvedFalseAccept.result.precision == approx(0.825688)
    assert Scenarios.SparseValidPoolDoubledValidRate.result.precision == approx(0.833333)
    # Precision is one minus the checker posterior of a defect given a pass.
    s = Scenarios.SparseValidPoolFirstPass
    post = Scenarios.PatchCheckerPosterior.variant(
        defect_rate=1 - s.valid_rate, false_accept_rate=s.false_accept_rate,
        valid_accept_rate=1 - s.false_reject_rate).result.posterior
    assert v.precision == approx(1 - post)
    g = Scenarios.ArgmaxScoreNoise.result
    assert abs(g.expected_max[1]) < 1e-6
    assert g.expected_max[4] == approx(1.02938) and g.expected_max[64] == approx(2.34373)
    assert g.asymptote[64] == approx(2.88405)
    assert g.exploit_presence == approx(0.274434)


def test_revision_tree_beam_policies_and_run_cost():
    r = Scenarios.ScopedPlanRevision.result
    assert (r.full_output, r.scoped_output, r.avoided_share) == (1800, 450, 0.75)
    t = Scenarios.StepCheckedSearchTree.result
    assert t.effective_branching == approx(1.6) and t.critical_rejection == approx(0.8)
    assert t.unpruned_nodes == 155 and t.pruned_nodes == approx(25.8)
    assert t.node_saving == approx(0.833548) and t.wide_leaves == 4 ** 10
    b = Scenarios.FlatVsBeamSearch.result
    assert b.frontier_counts == [3, 9, 12, 12] and (b.flat_waves, b.beam_waves) == (2, 7)
    assert (b.beam_output_tokens, b.beam_prompt_tokens, b.scorer_tokens) == (4608, 156288, 13440)
    assert b.flat_elapsed.to("s").magnitude == approx(22.5804)
    assert b.beam_elapsed.to("s").magnitude == approx(19.9229)
    assert b.flat_token_cost.to("USD").magnitude == approx(0.319488)
    assert b.beam_token_cost.to("USD").magnitude == approx(0.544704)
    assert b.breakeven_input_share == approx(0.172779)
    d = Scenarios.DeadlineBoundRepairPolicies.result
    assert d.rounds == 2
    assert d.round_time.to("s").magnitude == approx(31.0746)
    assert d.b_pass == approx(0.983204) and d.c_pass == approx(0.61)
    assert d.a_cost_per_pass.to("USD").magnitude == approx(0.05286)
    assert d.b_cost_per_pass.to("USD").magnitude == approx(0.172042)
    assert d.c_cost_per_pass.to("USD").magnitude == approx(0.0594138)
    assert d.c_time.to("s").magnitude == approx(49.7194)
    w = Scenarios.SweAgentClaude35SonnetRunCost.result
    assert (w.tasks, w.resolved) == (500, 168)
    assert w.cost_per_resolved == approx(5.10036) and w.unresolved_share == approx(0.773173)
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
def test_intent_model_evaluations():
    from mlsysim import Embodied
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator
    f = Scenarios.IntentModelEvaluationFp16.result
    assert f.weight_memory.to("GB").magnitude == approx(14.0)
    assert f.t_weight_sweep.to("ms").magnitude == approx(98.0392)
    assert f.t_prefill.to("ms").magnitude == approx(188.546)
    assert f.t_total.to("ms").magnitude == approx(776.781)
    assert f.prefill_bound == "compute"
    assert f.prefill_intensity.to("flop/byte").magnitude == approx(288)
    assert f.decode_intensity.to("flop/byte").magnitude == approx(1)
    assert f.ridge_intensity.to("flop/byte").magnitude == approx(210.784)
    assert f.prefill_attention_flops.to("flop").magnitude == approx(2 * 32 * 4096 * 288 ** 2)
    i = Scenarios.IntentModelEvaluationInt4.result
    assert i.t_total == rm.intent_inference_latency
    assert i.t_prefill == f.t_prefill
    assert i.prefill_share == approx(0.56181)
    assert i.prefill_ceiling.to("Hz").magnitude == approx(5.30374)
    assert Scenarios.IntentModelEvaluation8Bit.result.t_total.to("ms").magnitude == approx(482.664)
    assert Scenarios.IntentModelEvaluationThreeViews.result.t_total.to("ms").magnitude == approx(675.792)
    assert Scenarios.IntentModelEvaluationParallel.result.t_total.to("ms").magnitude == approx(194.215)
    five = Scenarios.IntentModelFiveHertz
    assert not five.fits
    assert five.prefill_budget.to("ms").magnitude == approx(52.9412)
    assert five.tokens_within == approx(80.8664)
    assert Scenarios.IntentModelParallelFiveHertz.slack_fraction == approx(0.028924)
    assert Scenarios.IntentModelHalfTokensFiveHertz.slack.to("ms").magnitude == approx(-41.0791)
    r = Scenarios.IntentModelDecodeRoofline
    assert r.attainable.to("TFLOPs/s").magnitude == approx(0.204)
    assert r.peak_fraction == approx(0.0047442)


def test_three_view_tokens_and_kv_history():
    from mlsysim import Embodied
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator
    t = Scenarios.ThreeViewVlaTokens
    assert t.patches_per_view == 256 and t.visual_tokens == 768 and t.total_tokens == 816
    assert t.single_view_tokens == rm.intent_input_tokens
    assert t.embedding_memory.to("MB").magnitude == approx(6.5536)
    assert t.kv_per_token.to("byte").magnitude == approx(524288)
    k = Scenarios.ThreeViewKvHistory
    assert k.frames == 50
    assert k.kv_per_step.to("MB").magnitude == approx(419.4304)
    assert k.visual_history.to("GB").magnitude == approx(20.1327)
    assert k.history.to("GB").magnitude == approx(20.9715)
    assert k.combined.to("GB").magnitude == approx(34.9715)
    assert k.combined_share == approx(0.50890)


def test_decoder_families_and_proposer_traffic():
    d = Scenarios.IntentBackboneDecoderFamilies.result
    assert d.binned_tokens == 112
    assert d.t_binned.to("ms").magnitude == approx(2909.13)
    assert d.t_fast.to("ms").magnitude == approx(899.331)
    assert d.t_parallel.to("ms").magnitude == approx(264.367)
    assert d.t_expert.to("ms").magnitude == approx(241.137)
    assert d.t_expert_step_bandwidth > d.t_expert_step_compute
    assert d.prefix_kv.to("MB").magnitude == approx(150.995)
    t = Scenarios.WarehouseProposerTraffic.result
    assert t.sustained_bandwidth.to("GB/s").magnitude == approx(142.8)
    assert t.amortized_sweep.to("ms").magnitude == approx(6.12745)
    assert t.single_step_demand.to("GB/s").magnitude == approx(560)
    assert t.single_step_overload == approx(3.92157)
    assert t.chunked_share == approx(0.245098)
    assert t.chunk_sweep.to("ms").magnitude == approx(1.12045)
    assert t.renewal_gap == approx(6.71210)
    assert t.intent_traffic.to("GB/s").magnitude == approx(49.0)
    assert t.proposer_traffic.to("GB/s").magnitude == approx(52.2)
    assert t.proposer_share == approx(0.365546)
    assert t.decode_duty == approx(0.294118)


def test_conveyor_evidence_horizon():
    e = Scenarios.ConveyorMugEvidenceHorizon.result
    assert e.untracked_deadline.to("ms").magnitude == approx(60)
    assert e.horizon.to("ms").magnitude == approx(219.089)
    assert e.age_at_replacement.to("ms").magnitude == approx(90)
    assert e.drift_at_replacement.to("mm").magnitude == approx(2.025)
    assert e.min_supply_setpoints == 4
    assert e.error_at_chunk_end.to("mm").magnitude == approx(43)
    assert e.outlast.to("ms").magnitude == approx(180.911)
    s4 = Scenarios.ConveyorMugEvidenceHorizonFourfoldSlip.result
    assert s4.horizon.to("ms").magnitude == approx(109.545)
    assert s4.drift_at_replacement.to("mm").magnitude == approx(8.1)
    half = Scenarios.ConveyorMugEvidenceHorizonHalfRenewal.result
    assert half.age_at_replacement.to("ms").magnitude == approx(140)
    assert half.min_supply_setpoints == 6


def test_quantization_ensemble_confidence_and_mlperf():
    q = Scenarios.OpenVlaOrinQuantization
    assert q.gap("FP16") == approx(-0.07526)
    assert q.gap("8-bit") == approx(0.024764)
    assert q.measured_speedup == approx(2.5)
    assert q.score_drop == approx(0.052)
    assert q.scale_overhead == approx(0.03125)
    a = Scenarios.ActTemporalEnsemble.result
    assert a["oldest_over_newest"] == approx(2.69123)
    assert a["weights"].sum() == approx(1.0)
    c = Scenarios.ProposerConfidenceRegimes
    assert c.gap("Near OOD") == approx(0.37)
    m = Scenarios.OrinDevKitSingleStream
    assert m.latest("resnet50").to("ms").magnitude == approx(0.640259)
    assert m.latest("gptj-99").to("s").magnitude == approx(4.17663)
    assert m.earliest("gptj-99").to("s").magnitude == approx(10.2045)
    assert m.span == approx(15566.8)
# --- Data engineering: keyword-spotter data, labels, agreement, ingest (added 2026-10-20) ---

def test_keyword_spotter_false_wake_budget_and_evidence():
    f = Scenarios.KeywordSpotterFalseWakes
    assert f.windows == approx(2_592_000)
    assert f.target_false_positive_rate == approx(3.858e-7)
    assert [f.false_alarms(r) for r in f.rejection_levels] == [approx(2592), approx(259.2), approx(25.92)]
    assert f.listening_time.to("hour").magnitude == approx(720)
    assert f.false_alarms_at_rate(0.1 / f.listening_time.units) == approx(72)
    e = Scenarios.KeywordSpotterFalseWakeEvidence
    assert e.events_bound == approx(2.99573)
    assert e.exposure_needed.to("hour").magnitude == approx(2156.93)
    assert e.recording_size.to("GB").magnitude == approx(248.478)


def test_keyword_spotter_data_budget_and_design_space():
    b = Scenarios.KeywordSpotterDataBudget
    assert b.labels == approx(750_000)
    assert b.allocation("storage and processing").to("USD").magnitude == approx(37_500)
    assert b.effective_label_cost.to("USD").magnitude == approx(0.12)
    assert Scenarios.KeywordSpotterMfccChoice.ratio == approx(3.0769)
    assert Scenarios.KeywordSpotterExampleCountChoice.ratio == approx(10)
    t = Scenarios.KeywordSpotterTargets
    assert t.model_memory.to("KB").magnitude == approx(256)
    assert t.preprocessing_memory.to("KB").magnitude == approx(16)


def test_labeling_cost_granularity_and_effort():
    r = Scenarios.ImageNetLabelingVersusResNetRun
    assert r.train_time.to("hour").magnitude == approx(1.25114)
    assert r.train_cost_low.to("USD").magnitude == approx(20.0182)
    assert r.ratio_low == approx(2497.72)
    assert r.ratio_high == approx(4995.45)
    g = Scenarios.FullHdMaskVersusBoxes
    assert g.mask_entries == 2_073_600
    assert g.mask_to_box_ratio == approx(51_840)
    assert Scenarios.CityscapesMasksForMillionFrames.person_years == approx(750)
    h = Scenarios.MswcByHand
    assert h.total_time.to("hour").magnitude == approx(65_000)
    assert h.person_years == approx(32.5)


def test_annotator_agreement_and_active_learning():
    a = Scenarios.BinaryLabelChanceAgreement
    assert a.chance_agreement == approx(0.82)
    assert a.kappa == approx(0.16667)
    assert Scenarios.WakeWordLabelQualityDrift.drop == approx(0.13)
    p = Scenarios.ActiveLearningTenMillionPool
    assert p.full_pool_scoring_cost.to("USD").magnitude == approx(100_000)
    assert p.candidate_pool == approx(500_000)
    assert p.affordable_labels == approx(90_000)
    assert p.active_total_low.to("USD").magnitude == approx(150_000)


def test_speech_commands_split_ingest_cost_and_corpus_growth():
    s = Scenarios.SpeechCommandsHashSplit
    assert s.train_percent == approx(80)
    assert s.released_total == approx(99_720)
    assert s.released_test_share == approx(0.049037)
    i = Scenarios.MillionEventBatchVersusStreaming
    assert i.throughput.to("GB/s").magnitude == approx(1)
    assert i.cost_ratio == approx(3)
    assert i.stream_per_core.to("MB/s").magnitude == approx(20)
    assert i.batch_per_core.to("MB/s").magnitude == approx(30)
    c = Scenarios.PretrainingCorpusGrowth
    assert c.growth == approx(2.8954)
    assert c.crossing(3e14) == approx(2027.65, rel=1e-4)


# --- Data engineering: delivery to the accelerator and drift (added 2026-10-20) ---

def test_petabyte_cross_region_move():
    t = Scenarios.PetabyteCrossRegionMove
    assert t.transfer_time("100 Gb/s").to("second").magnitude == approx(80_000)
    assert t.transfer_time("10 Gb/s").to("day").magnitude == approx(9.25926)
    assert t.egress_cost.to("USD").magnitude == approx(20_000)
    assert t.egress_compute_hours.to("hour").magnitude == approx(5_000)
    assert t.job_time.to("hour").magnitude == approx(20)


def test_resnet50_a100_input_feed():
    from mlsysim.core.units import Q_
    f = Scenarios.ResNet50A100InputFeed
    assert f.training_flops_per_sample.to("GFLOP").magnitude == approx(24.6)
    assert f.ceiling_rate.to("count/second").magnitude == approx(12_682.93, rel=1e-6)
    assert f.demand().to("MB/s").magnitude == approx(480)
    assert f.decoded_sample.to("KB").magnitude == approx(602.112)
    assert f.demand(decoded=True).to("GB/s").magnitude == approx(1.92676)
    assert f.feeding_loss() == approx(0.479167, rel=1e-5)
    assert f.feeding_loss(decoded=True) == approx(0.870248, rel=1e-5)
    assert f.supplied_fraction == approx(0.520833, rel=1e-5)
    assert f.ceiling_demand.to("GB/s").magnitude == approx(1.90244, rel=1e-5)
    assert f.samples_supplied(Q_(500, "MB/s")).to("count/second").magnitude == approx(3333.33, rel=1e-5)
    assert f.streams_needed(Q_(100, "MB/s")) == approx(19.0244, rel=1e-5)
    assert f.worker_crossing == approx(8.53333, rel=1e-5)
    assert f.workers_per_accelerator == 9
    e = Scenarios.ResNet50EightA100InputFeed
    assert e.total_rate.to("count/second").magnitude == approx(25_600)
    assert e.demand().to("GB/s").magnitude == approx(3.84)
    assert e.demand(decoded=True).to("GB/s").magnitude == approx(15.4141, rel=1e-5)
    assert e.workers_pooled == 69
    r = Scenarios.RecommendationFeatureLookups
    assert r.reads_per_second == approx(100_000)
    assert r.storage_reduction == approx(30)


def test_corpus_tiers_columns_and_codecs():
    c = Scenarios.MswcCorpusStorageTiers
    assert c.corpus_bytes.to("GB").magnitude == approx(748.8)
    assert c.monthly_cost("object store").to("USD/month").magnitude == approx(17.2224)
    assert c.monthly_cost("NVMe high").to("USD/month").magnitude == approx(224.64)
    assert c.load_time("object store").to("second").magnitude == approx(7488)
    assert c.load_time("NVMe").to("second").magnitude == approx(149.76)
    p = Scenarios.FraudModelColumnProjection
    assert p.read_fraction == approx(0.2)
    assert p.speedup == approx(5)
    g = Scenarios.GzipVersusSnappyEpochs
    assert g.decode_time("gzip").to("minute").magnitude == approx(13.8889, rel=1e-5)
    assert g.decode_time("Snappy").to("minute").magnitude == approx(3.33333, rel=1e-5)
    assert g.run_time_saved("gzip", "Snappy").to("hour").magnitude == approx(8.7963, rel=1e-5)
    assert g.rate_ratio("Snappy", "gzip") == approx(4.16667, rel=1e-5)


def test_loudness_drift_statistics():
    k = Scenarios.LoudnessDriftKs1000
    assert k.critical_value == approx(0.060821, rel=1e-5)
    assert k.flags
    assert Scenarios.LoudnessDriftKs100k.critical_value == approx(0.0060821, rel=1e-5)
    big, small = Scenarios.LoudnessShiftMatchingObservedGap, Scenarios.LoudnessShiftHarmless
    assert big.ks_statistic == approx(0.0796557, rel=1e-5)
    assert small.ks_statistic == approx(0.0199450, rel=1e-5)
    assert small.normed_wasserstein == approx(0.05)
    assert Scenarios.LoudnessDriftKs100k.flags_statistic(small.ks_statistic)
    assert not Scenarios.LoudnessDriftKs1000.flags_statistic(small.ks_statistic)
    b = Scenarios.QuieterMicrophoneLoudnessBins
    assert b.psi == approx(0.135403, rel=1e-5)
    assert b.terms[0] == approx(0.0405465, rel=1e-5)
    assert b.terms[-1] == approx(0.0693147, rel=1e-5)
    assert b.ratio(0) == approx(1.5)
    r = Scenarios.KeywordSpotterSilentRegression
    assert r.launch_value - r.drop == approx(0.94)


# ── Fleet goodput (added 2026-10-20) ──────────────────────────────────────────

def test_shared_pool_scheduling_snapshot():
    s = Scenarios.SharedPoolSchedulingSnapshot
    assert s.scheduling_goodput == approx(0.65)
    assert s.executing_share == approx(0.60)
    assert s.partial_hold_ceiling == approx(0.80)
    assert s.input_wait_of_held == approx(0.0769231)


def test_llama2_70b_runtime_goodput_128():
    sc = Scenarios.Llama2_70BRuntimeGoodput128
    r = sc.result
    assert r.mtbf.to("hour").magnitude == approx(302.419)
    assert r.write_time.to("s").magnitude == approx(9.8)
    assert r.interval.to("minute").magnitude == approx(76.9897)
    assert r.checkpoint_share == approx(0.00424299)
    assert r.startup_time.to("minute").magnitude == approx(5.16333)
    assert r.preemption_cost.to("minute").magnitude == approx(20.1633)
    assert r.runtime_goodput == approx(0.903980)
    assert r.interruption_checkpoint_interval.to("minute").magnitude == approx(20.8762)
    assert r.startup_share + r.input_wait_share + r.checkpoint_share + r.recovery_share \
        + r.preemption_share == approx(r.runtime_loss)
    assert sc.crossover(100) == 12928
    assert sc.at(16384).result.runtime_loss == approx(0.182919)


def test_llama2_70b_fleet_mfu_128():
    f = Scenarios.Llama2_70BFleetMfu128
    assert f.node_mfu == approx(0.649983)
    assert f.fleet_mfu == approx(0.480003)
    assert f.scaling_efficiency == approx(0.738486)
    assert f.fleet_useful_work.to("PFLOP").magnitude == approx(14.8873)
    assert f.fleet_peak_work.to("PFLOP").magnitude == approx(31.0150)
    assert f.extra_step_time.to("ms").magnitude == approx(65)


def test_shared_pool_goodput_and_llama3():
    g = Scenarios.Llama2_70BSharedPoolGoodput128.result
    assert g.mpg == approx(0.282044)
    assert g.scheduling_share + g.runtime_share + g.program_share + g.mpg == approx(1)
    assert g.point_value["P"] > g.point_value["S"] > g.point_value["R"]
    assert g.mpg_at_ceiling["P"] == approx(0.381922)
    assert g.mpg_at_ceiling["S"] == approx(0.347131)
    assert g.mpg_at_ceiling["R"] == approx(0.306044)
    assert g.cost_multiple == approx(3.54555)
    assert g.job_cost_multiple == approx(2.30461)
    assert g.useful_rate.to("USD/hour").magnitude == approx(7.09110)
    assert Scenarios.Llama2_70BSharedPoolGoodput128.trade_ratio(
        Scenarios.Llama2_70BTenthTwelfthTrade) == approx(1.0083333)
    l3 = Scenarios.Llama3_405BGoodput16K.result
    assert l3.mpg == approx(0.369)
    assert l3.runtime_share == approx(0.10)
    assert l3.program_share == approx(0.531)
    assert Scenarios.Llama3_405BOnSharedPool.result.mpg == approx(0.23985)


def test_decode_floor_layernorm_and_tp_exchanges():
    d = Scenarios.Llama2_70BDecodeFloorTp8.result
    assert d.memory_time.to("ms").magnitude == approx(5.22388)
    assert d.intensity.to("flop/byte").magnitude == approx(1.0)
    assert d.peak_fraction == approx(0.00338726)
    ln = Scenarios.LayerNormMeasuredH100
    assert ln.point_intensity.to("flop/byte").magnitude == approx(1.785714)
    assert ln.bandwidth_utilization == approx(0.835821)
    assert ln.ridge.to("flop/byte").magnitude == approx(295.224)
    t = Scenarios.Llama2_70BNodeTensorParallelStep.result
    assert t.payload.to("MB").magnitude == approx(36.1103)
    assert t.collectives_per_step == 320
    assert t.communication_time.to("ms").magnitude == approx(44.9373)
    m = Scenarios.Llama2_70BDecodeAllReduceTp8
    assert m.message.to("KB").magnitude == approx(32.768)
    assert m.startup_to_transfer == approx(68.6646)


def test_compilation_payback_trace_volume_and_software_gains():
    c = Scenarios.Decode13BCompilationPayback
    assert c.step_fraction == approx(0.635)
    assert c.compiled_step.to("ms").magnitude == approx(5.28955)
    assert c.saved_per_token.to("ms").magnitude == approx(3.04045)
    assert c.tokens_to_repay("autotune") == approx(296008.8)
    assert c.time_to_repay("autotune").to("minute").magnitude == approx(26.0959)
    assert c.shared_repay("autotune").to("s").magnitude == approx(12.2324)
    assert c.rollout_cost("autotune").to("hour").magnitude == approx(32)
    assert c.build_share("autotune") == approx(0.25)
    assert Scenarios.NsightTraceVolume1K.total.to("GB").magnitude == approx(512)
    sg = Scenarios.MlperfInferenceSoftwareGainsDgxH100
    assert sg.gain("gptj-99") == approx(2.69091)
    assert sg.gain("resnet50") == approx(0.999089, rel=1e-4)
    assert sg.months() == 19


# --- Serving at scale (added 2026-10-20) ---------------------------------------

def test_serving_cost_goodput_tax_and_knees():
    c = Scenarios.LlmServingFirstYear
    assert c.annual_requests == approx(18.25e9)
    assert c.annual_serving_cost == approx(18.25e6)
    assert c.serving_to_training == approx(9.125)
    assert c.efficiency_savings == approx(1.825e6)
    assert round(c.crossover_weeks) == 6
    g = Scenarios.SloGoodputTwoConfigurations
    assert g.goodput("X").magnitude == approx(840)
    assert g.goodput("Y").magnitude == approx(950)
    assert g.best == "Y"
    t = Scenarios.InteractiveRequestServingTax
    assert t.outside_model.to("ms").magnitude == approx(30)
    assert t.exchanges_low.to("ms").magnitude == approx(0.2)
    assert t.exchanges_high.to("ms").magnitude == approx(2)
    assert Scenarios.VisionBatchingKnee.knee == approx(2)
    assert Scenarios.LlmDecodeBatchingKnee.knee == approx(80)
    assert Scenarios.LlmDecodeBatchingKnee.throughput_share(80) == approx(0.5)


def test_littles_law_vision_and_chat():
    v = Scenarios.VisionServingLittlesLaw
    assert v.in_flight == approx(100)
    assert v.replica_throughput.to("1/s").magnitude == approx(100)
    assert v.throughput_floor == 10
    assert v.replicas_needed == 13
    assert v.mean_executing == approx(80)
    assert v.service_utilization == approx(0.769231)
    c = Scenarios.LlmChatLittlesLaw
    assert c.time_in_system.to("s").magnitude == approx(10.55)
    assert c.in_flight == approx(10550)


def test_request_traces_and_week_traffic():
    chat = Scenarios.LlmChatTrace2023.result
    assert chat.requests == 19366
    assert chat.arrival_rate.to("1/s").magnitude == approx(5.53042, rel=1e-5)
    assert (chat.median_prompt, chat.median_output, chat.max_output) == (1020, 129, 1000)
    assert chat.mean_output == approx(211.126, rel=1e-5)
    assert chat.fixed_batch_idle_share == approx(0.502929, rel=1e-5)
    assert Scenarios.LlmChatTrace2023.median_request_tokens == 1149
    code = Scenarios.LlmCodeTrace2023.result
    assert (code.median_prompt, code.median_output) == (1469, 13)
    assert code.fixed_batch_idle_share == approx(0.628734, rel=1e-5)
    assert code.per_second_mean == approx(2.56665, rel=1e-5)
    assert code.per_second_peak == 67
    w = Scenarios.LlmChatWeekTraffic2024.result
    assert w.peak_rate.to("1/s").magnitude == approx(79.685)
    assert w.min_rate.to("1/s").magnitude == approx(19.6283, rel=1e-5)
    assert w.weekday_trough_low.magnitude == approx(24.93)
    assert w.weekday_trough_high.magnitude == approx(37.98)
    assert w.weekday_peak_low.magnitude == approx(62.3)
    assert w.weekday_peak_to_trough == approx(2.24912, rel=1e-5)


def test_kv_admission_in_flight_and_handoff():
    long = Scenarios.Llama3_70BLongContextAdmissionDgxH100.result
    assert long.kv_per_token.to("byte").magnitude == 327_680
    assert long.head_dim == 128
    assert long.weights.to("GB").magnitude == approx(141.2)
    assert long.hbm_total.to("GB").magnitude == approx(687.194767)
    assert long.kv_budget.to("GB").magnitude == approx(525.994767)
    assert long.kv_per_request.to("GB").magnitude == approx(42.949673)
    assert long.max_requests == 12
    chat = Scenarios.Llama3_70BChatAdmissionDgxH100.result
    assert chat.kv_per_request.to("MB").magnitude == approx(376.50432)
    assert chat.max_requests == 1397
    f = Scenarios.LlmChatKvInFlight
    assert f.kv_in_flight.to("TB").magnitude == approx(4.35675, rel=1e-5)
    assert f.replicas_by_kv == approx(8.28288, rel=1e-5)
    m = Scenarios.MedianChatKvHandoff.result.times
    assert m["NVLink"].to("ms").magnitude == approx(0.836676, rel=1e-5)
    assert m["InfiniBand"].to("ms").magnitude == approx(7.53009, rel=1e-5)
    assert Scenarios.LongContextKvHandoff.result.times["NVLink"].to("ms").magnitude == approx(95.4437, rel=1e-5)


def test_prefix_sharing_and_precision_dividend():
    p = Scenarios.ChatbotSharedSystemPrompt
    assert p.kv_unshared.to("TB").magnitude == approx(6.5536)
    assert p.kv_shared.to("TB").magnitude == approx(1.315963)
    assert p.reduction == approx(0.7992)
    g = Scenarios.Llama2_70BPrecisionDividendGqaTp8.result
    assert g.kv_budget_per_device.to("GB").magnitude == approx(62.5)
    assert g.weights_per_device_low.to("GB").magnitude == approx(4.375)
    assert (g.max_batch, g.max_batch_low) == (372, 901)
    assert g.dividend == approx(2.42204, rel=1e-5)
    assert g.budget_gain == approx(1.21)
    m = Scenarios.Llama2_70BPrecisionDividendMhaTp8.result
    assert (m.max_batch, m.max_batch_low) == (46, 112)


def test_chunk_tp_decode_budget_and_speculation():
    assert Scenarios.LongPromptChunkedPrefill.iterations == 20
    t = Scenarios.Llama2_70BPrefillTensorParallel8
    assert t.sharded("attention").to("ms").magnitude == approx(1.5)
    assert t.sharded("feed-forward").to("ms").magnitude == approx(2.25)
    assert t.parallel_layer_time.to("ms").magnitude == approx(4.35)
    assert t.speedup == approx(6.89655, rel=1e-5)
    assert t.parallel_total.to("ms").magnitude == approx(348)
    b = Scenarios.Llama3_70B_ReplicaDecodeBudget
    assert b.totals == approx([85.0, 69.7, 49.6])
    assert b.floor.to("ms").magnitude == approx(5.223881, rel=1e-5)
    assert b.allreduces_per_step == 160
    assert b.allreduce_per_collective.to("microsecond").magnitude == approx(63.75)
    assert b.collective_estimate.to("microsecond").magnitude == approx(5.07282, rel=1e-5)
    s = Scenarios.Llama3_70B_ReplicaSpeculation
    assert s.expected_tokens == approx(3.52182, rel=1e-5)
    assert s.round_time.to("ms").magnitude == approx(53.6)
    assert s.per_token_time.to("ms").magnitude == approx(15.2194, rel=1e-5)
    assert s.tpot_gain == approx(3.259, rel=1e-4)


def test_fleet_routing_tail_cold_start_and_tenants():
    h = Scenarios.HeterogeneousH100A100Routing
    assert h.utilization("H100") == approx(0.681818, rel=1e-5)
    assert h.utilization("A100") == approx(h.utilization("H100"))
    f = Scenarios.ServingFanout100Leaves
    assert 1 - f.all_up == approx(0.633968, rel=1e-5)
    assert 1 - f.required_availability(0.99) == approx(1.00498e-4, rel=1e-4)
    assert Scenarios.ServingFanout100Leaves.variant(availability=0.9999).all_up == approx(0.990049, rel=1e-5)
    assert Scenarios.RecommenderTailEvents.events_per_second == approx(1000)
    c = Scenarios.Llama70BReplicaColdStart
    assert c.artifact.to("GB").magnitude == approx(140)
    assert c.fetch_time.to("minute").magnitude == approx(4.66667, rel=1e-5)
    assert c.cold_start_time.to("minute").magnitude == approx(6.66667, rel=1e-5)
    assert c.peak_saving == approx(1 / 3)
    n = Scenarios.SharedPoolNoisyNeighbor
    assert n.other_sequences == 75
    assert n.batch_reduction == approx(0.625)
    assert n.burst_multiple == approx(5)
    assert Scenarios.ReasoningTokenLatency.reasoning_time.to("s").magnitude == approx(12.8)
    b = Scenarios.FourRequestDecodeBatch
    assert b.idle_share == approx(0.375)
    assert (b.slot_iterations, b.useful_iterations, b.requests_advanced) == (800, 500, 7)
    assert b.total_tokens == [150, 280, 220, 240]
    p = Scenarios.RandomVsTwoChoicePlacement.result.max_load
    assert p[1][-1] == approx(7.6)
    assert p[2][-1] == approx(3.6)


def test_inferencex_frontier_and_token_cost():
    x = Scenarios.InferenceXServingFrontier
    assert x.config_count == 325
    b = x.solve("b200", 100)
    assert (b.peak_interactivity, b.peak_throughput) == approx((13.34, 6256.29))
    assert b.best_at == approx(208.02)
    assert x.best_at("h100", 50) == approx(82.1)
    assert x.best_at("gb200", 50) == approx(3653.27)
    assert x.max_concurrency_at("b200", 50, framework="sglang") == 64
    assert x.split_gain("b200", 100) == approx(148.18 / 208.02)
    k = Scenarios.H100OutputTokenCost
    assert k.cost_per_million(50).to("USD").magnitude == approx(13.5336, rel=1e-5)
    assert k.spread == approx(34.651, rel=1e-4)


# --- Retrieval from long-term memory (added 2026-10-20) ------------------------

def test_retrieval_records_in_allowance():
    r = Scenarios.RetrievalRecordsInAllowance.result
    assert r.allowance == 6144 and r.record == 576 and r.records == 10
    assert r.used == 5888 and r.remaining == 256
    assert r.body_only == 12 and r.body_only_used == 7040 and r.overshoot == 896 and r.overcount == 2
    assert r.scaled_metadata == 128 and r.scaled_metadata_records == 9


def test_bm25_short_against_long_file():
    r = Scenarios.Bm25ShortAgainstLongFile.result
    assert r.idf == approx(9.808187) and r.short_normalizer == approx(0.48)
    assert r.long_normalizer == approx(9.3)
    assert r.short_score == approx(14.579738) and r.long_score == approx(8.461966)


def test_exact_scan_ten_million_vectors():
    r = Scenarios.ExactScanTenMillionVectors.result
    assert r.footprint.to("GB").magnitude == approx(15.36) and r.bytes_per_vector.to("byte").magnitude == 1536
    assert r.flops.to("GFLOP").magnitude == approx(15.36)
    assert r.intensity.to("flop/byte").magnitude == approx(1.0) and r.ridge.to("flop/byte").magnitude == approx(20.0)
    assert r.stream_time.to("ms").magnitude == approx(307.2) and r.math_time.to("ms").magnitude == approx(15.36)
    assert r.trajectory_time.to("s").magnitude == approx(6.144)
    assert r.filtered_time.to("ms").magnitude == approx(0.3072)
    assert r.budget_vectors == approx(325520.83)


def test_ann_indexes_ten_million_vectors():
    r = Scenarios.AnnIndexesTenMillionVectors.result
    assert r.graph.to("GB").magnitude == approx(2.56) and r.graph_total.to("GB").magnitude == approx(17.92)
    assert r.codes.to("GB").magnitude == approx(0.64) and r.compression == approx(24.0)
    assert r.lookup_table.to("KB").magnitude == approx(65.536) and r.sub_dim == 12


def test_typed_traversal_depth_three():
    r = Scenarios.TypedTraversalDepthThree.result
    assert r.nodes_by_depth == [1, 4, 13, 40] and r.tokens_by_depth == [280, 820, 2440, 7300]
    assert r.allowance == 6144 and r.fit == 33 and r.packed == 6040
    assert r.shallow_nodes == 13 and r.shallow_required == 2440


def test_repair_edit_index_maintenance():
    r = Scenarios.RepairEditIndexMaintenance.result
    assert r.chunks == 16 and r.batches == 1
    assert r.local_time.to("ms").magnitude == approx(88.0)
    assert r.remote_batch_time.to("ms").magnitude == approx(210.0)
    assert r.remote_per_file_time.to("s").magnitude == approx(1.26)
    assert r.exclusion_write_time.to("us").magnitude == approx(0.4)
    assert r.exclusion_check_time.to("us").magnitude == approx(1.5)
    assert r.write_path_ratio == approx(220000.0)


def test_lexical_graph_against_dense_store():
    r = Scenarios.LexicalGraphAgainstDenseStore.result
    assert r.chunks == 100000 and r.edited_chunks == 30
    assert r.ratio_lexical_graph == approx(0.36) and r.ratio_lexical_dense == approx(8.52)
    assert r.reindex_lexical_graph.to("ms").magnitude == approx(34.0588)
    assert r.reindex_lexical_dense.to("ms").magnitude == approx(612.0)
    assert r.bytes_ratio == approx(23.6667) and r.time_ratio == approx(17.9689)


def test_maintained_memory_policy_cost():
    r = Scenarios.MaintainedMemoryPolicyCost.result
    assert r.baseline_cost == approx(2.0) and r.maintained_cost == approx(1.833333)
    assert r.change == approx(-0.083333) and r.break_even_acceptance == approx(0.55)


def test_word_overlap_scenarios():
    d = Scenarios.ConditionNotifyDocstrings.result
    assert d.shared == 7 and d.union == 13 and d.jaccard == approx(7 / 13)
    p = Scenarios.MeetingPreferenceRevision.result
    assert p.shared == 5 and p.first_words == 6 and p.union == 7


# --- Physical demonstration data (added 2026-10-20) ---

def test_correlated_log_and_imitation_bounds():
    e = Scenarios.PermissionTickTelemetryHour
    assert e.records == approx(3.6e6)
    assert e.effective_samples == approx(1800.9)
    assert e.oversampling == approx(1999.0)
    b = Scenarios.ImitationBoundsFiveSecondEpisode
    assert b.horizon_steps == 500
    assert b.open_loop == approx(0.5)
    assert b.cloning == approx(250)
    assert b.aggregation == approx(0.5)
    assert b.vacuous_error == approx(0.002)
    assert b.cloning_over_maximum == approx(0.5)
    assert Scenarios.ImitationBoundsTenSecondEpisode.vacuous_error == approx(0.001)


def test_pairing_lag_and_teleop_loop():
    p = Scenarios.LatchPairingLag
    assert p.displacement_transit.to("mm").magnitude == approx(17.5)
    assert p.displacement_guarded.to("mm").magnitude == approx(1.05)
    assert p.displacement_fast.to("mm").magnitude == approx(3.5)
    assert p.transit_over_clearance == approx(8.75)
    assert p.lag_ceiling_guarded.to("ms").magnitude == approx(66.667)
    assert p.lag_ceiling_fast.to("ms").magnitude == approx(20)
    assert p.skew_displacement_fast.to("mm").magnitude == approx(3.3)
    t = Scenarios.LatchPairingLagTightPlate
    assert t.lag_ceiling_fast.to("ms").magnitude == approx(10)
    assert t.fast_over_clearance == approx(3.5)
    loop = Scenarios.TeleopCorrectionLoop
    assert loop.loop.to("ms").magnitude == approx(340)
    assert loop.command_issued.to("ms").magnitude == approx(250)
    assert loop.human_share == approx(0.588235)


def test_teleop_ingestion():
    i = Scenarios.WarehouseTeleopIngestion
    assert i.record_bytes == 224
    assert i.kinematic_rate.to("MB/s").magnitude == approx(0.224)
    assert i.nav_rate.to("MB/s").magnitude == approx(2219.44)
    assert i.wrist_rate.to("MB/s").magnitude == approx(279.94)
    assert i.external_rate.to("MB/s").magnitude == approx(622.08)
    assert i.total_rate.to("MB/s").magnitude == approx(3121.68)
    assert i.storage_per_hour.to("TB").magnitude == approx(11.238)
    assert i.storage_per_shift.to("TB").magnitude == approx(89.904)
    assert i.nav_over_slow_drive == approx(4.0354)
    assert i.kinematic_share == approx(7.1756e-5)
    assert i.compressed_rate.to("MB/s").magnitude == approx(390.21)


def test_demonstration_campaign():
    c = Scenarios.LatchDemonstrationCampaign
    assert c.attempts_per_session == 52
    assert c.idle.to("s").magnitude == approx(20)
    assert c.retention == approx(0.68)
    assert c.kept_per_session == approx(35.36)
    assert c.kept_time.to("minute").magnitude == approx(14.733)
    assert c.kept_share == approx(0.245556)
    assert c.campaign_attempts == 7353
    assert c.campaign_time.to("hour").magnitude == approx(141.404)
    assert c.overhead_factor == approx(4.0724)
    assert c.gear_cycles == 102942
    assert c.cable_cycles == 58824
    assert c.gear_life_used == approx(0.0205884)
    assert c.cable_life_used == approx(0.29412)
    assert c.repeat_cable_life_used == approx(0.88236)


def test_hub_counts_and_pooled_corpus():
    h = Scenarios.LeRobotHubDatasetCounts
    assert h.count_at("2024-12") == 1085
    assert h.count_at("2025-12") == 26548
    assert h.latest_count == 80132
    assert h.counted == 59031
    assert h.small_share == approx(0.583676)
    assert h.large_share == approx(0.0191764)
    assert h.median_bin_low == 8
    assert h.median_bin_high == 16
    o = Scenarios.OpenXEmbodimentPoolComposition
    assert o.datasets == 72
    assert o.total == 2419193
    assert o.morphology_share("Single Arm") == approx(0.914399)
    assert o.morphology_share("Mobile Manipulator") == approx(0.0493495)
    assert o.top_datasets == 2
    assert o.top_share_within("Mobile Manipulator") == approx(0.945245)
    assert o.table_top_share == approx(0.891549)
    assert o.rate_share(10.0) == approx(0.569732)
    assert o.rate_share(None) == approx(0.276674)
    assert o.at_or_above_threshold_share == approx(0.0183892)


def test_coverage_grids_and_gripper_transfer():
    g = Scenarios.WarehouseManipulatorStateGrid
    assert g.dims == 28
    assert g.cells == approx(1e28)
    assert g.fine_cells == approx(1e56)
    assert g.records == approx(4.35495e23)
    assert g.max_coverage == approx(4.35495e-5)
    k = Scenarios.ContactAngleCoverage
    assert k.cells == 600
    assert k.occupied("cautious") == 8
    assert k.occupied("exploratory") == 510
    assert k.bin_span("cautious") == (13, 4, 4, 2)
    assert k.empty_share("cautious") == approx(0.986667)
    t = Scenarios.LatchScenarioTable
    assert (t.cells, t.supported, t.absences) == (6, 1, 5)
    r = Scenarios.LatchScenarioTableAfterRecovery
    assert (r.cells, r.supported, r.absences) == (6, 2, 4)
    s = Scenarios.LatchScenarioTableThirdSpeed
    assert (s.cells, s.supported, s.absences) == (9, 1, 8)
    gr = Scenarios.GripperStrokeTransfer
    assert gr.donor_opening.to("mm").magnitude == approx(80)
    assert gr.target_opening.to("mm").magnitude == approx(48)
    assert gr.stroke_excess.to("mm").magnitude == approx(20)
    assert gr.force_ratio == approx(2)
    assert gr.donor_clears and not gr.target_clears


# --- KV cache management (added 2026-10-20) ---------------------------------

def test_repair_request_zones():
    z = Scenarios.RepairRequestZones
    assert z.total_tokens == 32768 and z.shared_tokens == 14336
    assert z.shared_share == approx(0.4375)


def test_llama70b_dgx_h100_kv_admission_by_context():
    fits = {}
    for name in ("Llama3_70BShortContextAdmissionDgxH100", "Llama3_70BAgentContextAdmissionDgxH100",
                 "Llama3_70BLongContextAdmissionDgxH100"):
        sc = getattr(Scenarios, name)
        r = sc.result
        assert r.head_dim == 128 and r.kv_per_token.to("byte").magnitude == 327680
        assert r.hbm_total.to("GB").magnitude == approx(687.19477)
        assert r.weights.to("GB").magnitude == approx(141.2)
        assert r.kv_budget.to("GB").magnitude == approx(525.99477)
        fits[sc.context_tokens] = r.max_requests
    assert fits == {8192: 195, 32768: 48, 131072: 12}
    agent = Scenarios.Llama3_70BAgentContextAdmissionDgxH100.result
    assert agent.kv_per_request.to("GB").magnitude == approx(10.737418)


def test_kv_bytes_per_token_by_attention_design():
    r = Scenarios.KvBytesPerTokenByAttentionDesign.result
    assert r.design == {"GPT-3 (175B)": "MHA", "Llama-3.1-405B": "GQA", "Llama-3.1-70B": "GQA",
                        "Llama-2-70B": "GQA", "Mixtral-8x7B": "GQA", "DeepSeek-V3": "MLA"}
    assert r.per_token["GPT-3 (175B)"].to("kB").magnitude == approx(4718.592)
    assert r.per_token["DeepSeek-V3"].to("kB").magnitude == approx(70.272)
    assert r.mha_over_mla == approx(67.147541) and r.gqa_over_mla == approx(4.6630237)
    assert r.gqa_session.to("GB").magnitude == approx(42.949673)
    assert r.mla_session.to("GB").magnitude == approx(9.2107653)


def test_capped_requests_paged_allocation():
    r = Scenarios.CappedRequestsPagedAllocation.result
    assert r.reservations == 48 and r.final_length_efficiency == approx(0.0625)
    assert r.small_block.to("MB").magnitude == approx(5.24288)
    assert r.large_tail.to("MB").magnitude == approx(20.64384)
    assert r.large_tail_share == approx(0.0019226)
    assert r.small_waste == approx(0.0038961) and r.large_waste == approx(0.0154635)
    assert r.paged_fit == 780 and r.paged_gain == approx(16.25)


def test_forked_candidates_prefix_copy():
    assert Scenarios.ForkedCandidatesPrefixCopy.copied.to("GB").magnitude == approx(10.737418)


def test_invariant_prefix_rebuild_dgx_h100():
    r = Scenarios.InvariantPrefixRebuildDgxH100.result
    assert r.state.to("GB").magnitude == approx(1.3421773)
    assert r.linear_ops.to("flop").magnitude == approx(5.783552e14)
    assert r.attention_ops.to("flop").magnitude == approx(2.1990233e13)
    assert r.sustained_rate.to("TFLOPs/s").magnitude == approx(3560.4)
    assert r.rebuild_time.to("ms").magnitude == approx(168.61741)
    assert r.weight_read_time.to("ms").magnitude == approx(5.2686567)
    assert r.suffix_ops_time < r.weight_read_time


def test_chunked_prefill_among_decodes():
    r = Scenarios.ChunkedPrefillAmongDecodes.result
    assert r.decode_step.to("ms").magnitude == approx(5.2686567)
    assert r.monolithic_prefill.to("ms").magnitude == approx(162.44107)
    assert r.monolithic_gap.to("ms").magnitude == approx(167.70973)
    assert r.chunks == 8 and r.chunked_iteration.to("ms").magnitude == approx(21.574205)
    assert r.chunked_done.to("ms").magnitude == approx(172.59364)
    assert r.gap_ratio == approx(7.7736227)


def test_repair_tool_wait_paused_state():
    r = Scenarios.RepairToolWaitPausedState.result
    assert r.blocks == 2048 and r.state.to("byte").magnitude == 10737418240
    assert r.link.to("GB/s").magnitude == approx(512)
    assert r.copy_in.to("ms").magnitude == approx(20.97152)
    assert r.recompute.to("s").magnitude == approx(1.2995286)
    assert r.recompute_over_reload == approx(61.966352)
    assert r.recompute_over_reload_derated == approx(15.491588)
    assert r.held_occupancy.to("GB*s").magnitude == approx(483.18382)
    assert r.released_on_demand.to("s").magnitude == approx(44.979028)
    assert r.released_prefetch.to("s").magnitude == approx(44.958057)
    assert r.burst.to("GB").magnitude == approx(343.59738)
    assert r.burst_time.to("s").magnitude == approx(0.67108864) and r.burst_slowdown == approx(32)


def test_repair_block_sharing_residency():
    r = Scenarios.RepairBlockSharingResidency.result
    assert r.usable_blocks == 90292 and r.unshared_blocks == 2176
    assert r.common_blocks == 1536 and r.private_blocks == 640
    assert r.fit_unshared == 41 and r.fit_shared == 138
    assert r.retained_rate.to("1/s").magnitude == approx(0.82)
    assert r.released_rate.to("1/s").magnitude == approx(8.2)
    assert r.shared_retained_rate.to("1/s").magnitude == approx(2.76)
    assert r.shared_released_rate.to("1/s").magnitude == approx(27.6)
    assert r.residency_factor == approx(10)
    assert r.shared_fraction == approx(0.7058824) and r.share_gain == approx(3.3658537)
    assert r.share_bound == approx(3.4)
    assert Scenarios.RepairBlockSharingResidencyLongTest.result.residency_factor == approx(19)


def test_repair_and_support_residency():
    r = Scenarios.RepairAgentResidency.result
    assert (r.fit_unshared, r.fit_shared) == (48, 60)
    assert r.shared_fraction == approx(0.1875) and r.residency_factor == approx(10)
    assert r.recompute.to("s").magnitude == approx(1.2995286)
    assert r.reload.to("ms").magnitude == approx(20.97152)
    assert r.recompute_share == approx(0.0259906)
    s = Scenarios.SupportAgentResidency.result
    assert (s.fit_unshared, s.fit_shared) == (195, 780)
    assert s.shared_fraction == approx(0.75) and s.residency_factor == approx(11)
    assert s.recompute.to("s").magnitude == approx(0.32488215)
    assert s.reload.to("ms").magnitude == approx(5.24288)
def test_permission_path_phase_and_response_time():
    p = Scenarios.WarehouseJointLoopThroughTick.result
    assert p.phase_hold.to("degree").magnitude == approx(3.6)
    assert p.phase_compute.to("degree").magnitude == approx(1.8)
    assert p.phase_transport.to("degree").magnitude == approx(7.2)
    assert p.phase_total.to("degree").magnitude == approx(12.6)
    assert p.total_delay.to("ms").magnitude == approx(1.75)
    assert p.allowed_delay.to("ms").magnitude == approx(0.69444)
    assert p.crossover_ceiling.to("Hz").magnitude == approx(7.93651)
    assert p.over_allowance == approx(2.52)
    assert p.phase_per_delay.to("degree/ms").magnitude == approx(7.2)
    assert not p.fits
    assert Scenarios.WarehouseJointLoopNextTickOutput.result.phase_total.to("degree").magnitude == approx(18.0)
    assert Scenarios.WarehouseJointLoopSlowTick.result.phase_hold.to("degree").magnitude == approx(7.2)
    r = Scenarios.WarehousePermissionTickResponse.result
    assert r.response.to("us").magnitude == approx(340)
    assert r.margin.to("us").magnitude == approx(60)
    assert r.blocking_limit.to("us").magnitude == approx(110)
    assert r.wcet_claim.to("us").magnitude == approx(115)
    assert r.meets_deadline
    b = Scenarios.WarehousePermissionTickLongBlocking.result
    assert b.response.to("us").magnitude == approx(410)
    assert b.margin.to("us").magnitude == approx(-10)
    assert not b.meets_deadline


def test_servo_fieldbus_occupancy():
    one = Scenarios.WarehouseServoCanFdOneBus.result
    assert one.frame_time.to("us").magnitude == approx(56.6)
    assert (one.bits_nominal, one.bits_data, one.frames) == (34, 113, 18)
    assert one.occupancy.to("us").magnitude == approx(1018.8)
    assert not one.fits
    assert one.axes_capacity == 8
    arm = Scenarios.WarehouseServoCanFdArmBus.result
    assert arm.occupancy.to("us").magnitude == approx(792.4)
    assert arm.unallocated.to("us").magnitude == approx(207.6)
    assert Scenarios.WarehouseServoCanFdBaseBus.result.occupancy.to("us").magnitude == approx(226.4)
    e = Scenarios.WarehouseServoEtherCat.result
    assert e.occupancy.to("us").magnitude == approx(22.04)
    assert e.unallocated.to("us").magnitude == approx(977.96)
    assert e.utilization == approx(0.02204)


def test_chunk_lease_budget_and_rule():
    r = Scenarios.WarehouseChunkLease.result
    assert r.delay_ceiling.to("ms").magnitude == approx(258.3333)
    assert r.body_terms_distance.to("mm").magnitude == approx(742.5)
    assert r.lease_path_travel.to("mm").magnitude == approx(93)
    assert r.running_total.to("mm").magnitude == approx(835.5)
    assert r.spare.to("mm").magnitude == approx(264.5)
    assert r.admission_to_brake_onset.to("ms").magnitude == approx(82)
    assert r.delay_spare.to("ms").magnitude == approx(176.3333)
    assert r.lease_floor.to("ms").magnitude == approx(55)
    assert r.lease_ceiling.to("ms").magnitude == approx(236.3333)
    assert r.lease_ceiling_after_age.to("ms").magnitude == approx(184.7333)
    assert r.lease_ceiling_at_admission.to("ms").magnitude == approx(121.4226)
    assert r.fits
    assert (r.stall_last_admission.to("ms").magnitude, r.stall_lease_expiry.to("ms").magnitude,
            r.stall_brake_onset.to("ms").magnitude) == (approx(90), approx(150), approx(172))
    long = Scenarios.WarehouseChunkLeaseLengthened.result
    assert long.running_total.to("mm").magnitude == approx(895.5)
    assert long.admission_to_brake_onset.to("ms").magnitude == approx(122)
    assert long.fits
    stop = Scenarios.WarehouseBaseStopFullDelay.result
    assert stop.delay_ceiling.to("ms").magnitude == approx(258.3333)
    assert Scenarios.WarehouseBaseHubDriveElectrical.result.pole_frequency.to("Hz").magnitude == approx(59.1026)


def test_latency_tail_model():
    from mlsysim.core.units import Q_
    from mlsysim.solvers import LatencyTailModel
    t = LatencyTailModel().solve(list(range(1, 1001)), threshold=Q_(0.9, "ms"))
    assert t.count == 1000
    assert t.p50.to("us").magnitude == approx(500.5)
    assert t.maximum.to("us").magnitude == approx(1000)
    assert t.over_threshold == 100


# --- Tool calling (added 2026-10-20) -------------------------------------------

def test_blind_retry_two_percent_loss():
    r = Scenarios.BlindRetryTwoPercentLoss.result
    assert r.timeout_probability == approx(0.0396)
    assert r.duplicate_probability == approx(0.019208)
    assert r.calls == 200_000 and r.duplicates == approx(3841.6)
    assert r.timeouts == approx(7920)


def test_twenty_four_megabyte_log_observation():
    r = Scenarios.TwentyFourMegabyteLogObservation.result
    assert r.raw_tokens.to("token").magnitude == approx(8_000_000)
    assert r.raw_windows == approx(62.5) and r.shaped_share == approx(0.03125)
    assert r.carried_tokens.to("token").magnitude == 120_000
    assert r.carried_cost_uncached.to("USD").magnitude == approx(0.36)
    assert r.carried_cost_cached.to("USD").magnitude == approx(0.036)


def test_model_polling_intervals():
    r = Scenarios.ModelPollingEveryFiveSeconds.result
    assert r.polls == 59 and r.input_tokens == 1_933_312 and r.output_tokens == 2_360
    assert r.input_cost.to("USD").magnitude == approx(0.84099072)
    assert r.output_cost.to("USD").magnitude == approx(0.0354)
    assert r.cost.to("USD").magnitude == approx(0.87639072)
    assert r.worst_delay.to("s").magnitude == approx(5)
    costs = [s.result.cost.to("USD").magnitude for s in (
        Scenarios.ModelPollingEveryTenSeconds, Scenarios.ModelPollingEveryTwentySeconds,
        Scenarios.ModelPollingEveryThirtySeconds, Scenarios.ModelPollingEverySixtySeconds)]
    assert costs == approx([0.43076832, 0.20795712, 0.13368672, 0.05941632])


def test_fifty_tool_catalog_six_selected():
    r = Scenarios.FiftyToolCatalogSixSelected.result
    assert (r.full_tokens, r.selected_tokens, r.released_tokens) == (15_000, 2_200, 12_800)
    assert r.unused_descriptor_tokens == 13_200 and r.break_even_unused == approx(4 / 3)


def test_shell_against_twelve_structured_tools():
    r = Scenarios.ShellAgainstTwelveStructuredTools.result
    assert r.saved_tokens == 111_000 and r.saved_cost.to("USD").magnitude == approx(0.333)
    assert r.shell_attempt_cost.to("USD").magnitude == approx(0.867)
    assert r.structured_cost_per_accepted.to("USD").magnitude == approx(1.5)
    assert r.shell_cost_per_accepted.to("USD").magnitude == approx(1.1115385)
    assert r.structured_cost_per_accepted_with_loss.to("USD").magnitude == approx(4.0)
    assert r.shell_cost_per_accepted_with_loss.to("USD").magnitude == approx(11.367949)
    assert r.break_even_change_probability == approx(0.0011265)


def test_three_hosts_four_tool_services():
    r = Scenarios.ThreeHostsFourToolServices.result
    assert (r.pairwise, r.shared) == (12, 7)


def test_mini_swe_agent_run_truncation():
    s = Scenarios.MiniSweAgentSonnet4Run
    r = s.result
    assert (r.observations, r.truncated_outputs, r.truncated_tasks) == (18_076, 298, 253)
    assert s.head_chars + s.tail_chars == s.truncation_chars == 10_000


# --- Agent sandboxes (added 2026-10-20) ---
def test_syscall_tax_per_million_calls():
    r = Scenarios.SyscallTaxPerMillionCalls.result
    assert r.container_total.to("s").magnitude == approx(0.15)
    assert r.microvm_total.to("s").magnitude == approx(0.10)
    assert r.userspace_total.to("s").magnitude == approx(3.6)
    assert r.userspace_vs_microvm == approx(36)
    assert r.microvm_call < r.container_call < r.userspace_call


def test_parser_calls_envelope_setup():
    r = Scenarios.ParserCallsEnvelopeSetup.result
    assert r.useful_work.to("ms").magnitude == approx(300)
    assert r.per_call_total.to("ms").magnitude == approx(1800)
    assert r.per_trajectory_total.to("ms").magnitude == approx(315)
    assert r.setup_saved.to("ms").magnitude == approx(1485)
    assert r.reuse_speedup == approx(5.7142857)
    assert r.setup_share == approx(0.8333333)
    assert r.work_for_cap.to("ms").magnitude == approx(135)


def test_copy_on_write_repair_workspaces():
    r = Scenarios.CopyOnWriteRepairWorkspaces.result
    assert r.full_copy.to("GB").magnitude == approx(76.8)
    assert r.cow_written.to("GB").magnitude == approx(1.28)
    assert r.full_copy_time.to("s").magnitude == approx(10.971429)
    assert r.cow_time.to("s").magnitude == approx(0.18285714)
    assert r.copy_vs_cow == approx(60)
    assert r.file_copy.to("GB").magnitude == approx(38.4)
    assert r.block_copy.to("MB").magnitude == approx(2.048)
    assert r.logical_changes.to("byte").magnitude == approx(32)
    assert r.file_vs_block == approx(18750)


def test_coding_sandbox_pool_two_per_second():
    r = Scenarios.CodingSandboxPoolTwoPerSecond.result
    assert r.active == approx(600) and r.restore_arrivals == approx(0.03)
    assert r.initial_miss == approx(0.029554466) and r.reserve == 2
    assert r.reserve_miss == approx(0.00044110045)
    assert r.private_memory.to("GB").magnitude == approx(322.12255)
    assert r.pool_memory.to("GB").magnitude == approx(38.952871)
    assert r.memory_ratio == approx(8.2695457)
    assert r.max_restore.to("ms").magnitude == approx(22.701009)
    assert r.slow_arrivals == approx(0.3) and r.slow_reserve == 4


def test_firecracker_preconfigured_boot():
    s = Scenarios.FirecrackerPreconfiguredBootSerial.result
    assert s.median.to("ms").magnitude == approx(110) and s.tail.to("ms").magnitude == approx(115)
    h = Scenarios.FirecrackerPreconfiguredBootHundred.result
    assert h.median.to("ms").magnitude == approx(120) and h.tail.to("ms").magnitude == approx(153)


# --- On-device learning and federated rounds (added 2026-10-20) ---

def test_phone_full_update_working_set():
    w = Scenarios.PhoneFullUpdateWorkingSet
    assert w.weights.to("MB").magnitude == approx(40)
    assert w.optimizer_state.to("MB").magnitude == approx(80)
    assert w.total.to("MB").magnitude == approx(360)
    assert (w.amplification, w.floor, w.ceiling) == (approx(9), approx(4), approx(12))
    assert w.step_traffic.to("MB").magnitude == approx(800)
    assert w.traffic_ratio == approx(20) and w.weight_read_share == approx(0.15)
    assert w.step_floor.to("ms").magnitude == approx(15.625)


def test_phone_background_fine_tune_battery():
    b = Scenarios.PhoneBackgroundFineTuneBattery
    assert b.required_energy.to("Wh").magnitude == approx(2.25)
    assert b.budget_ratio == approx(0.15)


def test_parameter_efficient_updates():
    lr = Scenarios.TransformerLayerRank4Update
    assert (lr.full_params, lr.update_params) == (589_824, 6_144)
    assert lr.reduction == approx(0.98958333)
    ad = Scenarios.CompactEncoderResidualAdapters
    assert (ad.params_per_adapter, ad.params_total) == (25_360, 304_320)
    assert ad.adapter_bytes.to("KB").magnitude == approx(50.72)
    assert ad.total_bytes.to("MB").magnitude == approx(0.60864)


def test_camera_model_personalization_storage():
    p = Scenarios.CameraModelTenContextPersonalization
    assert p.total_copies.to("MB").magnitude == approx(400)
    assert p.total_shared.to("MB").magnitude == approx(42)
    assert p.total_savings == approx(9.5238095) and p.per_context_savings == approx(200)


def test_federated_uplink_and_adapter_upload():
    f = Scenarios.PhoneImagesVersusQuantizedUpdates
    assert f.update_size.to("MB").magnitude == approx(2.5)
    assert f.reduction == approx(80) and f.break_even_rounds == approx(80)
    assert f.weekly_upload("daily").to("MB").magnitude == approx(17.5)
    assert f.weekly_upload("hourly").to("MB").magnitude == approx(420)
    t = Scenarios.FederatedAdapterUploadMobileUplink
    assert t.transfer_time("low").to("s").magnitude == approx(419.4304)
    assert t.transfer_time("high").to("s").magnitude == approx(41.94304)


def test_training_second_versus_wake_word_checks():
    e = Scenarios.TrainingSecondVersusWakeWordChecks
    assert e.checks_low == approx(12779.553) and e.checks_high == approx(95238.095)
    assert e.detector_power.to("mW").magnitude == approx(1.575)
    assert e.time_to_training_second_low.to("min").magnitude == approx(21.164021)
    assert e.time_to_training_second_high.to("min").magnitude == approx(31.746032)


def test_federated_round_mechanics():
    d = Scenarios.ForegroundDeadlinesVersusRoundTrip
    assert [d.local_only(k) for k in d.deadlines] == [True, True, True, False]
    c = Scenarios.KeyboardFleetParticipationCoverage
    assert (c.total("deployed"), c.total("admitted"), c.total("selected"), c.total("completed")) == (100, 40, 26, 20)
    assert c.coverage_ratio("flagship") == approx(2.0) and c.coverage_ratio("budget") == approx(0.25)
    assert c.share("budget", "completed") == approx(0.05) and c.stage_ratio("selected", "completed") == approx(1.3)
    m = Scenarios.TwoClientPairwiseMask
    assert m.total == [approx(0.3), approx(0.2), approx(0.2), approx(0.4)]
    assert m.upload_a == [approx(7.5), approx(-4.7), approx(3.3), approx(9.4)]
    g = Scenarios.AdaptedModelShadowGate
    assert g.disagreements == 2 and g.disagreement_rate == approx(0.25) and g.hold
    s = Scenarios.PhoneSocAiScoreSpread
    assert s.spread("fleet") == approx(411.47273) and s.spread("generation") == approx(14.070715)
    assert s.lowest("fleet")["processor"] == "Snapdragon 460"


def test_on_device_learning_literature():
    from mlsysim import Literature
    L = Literature.OnDeviceLearning
    assert (L.GboardNgramRecallClientPct, L.GboardServerRecallClientPct, L.GboardFederatedRecallClientPct) == (12.5, 15.0, 15.8)
    assert (L.FlSessionsAccepted, L.FlSessionsRejectedLate, L.FlSessionsInterrupted) == (1_116_401, 327_478, 29_771)
    rows = Literature.Series.FedAvgMnistCnnRounds.entries
    assert (rows[0]["iid_rounds"], rows[0]["non_iid_rounds"], rows[-1]["iid_rounds"]) == (626, 483, 18)


# --- Policy training (added 2026-10-20) ---

def test_latch_opening_discount():
    d = Scenarios.LatchOpeningDiscount
    assert d.decisions == 320
    assert d.terminal_weight == approx(0.0401109, rel=1e-5)
    assert d.horizon_steps == approx(100.0)
    assert d.horizon_time.to("s").magnitude == approx(2.5)
    assert d.retaining_discount == approx(0.9978363, rel=1e-6)
    fast = Scenarios.LatchOpeningDiscountDoubledRate
    assert fast.decisions == 640
    assert fast.horizon_time.to("s").magnitude == approx(1.25)


def test_latch_decoder_budget():
    b = Scenarios.LatchDecoderBudget
    assert b.per_evaluation.to("ms").magnitude == approx(2.5)
    assert b.flow_latency.to("ms").magnitude == approx(18.0)
    assert b.one_pass_slack.to("ms").magnitude == approx(10.0)
    assert b.denoising_slack.to("ms").magnitude == approx(2.0)
    assert b.denoising_overrun_under_jitter.to("ms").magnitude == approx(3.0)
    assert b.one_pass_end_under_jitter.to("ms").magnitude == approx(45.0)
    assert (b.evaluations_within_step, b.evaluations_within_renewal) == (6, 14)
    assert b.evaluation_room(b.setpoint_step) == approx(6.8)
    assert b.evaluation_room(b.renewal_less_jitter) == approx(14.8)
    assert b.latency_at(14).to("ms").magnitude == approx(43.0)


def test_latch_trial_campaign_and_parallel_sim():
    r = Scenarios.LeggedGymQuadrupedRun
    assert (r.batch, r.samples) == (98_304, 147_456_000)
    assert r.throughput.to("1/s").magnitude == approx(122_880)
    assert r.simulated_experience.to("day").magnitude == approx(34.133333)
    c = Scenarios.LatchTrialCampaign
    assert c.execution_time.to("day").magnitude == approx(92.592593)
    assert c.automated_time.to("day").magnitude == approx(138.888889)
    assert c.manual_time.to("day").magnitude == approx(439.814815)
    assert c.farm_manual_time.to("day").magnitude == approx(31.415344)
    assert c.manual_reset_share == approx(30 / 38)
    assert c.reversals == 4_000_000
    assert (c.seeded_trials, c.seeded_reversals) == (750, 3000)
    assert c.simulated_steps == 320_000_000
    assert c.simulated_time.to("minute").magnitude == approx(43.402778)
    assert c.simulation_speedup > 1000


def test_latch_contact_deadlines_and_exploit():
    x = Scenarios.LatchContactDeadlines
    assert x.deadline_fast.to("ms").magnitude == approx(2.5)
    assert x.deadline_guarded.to("ms").magnitude == approx(8.333333)
    assert x.brake_onset_fast.to("ms").magnitude == approx(2.375)
    assert x.time_left_fast.to("ms").magnitude == approx(0.125)
    assert x.time_left_guarded.to("ms").magnitude == approx(5.083333)
    e = Scenarios.LatchSoftSpringExploit
    assert e.moving_mass.to("kg").magnitude == approx(9.0)
    assert e.penalty_stiffness.to("N/m").magnitude == approx(15_625)
    assert e.simulated_peak.to("N").magnitude == approx(37.5)
    assert e.rigid_peak.to("N").magnitude == approx(189.736660)
    assert e.guarded_rigid_peak.to("N").magnitude == approx(56.920998)
    assert e.break_even_arm_stiffness.to("N/m").magnitude == approx(153_846.15)
    assert e.chain_peak_force(e.break_even_arm_stiffness).to("N").magnitude == approx(100.0)
    assert e.plate_over_penalty == approx(25.6)
    assert e.penalty_timescale.to("ms").magnitude == approx(24.0)


def test_linear_drive_gap_fine_tune_and_dactyl():
    g = Scenarios.LinearDriveRealityGap
    assert g.simulated_time_to_target.to("ms").magnitude == approx(20.0)
    assert g.physical_time_to_target.to("ms").magnitude == approx(40.0)
    assert g.unmodeled_travel.to("mm").magnitude == approx(9.9)
    f = Scenarios.LatchResidualFineTune
    assert f.setpoint_rate.to("Hz").magnitude == approx(40.0)
    assert f.active_time.to("s").magnitude == approx(500.0)
    assert f.exceed_samples == 300
    a = Scenarios.DactylPhysicalAblation
    for table in ("4", "5"):
        for cond in a.conditions(table):
            assert len(a.trials(cond, table)) == 10
            assert a.median(cond) == approx(a.printed_median(cond))


# --- Network architectures: dense, convolutional, and recurrent layers (added 2026-10-20) ---

def test_mnist_dense_layers():
    mlp = Scenarios.MnistMlpLayerStack
    assert mlp.widths == [784, 128, 64, 10]
    assert mlp.weights == 109184 and mlp.macs_per_example_dense == 109184
    l1 = Scenarios.MnistMlpFirstLayer
    assert l1.weights == 100352 and l1.parameters == 100480
    assert l1.weight_bytes.to("KB").magnitude == approx(401.408)
    assert l1.intensity == approx(0.5) and l1.batch_for_intensity(100) == approx(200)
    assert Scenarios.MnistMlpFirstLayerDoubledInput.weights == 4 * l1.weights
    wide = Scenarios.Dense2048Layer
    assert wide.weights == 4194304 and wide.weight_bytes.to("MB").magnitude == approx(16.777216)
    assert Scenarios.Dense1024LayerFp16.weight_bytes.to("MB").magnitude == approx(2.097152)


def test_four_pixel_layer_by_hand():
    toy = Scenarios.FourPixelThreeNeuronLayer
    assert toy.multiply_adds == 12
    assert toy.terms(0) == [approx(0.40), approx(-0.06), approx(0.18), approx(0.07)]
    assert toy.preactivations == [approx(0.59), approx(-0.09), approx(0.45)]
    assert toy.activations == [approx(0.59), 0.0, approx(0.45)]


def test_mnist_conv_against_dense():
    first = Scenarios.MnistFirstConvLayer
    assert first.dense_weights == 25088 and first.weights == 288
    assert first.sharing_ratio == approx(87.111111) and first.positions == 784
    cnn = Scenarios.MnistSmallCnn
    assert cnn.weights == 421408 and cnn.dense_share == approx(0.9555775)


def test_imagenet_conv_layer():
    rgb = Scenarios.ImageNetRgbFilterVersusDense
    assert rgb.dense_weights == 150528 and rgb.weights == 27 and rgb.sharing_ratio == approx(5575.1111)
    c = Scenarios.ImageNet64ChannelConv
    assert c.weights == 36864 and c.dense_weights == 205520896
    assert c.positions == 50176 and c.macs == 1849688064 and c.outputs == 3211264
    assert c.weight_bytes.to("KB").magnitude == approx(147.456)
    assert c.activation_bytes.to("MB").magnitude == approx(12.845056)
    assert c.activation_to_weight_ratio == approx(87.111111)
    assert c.variant(in_channels=1).weights == 576
    assert (c.depthwise_weights, c.pointwise_weights, c.separable_weights) == (576, 4096, 4672)
    assert c.separable_ratio == approx(7.890411) and c.separable_fraction == approx(1 / 64 + 1 / 9)
    assert c.separable_limit == 9 and c.max_copies == 9
    assert Scenarios.ImageNet64ChannelConvAt1024.dense_weights / c.dense_weights == approx(20.897959)
    assert Scenarios.ImageNet64ChannelConvUnpadded.positions == 49284


def test_im2col_alexnet_receptive_field():
    mini = Scenarios.Im2colMiniature
    assert (mini.input_values, mini.positions, mini.unfolded_entries) == (9, 4, 16)
    assert Scenarios.AlexNetFirstConv.weights == 34848
    rf = Scenarios.ThreeByThreeReceptiveField
    assert (rf.growth_per_layer, rf.field, rf.layers_for_target) == (2, 7, 10)


def test_depth_and_residual_add():
    assert Scenarios.PlainStackGradientDecay.product == approx(0.0051538)
    add = Scenarios.ResidualAddFp32
    assert add.bytes_per_element.to("byte").magnitude == 12 and add.intensity == approx(1 / 12)
    assert Scenarios.ResNet50ConvKernelShare.share == approx(0.4795231)


def test_resnet50_thirty_fps_stream():
    s = Scenarios.ResNet50ThirtyFpsStream
    assert s.required_rate.to("GFLOP/s").magnitude == approx(246)
    assert s.sustained_low.to("TFLOP/s").magnitude == approx(5) and s.sustained_high.to("TFLOP/s").magnitude == approx(6)
    assert s.headroom == approx(20.325203) and s.alternative_headroom == approx(1.6666667)
    assert s.alternative_required_rate.to("TFLOP/s").magnitude == approx(3)
    assert s.peak_frame_rate.to("1/s").magnitude == approx(1219.5122)


def test_keyword_spotter_on_nrf52840():
    fp32, int8 = Scenarios.KwsDsCnnOnNrf52840Fp32, Scenarios.KwsDsCnnOnNrf52840Int8
    assert fp32.weight_bytes.to("KB").magnitude == approx(800) and int8.weight_bytes.to("KB").magnitude == approx(200)
    assert fp32.flash_share == approx(0.7629395) and int8.flash_share == approx(0.1907349)
    assert fp32.sram_ratio == approx(3.0517578)


def test_recurrent_steps():
    rnn, lstm = Scenarios.ElmanRnnStep, Scenarios.LstmStep
    assert (rnn.recurrent_macs, rnn.input_macs, rnn.macs_per_step) == (16384, 12800, 29184)
    assert lstm.gate_rows == 512 and lstm.weights == 4 * rnn.weights
    assert rnn.training_states == 1000
    assert Scenarios.RecurrentState512Fp32.state_bytes.to("KiB").magnitude == approx(2)


# --- Neural computation: the running MNIST network (added 2026-10-20) ---

def test_mnist_dense_network_counts():
    r = Scenarios.MnistDenseNetwork.result
    assert r.layer_widths == [784, 128, 64, 10] and r.layers == 3
    assert r.weights_per_layer == [100352, 8192, 640] and r.biases == 202
    assert r.macs_per_example == 109184 and r.flops_per_example == 218368
    assert r.parameters == 109386
    assert r.parameter_bytes.to("byte").magnitude == 437544
    assert r.first_layer_share == approx(0.919119)
    assert r.matmul_flops_per_layer == [6422528, 524288, 40960]
    assert r.elementwise_flops == 12928 and r.forward_flops == 7000704
    assert r.matmul_share == approx(0.998153)
    assert r.traffic_bytes.to("byte").magnitude == 588328
    assert r.intensity.to("flop/byte").magnitude == approx(11.89932)
    assert r.input_gradient_flops == 565248 and r.training_step_flops == 14540800
    assert r.training_flops_ratio == approx(2.080891)
    assert r.saved_activation_bytes.to("KB").magnitude == approx(126.208)
    assert r.inference_activation_bytes.to("KB").magnitude == approx(116.736)
    assert r.training_memory.to("KB").magnitude == approx(1876.384)
    assert r.inference_memory.to("KB").magnitude == approx(554.28)
    assert r.training_memory_ratio == approx(3.385264)
    assert r.steps_per_epoch == 1875


def test_mnist_dense_network_variants():
    b1 = Scenarios.MnistDenseNetwork.variant(batch_size=1).result
    assert b1.traffic_bytes.to("byte").magnitude == 442256
    assert b1.intensity.to("flop/byte").magnitude == approx(0.494673)
    wide = Scenarios.MnistDenseNetworkWiderHidden.result
    assert wide.layer_widths == [784, 256, 128, 10] and wide.forward_flops == 15049344
    assert Scenarios.MnistDenseNetworkDoubledInputSide.result.weights_per_layer[0] == 401408


def test_dense_width_depth_sweep():
    s = Scenarios.DenseWidthDepthSweep
    assert s.width_double_factor == approx(3.441239)
    assert s.depth_double_factor == approx(1.966722)
    assert s.saved_activations_double_factor == approx(1.911640)


def test_mnist_epoch_cpu_versus_h100():
    r = Scenarios.MnistEpochCpuVersusH100.result
    assert r.steps == 1875 and r.ops_per_step == 12
    assert r.step_flops.to("Mflop").magnitude == approx(14.5408)
    assert r.ridge["host"].to("flop/byte").magnitude == approx(20.0)
    assert r.rate["accelerator"].to("Tflop/s").magnitude == approx(39.86273)
    assert r.rate_batch1["host"].to("Gflop/s").magnitude == approx(24.73364)
    assert r.batching_speedup["host"] == approx(24.05487)
    assert r.work_time["host"].to("ms").magnitude == approx(45.82446)
    assert r.work_time["accelerator"].to("ms").magnitude == approx(0.683947)
    assert r.overhead_time["host"].to("ms").magnitude == approx(225.0)
    assert r.overhead_time["accelerator"].to("ms").magnitude == approx(337.5)
    assert r.work_speedup == approx(67.0)
    assert r.overhead_share_accelerator == approx(0.997978)
    assert r.movement_to_arithmetic_energy == approx(5.846150)
    assert r.movement_to_arithmetic_energy_batch1 == approx(140.6288)


def test_digit_paradigms_and_activation_cost():
    p = Scenarios.DigitRecognitionParadigms
    assert p.network_to_rule_ratio == approx(1091.84) and p.feature_to_rule_ratio == approx(80)
    assert p.feature_count == 441 and p.rule_memory.to("byte").magnitude == 784
    assert Scenarios.ReluVersusSigmoidLogic.ratio == approx(50)


def test_vanishing_gradient_and_formats():
    g = Scenarios.TwentySigmoidLayers
    assert g.peak_slope == approx(0.25) and g.surviving_fraction == approx(9.094947e-13)
    f = Scenarios.VanishedGradientFormats
    assert f.range("fp16") == (65504.0, approx(6.103516e-05), approx(5.960464e-08))
    assert f.range("fp32")[0] == approx(3.402823e38) and f.range("fp32")[1] == approx(1.175494e-38)
    assert f.stored("fp16") == 0.0 and f.stored("fp32") == approx(9.094947e-13)
    assert f.integer_max == 32767


def test_cross_entropy_and_backprop_trace():
    c = Scenarios.DigitCrossEntropyExamples
    assert c.loss("confident") == approx(0.1053605) and c.loss("correct") == approx(0.2231436)
    assert c.uniform_loss == approx(2.302585) and c.misread_loss == approx(2.302585)
    assert c.misread_gradient_true == approx(-0.9) and c.misread_gradient_wrong == approx(0.6)
    t = Scenarios.TwoTwoOneBackpropTrace
    assert t.y_hat == approx(0.54) and t.loss == approx(0.1058) and t.error == approx(-0.46)
    assert t.w2_gradient[0] == approx(-0.414) and t.sent_back == [approx(-0.276), approx(0.184)]
    assert t.delta[1] == 0.0 and t.w1_gradient[1][0] == approx(-0.138)
    assert t.w1_updated(0, 0) == approx(0.5276)


def test_training_state_fit():
    g = Scenarios.Gpt2Fp32AdamOnH100
    assert g.weights.to("GB").magnitude == approx(6.0) and g.total.to("GB").magnitude == approx(24.0)
    assert g.copies == approx(4.0) and g.fits
    r = Scenarios.ResNet50Fp32SgdMomentum  # follows the registry's ResNet-50 count
    params = r.workload.parameters.to("param").magnitude
    assert r.weights.to("byte").magnitude == approx(4 * params)
    assert r.optimizer_state == r.weights and r.copies == approx(3.0)


def test_notable_model_parameter_growth():
    r = Scenarios.NotableModelParameterGrowth.result
    assert r.model_count == 725 and r.split_year == 2010
    assert r.early_doubling_months == approx(43.17425)
    assert r.late_doubling_months == approx(10.47560)
    assert r.largest_parameters >= 1e12


def test_mnist_large_batch_saved_activations():
    big = Scenarios.MnistDenseNetworkLargeBatch.result
    small = Scenarios.MnistDenseNetwork.result
    assert big.batch_size == 1024
    assert big.saved_activation_values / small.saved_activation_values == approx(32)


# ── ML operations at scale (added 2026-10-20) ────────────────────────────────

def test_operational_load_sketch_crosses_capacity_near_fifty_models():
    o = Scenarios.OperationalLoadSketch
    assert o.crossing_models == 51
    assert o.conflicts(100) == approx(5000) and o.alerts(100) == approx(2000)


def test_feature_history_and_lookups():
    h = Scenarios.RecommendationFeatureHistory
    assert h.values == 876_000_000_000_000
    assert h.row_bytes.to("byte").magnitude == approx(32)
    assert h.total_bytes.to("PB").magnitude == approx(28.032)
    assert h.metadata_share == approx(0.75)
    assert Scenarios.ClicksBeforeNoonJoin.leakage_inflation == approx(2.5)
    assert Scenarios.DailyBillionRecommendationLookups.reads_per_second == approx(1157407.4)


def test_incident_costs():
    r = Scenarios.AggregationWindowRevenueIncident
    assert r.undetected_loss.to("USD").magnitude == approx(857142.86)
    assert r.alerted_loss.to("USD").magnitude == approx(28571.43)
    assert r.window_ratio == approx(30) and r.impact_reduction == approx(0.966667)
    assert r.feature_shift == approx(-0.261905)
    c = Scenarios.DelayedAlertClickLoss
    assert c.requests == approx(432e6) and c.lost_clicks == approx(2.16e6)
    assert c.loss.to("USD").magnitude == approx(1.08e6)
    assert c.loss_rate.to("USD/hour").magnitude == approx(45000)


def test_rollout_exposure_and_canary_schedule():
    assert Scenarios.CanaryVersusCutoverExposure.exposure_ratio == approx(20)
    k = Scenarios.CanaryGuardrailSchedule
    assert k.stage_minimum(0.01).to("hour").magnitude == approx(1.0)
    assert k.stage_minimum(0.05).to("minute").magnitude == approx(12.0)
    assert k.total.to("hour").magnitude == approx(4.0) and k.first_stage_buffer == approx(2.0)


def test_experiment_sizing():
    e = Scenarios.CtrLiftExperiment
    assert e.per_variant == approx(29792) and e.collection_time.to("hour").magnitude == approx(1.430016)
    assert e.familywise_error == approx(0.641514)
    s = Scenarios.SmallCtrLiftExperiment
    assert s.per_variant == approx(620928) and s.collection_time.to("hour").magnitude == approx(29.804544)


def test_monitor_fleet_alerts():
    a, b = Scenarios.ThreeSigmaModelFleetAlerts, Scenarios.TwoSigmaModelFleetAlerts
    assert a.monitors == 1000 and a.checks_per_day == approx(288)
    assert a.daily_false_alerts == approx(864) and b.daily_false_alerts == approx(14400)
    assert b.remaining_after_dedup == approx(144)
    assert a.any_false_alert(1000) == approx(0.950437) and b.any_false_alert(100) == approx(0.994079)


def test_slo_budget_vector_and_burn_rates():
    v = Scenarios.RecommendationSloBudgetVector
    assert v.mean == approx(0.70) and v.worst_dimension == "freshness" and not v.promote
    sre = Scenarios.SreMultiwindowBurnAlerts
    assert [sre.budget_spent(t) for t in ("page_fast", "page_slow", "ticket")] == [approx(0.02), approx(0.05), approx(0.10)]


def test_own_versus_rent():
    c = Scenarios.OwnVsRentH100Cluster1K
    assert c.upfront.to("USD").magnitude == approx(59.8e6)
    assert c.rent_per_month.to("USD").magnitude == approx(2392064)
    assert c.crossover_months == approx(26.37773) and c.breakeven_utilization == approx(0.597346)
    assert Scenarios.OwnVsRentH100Cluster1KHalfBusy.crossover_months == approx(43.64836)
    assert Scenarios.OwnVsRentDgxH100Node.breakeven_utilization == approx(0.424751)
    t = Scenarios.OwnVsRentH100Cluster10K
    assert t.upfront.to("USD").magnitude == approx(537.5e6)
    assert t.electricity_per_year_full.to("USD").magnitude == approx(4721640)
    assert t.owned_total.to("USD").magnitude == approx(566664920)
    assert t.rented_total.to("USD").magnitude == approx(840960000)
    assert t.breakeven_utilization == approx(0.539065)
    b = Scenarios.BurstyTrainingOn1KCluster
    assert b.owned_cost_per_run.to("USD").magnitude == approx(3572222.2)
    assert b.rented_cost_per_run(b.run_low).to("USD").magnitude == approx(1376256)


def test_lifecycle_tco_payback_and_platform_benefit():
    s, p = Scenarios.StartupOneModelTco, Scenarios.FiftyModelCompanyTco
    assert s.total.to("USD/month").magnitude == approx(47500) and s.dominant == "iteration"
    assert p.total.to("USD/month").magnitude == approx(980000) and p.dominant == "inference"
    assert round(100 * s.shares["iteration"]) == 84 and round(100 * p.shares["inference"]) == 41
    assert p.ratios_to(s)["total"] == approx(20.631579)
    assert Scenarios.Int8InferencePayback.payback.to("month").magnitude == approx(0.5)
    assert Scenarios.TrainingOptimizationPayback.payback.to("month").magnitude == approx(1.777778)
    assert Scenarios.SharedPlatformBenefit50.benefit_cost_ratio == approx(1.25)
    assert Scenarios.SharedPlatformBenefit20.benefit_cost_ratio == approx(0.5)


def test_latch_evaluation_campaign():
    c = Scenarios.LatchEvaluationCampaign
    assert c.lower_bound == approx(0.860892) and c.failure_upper == approx(0.139108)
    assert c.rule_of_three == approx(0.15)
    assert [c.runs_for(t) for t in c.nine_targets] == [29, 299, 2995]
    assert c.target_runs == 2995 and c.joint_runs == 3688
    assert c.estimate_runs == 598 and c.estimate_rule_of_three_runs == approx(600)
    assert c.weak_policy_clean_probability == approx(0.0692088)
    assert c.lower_bound_with_failures == approx(0.783894)
    assert c.fallacy_lower_bound == approx(0.997009) and c.any_false_positive == approx(0.226219)
    assert c.cycle.to("s").magnitude == approx(40)
    assert c.campaign_time(c.target_runs).to("hour").magnitude == approx(33.27778)
    assert c.operator_time(c.target_runs).to("hour").magnitude == approx(66.55556)
    assert c.shifts(c.target_runs) == approx(8.319444)
    assert c.jam_reset_share == approx(60 / 85)
    assert (c.table.cells, c.table.supported, c.table.absences) == (6, 1, 5)


def test_latch_sequential_test_and_comparison():
    s = Scenarios.LatchSequentialTest
    assert s.success_increment == approx(0.0953102) and s.failure_increment == approx(-2.302585)
    assert s.accept == approx(2.890372) and s.reject == approx(-2.251292)
    assert s.runs_to_accept == 31 and s.trace_score == approx(0.0801694)
    assert s.false_accept_bound == approx(0.05 / 0.9)
    assert s.fixed_runs == 29 and s.fixed_rejects_acceptable == approx(0.252828)
    a = Scenarios.LatchCheckpointComparison
    assert a.fisher_p == approx(0.106029) and a.all_in_one_arm == approx(0.0530146)
    assert a.failures == 4 and a.cohens_h == approx(0.283794)
    assert a.runs_per_arm == 195 and a.runs_per_arm_for(0.2) == 393


def test_hazard_exposure_and_missions():
    w = Scenarios.HazardExposureWall
    assert [w.exposure(r).to("hour").magnitude for r in w.rates] == [approx(29957.32), approx(2.995732e6), approx(2.995732e9)]
    assert w.rig_rate.to("1/hour").magnitude == approx(3.417445e-4)
    assert w.shortfall(w.rates[1]) == approx(341.7445) and w.fleet_time(w.rates[2]).to("year").magnitude == approx(3417.445)
    m = Scenarios.WarehouseMissionExposure
    assert m.mission_distance.to("m").magnitude == approx(540)
    assert m.missions_needed == approx(299571.73) and m.calendar_time.to("hour").magnitude == approx(39942.90)
    assert m.wheel_revolutions == approx(1.935812e8, rel=1e-5) and m.relaxed_exposure.to("hour").magnitude == approx(2995.732)
    i = Scenarios.WarehouseSupervisedMissions
    assert i.active_time.to("hour").magnitude == approx(10) and i.distance.to("km").magnitude == approx(54)
    assert i.mtbi.to("minute").magnitude == approx(15.78947) and i.density.to("1/km").magnitude == approx(0.703704)
    lo, hi = i.count_interval
    assert lo == approx(26.8911) and hi == approx(52.1580)


def test_heading_drift_contact_and_force_tail():
    h = Scenarios.AisleHeadingDrift
    assert h.angular_mse.magnitude == approx(3.046174e-4)
    assert h.time_to_clearance.to("s").magnitude == approx(5.729869) and h.distance_to_clearance.to("m").magnitude == approx(8.594803)
    assert h.aisle_fraction == approx(0.286493)
    k = Scenarios.StiffSurfaceResidual
    assert k.residual_force.to("N").magnitude == approx(1000)
    assert k.tripwire_penetration.to("mm").magnitude == approx(0.03) and k.limit_penetration.to("mm").magnitude == approx(0.2)
    f = Scenarios.LatchForceTail
    assert f.tail_over_limit.to("N").magnitude == approx(15) and f.tail_over_mean == approx(115 / 12)


def test_road_exposure_and_crash_rate_denominator():
    r = Scenarios.RoadMilesToDemonstrate
    for e in ("fatalities", "injuries", "crashes"):
        assert r.recomputed_zero_failure(e) == approx(r.zero_failure[e], rel=0.02)
    assert r.shortfall("fatalities") == approx(4.850088) and r.precision_over_fleet("fatalities") == approx(155.2028)
    assert sorted(r.cleared) == [1.6, 3.9, 51.0]
    assert r.requirements("crashes") == (1.6, 51.0, 65.0)
    d = Scenarios.AutosteerCrashRateDenominator
    assert d.reported_change == approx(-0.384615) and d.cohort_change == approx(0.592105)
    assert d.exact_exposure_share == approx(0.13)


def test_closed_loop_evaluation_statistics():
    from mlsysim.physics import (calc_clopper_pearson_lower_bound, calc_clopper_pearson_zero_failure_bound,
                                 calc_poisson_exact_interval, calc_fisher_exact_two_sided)
    assert calc_clopper_pearson_lower_bound(20, 20) == approx(calc_clopper_pearson_zero_failure_bound(20))
    assert calc_clopper_pearson_lower_bound(0, 20) == 0.0
    lo, hi = calc_poisson_exact_interval(0)
    assert lo == 0.0 and hi == approx(3.688879)
    assert calc_fisher_exact_two_sided(10, 0, 0, 10) == approx(2 / 184756)
