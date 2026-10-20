"""Calculation scenarios: named inputs plus the solver that owns the arithmetic.

Chapters and lecture slides quote these values; pinning them here means a
registry or solver change that moves one shows up as a failing test instead
of a silent change in print. Values were checked against the hand-derived
chapter results when the scenarios were added (2026-10-20).
"""

import pytest

from mlsysim import Models, Scenarios


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


def test_llama2_70b_reference_replica_decode():
    d = Scenarios.Llama2_70B_DecodeH100
    assert d.workload.name == "Llama-2-70B"
    assert d.result.memory_time.to("ms").magnitude == approx(41.79104, rel=1e-5)


def test_resnet50_macs_from_architecture():
    from mlsysim.physics.layers import calc_resnet_bottleneck_macs
    v1 = calc_resnet_bottleneck_macs()
    assert v1["3x3"] == 1_849_688_064
    assert v1["total"] == 3_857_973_248
    assert calc_resnet_bottleneck_macs(stride_on_3x3=True)["total"] == approx(4.0892e9, rel=1e-4)
    assert Scenarios.ResNet50ConvKernelShare.share == approx(0.47945, rel=1e-4)
# --- Network architectures: attention, transformer blocks, decode, recommenders (added 2026-10-20) ---

def test_toy_attention_head():
    t = Scenarios.FourTokenToyAttentionHead
    assert t.scores[0] == pytest.approx([2 ** 0.5, 0, 2 ** 0.5 / 2, 0])
    assert t.weights[0] == pytest.approx([0.5052, 0.1228, 0.2491, 0.1228], abs=1e-4)
    assert t.outputs[0] == pytest.approx([0.7543, 0.3720], abs=1e-4)
    assert t.outputs[1] == pytest.approx(t.outputs[0][::-1]) and t.outputs[3] == pytest.approx(t.outputs[0])
    assert t.macs_per_product == 32


def test_packed_qkv_projection():
    p = Scenarios.Gpt2SmallSixTokenQkvProjection
    assert (p.width, p.packed_width, p.workload.head_width) == (768, 2304, 64)
    assert p.macs == 10_616_832


def test_gpt2_xl_block_counts():
    b = Scenarios.Gpt2XlBlockFullWindow
    assert (b.weights_per_block, b.block_coefficient) == (30_720_000, 12)
    assert b.block_weights == 1_474_560_000 and b.token_table == 80_411_200
    assert b.attention_weights == 491_520_000 and b.mlp_weights == 983_040_000
    assert b.exact_parameters == 1_557_611_200
    assert b.flops_per_token.to("GFLOP").magnitude == approx(2.94912)
    assert (b.dense_macs, b.attention_macs) == (31_457_280_000, 3_355_443_200)
    assert b.attention_share == approx(0.0963855) and b.crossover == 9600
    assert b.prompt_flops.to("TFLOP").magnitude == approx(3.3420214)
    assert Scenarios.Gpt2XlBlockAt100kTokens.attention_to_dense == approx(10.416667)
    assert Scenarios.Gpt2XlBlockTwiceAsWide.crossover == 19200


def test_materialized_scores_and_growth():
    a = Scenarios.TwelveHeadScoresAt100kTokens
    assert a.scores_per_layer == 120_000_000_000
    assert a.layer_bytes.to("GB").magnitude == approx(240) and a.total_bytes.to("GB").magnitude == approx(7680)
    assert a.layer_device_multiple == approx(3)
    assert Scenarios.TwelveHeadScoresAt8192Tokens.layer_bytes / Scenarios.TwelveHeadScoresAt4096Tokens.layer_bytes == approx(4)
    assert Scenarios.OneHeadScoresAtMillionTokens.layer_bytes.to("TB").magnitude == approx(2)
    assert Scenarios.BertWindowToFourK.score_ratio == 64 and Scenarios.BertWindowToTwoK.score_ratio == 16
    assert Scenarios.Gpt3ToGemini15PreviewWindow.ratio == approx(488.28125)


def test_kv_footprints():
    assert Scenarios.Gpt2XlKvFullWindow.per_token.to("byte").magnitude == 307_200
    assert Scenarios.Gpt2XlKvFullWindow.total.to("MB").magnitude == approx(314.5728)
    assert Scenarios.Llama2_7BKvAt2048Tokens.total.to("GB").magnitude == approx(1.073741824)
    assert Scenarios.Llama3_8BKvFullWindow.total.to("GB").magnitude == approx(17.179869)
    assert Scenarios.Llama3_8BKvAtMillionTokens.total.to("GB").magnitude == approx(131.072)
    o = Scenarios.Opt13BKvOnA100_40GB
    assert o.per_token.to("byte").magnitude == 819_200 and o.requests == approx(7.1525574)
    assert Scenarios.Opt13BKvOnA100_40GBHalfWidth.requests == approx(2 * o.requests)
    assert Models.Language.PaLM_540B.get_kv_cache_size(1, 1).to("byte").magnitude == 120_832
    assert Models.Language.Llama2_70B.queries_per_kv_head == 8


def test_context_memory_crossings():
    c = Scenarios.Gpt2XlContextMemoryOnH100
    assert c.kv_crossing == approx(9765.625) and c.kv_score_crossing == approx(6144)
    assert c.score_crossing == approx(41448.606)


def test_decode_weight_stream():
    one, two = Scenarios.Llama2_70B_DecodeH100.result, Scenarios.Llama2_70B_DecodeTwoH100.result
    assert one.weight_bytes.to("GB").magnitude == approx(140) and two.weight_bytes.to("GB").magnitude == approx(70)
    assert two.memory_time.to("ms").magnitude == approx(20.895522)
    assert Scenarios.H100DecodeRoofline.ridge.magnitude == approx(295.22388)
    assert Scenarios.Gpt2XlDecodeH100.result.memory_time.to("ms").magnitude == approx(0.8955224)


def test_lighthouse_training_state():
    assert Scenarios.ResNet50Fp32Adam.total.to("MB").magnitude == approx(408.912512)
    assert Scenarios.MobileNetV2Fp32Adam.total.to("MB").magnitude == approx(56.077952)
    assert Scenarios.KwsDsCnnFp32Adam.total.to("MB").magnitude == approx(3.2)


def test_recommender_gather_and_tables():
    g = Scenarios.CriteoShapedRequestOnH100
    assert g.request_bytes.to("byte").magnitude == 13_312 and (g.pairs, g.interaction_macs) == (351, 44_928)
    assert g.intensity.magnitude == approx(6.75) and g.bandwidth_ratio == approx(52.34375)
    assert g.device_rate.to("1/s").magnitude == approx(251_652_644.2)
    e = Scenarios.HundredMillionItemTableOnA100
    assert e.table_bytes.to("GB").magnitude == approx(51.2) and e.labeled_share == approx(0.64)
    assert not Scenarios.TwoHundredMillionRowTablesOnA100.fits
    assert Scenarios.BillionUserTable.table_bytes.to("GB").magnitude == approx(512)
def test_framework_dispatch_and_fusion():
    g = Scenarios.Gpt2XlEagerDecodeDispatch
    assert g.operations == 576 and g.launches == 576
    assert g.device_time.to("ms").magnitude == approx(0.895522, rel=1e-5)
    assert g.launch_time.to("ms").magnitude == approx(2.88)
    assert g.host_time.to("ms").magnitude == approx(8.64)
    assert g.overhead_ratio == approx(9.648) and g.launch_ratio == approx(3.216)
    assert Scenarios.SmallKernelDispatch.host_share == approx(0.9375)
    assert Scenarios.LargeKernelDispatch.host_share == approx(0.130435, rel=1e-5)
    small = Scenarios.SmallMlpDispatch
    assert small.host_time.to("microsecond").magnitude == approx(90)
    assert small.overhead_ratio == approx(34.615385) and small.speedup_limit == approx(35.615385)
    assert Scenarios.Gpt3BlockDispatch.overhead_ratio == approx(0.0018)
    pair, chain = Scenarios.ElementwisePairFusion, Scenarios.LayerNormDropoutReluFusion
    assert pair.overhead_speedup == approx(2) and pair.traffic_ratio == approx(2)
    assert chain.unfused_trips == 6 and chain.unfused_launches == 3 and chain.traffic_ratio == approx(3)


def test_framework_compile_breakeven_and_jacobian():
    c = Scenarios.ResNet50CompileBreakeven
    assert c.saving_per_item.to("ms").magnitude == approx(0.224539, rel=1e-5)
    assert c.breakeven_items == approx(133607.14, rel=1e-6)
    assert c.experiment_eager.to("second").magnitude == approx(6.896552, rel=1e-6)
    assert c.experiment_compiled_total.to("second").magnitude == approx(304.651163, rel=1e-6)
    assert c.experiment_slowdown == approx(44.174419, rel=1e-6)
    j = Scenarios.Dense100kLayerJacobianFp32
    assert j.elements == 10 ** 10 and j.size.to("GB").magnitude == approx(40)


def test_framework_training_memory_and_host_feed():
    lo, hi = Scenarios.ResNet50Fp32AdamBatch64, Scenarios.ResNet50Fp32AdamBatch64HighActivations
    assert lo.weights.to("MB").magnitude == approx(102.228128)
    assert lo.optimizer_state.to("MB").magnitude == approx(204.456256)
    assert lo.training_total.to("GB").magnitude == approx(8.408913, rel=1e-6)
    assert hi.training_total.to("GB").magnitude == approx(12.408913, rel=1e-6)
    assert lo.training_copies == approx(82.256348, rel=1e-6) and hi.training_copies == approx(121.384523, rel=1e-6)
    assert lo.activations_at(512).to("GB").magnitude == approx(64) and hi.activations_at(512).to("GB").magnitude == approx(96)
    t = Scenarios.Fp32MillionElementTensorAcrossH100Links.result.times
    assert t["PCIe"].to("microsecond").magnitude == approx(62.5)
    assert t["NVLink"].to("microsecond").magnitude == approx(8.888889, rel=1e-6)
    f = Scenarios.ImageNetBatch64HostFeed
    assert f.batches_in_flight == 8 and f.raw_feed.to("MB/s").magnitude == approx(150.528)
    assert f.batch_bytes.to("MB").magnitude == approx(38.535168)
    assert f.copy_time.to("ms").magnitude == approx(0.602112)
    assert f.copy_share("large model") == approx(0.1204224)


def test_framework_mlp_training_step_and_paths():
    m = Scenarios.MnistTwoLayerTrainingStepOnH100
    assert m.parameters == 101770
    assert m.total_flops.to("flop").magnitude == approx(13100608)
    assert m.total_bytes.to("byte").magnitude == approx(1191936)
    assert m.dense_training_ratio == approx(2.012594, rel=1e-6)
    assert m.optimizer_step_bytes.to("MB").magnitude == approx(2.84956)
    assert m.host.operations == 5 and m.host.launches == 9
    assert m.host.host_time.to("microsecond").magnitude == approx(95)
    assert m.compute_time.to("microsecond").magnitude == approx(0.195531, rel=1e-5)
    p = Scenarios.EmbeddedLinuxFrameworkPaths
    assert p.passing == ["PyTorch to ONNX Runtime", "TensorFlow to TensorFlow Lite"]
    assert not p.variant(latency_budget=p.paths["PyTorch to ONNX Runtime"]["latency"] / 2).passing


def test_framework_batch512_activation_scaling():
    b = Scenarios.ResNet50Fp32AdamBatch512
    assert b.batch_size == 512 and b.activations.to("GB").magnitude == approx(64)
    hi = Scenarios.ResNet50Fp32AdamBatch64HighActivations.activations_at(b.batch_size)
    assert b.training_total < b.hardware.memory.capacity < hi
def test_query_extraction_budget():
    e = Scenarios.ResNet50ApiExtractionBudget
    assert e.collection_days == approx(500)
    assert e.full_set_tier.to("USD").magnitude == approx(833.333333)
    assert e.full_set_pay_per_query.to("USD").magnitude == approx(5000)
    assert e.allowance_fees.to("USD").magnitude == approx(16.666667)
    assert e.full_set_lifetime.to("USD").magnitude == approx(4916.666667)
    assert e.lifetime_cost(5e4).to("USD").magnitude == approx(e.tier_cost(5e4).to("USD").magnitude)


def test_output_shaping_and_attack_surface():
    o = Scenarios.ImageNetTop5RoundedScores
    assert o.scores_reduction == approx(200) and o.entries_removed == approx(0.995)
    assert o.bits_removed == approx(13.287712) and o.retained_bits == approx(6.658211)
    r = Scenarios.ReferenceJobAttackSurface
    assert r.accelerators == 1024 and r.nodes == 128 and r.accelerators_per_replica == approx(16)
    assert r.gradient_bytes.to("GB").magnitude == approx(350)
    assert r.checkpoint_bytes.to("TB").magnitude == approx(2.45)
    assert r.weight_bytes.to("GB").magnitude == approx(350)
    assert r.allreduce_factor == approx(1.96875) and r.gather_traffic_ratio == approx(32)
    b = Scenarios.ByzantineMeanOnReferenceJob
    assert b.mean() == approx(157.234375) and b.median() == approx(1.0)


def test_isolation_enclave_and_encryption_costs():
    assert Scenarios.SecureMultiTenantServing.overhead == approx(0.15)
    f = Scenarios.ResNet50Fp32InSgxEnclave
    assert f.weight_bytes.to("MB").magnitude == approx(102.228128)
    assert f.capacity_share == approx(0.761659)
    assert Scenarios.ResNet18Int8InSgxEnclave.weight_bytes.to("MB").magnitude == approx(11.7, rel=1e-3)
    h = Scenarios.HealthMonitoringEncryptedInference
    assert h.encrypted_transport_latency.to("ms").magnitude == approx(20.5)
    assert h.fhe_latency.to("second").magnitude == approx(200)
    p = Scenarios.FiveBytePasswordPowerAnalysis
    assert p.brute_force_guesses == 256 ** 5 and p.bytewise_guesses == 1280


def test_differential_privacy_scenarios():
    s = Scenarios.SalaryMeanLaplaceRelease
    assert s.noise_scale.to("USD").magnitude == approx(200000)
    assert s.mean_error().to("USD").magnitude == approx(200)
    assert s.small_group_error.to("USD").magnitude == approx(2000)
    assert Scenarios.TenReleasesBasicComposition.total_epsilon == approx(10)
    a = Scenarios.DpSgdAccountantComparison
    assert a.tightening(100) == approx(7.412698) and a.tightening(400) == approx(9.498039)
    m = Scenarios.MembershipAuditAtEpsilonOne
    assert all(m.audit(i / 200) <= m.ceiling(i / 200) + 1e-12 for i in range(201))
    c = Scenarios.Cifar10DpSgdRun
    assert c.sampling_rate == approx(0.08) and c.steps == 750
    assert Scenarios.LaionWebScalePoisoning.poisoned_items == approx(40000)
def test_robust_ai_drift_and_silent_errors():
    assert Scenarios.FleetSilentErrorHour1K.p_step == approx(0.0951671)
    assert Scenarios.FleetSilentErrorHour10K.p_step == approx(0.6321390)
    m = Scenarios.FeatureMeanShiftTest
    assert m.standard_error == approx(0.0094868) and m.z_score == approx(5.270463)
    assert m.p_value < m.significance_bound
    p = Scenarios.FraudCountryPsi
    assert p.psi == approx(0.0653018) and p.term("Other") == approx(0.0470229)
    assert p.shares("Other") == (0.10, 0.18)
    d = Scenarios.FraudAmountDrift
    assert d.kl_log_term == approx(0.1670541) and d.kl_ratio_term == approx(0.3846154)
    assert d.kl == approx(0.0516695) and d.triggers
    k = Scenarios.FraudAmountKs
    assert k.coefficient == approx(1.6276236) and k.critical_value == approx(0.0230181)
    assert k.scale == approx(0.0141421) and k.flags


def test_robust_ai_answering_attacks_and_faults():
    s = Scenarios.SelectiveAnswering90
    assert s.risk == approx(0.0651441) and s.full_risk == approx(0.112)
    assert s.fallback_rate.to("1/s").magnitude == approx(1000)
    c = Scenarios.ConformalThreshold1K
    assert c.rank_bound == approx(900.9) and c.rank == 901
    assert Scenarios.PgdSevenTraining.passes == 8 and Scenarios.PgdSevenTraining.slowdown == 8
    assert Scenarios.FgsmCleanPlusAdversarialTraining.passes == 3
    assert Scenarios.ResNet50CleanVersusRobustAttacked.gap == approx(26.13, rel=1e-4)
    r = Scenarios.SmoothingRadiusCifar10
    assert r.radius == approx(1.5451162) and r.quantile == approx(3.0902323)
    assert r.coordinates == 3072 and r.box_corner_l2 == approx(1.7388432) and not r.covers_box
    f = Scenarios.Fp32WeightExponentFlip
    assert f.flipped_bit == 30 and f.word_bits == 32 and f.scale_log2 == 128
    assert f.corrupted == approx(1.4291859e37)
    t = Scenarios.ImageNetTestbedV2Shift.result
    assert t.model_count == 212 and t.baseline_count == 0
    assert (t.gap_min, t.gap_max, t.gap_median) == (approx(7.278), approx(13.572), approx(12.010))
    b = Scenarios.RobustBenchCifar10LinfProgress
    assert b.best_first["autoattack_acc"] == approx(44.04) and b.best_last["autoattack_acc"] == approx(75.28)
    assert b.best_last["clean_acc"] == approx(93.6) and len(b.broken) == 4
# --- Agent security: injection exposure and defense evaluation (added 2026-10-20) ---
def test_injection_exposure_by_read_volume():
    r = Scenarios.InjectionExposureByReadVolume.result
    assert r.exposure["repair"] == approx(0.18209306)
    assert r.exposure["research"] == approx(0.63396766)
    assert r.exposure["support"] == approx(0.05851985)
    assert r.exposure["remediation"] == approx(0.09561792)
    assert r.linear["repair"] == approx(0.2)
    assert r.screened_rate == approx(0.0005)
    assert r.screened["repair"] == approx(0.0099526422)
    assert r.screened["research"] == approx(0.04878247)
    assert r.screened_per_cohort["research"] == approx(48.78247)


def test_repair_trajectory_turn_exposure():
    r = Scenarios.RepairTrajectoryTurnExposure.result
    assert r.per_turn == approx(0.0199) and r.reads == 60
    assert r.trajectory == approx(0.45284336)
    assert r.same_reads == approx(r.trajectory, rel=1e-12)


def test_fixture_release_policy_comparison():
    r = Scenarios.FixtureReleasePolicyComparison.result
    assert (r.baseline_lower, r.candidate_lower) == (approx(0.20), approx(0.05))
    assert (r.baseline_upper, r.candidate_upper) == (approx(0.25), approx(0.15))
    assert r.reduction_observed == approx(0.75)
    assert r.reduction_upper == approx(0.40)
    assert r.reduction_least_favorable == approx(0.25)
    assert r.cost_rise == approx(0.125)
    assert r.lost_benign == r.extra_false_blocks == 10
    assert r.baseline_cost_per_accepted == approx(2.3333333)
    assert r.candidate_cost_per_accepted == approx(2.625)


def test_clean_attack_suites():
    r50 = Scenarios.CleanAttackSuite50.result
    assert r50.rate_upper_exact == approx(0.058155079)
    assert r50.rate_upper == approx(0.059914645)
    assert r50.trials_for_target == 299
    assert r50.trials_for_target_approx == approx(299.57323)
    r100 = Scenarios.CleanAttackSuite100.result
    assert r100.rate_upper_exact == approx(0.02951305)


def test_prefix_cache_probe_and_delegated_grant():
    probe = Scenarios.PrefixCacheProbe2K.result
    assert probe.rebuild_time.to("ms").magnitude == approx(82.764621)
    assert probe.weight_read_time.to("ms").magnitude == approx(5.2686567)
    assert Scenarios.DelegatedRepositoryGrant.refused == 49
# --- The agent harness (added 2026-10-20) ------------------------------------

def test_freeze_rule_and_turn_boundary_wait():
    r = Scenarios.FreezeRuleOverStepBudget.result
    assert r.success[30] == approx(0.739700)
    assert Scenarios.CanceledStreamDecodeSavings.allowance_decode_time.to("s").magnitude == approx(86.32167)


def test_approval_escrow_by_workload():
    r = Scenarios.RepairRemediationApprovalEscrow.result
    assert r.held == approx(12) and r.retained_memory.to("GiB").magnitude == approx(6)
    assert r.remediation_held == approx(0.5) and r.time_to_mitigate.to("minute").magnitude == approx(30)
    assert r.approval_share == approx(0.5)
    f = Scenarios.RepairRemediationEscrowFasterReview.result
    assert f.retained_memory.to("GiB").magnitude == approx(3) and f.time_to_mitigate.to("minute").magnitude == approx(22.5)


def test_deficit_round_robin_shares():
    r = Scenarios.InteractiveMaintenanceDrr.result
    assert r.calls_per_round["interactive"] == approx(9.6) and r.rounds_per_call["maintenance"] == approx(5.25)
    assert r.charge_share["interactive"] == approx(0.75) and r.call_share["interactive"] == approx(0.980545)
    assert r.call_ratio == approx(50.4)
    assert r.simulated_call_share["interactive"] == approx(r.call_share["interactive"], rel=1e-4)
    assert Scenarios.InteractiveMaintenanceDrrCompacted.result.rounds_per_call["maintenance"] == approx(2.75)


def test_bounded_liability_admission():
    r = Scenarios.RepairNextCallAdmission.result
    assert r.liability == approx(0.21144) and r.spend_headroom == approx(0.92) and r.headroom == approx(0.12)
    assert r.spend_check_admits and not r.admitted
    assert r.finalization_shortfall == approx(0.09144) and r.input_share == approx(0.709421)
    assert r.max_input_tokens == approx(19520) and r.reservation_share == approx(0.142857)


def test_trajectory_budget_meter():
    r = Scenarios.RepairTrajectoryBudgetMeter.result
    assert r.input_tokens == 892500 and r.cached_tokens == 841000 and r.total_tokens == 898500
    assert r.cost_with_reuse == approx(0.4968) and r.cost_without_reuse == approx(2.7675)
    assert r.used_with_reuse["tokens"] == approx(0.44925) and r.used_without_reuse["dollars"] == approx(0.790714)
    assert r.leading_with_reuse == "turns" and r.leading_without_reuse == "dollars"


def test_call_and_step_ceilings_from_public_runs():
    r = Scenarios.SweBenchCallCeilingP90.result
    assert r.runs == 2100 and len(r.configurations) == 5 and r.longer_when_failed
    assert r.failed_halted_low == approx(0.148649) and r.failed_halted_high == approx(0.310241)
    assert r.accepted_halted_high == approx(0.100529)
    assert r.cap_low == approx(46.3) and r.cap_high == approx(101)
    assert r.median_calls_low == approx(23) and r.median_calls_high == approx(64)
    c = Scenarios.MiniSweAgentOpus46RunCost.result
    assert c.cost_per_resolved == approx(0.729527) and c.over_limit_runs == 8
    assert c.mean_overshoot == approx(0.057131) and c.max_overshoot == approx(0.286232)
    s = Scenarios.OSWorldStepCeilingReturn.result
    assert s.steps == [15, 50, 100] and s.flat_after_middle == 3
    assert s.best_gain == approx(20.0) and s.best_model == "claude-sonnet-4-5-20250929"
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
def test_sensor_perception_capture_and_age():
    a = Scenarios.WarehouseNavCameraAge.result
    assert [round(st[2].to("ms").magnitude, 1) for st in a.stages] == [8.0, 24.6, 25.4, 26.6, 29.1, 51.1, 51.6]
    assert a.age.to("ms").magnitude == approx(51.6) and a.age_travel.to("mm").magnitude == approx(77.4)
    assert a.arrival_offset.to("ms").magnitude == approx(26.6) and a.arrival_travel.to("mm").magnitude == approx(39.9)
    assert a.trigger_travel.to("mm").magnitude == approx(12.0) and a.exposure_travel.to("mm").magnitude == approx(24.0)
    assert a.sensor_age_middle_row.to("ms").magnitude == approx(16.3)
    assert a.top_row_excess.to("ms").magnitude == approx(7.9333)
    assert a.frame_travel.to("mm").magnitude == approx(25.0)
    assert a.batch_wait.to("ms").magnitude == approx(50.0) and a.batched_age.to("ms").magnitude == approx(101.6)
    assert round(100 * a.capture_share) == 48 and round(100 * a.backbone_share) == 43
    r = Scenarios.WarehouseNavCameraRollingShutter.result
    assert r.row_interval.to("us").magnitude == approx(5.4605) and r.skew.to("mm").magnitude == approx(24.9)
    assert r.lateral_skew.to("mm").magnitude == approx(9.13) and r.lateral_share == approx(0.060867)
    assert Scenarios.WarehouseNavCameraRollingShutterAisleSpeed.result.skew.to("mm").magnitude == approx(21.58)
    e = Scenarios.WarehouseRackEndTimingAttitude.result
    assert e.timing_travel.to("mm").magnitude == approx(1.5) and e.timing_lateral.to("mm").magnitude == approx(0.55)
    assert e.angle_lateral.to("mm").magnitude == approx(19.198)
    m = Scenarios.WarehouseLongAisleMountError.result
    assert m.angle_lateral.to("mm").magnitude == approx(261.80) and m.angle_lateral_share == approx(1.7453)
    assert Scenarios.WarehouseLidarExtrinsicShift.result.angle_lateral_share == approx(1.0)
    l = Scenarios.WarehouseRackEndLidarAge.result
    assert l.sweep_travel.to("mm").magnitude == approx(75.0) and l.sweep_rotation.to("mrad").magnitude == approx(25.0)
    assert l.sector_angle.to("deg").magnitude == approx(61.158) and l.age_streamed.to("ms").magnitude == approx(19.494)
    assert l.age_full.to("ms").magnitude == approx(61.0) and l.slack_streamed.to("ms").magnitude == approx(32.106)
    assert Scenarios.WarehouseChunkLease.result.renewal_lateness_limit.to("ms").magnitude == approx(10.0)
    c = Scenarios.WarehouseLidarClockConversion.result
    assert c.drift_per_interval.to("us").magnitude == approx(10.0)
    assert c.dispatch_age_limit.to("ms").magnitude == approx(51.8) and c.bound_travel.to("mm").magnitude == approx(0.3)


def test_sensor_perception_error_scales():
    s = Scenarios.WarehouseStereoRigDepthError.result
    assert s.sigma_near.to("mm").magnitude == approx(12.5) and s.sigma_far.to("mm").magnitude == approx(312.5)
    assert s.sigma_target.to("mm").magnitude == approx(3.78125) and s.sigma_growth == approx(25.0)
    assert s.inset_near.to("mm").magnitude == approx(162.5) and s.inset_far.to("mm").magnitude == approx(462.5)
    assert s.range_at_loc_allowance.to("m").magnitude == approx(4.0)
    assert s.range_at_side_clearance.to("m").magnitude == approx(6.9282)
    assert s.range_at_side_clearance_half_baseline.to("m").magnitude == approx(4.8990)
    assert s.baseline_bias.to("mm").magnitude == approx(10.0)
    p = Scenarios.VisualServoDelayPhaseMargin.result
    assert p.delay_ceiling.to("ms").magnitude == approx(78.540) and p.share_used == approx(0.76394)
    assert p.crossover_frequency.to("Hz").magnitude == approx(1.5915)
    k = Scenarios.ClearanceInsetConformalRank.result
    assert (k.min_count, k.rank, k.rank_from_top) == (99, 991, 10)
    assert Scenarios.DenseFogVisibility.result.visibility_range.to("m").magnitude == approx(59.915)
    assert Scenarios.AluminumLinkWarming.result.expansion.to("mm").magnitude == approx(0.231)


def test_gpt2_xl_training_run_and_step_on_one_h100():
    r = Scenarios.Gpt2XlRunOnOneH100
    assert r.training_time.to("day").magnitude == approx(35.108)
    assert r.step_operations.to("TFLOP").magnitude == approx(294.912)
    assert r.step_time.to("s").magnitude == approx(0.59638)
    assert r.steps == approx(5.0863e6)
    assert r.energy.to("kWh").magnitude == approx(589.82) and r.step_energy.to("J").magnitude == approx(417.47)
    assert r.rental_cost.to("USD").magnitude == approx(3370.4)
    assert r.training_tokens.to("count").magnitude == approx(1.6667e11)
    assert Scenarios.Gpt2XlRunOnOneH100HalfUtilization.step_time.to("s").magnitude == approx(1.1928)
    assert Scenarios.Gpt2XlRunOnOneDgxH100.training_time.to("day").magnitude == approx(4.3886)
    a2 = Scenarios.Gpt2XlBatch32ActivationsAt2048
    assert a2.per_token_activations.to("GB").magnitude == approx(171.13)
    assert a2.score_activations.to("GB").magnitude == approx(805.31)
    b = Scenarios.Gpt2XlBatch32Block
    assert b.layer_forward_flops.to("GFLOP").magnitude == approx(2228.0)
    assert b.training_step_flops.to("TFLOP").magnitude == approx(320.83)
    assert b.six_p_step_flops.to("TFLOP").magnitude == approx(294.91)
    assert b.attention_step_flops.to("TFLOP").magnitude == approx(30.924)


def test_gpt2_xl_training_memory():
    a = Scenarios.Gpt2XlBatch32Activations
    assert a.activations("none").to("GB").magnitude == approx(286.89)
    assert a.per_token_activations.to("GB").magnitude == approx(85.564)
    assert a.score_activations.to("GB").magnitude == approx(201.33)
    assert a.projection_activations_per_layer.to("MB").magnitude == approx(419.43)
    assert a.ffn_activations_per_layer.to("MB").magnitude == approx(419.43)
    assert a.line_items_per_layer.to("MB").magnitude == approx(5043.2)
    assert a.line_items_total.to("GB").magnitude == approx(242.07)
    m = Scenarios.Gpt2MixedPrecisionAdamOnH100
    assert m.total.to("GB").magnitude == approx(24.0) and m.master_weights.to("GB").magnitude == approx(6.0)
    assert m.bytes_per_trained_parameter.to("byte").magnitude == 16
    assert m.training_total.to("GB").magnitude == approx(310.89)
    assert m.capacity_share == approx(0.27940) and m.training_capacity_multiple == approx(3.6192)
    assert m.checkpoint.to("GB").magnitude == approx(21.0)
    f = Scenarios.Gpt2Fp32AdamBatch32OnH100
    assert f.training_total.to("GB").magnitude == approx(597.78) and f.training_capacity_multiple == approx(6.9591)
    u = Scenarios.Gpt2Fp32AdamOnH100
    assert u.update_bytes_per_parameter.to("byte").magnitude == 28
    assert u.update_traffic.to("GB").magnitude == approx(42.0) and u.update_time.to("ms").magnitude == approx(12.537)
    assert Scenarios.Llama2_7BMixedPrecisionAdam.total.to("GB").magnitude == approx(112.0)
    p = Scenarios.Palm540BMixedPrecisionAdam
    assert p.total.to("TB").magnitude == approx(8.64) and p.checkpoint.to("TB").magnitude == approx(7.56)
    assert p.capacity_share == approx(100.58)


def test_llama3_8b_fine_tuning_scope_on_one_h100():
    full = Scenarios.Llama3_8BFullFineTuneOnH100
    assert full.total.to("GB").magnitude == approx(128.48)
    assert full.activations.to("GB").magnitude == approx(36.507)
    assert full.training_total.to("GB").magnitude == approx(164.99) and not full.fits
    lora = Scenarios.Llama3_8BLoraOnH100
    assert lora.adapter_parameters == 6_815_744 and lora.adapter_fraction == approx(8.4879e-4)
    assert lora.adapter_state.to("GB").magnitude == approx(0.10905)
    assert lora.training_total.to("GB").magnitude == approx(52.676)


def test_training_input_and_loader_stalls():
    f = Scenarios.ResNet50H100InputFeed
    assert f.sustained_rate.to("count/s").magnitude == approx(10143.6)
    assert f.host_supply.to("count/s").magnitude == approx(5250.0) and f.busy_fraction == approx(0.51757)
    g = Scenarios.GilThreadedLoaderStall.result
    assert g.sequential_step.to("ms").magnitude == approx(262.0) and 1 - g.sequential_stall == approx(0.045802)
    assert Scenarios.GilThreadedLoaderFasterAccelerator.result.sequential_step.to("ms").magnitude == approx(256.0)


def test_llama31_405b_model_card_run():
    c = Scenarios.Llama31_405BModelCardRun
    assert c.training_operations.to("flop").magnitude == approx(3.7908e25)
    assert c.achieved_rate.to("TFLOP/s").magnitude == approx(341.44)
    assert c.energy.to("GWh").magnitude == approx(21.588)
def test_responsible_ai_group_metrics():
    from mlsysim import Literature
    loan = Scenarios.LoanScreeningTwoGroups
    assert loan.positive_rate("A") == approx(0.70) and loan.positive_rate("B") == approx(0.40)
    assert loan.tpr("A") == approx(40 / 45) and loan.tpr("B") == approx(0.60)
    assert loan.fpr("A") == approx(30 / 55) and loan.fpr("B") == approx(0.20)
    assert loan.ppv("A") == approx(40 / 70) and loan.ppv("B") == approx(0.75)
    assert loan.disparate_impact == approx(4 / 7) and loan.gap("positive_rate") == approx(0.30)
    assert loan.average_odds_difference == approx(0.5 * (40 / 45 - 0.6 + 30 / 55 - 0.2))
    fraud = Scenarios.FraudFlagTwoGroups
    assert fraud.positive_rate("A") == approx(0.5) and fraud.disparate_impact == approx(0.5)
    c = Scenarios.CompasTwoYearRecidivism
    assert c.fpr("African-American") == approx(641 / 1514) and c.fpr("Caucasian") == approx(282 / 1281)
    assert round(100 * c.base_rate("African-American"), 1) == 52.3
    assert Literature.Fairness.FourFifthsRuleRatio == approx(0.8)


def test_responsible_ai_gap_evidence():
    t = Scenarios.LoanApprovalGapTest
    assert t.pooled_rate == approx(0.55) and t.z_critical == pytest.approx(1.95996, abs=1e-3)
    assert t.z == pytest.approx(4.2640, abs=1e-3) and t.z_at(10) == pytest.approx(1.3484, abs=1e-3)
    lo, hi = t.interval
    assert (round(lo, 2), round(hi, 2)) == (0.17, 0.43)
    slo, shi = t.interval_at(10)
    assert (round(slo, 2), round(shi, 2)) == (-0.12, 0.72)
    assert t.n_at_critical == pytest.approx(21.128, abs=1e-3)
    d = Scenarios.GroupThresholdSketch
    assert d.tpr("A", 0.75) == 0.75 and d.tpr("B", 0.75) == 0.25 and d.tpr("B", 0.45) == 0.75
    assert d.fpr("B", 0.75) == 0 and d.fpr("B", 0.45) == 0.25
    tax = Scenarios.CreditParityAccuracyCost
    assert tax.accuracy_cost == approx(4.0) and tax.relative_cost == approx(4 / 85)
    m = Scenarios.FairnessMonitor10K
    assert m.decisions_per_day == approx(864e6) and m.records_per_day == approx(8.64e6)
    assert m.log_per_day.to("GB").magnitude == approx(1.728) and m.false_alerts_per_interval == approx(7.5)
    assert m.false_alerts_bonferroni == approx(0.05)
    o = Scenarios.OfqualStandardization2020
    assert o.split("B") == (1, 59, 37, 3) and o.split("U") == (20, 79, 0, 0)
    assert o.entries_lowered_pct == pytest.approx(38.949, abs=1e-3)
    assert o.candidate_rows_sum == 246_100 and o.total_candidates == 246_110
    assert round(100 * o.at_least_one_lowered_share, 1) == 68.1


def test_responsible_ai_safeguard_costs():
    s = Scenarios.ExactShapEnumeration
    assert s.subsets(20) == 1_048_576 and s.time(20).to("s").magnitude == approx(1.048576)
    r = Scenarios.InlineShapRetrofit
    assert r.latency_low.to("ms").magnitude == approx(150) and r.latency_high.to("ms").magnitude == approx(300)
    assert r.capacity_low == approx(1.5) and r.capacity_high == approx(3.0)
    x = Scenarios.ImageExplanationLog
    assert x.monthly_low.to("TB").magnitude == approx(30) and x.monthly_high.to("TB").magnitude == approx(300)
    u = Scenarios.Gpt3ShardedUnlearning
    assert u.full_retrain_time.to("day").magnitude == pytest.approx(35.548, abs=1e-3) and u.planning_days == 36
    assert u.shard_cost.to("USD").magnitude == approx(46_000) and u.shard_time.to("hour").magnitude == approx(8.64)
    assert u.p_shard_untouched == approx(4.3171e-5, rel=1e-3) and u.shards_touched_per_day == pytest.approx(99.9957, abs=1e-3)
    p = Scenarios.Gpt3PrivacyPlan
    assert p.cost_strong.to("USD").magnitude == approx(13.8e6)
    b = Scenarios.RewardModelLabelBudget
    assert b.budget_low.to("USD").magnitude == approx(25_000) and b.spread == approx(100)
    a = Scenarios.RadiologyAutomationBias
    assert a.accepted_misses() == approx(0.035) and a.team_sensitivity_bound == approx(0.965)
    assert a.accepted_misses(a.acceptance_high) == approx(0.04) and a.gain_over_ai == approx(0.015)
    assert Scenarios.SafetyComplianceAtBillionRequests.daily_failures == approx(1e6)


def test_responsible_ai_literature_series():
    from mlsysim import Literature
    dp = Literature.Series.AbadiDpSgdAccuracy.entries
    cifar8 = next(r for r in dp if r["dataset"] == "CIFAR-10" and r["epsilon"] == 8)
    assert cifar8["accuracy_pct"] == Literature.DifferentialPrivacy.Cifar10AbadiAccuracyAtEps8Pct
    assert len(Literature.Series.OfqualGradeMatrix2020.entries) == 7
    assert Literature.Fairness.OfqualReversalDayAugust2020 - Literature.Fairness.OfqualResultsDayAugust2020 == 4
def test_durable_execution_log_and_snapshots():
    g = Scenarios.IntentQuorumGroupCommit.result
    assert g.work_per_record.to("ms").magnitude == approx(0.25)
    assert g.first_wait.to("ms").magnitude == approx(6.0) and g.last_wait.to("ms").magnitude == approx(4.0)
    s = Scenarios.GrowingTrajectorySnapshots.result
    assert s.checkpoints == 100 and s.ratio == approx(50.5)
    assert s.full_bytes.to("GB").magnitude == approx(3.232) and s.delta_bytes.to("MB").magnitude == approx(64.0)


def test_durable_execution_checkpoint_cadence():
    t = Scenarios.TwoCrashCheckpointTrace.result
    assert [t.total_time[k].to("s").magnitude for k in (10, 25, 50)] == [approx(8.0), approx(6.0), approx(10.0)]
    assert t.lost_turns == {10: (7, 3), 25: (12, 8), 50: (37, 33)} and t.best_stride == 25
    assert t.every_turn_time.to("s").magnitude == approx(52.0) and t.crash_rate == approx(0.02)
    assert t.stride_logged == approx(22.361) and t.stride_reexecuted == approx(1.5811)
    assert t.stride_gain == approx(14.142)
    over = Scenarios.TwoCrashCheckpointTrace.stride_overhead(10, Scenarios.TwoCrashCheckpointTrace.rebuild_per_turn)
    assert over.to("s").magnitude == approx(0.06)
    y = Scenarios.ReexecutedWorkCheckpointInterval.result
    assert y.interval.to("s").magnitude == approx(48.990) and y.waste == approx(0.081650)


def test_durable_execution_resume_replay_retention():
    assert Scenarios.ResumedContextColdPrefill.result.prefill_time.to("s").magnitude == approx(2.3380)
    assert Scenarios.GreedyRerunReproduction.result.success[30] == approx(0.73970)
    r = Scenarios.RepairTrajectoryRetention.result
    assert r.log_bytes.to("MB").magnitude == approx(3.96) and r.total_bytes.to("MB").magnitude == approx(53.96)
    assert r.cost.to("USD").magnitude == approx(0.014893) and r.attempt_share == approx(0.0070919)
    assert r.log_share == approx(0.00052046) and r.workspace_share == approx(0.92661)


def test_durable_execution_flat_optimum():
    y = Scenarios.ReexecutedWorkCheckpointInterval
    assert y.miss_factors == [2, 10]
    assert y.miss_penalty(2) == approx(0.25) and y.miss_penalty(10) == approx(4.05)
def test_failure_recovery_repair_and_pivot():
    r = Scenarios.TestConfigRepairVsRebuild.result
    assert r.fallback_time.to("s").magnitude == approx(86.6) and r.fallback_cost.to("USD").magnitude == approx(0.162)
    assert r.repair_time.to("s").magnitude == approx(4.09) and r.repair_cost.to("USD").magnitude == approx(0.00987)
    assert r.break_even_time == approx(4.09 / 86.6) and r.break_even_cost == approx(0.00987 / 0.162)
    assert r.expected_time.to("s").magnitude == approx(47.39) and r.time_saved_fraction == approx(0.452771, rel=1e-5)
    assert r.one_in == 16
    p = Scenarios.RemediationCanaryPivotLoss.result
    assert p.loss_without_check.to("USD").magnitude == approx(480) and p.loss_with_check.to("USD").magnitude == approx(134.4)
    assert p.loss_caught.to("USD").magnitude == approx(38.4) and p.loss_missed.to("USD").magnitude == approx(96)
    assert p.reduction == approx(480 / 134.4) and p.missed_share == approx(96 / 134.4)
    assert p.loss_per_catch_step.to("USD").magnitude == approx(43.2)
    assert p.loss_with_check_raised.to("USD").magnitude == approx(91.2)
    assert p.attempt_cost.to("USD").magnitude == approx(20.45) and p.loss_vs_attempt == approx(134.4 / 20.45)


def test_failure_recovery_loop_and_containment():
    from mlsysim.physics import calc_cleanup_critical_path, calc_layered_retry_attempts
    lp = Scenarios.RepairLoopToTurnCeiling.result
    assert (lp.ceiling_input, lp.stop_input, lp.peak_request) == (1_397_500, 10_600, 55_200)
    assert lp.ceiling_cost.to("USD").magnitude == approx(4.4175) and lp.stop_cost.to("USD").magnitude == approx(0.0498)
    assert lp.cost_ratio == approx(88.7048, rel=1e-5) and lp.fits_context
    assert lp.input_by_turn[50] == lp.ceiling_input and lp.input_by_turn[4] == lp.stop_input
    assert lp.half_ratio == approx(1_397_500 / 355_000)
    t = Scenarios.ThreeLayerRetryAmplification.result
    assert t.by_layer == [1, 3, 9, 27] and t.coordinated == 3 and calc_layered_retry_attempts(2, 3) == 8
    b = Scenarios.BlockingCallBulkhead.result
    assert (b.uncapped_free, b.queued, b.rejected, b.protected_workers) == (44, 32, 160, 236)
    j = Scenarios.RetryCohortJitter.result
    assert j.synchronized_rate.to("1/s").magnitude == approx(64_000) and j.jittered_mean_rate.to("1/s").magnitude == approx(64)
    assert j.spread_ratio == approx(1000) and j.expected_in_window == approx(0.128)
    c = Scenarios.TestEnvironmentCleanupGraph.result
    assert c.stack_time.to("s").magnitude == 105 and c.graph_time.to("s").magnitude == 75
    assert c.blocked == ["release", "revoke"] and c.settleable == ["preserve", "scratch"]
    assert calc_cleanup_critical_path({"a": 1, "b": 2, "c": 3}, {"c": ["a", "b"]}) == 5
    w = Scenarios.TestServiceBreakerWindow.result
    assert (w.outcomes, w.classified_failures, w.naive_failures) == (50, 20, 35)
    assert not w.classified_trips and w.naive_trips
    m = Scenarios.RetrySustainedMetastableOutages.result
    assert (m.incidents, m.retry_sustained, m.over_hours) == (21, 12, 3) and m.without_duration == ["SPF1"]
    assert m.with_duration[0] == ("IBM1", 73.53)
def test_spatial_memory_estimation():
    f = Scenarios.WarehouseRackEndRangeFusion.result
    assert f.process_sigma.to("mm").magnitude == approx(5.0) and f.predicted_sigma.to("mm").magnitude == approx(20.6155)
    assert f.first_gain == approx(0.320755) and f.first_sigma.to("mm").magnitude == approx(16.9906)
    assert f.second_gain == approx(0.952809) and f.fused_sigma.to("mm").magnitude == approx(3.69095)
    assert f.misdated_shift.to("mm").magnitude == approx(38.0171) and f.misdated_shift_sigmas == approx(10.3001)
    assert f.nis_expected == approx(6.25456)
    assert f.comparison_gain == approx(0.834645) and f.comparison_sigma.to("mm").magnitude == approx(6.90902)
    c = Scenarios.RackEndMisdatedUpdateConsistency.result
    assert (c.band_low, c.band_high) == (approx(0.647147), approx(1.428404))
    assert c.gates == [approx(3.841459), approx(6.634897)] and c.detection == [approx(0.630186), approx(0.388379)]
    assert Scenarios.WarehouseObservationReplayBuffer.result.slots == 52
    i = Scenarios.WarehouseImuOneSecondCoast.result
    assert i.heading_noise.to("deg").magnitude == approx(0.014) and i.lateral_bias.to("mm").magnitude == approx(19.1986)
    assert i.position_noise.to("mm").magnitude == approx(0.905900) and i.position_bias.to("mm").magnitude == approx(98.0665)
    o = Scenarios.WarehouseUmbmarkOdometryDrift.result
    assert o.rate_uncalibrated == approx(0.0198125) and o.distance_calibrated.to("m").magnitude == approx(38.0952)
    assert o.segment_error_uncalibrated.to("mm").magnitude == approx(594.375) and o.speed_ratio == approx(6.5)
    k = Scenarios.WarehouseKittiOdometryDrift.result
    assert (k.count, k.best_pct, k.laser_median_pct, k.count_below_threshold) == (127, 0.53, 0.68, 0)
    assert k.distance_best.to("m").magnitude == approx(9.43396)
    t = Scenarios.WarehouseTrackConfirmationDelay.result
    assert t.wait.to("ms").magnitude == approx(33.3333) and t.travel.to("mm").magnitude == approx(50.0)


def test_spatial_memory_belief_expiry():
    b = Scenarios.LatchSlotFingerBelief.result
    assert b.horizon.to("ms").magnitude == approx(65.9412) and b.constant_velocity_horizon.to("ms").magnitude == approx(75.0)
    assert b.error_at_constant_velocity_horizon.to("mm").magnitude == approx(1.140625) and b.overshoot == approx(0.140625)
    assert Scenarios.LatchSlotFingerBeliefTightSlot.result.horizon.to("ms").magnitude == approx(16.1250)
    s = Scenarios.LatchSlotStaleWristTransform.result
    assert s.drift_term.to("mm").magnitude == approx(0.2) and s.lever_term.to("mm").magnitude == approx(0.75)
    assert s.total.to("mm").magnitude == approx(1.05) and s.same_epoch_error.to("mm").magnitude == approx(0.306944)
    m = Scenarios.ConveyorMugHandleBelief.result
    assert m.horizon.to("ms").magnitude == approx(60.0) and m.error_at_use.to("mm").magnitude == approx(10.04)
    assert m.time_left.to("ms").magnitude == approx(24.8) and m.latest_use_age.to("ms").magnitude == approx(59.8)
    assert Scenarios.CoworkerBesideArmBelief.result.horizon.to("ms").magnitude == approx(84.375)
    assert Scenarios.CoworkerBesideArmBeliefTightTolerance.result.horizon.to("ms").magnitude == approx(53.125)
    r = Scenarios.WarehouseRackEndFreeSpace.result
    assert r.horizon_ahead.to("ms").magnitude == approx(593.75) and r.horizon_abreast.to("ms").magnitude == 0.0
    assert r.time_left_at_dispatch.to("ms").magnitude == approx(542.15) and r.frames_bridged == approx(32.529)
    assert r.depth_eroded_by_age.to("mm").magnitude == approx(82.56) and r.walker_intrusion.to("m").magnitude == approx(1.41376)
    assert r.timeout_refused.to("ms").magnitude == approx(493.75)
    assert Scenarios.ArmSensorEpochGap.result.travel.to("mm").magnitude == approx(4.25)
    k = Scenarios.WarehouseSkewedRecordPair.result
    assert k.travel.to("mm").magnitude == approx(62.0) and k.slack.to("mm").magnitude == approx(50.0)
    w = Scenarios.WristForceBiasDrift.result
    assert w.shift.to("N").magnitude == approx(3.0) and w.lost_share == approx(0.6)
    z = Scenarios.ConveyorMugFrozenTransform.result
    assert z.error.to("mm").magnitude == approx(20.0) and z.passes_freshness and z.exceeds_tolerance


def test_spatial_memory_maps():
    q = Scenarios.WarehousePermissionClearanceQuery.result
    assert q.query_time.to("us").magnitude == approx(2.0) and q.reads_budget == approx(800.0)
    assert q.tick_spare_share == approx(0.9) and q.wcet_share == approx(0.4)
    assert [round(q.rows[p].occupancy_reads) for p in (2, 5, 10)] == [22449, 1437, 180]
    assert q.rows[5].octree_depth == 8 and round(q.rows[5].octree_reads) == 11494
    assert q.rows[5].esdf_bytes.to("kB").magnitude == approx(182.88) and q.rows[10].quantization.to("mm").magnitude == approx(86.6025)
    h = Scenarios.WorkcellHostMapSizing.result
    assert h.dense_bytes.to("GB").magnitude == approx(2.4) and h.hashed_bytes.to("MB").magnitude == approx(76.8)
    assert h.blocks == approx(18750) and h.ratio == approx(31.25)
    r = Scenarios.WarehouseDepthRaymarchTraffic.result
    assert r.traffic.to("GB/s").magnitude == approx(7.3728) and r.share == approx(0.0361412)
    assert Scenarios.WarehouseFoundationEncoderAtCameraRate.result.shortfall == approx(4.18605)
# --- Grounded intent: the intent lease and its gates (added 2026-10-20) -------

def test_conveyor_mug_intent_cadence():
    c = Scenarios.ConveyorMugIntentCadence.result
    assert c.margin.to("mm").magnitude == approx(12.0)
    assert c.untracked_horizon.to("ms").magnitude == approx(60.0)
    assert c.tracked_horizon.to("ms").magnitude == approx(219.089023)
    assert c.intent_period.to("ms").magnitude == approx(500.0)
    assert c.error_at_period.to("mm").magnitude == approx(103.0)
    assert c.excess_at_period.to("mm").magnitude == approx(88.0)
    assert c.renewal_age == c.intent_period + Scenarios.ConveyorMugIntentCadence.machine.intent_inference_latency
    assert c.renewal_speed_ceiling.to("mm/s").magnitude == approx(c.margin.magnitude / c.renewal_age.to("s").magnitude)
    assert c.stall_error_untracked.to("mm").magnitude == approx(43.0)
    assert c.stall_error_tracked.to("mm").magnitude == approx(13.0)
    assert c.stop_onset.to("ms").magnitude == approx(22.0)
    assert Scenarios.ConveyorMugSettlingDrift.result.untracked_horizon.to("ms").magnitude == approx(600.0)
    assert Scenarios.ConveyorMugSlowBelt.result.untracked_horizon.to("ms").magnitude == approx(240.0)


def test_conveyor_mug_reach_occupancy_and_lease_expiry():
    r = Scenarios.ConveyorMugDynamicReach.result
    assert r.transit_floor.to("ms").magnitude == approx(300.0)
    assert r.reach_untracked.to("mm").magnitude == approx(60.0)
    assert not r.fits and r.fresh_captures == 2
    o = Scenarios.AisleCoworkerForwardOccupancy.result
    assert o.growth_untracked.to("mm").magnitude == approx(96.0)
    assert o.growth_tracked.to("mm").magnitude == approx(350.542437)
    assert o.growth_forecast.to("m").magnitude == approx(3.2)
    assert o.aisle_span_horizon.to("ms").magnitude == approx(812.5)
    e = Scenarios.ConveyorMugIntentLeaseExpiry.result
    assert e.credible_deceleration.to("m/s**2").magnitude == approx(3.079710)
    assert e.stop_distance.to("mm").magnitude == approx(184.352941)
    assert e.stopping_deadline.to("ms").magnitude == approx(115.647059)
    assert (e.binding_untracked, e.binding_tracked) == ("evidence", "administrative")
    assert e.expiry_tracked.to("ms").magnitude == approx(100.0)


def test_referent_sets_confidence_region_and_latch_stall():
    s = Scenarios.MugJarReferentSet.result
    assert s.normalized == approx([0.563636, 0.436364], rel=1e-5)
    assert s.entropy == approx(0.685026, rel=1e-5) and s.rank == 181
    assert s.set_size == 2 and s.refuse
    low = Scenarios.MugJarReferentSetLowJar.result
    assert low.set_size == 1 and low.normalized[1] == approx(0.138889, rel=1e-5)
    alone = Scenarios.JarAloneReferentSet.result
    assert alone.normalized == [1.0] and alone.set_size == 1 and alone.entropy == 0.0
    g = Scenarios.DistantHandleConfidenceRegion.result
    assert g.quantile == approx(7.814728, rel=1e-6)
    assert [a.to("mm").magnitude for a in g.semi_axes] == approx([11.181934, 13.977417, 33.545802], rel=1e-6)
    assert g.axes_within == 2
    x = Scenarios.LatchUnbrakedStall
    assert x.force_at_brake_onset_guarded.to("N").magnitude == approx(39.0)
    assert x.compression_left_guarded.to("mm").magnitude == approx(0.1525)
    assert x.stall_travel.to("mm").magnitude == approx(6.0)
    assert x.stall_force.to("N").magnitude == approx(2400.0)
    assert x.stall_over_limit == approx(24.0)
    gs = Scenarios.ConveyorMugGoalSetOffsets
    assert gs.norms == approx([0.666667, 1.333333], rel=1e-5) and gs.inside == [True, False]

# --- Data selection (added 2026-10-20) ---

def test_gpt2_webtext_data_wall():
    g = Scenarios.Gpt2XlWebTextCorpus
    assert g.unique_tokens.magnitude == approx(10e9) and g.optimal_tokens.magnitude == approx(30e9)
    assert g.shortfall == approx(3.0) and g.unique_tokens_per_param == approx(6.6667)
    assert g.tokens_processed.magnitude == approx(166.67e9) and g.passes == approx(16.667)
    assert g.trained_tokens_per_param == approx(111.11)
    assert g.run_time.to("day").magnitude == approx(4.3886) and g.run_energy.to("kWh").magnitude == approx(589.82)
    n, d = g.compute_optimal
    assert n == approx(3.5355e9) and d == approx(70.711e9)
    assert Scenarios.Gpt2XlRunOnOneH100.training_time.to("day").magnitude == approx(35.108)
    assert Scenarios.ChinchillaComputeDoubling.growth == approx(0.41421)


def test_gpt2_deduplication_and_minhash_banding():
    d = Scenarios.Gpt2WebTextNearDuplicates
    assert d.time_saved("C4").to("hour").magnitude == approx(3.2020)
    assert d.time_saved("RealNews").to("hour").magnitude == approx(14.356)
    assert d.energy_saved("C4").to("kWh").magnitude == approx(17.931)
    assert d.energy_saved("RealNews").to("kWh").magnitude == approx(80.394)
    assert d.signature_bytes.to("GB").magnitude == approx(4.096) and d.read_time.to("s").magnitude == approx(5.7143)
    assert d.all_pairs == approx(3.2e13)
    b = Scenarios.FineWebMinHashBanding
    assert b.hashes == 112 and b.threshold == approx(0.71901)
    assert b.candidate_probability(0.5) == approx(0.053320) and b.candidate_probability(0.8) == approx(0.92355)


def test_resnet50_coreset_selection_and_data_path():
    i = Scenarios.ResNet50HalfImageNetIcr
    assert i.epoch_flops.to("flop").magnitude == approx(3.1517e16) and i.icr_ratio == approx(1.8)
    c = Scenarios.ResNet50TenPercentCoresetH100
    assert c.utilization == approx(0.25231) and c.full_time.to("minute").magnitude == approx(210.51)
    assert c.score_time().to("s").magnitude == approx(42.101) and c.score_time(c.proxy).to("s").magnitude == approx(18.483)
    assert c.saving_fraction() == approx(0.89667) and c.saving_fraction(c.proxy) == approx(0.89854)
    assert c.rescoring_time.to("minute").magnitude == approx(70.168)
    assert c.rescoring_saving_fraction == approx(0.56667) and c.rescoring_over_training == approx(3.3333)
    assert c.run_time.to("hour").magnitude == approx(3.5084)
    e = Scenarios.ResNet50HalfCoresetEpochH100
    assert e.epoch_time.to("s").magnitude == approx(126.30) and e.epoch_energy.to("kJ").magnitude == approx(88.412)
    assert e.sample_rate == approx(10144) and e.storage_time.to("s").magnitude == approx(21.168)
    assert e.link_bytes.to("GB").magnitude == approx(192.85) and e.link_time.to("s").magnitude == approx(3.0133)
    assert e.host_rate.to("count/second").magnitude == approx(5250) and e.host_epoch_time.to("s").magnitude == approx(244.03)
    assert e.run_saving_time.to("hour").magnitude == approx(1.5788) and e.run_saving_energy.to("MJ").magnitude == approx(3.9785)
    assert e.saving_over_score_energy == approx(135.0) and e.materialized_bytes.to("GB").magnitude == approx(74.088)
    h = Scenarios.ResNet50EchoingH100
    assert h.ratio == approx(1.9238) and h.utilization == approx(0.51980)
    assert h.run_time.to("hour").magnitude == approx(6.1008) and h.echoed_run_time.to("hour").magnitude == approx(3.1712)


def test_selection_budgets_coverage_and_curriculum():
    p = Scenarios.ProductionModelScoringCheck
    assert not p.pays and p.total_time.to("hour").magnitude == approx(13) and p.breakeven_selection.to("hour").magnitude == approx(9)
    b = Scenarios.ExpensiveSelectionPitfall
    assert not b.pays and b.overhead_ratio == approx(5.0)
    assert Scenarios.ProxySelectionPitfall.selection_share == approx(0.0625)
    v = Scenarios.RareClassTenPercentCoreset
    assert v.rare == approx(1000) and v.expected_kept == approx(100) and v.shortfall == approx(50)
    k = Scenarios.Cifar10LinearCurriculum
    assert k.epoch_reduction == approx(0.23333) and k.warmup_samples == approx(5.5) and k.warmup_share == approx(0.55)
    assert Scenarios.MentorNetCurriculum.epoch_reduction == approx(0.22222)


def test_labeling_and_data_cost():
    a = Scenarios.ActiveLearningTenRounds
    assert a.active_labels == 2000 and a.random_cost.magnitude == approx(50_000)
    assert a.active_cost.magnitude == approx(20_500) and a.roi == approx(1.4390)
    m = Scenarios.MedicalScanActiveLearning
    assert m.full_cost.magnitude == approx(5e6) and m.random_labels.magnitude == approx(100_000)
    assert m.label_ratio == approx(2.0) and m.saving.magnitude == approx(250_000) and m.over_budget_ratio == approx(10)
    s = Scenarios.FixMatchCifar10Cost
    assert s.supervised_total.magnitude == approx(4050) and s.ssl_total.magnitude == approx(500)
    assert s.cost_reduction == approx(8.1) and s.compute_multiple == approx(5) and s.label_reduction == approx(16)
    f = Scenarios.FoundationModelTenTasks
    assert f.crossover_tasks == approx(10.526) and f.marginal_compute_reduction == approx(20)
    assert f.foundation_total.to("hour").magnitude == approx(10_500)
    c = Scenarios.ImageNetDataCostBreakdown
    assert c.label_cost.to("USD").magnitude == approx(64_058.35) and c.storage_cost.to("USD").magnitude == approx(35.562)
    assert c.accelerator_hours.to("hour").magnitude == approx(105.25) and c.compute_cost.to("USD").magnitude == approx(421.01)
    assert c.total.to("USD").magnitude == approx(114_514.9) and c.compute_share.to("").magnitude == approx(0.0036765)
    r = Scenarios.MinHashPipelineAmortization
    assert r.investment.magnitude == approx(55_000) and r.breakeven_runs == approx(5.5)
    assert r.net_return(1) == approx(-0.81818) and r.savings_multiple(50) == approx(9.0909)
    assert Scenarios.TenfoldDataFourPoints.scale == approx(10)
# --- Sustainability (added 2026-10-20) ---------------------------------------

def test_llama31_405b_training_footprint():
    r = Scenarios.Llama31_405BTrainingFootprint.result
    assert r.power.accelerator_power.to("MW").magnitude == approx(11.4688)
    assert r.power.it_power.to("MW").magnitude == approx(15.7696)
    assert r.power.facility_power.to("MW").magnitude == approx(17.662)
    assert r.accelerator_energy.to("MWh").magnitude == approx(21588.0)
    assert r.it_energy.to("MWh").magnitude == approx(29683.5)
    assert r.facility_energy.to("MWh").magnitude == approx(33245.52)
    assert r.duration.to("hour").magnitude == approx(1882.32)
    assert r.carbon["US average"].to("metric_ton").magnitude == approx(14262.3)
    assert r.carbon["Norway"].to("metric_ton").magnitude == approx(332.455)
    assert r.carbon["Poland"].to("metric_ton").magnitude == approx(27261.3)
    assert r.embodied_manufactured.to("metric_ton").magnitude == approx(2686.98)
    assert r.embodied_allocated.to("metric_ton").magnitude == approx(192.457)
    assert r.embodied_share["Quebec"] == approx(0.22447)
    assert r.water["Google fleet"].to("liter").magnitude == approx(3.4136e7)
    assert r.water["evaporative"].to("liter").magnitude == approx(5.3430e7)


def test_facility_power_chain_and_rollback():
    m = Scenarios.Mega100KFacilityPower.result
    assert m.accelerator_power.to("MW").magnitude == approx(70.0)
    assert m.it_power.to("MW").magnitude == approx(96.25) and m.facility_power.to("MW").magnitude == approx(107.8)
    assert Scenarios.Reference25KFacilityPower.result.accelerator_power.to("MW").magnitude == approx(17.5)
    assert Scenarios.FeedSizing14KGpus.result.it_power.to("MW").magnitude == approx(13.475)
    e = Scenarios.Training10KCoolingRollback.result.energy
    assert e["low"].to("MWh").magnitude == approx(3.5) and e["high"].to("MWh").magnitude == approx(7.0)


def test_pue_retrofit_and_site_rates():
    r = Scenarios.PueRetrofit2MW.result
    assert r.facility_power_before.to("MW").magnitude == approx(3.16)
    assert r.saved_power.to("MW").magnitude == approx(0.96)
    assert r.saved_energy.to("MWh").magnitude == approx(8409.6)
    assert r.saved_cost.to("USD").magnitude == approx(588672.0)
    rates = Scenarios.PueVersusGridSites.result.rates
    assert rates["A"].to("metric_ton/hour").magnitude == approx(4.86)
    assert rates["B"].to("metric_ton/hour").magnitude == approx(0.675)


def test_grid_placement():
    q = Scenarios.QuebecVersusPolandPlacement.result
    assert q.emissions["Quebec"].to("metric_ton").magnitude == approx(200.0)
    assert q.emissions["Poland"].to("metric_ton").magnitude == approx(8200.0)
    assert q.emissions_after_cut["Poland"].to("metric_ton").magnitude == approx(6970.0)
    assert q.spread == approx(41.0)
    g = Scenarios.RegionalGridPlacement.result
    assert (g.lowest, g.highest) == ("Norway", "Poland") and g.spread == approx(82.0)


def test_embodied_payback_and_life_extension():
    r = Scenarios.H100ReplacementPayback.result
    assert r.saved_energy_per_day.to("kWh").magnitude == approx(8.4)
    assert r.avoided_per_day["US average"].to("kg").magnitude == approx(3.6036)
    assert r.payback["US average"].to("week").magnitude == approx(6.5014)
    assert r.payback["Quebec"].to("day").magnitude == approx(976.19)
    assert Scenarios.AcceleratorLifeExtension.result.annual_charge_reduction == approx(0.4)


def test_power_cap_and_serving_crossover():
    c = Scenarios.FeedCappedTrainingRun.result
    assert c.concurrency == approx(2 / 3) and c.stretched_duration.to("day").magnitude == approx(4.5)
    s = Scenarios.GeminiPromptsAgainstLlama31Training.result
    assert s.daily_energy.to("MWh").magnitude == approx(240.0)
    assert s.crossover_days == approx(138.52)


def test_edge_energy_models():
    c = Scenarios.EdgeFirstImageCascade.result
    assert c.mean_energy.to("mJ").magnitude == approx(6.5) and c.saving == approx(0.89167)
    w = Scenarios.WakeWordHierarchyPower.result
    assert w.average_power.to("uW").magnitude == approx(15.5) and w.reduction == approx(645.16)
    d = Scenarios.BatchOneDecodeWeightEnergy.result
    assert d.movement_energy.to("pJ").magnitude == approx(320.0) and d.compute_energy.to("pJ").magnitude == approx(20.0)
    assert d.movement_share == approx(0.94118) and d.crossover_intensity == approx(16.0)
    assert d.operand_energy.to("pJ").magnitude == approx(640.0) and d.operand_over_flop == approx(64.0)


def test_demand_rebound_sweep_and_scaling():
    assert Scenarios.HalvedQueryEnergyRebound.result.net_factor == approx(2.0)
    assert Scenarios.QuarterTaskEnergyRebound.result.net_factor == approx(5.0)
    assert Scenarios.TokenPriceRebound.result.net_factor == approx(3.3333)
    assert Scenarios.CandidateSweep50.result.final_run_equivalents == approx(5.0)
    assert Scenarios.KaplanTenPercentLowerLoss.reduction_multiple == approx(8.2253)


def test_eu_week_fossil_share_and_water():
    e = Scenarios.EuropeWeek47FossilShare
    r = e.result
    assert min(e.shares) == approx(20.0006) and max(e.shares) == approx(37.8716)
    assert r.low == approx(22.7386) and r.high == approx(35.7098) and r.ratio == approx(1.57045)
    w = Scenarios.ReferenceItLoadAnnualWater.result.water["evaporative"]
    assert w.to("liter").magnitude == approx(1.5768e8)


def test_carbon_priced_region_placement():
    r = Scenarios.CarbonPricedRegionPlacement.result
    assert r.best_region == "Quebec" and r.best_region_untaxed == "Quebec"
    assert r.objective_value == approx(25336888.4)
    assert r.objective["US_Avg"] - r.objective["Quebec"] == approx(531280.0)


def test_cluster_expansion_doubling():
    e = Scenarios.Gpt3ClusterDoubling1K
    assert e.expanded_devices == 2048
    assert e.mtbf.to("hour").magnitude == approx(48.828125)
    assert e.expanded_mtbf.to("hour").magnitude == approx(24.4140625)
    assert e.checkpoint.t_write.to("second").magnitude == approx(24.5)
    assert e.overhead == approx(0.016696) and e.expanded_overhead == approx(0.023612)
    assert e.overhead_growth == approx(2 ** 0.5)
    assert e.breakeven_speedup == approx(1.007083)
    assert e.accelerator_hours_change(e.factor) == approx(0.007083)
    assert e.superlinear_threshold == approx(2.014166)
    assert e.interval.to("minute").magnitude == approx(48.914) and e.expanded_interval.to("minute").magnitude == approx(34.587)
    assert e.run_time_change(176900 / 95100) == approx(-0.45861, rel=1e-4)


def test_fleet_scale_recall_scenarios():
    s = Scenarios.SlowWorker8K
    assert s.throughput == approx(0.8) and (s.slow_step / s.step_time).to("").magnitude == approx(1.25)
    assert Scenarios.OneInAMillionAtMillionDaily.events_per_second * 86400 == approx(1.0)
    k = Scenarios.KaplanComputeDoubling
    assert k.multiple_factor == approx(0.965936) and k.multiple_reduction == approx(0.034064)
    assert Scenarios.KaplanSixteenfoldCompute.multiple_reduction == approx(0.129449)
    d = Scenarios.DeepSeekV3WeightTraffic
    assert d.resident_bytes.to("GB").magnitude == approx(1342) and d.active_bytes.to("GB").magnitude == approx(74)
    assert d.dense_bytes.to("GB").magnitude == approx(800) and d.resident_to_active == approx(18.135135)
    assert Scenarios.OpticalLinkEnergyShare.saving == approx(0.05)
    assert Scenarios.OpticalLinkEnergyShare.ceiling == approx(0.10)
    assert Scenarios.OpticalLinkEnergyDoubleShare.saving == approx(0.10)


def test_megascale_and_tail_at_scale_series():
    from mlsysim import Literature
    rows = {r["gpus"]: r for r in Literature.Series.MegaScaleStrongScaling.entries}
    assert rows[1024]["megascale_tokens_s"] / rows[512]["megascale_tokens_s"] == approx(1.86015, rel=1e-4)
    tail = {(r["leaves_awaited"], r["percentile"]): r["latency_ms"]
            for r in Literature.Series.TailAtScaleLeafLatency.entries}
    assert tail[("all leaves", 99.0)] / tail[("one random leaf", 99.0)] == approx(14.0)
def test_warehouse_finished_stopping_budget():
    d = Scenarios.WarehouseFinishedStopAtDriveLimit.result
    assert d.jerk.to("m/s**3").magnitude == approx(64.0 / 9.0)
    assert d.jerk_ramp_travel.to("mm").magnitude == approx(210.9375)
    assert d.required_distance.to("mm").magnitude == approx(1163.8375)
    assert d.spare.to("mm").magnitude == approx(-63.8375)
    assert d.stop_time.to("ms").magnitude == approx(1031.25)
    assert d.speed_ceiling.to("m/s").magnitude == approx(1.43669, rel=1e-5)
    assert d.ramp_delay.to("ms").magnitude == approx(140.625)
    a = Scenarios.WarehouseFinishedStopAtAisleSpeed.result
    assert a.required_distance.to("mm").magnitude == approx(968.9925)
    assert a.spare.to("mm").magnitude == approx(131.0075)
    assert a.delay_spare.to("ms").magnitude == approx(100.775)
    assert a.stop_time.to("ms").magnitude == approx(931.25)
    w = Scenarios.WarehouseFinishedStopWithDispatch.result
    assert w.spare.to("mm").magnitude == approx(48.7036)
    assert w.speed_ceiling.to("m/s").magnitude == approx(1.34872, rel=1e-5)
    assert Scenarios.WarehouseFinishedStopInspectedFloor.result.speed_ceiling.to("m/s").magnitude == approx(1.23087, rel=1e-5)
    assert Scenarios.WarehouseFinishedStopWalkingPerson.result.speed_ceiling.to("m/s").magnitude == approx(0.25994, rel=1e-5)
    assert (Scenarios.WarehouseFinishedStopWalkingPersonWithDispatch.result.speed_ceiling.to("m/s").magnitude
            == approx(0.19188, rel=1e-4))
    assert (Scenarios.WarehouseFinishedStopInspectedFloorWithDispatch.result.speed_ceiling.to("m/s").magnitude
            == approx(1.17093, rel=1e-5))
    assert Scenarios.WarehouseFinishedStopTenMsSlower.result.speed_ceiling.to("m/s").magnitude == approx(1.42231, rel=1e-5)
    assert Scenarios.WarehouseHostLeaseStall.result.added_delay_travel.to("mm").magnitude == approx(182.0)
    assert Scenarios.WarehouseIntegratorUnwind.result.added_delay_travel.to("mm").magnitude == approx(195.0)


def test_loaded_joint_barrier_projection_and_cycle():
    g = Scenarios.LoadedArmJointGravityStop
    assert g.bias.to("N*m").magnitude == approx(65.25)
    assert g.alpha_descending.to("1/s**2").magnitude == approx(7.25)
    assert g.tool_stop_free.to("mm").magnitude == approx(19.8621)
    assert g.tool_stop_descending.to("mm").magnitude == approx(79.4483)
    assert g.tool_stop_ascending.to("mm").magnitude == approx(11.3498)
    assert g.stop_time_descending.to("ms").magnitude == approx(165.517)
    far, near = g.barrier_demands
    assert far.to("N*m").magnitude == approx(80.078125) and near.to("N*m").magnitude == approx(89.140625)
    assert g.barrier_torque_demand(g.tool_stop_descending).to("N*m").magnitude == approx(87.0)
    assert g.separation_total.to("mm").magnitude == approx(686.292) and g.separation_person_share == approx(0.69735)
    p = Scenarios.TwoJointTorqueProjection
    assert p.projection["u_projected"] == [1.5, 2.0] and p.projection["lambda_star"] == approx(1.0)
    assert p.breach == approx(5.0) and p.clamp_value == approx(7.0)
    c = Scenarios.WarehouseEnforcerCycleBudget
    assert c.total.to("us").magnitude == approx(250.0) and c.headroom.to("us").magnitude == approx(150.0)
    assert Scenarios.WarehouseBaseHoldDrift.drift.to("mm").magnitude == approx(1.501)
# --- Agent evaluation (added 2026-10-20) ---------------------------------------
def test_recurring_task_consistency():
    r = Scenarios.RecurringTaskConsistency.result
    assert r.success[8] == approx(0.43047) and 1 - r.required_step_error == approx(0.98692)
    assert r.error_ratio == approx(7.6431)
    assert Scenarios.RecurringTaskConsistencyLowerRate.result.success[8] == approx(0.16777)
    assert Scenarios.RecurringTaskConsistencyHalvedFailure.result.success[8] == approx(0.66342)
    assert Scenarios.RefundAnswerFiveSessions.result.success[5] == approx(0.77378)


def test_verdicts_grader_and_agreement():
    v = Scenarios.UnresolvedTrialsOfHundred.result
    assert (v.default_rate, v.upper_bound) == (0.6, 0.75) and v.dropped_rate == approx(0.70588)
    g = Scenarios.ResearchReportJudgeCalibration.result
    assert g.false_accept_rate == approx(0.2) and g.false_accept_low == approx(0.12703)
    assert g.false_accept_high == approx(0.30046) and g.failure_among_accepted == approx(0.12308)
    assert g.false_reject_rate == approx(0.05)
    j = Scenarios.IndependentJudgesNinetyPercent
    assert j.agreement == approx(0.82) and abs(j.kappa) < 1e-12
    assert Scenarios.IndependentJudgesEightyPercent.agreement == approx(0.68)


def test_task_repeat_budget_and_intervals():
    t = Scenarios.RepairSuiteTaskRepeatBudget.result
    assert t.within_variance == approx(0.16) and t.variance_ratio == approx(2.0)
    assert t.free_se[1] == approx(0.02) and t.free_se[3] == approx(0.025820) and t.free_se[6] == approx(0.032660)
    assert (t.best_runs, t.best_tasks) == (5, 40) and t.best_se == approx(0.052915)
    assert t.paid_tasks[1] == 54 and t.paid_se[1] == approx(0.066667)
    w = Scenarios.FiftyTaskIntervals.result
    assert w.wald[0] == (0.0, 0.0) and w.wilson[0][1] == approx(0.071348)
    assert w.wald[2][0] == approx(-0.014316) and w.wilson[2][0] == approx(0.011039) and w.wilson[2][1] == approx(0.13460)
    p = Scenarios.AcceptanceRatePlanning.result
    assert p.half_width == approx(0.097998) and p.wilson_half_width == approx(0.096168)
    assert p.samples_for_target == 2401 and p.per_arm[0.04] == approx(2450) and p.per_arm[0.02] == approx(9800)


def test_paired_comparison_and_published_newcombe_examples():
    from mlsysim.physics import calc_newcombe_paired_difference, calc_mcnemar_exact_two_sided
    r = Scenarios.RepairCandidatePairedComparison.result
    assert r.difference == approx(0.04) and r.lower == approx(-0.032063) and r.upper == approx(0.11121)
    assert not r.noninferior and r.exact_p == approx(0.36159)
    assert (r.pairs_if_gain_holds, r.pairs_if_no_gain) == (217, 1948)
    assert r.paired_share == approx(0.40464) and r.independent_multiple == approx(2.4713)
    assert r.paired_half_width == approx(0.071281)
    # Newcombe (1998), Table III: (a, b, c, d) = (20, 2, 12, 16) and (27, 0, 0, 27), method 10.
    z = 1.959963984540054
    _, lo, hi = calc_newcombe_paired_difference(20, 12, 2, 16, z)
    assert lo == pytest.approx(0.0562, abs=5e-5) and hi == pytest.approx(0.3292, abs=5e-5)
    _, lo, hi = calc_newcombe_paired_difference(27, 0, 0, 27, z)
    assert lo == pytest.approx(-0.0351, abs=5e-5) and hi == pytest.approx(0.0351, abs=5e-5)
    assert calc_mcnemar_exact_two_sided(0, 0) == 1.0


def test_leaderboards_and_selection_cost():
    s = Scenarios.TerminalBenchScaffoldSpread.result
    assert s.models == 16 and s.widest == "claude-opus-4-6" and s.widest_count == 11
    assert s.median_spread == approx(0.13589) and s.median_se == approx(0.027)
    t = Scenarios.SweBenchVerifiedRankTies.result
    assert (t.models, t.pairs, t.tied) == (33, 24, 24) and t.needed_gap == approx(0.054308)
    m = Scenarios.TwoTaskTypeConsistency.result
    assert m.same_task_all == approx(0.53) and m.new_tasks_all == approx(0.49)
    c = Scenarios.CappedCandidateSelection.result
    assert c.availability == approx(0.784) and c.expected_candidates == approx(1.96)
    assert c.batch_per_accepted.to("USD").magnitude == approx(0.15306)
    assert c.sequential_per_accepted.to("USD").magnitude == approx(0.1) and c.batch_premium == approx(0.53061)


def test_pass_pow_k_and_paired_run_summary():
    from mlsysim.physics import calc_pass_pow_k_unbiased
    from mlsysim.solvers import PairedRunEvaluationModel
    assert calc_pass_pow_k_unbiased(8, 6, 2) == approx(15 / 28) and calc_pass_pow_k_unbiased(8, 1, 2) == 0.0
    rows = []
    for task, base, cand in (("a", (1, 1), (1, 1)), ("b", (0, 1), (1, 1))):
        for run in range(2):
            for cfg, ok in (("base", base[run]), ("cand", cand[run])):
                rows.append(dict(task=task, run=run, config=cfg, verdict="pass" if ok else "fail",
                                 visible="pass", calls=1, input_tokens=10, output_tokens=5, seconds=1.0))
    r = PairedRunEvaluationModel().solve(rows, ["a", "b"], 2, "base", "cand")
    assert r.accepted == {"base": 3, "cand": 4} and (r.both, r.candidate_only, r.baseline_only, r.neither) == (3, 1, 0, 0)
    assert r.pass_pow["base"][2] == approx(0.5) and r.pass_at["base"][2] == approx(1.0)
    assert r.visible_pass_sealed_reject["base"] == 1 and r.tokens_per_accepted_ratio == approx(0.75)
# --- Agent operations: traces, retention, postmortems, staged release (added 2026-10-20) ---

def test_repair_trial_span_timeline():
    r = Scenarios.RepairTrialSpanTimeline.result
    assert r.leaf_sum.to("s").magnitude == approx(13.2) and r.critical_path.to("s").magnitude == approx(11.2)
    assert r.elapsed.to("s").magnitude == approx(11.7) and r.unexplained.to("s").magnitude == approx(0.5)
    assert r.changed_elapsed.to("s").magnitude == approx(7.7) and r.saving.to("s").magnitude == approx(4.0)
    assert r.changed_critical == ["retrieve", "propose", "lint", "synthesize"]
    lint = Scenarios.RepairTrialSpanTimeline.variant(intervention={"lint": Scenarios.RepairTrialSpanTimeline.intervention["tests"]})
    assert lint.result.saving.to("s").magnitude == pytest.approx(0.0, abs=1e-12)


def test_head_sampling_miss():
    assert Scenarios.RareFailureUnderHeadSampling.miss == approx(0.59049)
    assert Scenarios.RecurringFailureUnderHeadSampling.miss == approx(0.0051538)
    assert Scenarios.FailureUnderFivePercentHeadSample.miss == approx(0.35849)
    assert Scenarios.FailureUnderOnePercentHeadSample.miss == approx(0.90438)


def test_verdict_aware_trace_retention():
    r = Scenarios.VerdictAwareTraceRetention.result
    assert (r.head_count, r.head_failed, r.head_unresolved, r.tail_control, r.tail_count) == (500, 20, 5, 190, 690)
    assert r.full_volume.to("GB").magnitude == approx(336.0) and r.head_volume.to("GB").magnitude == approx(16.8)
    assert r.tail_volume.to("GB").magnitude == approx(23.184) and r.full_over_tail == approx(14.4928)
    assert r.floor_volume.to("GB").magnitude == approx(16.8) and r.per_point_volume.to("GB").magnitude == approx(3.192)


def test_interface_repair_branches_wilson():
    b = Scenarios.InterfaceRepairBranches
    assert b.failure_upper("paginated") == approx(0.16113) and b.clean_runs_for_target == 73
    assert b.clean_failure_upper(72) >= 0.05 > b.clean_failure_upper(73)
    lo, hi = b.acceptance_interval("original")
    assert lo == approx(0.0088814) and hi == approx(0.23613)
    lo, hi = b.acceptance_interval("warning")
    assert lo == approx(0.21881) and hi == approx(0.61342)


def test_repair_candidate_paired_comparison_as_read_by_operations():
    """The Agent Evaluation comparison that the release decision record cites (shared with Agent Operations)."""
    r = Scenarios.RepairCandidatePairedComparison.result
    assert r.difference == approx(0.04) and r.lower == approx(-0.032062854, rel=1e-6)
    assert r.upper == approx(0.111211210, rel=1e-6) and not r.noninferior
    assert r.pairs_if_gain_holds == 217


def test_tool_drift_and_familywise_error():
    d = Scenarios.TestLintToolDrift
    assert d.divergence == approx(0.021006) and d.share_of_maximum == approx(0.030305)
    f = Scenarios.ReleaseGateFamilywiseError
    assert f.union_bound == approx(0.20) and f.independent == approx(0.18549) and f.per_test == approx(0.0125)


def test_acceptance_canary_monitor_and_ladder():
    m = Scenarios.AcceptanceCanaryMonitor
    r = m.result
    assert r.upper == approx(4.553877) and r.lower == approx(-2.985682)
    assert r.failure_step == approx(0.916291) and r.success_step == approx(-0.182322)
    assert r.expected_samples == approx(45.23806) and r.failures_to_hold == 5
    assert r.decision_time.to("hour").magnitude == approx(15.579354)
    assert r.excess_failures == approx(7.01071) and r.full_excess_failures == approx(11.28571)
    assert m.variant(fraction=0.10).result.decision_time.to("hour").magnitude == approx(8.03968)
    assert m.variant(fraction=1.0).result.decision_time.to("hour").magnitude == approx(1.25397)
    w = Scenarios.WebCanaryLadderOnAgentTasks
    assert w.total.to("minute").magnitude == approx(40.0) and w.labeled_by_completion == approx(2.0)
    assert w.admitted_by_completion == approx(20.0)


def test_openai_sycophancy_timeline_and_dapper():
    from mlsysim import Literature
    inc = Literature.IncidentReports
    assert (inc.OpenAiSycophancyRolloutStartDay, inc.OpenAiSycophancyRollbackStartDay) == (24, 28)
    assert inc.OpenAiSycophancyRollbackHours == 24
    rows = {r["sampling_frequency"]: r for r in Literature.DapperSamplingOverhead.entries}
    assert rows["1/1"]["latency_change_pct"] == 16.3 and rows["1/1024"]["latency_change_pct"] == -0.20
    assert Literature.DistributedTracing.DapperFirstProductionSamplingDenominator == 1024
def test_gpt2_xl_step_fits_one_h100():
    flash = Scenarios.Gpt2MixedPrecisionFlashAttentionOnH100
    assert flash.training_total.to("GB").magnitude == approx(109.5638)
    assert flash.resident_batch_limit == approx(23.1497)
    assert flash.max_resident_batch == 23
    assert Scenarios.Gpt2MixedPrecisionTwoMicroBatchesOnH100.training_total.to("GB").magnitude == approx(66.7819)
    assert Scenarios.Gpt2MixedPrecisionSegmentCheckpointedOnH100.training_total.to("GB").magnitude == approx(35.5343)
    ckpt = Scenarios.Gpt2XlBatch32SegmentCheckpointed
    assert ckpt.segment_layers == 6
    assert ckpt.segment_input.to("MB").magnitude == approx(104.8576)
    assert ckpt.segment_inputs.to("GB").magnitude == approx(0.83886)
    assert ckpt.live_segment.to("GB").magnitude == approx(10.69548)
    assert ckpt.checkpoint_memory_units(7) == approx(13.857143)
    act = Scenarios.Gpt2XlBatch32Activations
    assert act.per_token_bytes_coefficient == approx(34)
    assert act.score_bytes_coefficient == approx(80)
    assert act.score_share == approx(0.701754)
    assert act.per_sequence_activations.to("GB").magnitude == approx(2.67387)
    assert Scenarios.Gpt2XlRunOnOneH100Fp32.training_time.to("day").magnitude == approx(518.2421)
    assert Scenarios.Gpt2XlRunOnOneH100Tf32.rental_cost.to("USD").magnitude == approx(6740.816)
    assert Scenarios.Gpt2XlSingleCoreTokenizerFeed.busy_fraction == approx(0.476441)


def test_number_formats_and_loss_scaling():
    f = Scenarios.Fp16WeightSmallUpdates
    assert f.spacing("fp16") == 2 ** -11
    assert f.spacing("bf16") == 2 ** -8
    assert f.stored_after_one("fp16") == 0.5
    assert f.final("fp16") == 0.5
    assert f.final("fp32") == approx(1.50017)
    assert f.vanishing_ratio("fp16") == 2048
    s = Scenarios.SsdGradientLossScaling
    assert s.below_subnormal_pct == approx(26.2213)
    assert s.below_share_of_nonzero == approx(0.815987)
    assert s.subnormal_pct == approx(5.85519)
    assert s.nonzero_pct == approx(32.1344)
    assert s.shift == 3
    assert s.rescued_pct == approx(5.83949)
    assert s.headroom_doublings == approx(25.9993)
    assert s.dynamic_overflow_threshold == approx(63.96875)


def test_online_softmax_and_attention_tiles():
    o = Scenarios.OnlineSoftmaxFourScores
    assert o.blocks[0]["sum"] == approx(1.135335)
    assert o.blocks[1]["rescale"] == approx(0.135335)
    assert o.blocks[1]["sum"] == approx(o.whole_row["sum"])
    assert o.blocks[1]["output"] == approx(44.38347)
    assert o.weighted_average == approx(36.8810)
    a = Scenarios.AttentionScores4096TokensFp32
    assert a.score_matrix.to("MB").magnitude == approx(67.108864)
    assert a.tile_bytes.to("KB").magnitude == approx(65.536)
    assert a.tile_count == 1024
    assert a.stored_standard.to("MB").magnitude == approx(138.412)
    assert a.stored_flash.to("MB").magnitude == approx(4.22707)
    assert a.stored_ratio == approx(32.7442)
def test_model_compression_scenarios():
    from mlsysim import Literature
    from mlsysim.physics import (calc_csr_bytes, calc_pareto_frontier, calc_sparse_breakeven_zero_share,
                                 calc_sparse_bytes_per_nonzero)
    from mlsysim.core.units import BYTES_FP32, BYTES_INT32
    t = Scenarios.Llama2_7BTokenOnFlagshipPhone
    assert t.weight_bytes.to("GB").magnitude == approx(14.0) and t.read_time.to("ms").magnitude == approx(273.4375)
    assert t.read_energy.to("J").magnitude == approx(2.24) and t.read_power.to("W").magnitude == approx(8.192)
    assert Scenarios.MobileNetV2Fp32WeightReadPhone.read_time.to("ms").magnitude == approx(0.27382, rel=1e-4)
    assert Scenarios.ResNet50Fp32WeightReadPhone.weight_bytes.to("MB").magnitude == approx(102.228128)
    assert Scenarios.Gpt2XlFp16WeightReadPhone.read_time.to("ms").magnitude == approx(58.59375)
    assert Scenarios.Gpt2XlInt4WeightReadPhone.read_time.to("ms").magnitude == approx(14.6484375)
    b = Scenarios.MobileNetV2BlurPipelinePhone
    assert b.cpu_frame.to("ms").magnitude == approx(125.0) and b.flop_fraction == approx(418 / 600)
    assert b.pruned_cpu_frame.to("ms").magnitude == approx(87.0833, rel=1e-5)
    assert b.pruned_bytes.to("MB").magnitude == approx(10.44) and b.pruned_int_bytes.to("MB").magnitude == approx(2.61)
    m = Scenarios.SmallestModelAtSixtyPercentMmlu
    assert m.first["model"] == "Gopher 280B" and m.last["model"] == "Qwen2.5-Coder-3B"
    assert m.shrink == approx(93.333, rel=1e-4) and m.span_years == approx(2.7789, rel=1e-4)
    r = Scenarios.Rank128Factorization4096
    assert r.full_bytes.to("MB").magnitude == approx(67.108864) and r.compression == approx(16.0)
    f = Scenarios.ResNet50Conv256FusionH100
    assert f.unfused_traffic.to("MB").magnitude == approx(7.17824) and f.fused_traffic.to("MB").magnitude == approx(3.964928)
    assert f.traffic_cut == approx(0.447646, rel=1e-5) and f.time_ratio == approx(1.810434, rel=1e-5)
    s = Scenarios.SparseMatVec4x4
    assert s.nonzeros == 6 and s.csr_bytes.to("byte").magnitude == 68 and s.dense_bytes.to("byte").magnitude == 64
    p = Scenarios.MagnitudePruned3x3
    assert p.survivors == [0.8, -0.7, -0.9, -0.6] and p.csr_bytes.to("byte").magnitude == 80
    w = Scenarios.WearableMobileNetV2SparseFlash
    assert w.stored_bytes.to("MB").magnitude == approx(3.0) and not w.fits
    assert Scenarios.Channels256To250Tile8.last_tile_lanes == 2 and Scenarios.Channels256To248Tile8.remainder == 0
    assert [round(100 * x) for x in Scenarios.TeacherSoftTargetsT4.softened] == [48, 28, 24]
    assert Scenarios.PrunedInt8PaperVsMeasured.paper == approx(8.0)
    assert Scenarios.PrunedInt8PaperVsMeasured.realized_share == approx(0.1875)
    assert Scenarios.BertBaseMobileCompression.compression_ratio == approx(16.0)
    assert Scenarios.ModelFifthOfRequestLatency.result.overall_speedup == approx(1.25)
    assert Scenarios.ModelThreeFifthsFourTimesFaster.result.overall_speedup == approx(1 / 0.55)
    csr = calc_sparse_bytes_per_nonzero(BYTES_FP32, BYTES_INT32)
    coo = calc_sparse_bytes_per_nonzero(BYTES_FP32, BYTES_INT32, indices=2)
    assert calc_sparse_breakeven_zero_share(BYTES_FP32, csr) == approx(0.5)
    assert calc_sparse_breakeven_zero_share(BYTES_FP32, coo) == approx(2 / 3)
    assert calc_csr_bytes(6, 4, BYTES_FP32, BYTES_INT32).to("byte").magnitude == 68
    t3 = Literature.Series.MobileNetV3Table3.entries
    front = [t3[i]["model"] for i in calc_pareto_frontier([(r["pixel1_ms"], r["top1"]) for r in t3])]
    assert "V2 1.0" not in front and "V3-Large 0.75" in front
    assert Literature.CompressionResults.SanhDistilBertRetainedPct == 97
# --- Hardware acceleration on one accelerator (added 2026-10-20) ---


def test_h100_over_reference_cpu_and_systolic_reuse():
    g = Scenarios.H100OverReferenceCpu
    assert g.ratio_low == approx(494.5) and g.ratio_high == approx(989.0)
    e = Scenarios.SystolicArrayReuseEnergy
    assert e.naive_energy.to("pJ").magnitude == approx(2561.0)
    assert e.systolic_accesses == approx(0.015625)
    assert e.systolic_energy.to("pJ").magnitude == approx(11.0)
    assert e.ratio == approx(232.818)


def test_amdahl_on_h100_against_a_server_cpu():
    rn, gpt, bw = (Scenarios.ResNet50InferenceH100OverServerCpu, Scenarios.Gpt2DecodeH100OverServerCpu,
                   Scenarios.Gpt2DecodeH100OverServerCpuBandwidth)
    assert rn.gain == approx(247.375) and rn.speedup == approx(18.5734)
    assert gpt.speedup == approx(4.92044) and gpt.ceiling == approx(5.0)
    assert bw.gain == approx(67.0) and bw.speedup == approx(4.71831)


def test_resnet50_layers_against_the_h100_ridges():
    assert Scenarios.ResNet50Stage3ConvBatch32.flops.to("GFLOP").magnitude == approx(7.39875)
    assert Scenarios.ResNet50Stage3ConvBatch32.traffic_bytes.to("MB").magnitude == approx(13.13997)
    assert Scenarios.ResNet50Stage3ConvBatch32.traffic_intensity == approx(563.072)
    assert Scenarios.ResNet50Stage3ConvBatch32Int8.traffic_intensity == approx(1126.14)
    assert Scenarios.ResNet50Stage3ConvBatch32Int8Weights.traffic_intensity == approx(569.463)
    assert Scenarios.ResNet50Stage2ConvBatch32.traffic_intensity == approx(287.176)
    assert Scenarios.ResNet50Stage2ConvBatch32Int8.traffic_intensity == approx(574.352)
    assert Scenarios.ResNet50Stage2ExpansionBatch32.traffic_intensity == approx(51.1739)
    assert Scenarios.ResNet50Stage2ExpansionBatch32Int8.traffic_intensity == approx(102.348)
    assert Scenarios.ResNet50Batch1H100Int8PeakRate.result.latency_compute.to("us").magnitude == approx(4.14351)
    assert Scenarios.ResNet50Batch1H100Fp16PeakRate.result.latency_compute.to("us").magnitude == approx(8.29120)


def test_five_layers_on_the_h100_roofline():
    qkv, head = Scenarios.Gpt2XlQkvProjectionPrefill, Scenarios.ResNet50ClassifierHeadBatch32
    assert qkv.flops.to("GFLOP").magnitude == approx(125.829) and qkv.traffic_intensity == approx(1046.68)
    assert head.traffic_bytes.to("MB").magnitude == approx(4.29107) and head.traffic_intensity == approx(30.5453)
    assert Scenarios.Gpt2XlLayerNormPrefill.intensity == approx(1.99976)
    assert Scenarios.Gpt2XlSoftmaxPrefill.intensity == approx(0.75)
    assert Scenarios.Gpt2XlDecodeH100Roofline.peak_fraction == approx(0.0033873)
    assert Scenarios.Gpt2XlLayerNormH100Roofline.attainable.to("TFLOP/s").magnitude == approx(6.69918)
    assert Scenarios.ResNet50ClassifierHeadH100Roofline.peak_fraction == approx(0.103465)
    assert Scenarios.ResNet50Stage3ConvH100Roofline.peak_fraction == approx(1.0)
    a, h = Scenarios.KernelAt200FlopPerByteA100, Scenarios.KernelAt200FlopPerByteH100
    assert a.attainable.to("TFLOP/s").magnitude == approx(312.0) and h.attainable.to("TFLOP/s").magnitude == approx(670.0)


def test_batch_raises_dense_layer_intensity():
    b1, b32, b512 = Scenarios.Dense2048Fp16Batch1, Scenarios.Dense2048Fp16Batch32, Scenarios.Dense2048Fp16Batch512
    assert b1.traffic_intensity == approx(0.999024) and b32.traffic_intensity == approx(31.0303)
    assert b512.traffic_intensity == approx(341.333)
    assert b1.batch_for_traffic_intensity(295.2238806) == approx(414.818)
    assert b1.variant(batch_size=415).traffic_intensity == approx(295.2238806, rel=1e-3)


def test_accelerator_trends_and_latency_in_cycles():
    w = Scenarios.DenseParametersVsBandwidthSince2014
    assert w.parameter_growth == approx(3902.93) and w.bandwidth_growth == approx(33.3333)
    e = Scenarios.AcceleratorEnvelopeSinceTpuV2
    assert e.device_count == 52 and e.compute_growth == approx(54.7087) and e.bandwidth_growth == approx(13.3333)
    assert e.best_bandwidth_since == "2025-02-15"
    g = Scenarios.NvidiaGenerationsV100ToB200
    assert g.compute_growth == approx(18.0) and g.bandwidth_growth == approx(8.88889) and g.gap_growth == approx(2.025)
    c = Scenarios.HbmAccessAt2Ghz
    assert c.cycle_time.to("ps").magnitude == approx(500.0) and c.cycles == approx(545.64)
    t = Scenarios.TensorCoreMmaM16n8k16
    assert t.macs == 2048 and t.macs_per_thread == approx(64.0)
def test_trajectory_planning_joint_rates_handoffs_and_replacement():
    good = Scenarios.WarehouseArmRightAngleElbow.result
    assert good.reach.to("m").magnitude == approx(0.700)
    assert good.joint_rate_norm.to("rad/s").magnitude == approx(4.896751, rel=1e-6)
    assert good.joint_rate_one_joint.to("rad/s").magnitude == approx(3.462526, rel=1e-6)
    assert good.exceeds_joint_limit
    assert good.damped_joint_rate.to("rad/s").magnitude == approx(4.619815, rel=1e-6)
    assert good.realized_fraction == approx(0.943445, rel=1e-5)
    assert good.damped_ceiling.to("rad/s").magnitude == approx(10.0)
    bad = Scenarios.WarehouseArmNearStraightElbow.result
    assert bad.joint_rate_norm.to("rad/s").magnitude == approx(37.760396, rel=1e-6)
    assert bad.damped_joint_rate.to("rad/s").magnitude == approx(8.272410, rel=1e-6)
    assert bad.realized_fraction == approx(0.219076, rel=1e-5) and bad.damped_exceeds_joint_limit
    m = Scenarios.GenericReplannerMissionExposure.result
    assert m.handoffs == 3000 and m.expected_late == approx(30.0)
    assert m.all_on_time == approx(8.046070e-14, rel=1e-5)
    assert m.per_handoff_budget == approx(1e-3 / 3000)
    assert m.orders_deeper == approx(4.477121, rel=1e-6) and m.handoffs_per_expected_late == approx(100.0)
    r = Scenarios.WarehouseArmChunkReplacement.result
    assert r.latency.to("ms").magnitude == approx(114.91072)
    assert r.other_stages.to("ms").magnitude == approx(5.31072)
    assert r.requests_in_flight == 3 and r.latency_beyond_lease.to("ms").magnitude == approx(54.91072)
    assert r.tail_over_inference == approx(3.75)
    s = Scenarios.WarehouseArmStaleAnchor.result
    assert s.gap.to("rad").magnitude == approx(0.040) and s.tool_displacement.to("mm").magnitude == approx(28.0)
    w = Scenarios.WarehouseBaseDynamicWindow.result
    assert w.half_width.to("m/s").magnitude == approx(0.1)
    assert w.admissible_speed.to("m/s").magnitude == approx(2.097618, rel=1e-6) and w.admits_drive_limit


def test_trajectory_planning_jerk_limited_stops_and_deadlines():
    arm = Scenarios.WarehouseArmStallStop.result
    assert arm.jerk.to("m/s**3").magnitude == approx(200 / 3)
    assert arm.delay.to("ms").magnitude == approx(82.0) and arm.delay_travel.to("mm").magnitude == approx(82.0)
    assert arm.scurve_travel.to("mm").magnitude == approx(137.5) and arm.stop_time.to("ms").magnitude == approx(275.0)
    assert arm.braking_travel.to("mm").magnitude == approx(100.0) and arm.jerk_ramp_travel.to("mm").magnitude == approx(37.5)
    assert arm.polynomial_distance.to("mm").magnitude == approx(150.0)
    assert arm.required_distance.to("mm").magnitude == approx(219.5) and arm.spare.to("mm").magnitude == approx(80.5)
    assert arm.rest_time.to("ms").magnitude == approx(357.0) and arm.fits
    assert arm.stop_distance.to("mm").magnitude == approx(219.5) and arm.ramp_delay.to("ms").magnitude == approx(37.5)
    mug = Scenarios.WarehouseArmOpenMugStop.result
    assert mug.spill_deceleration.to("m/s**2").magnitude == approx(2.4516625)
    assert mug.rise_at_free_deceleration.to("mm").magnitude == approx(20.394324, rel=1e-6)
    assert mug.stop.jerk.to("m/s**3").magnitude == approx(16.028397, rel=1e-6)
    assert mug.stop.scurve_travel.to("mm").magnitude == approx(280.421959, rel=1e-6)
    assert mug.stop.required_distance.to("mm").magnitude == approx(362.421959, rel=1e-6)
    assert mug.reused_jerk_stop_distance.to("mm").magnitude == approx(222.330711, rel=1e-6)
    assert mug.carry_speed.to("m/s").magnitude == approx(0.885025, rel=1e-5)
    base = Scenarios.WarehouseBaseJerkLimitedStop.result
    assert base.jerk.to("m/s**3").magnitude == approx(64 / 9)
    assert base.scurve_travel.to("mm").magnitude == approx(773.4375) and base.stop_time.to("ms").magnitude == approx(1031.25)
    assert base.polynomial_distance.to("mm").magnitude == approx(843.75)
    assert base.jerk_ramp_travel.to("mm").magnitude == approx(210.9375)
    assert base.ramp_delay.to("ms").magnitude == approx(140.625)
    assert base.constant_required.to("mm").magnitude == approx(912.9)
    assert base.required_distance.to("mm").magnitude == approx(1123.8375)
    assert base.polynomial_required.to("mm").magnitude == approx(1194.15)
    assert base.spare.to("mm").magnitude == approx(-23.8375) and not base.fits
    d = Scenarios.WarehouseArmCommitmentSchedule.result
    assert d.preferred_admission.to("ms").magnitude == approx(240.0) and d.reserve.to("ms").magnitude == approx(42.0)
    assert d.minimum_onset.to("ms").magnitude == approx(262.0) and d.admission_cutoff.to("ms").magnitude == approx(278.0)
    assert d.rest.to("ms").magnitude == approx(575.0) and d.nominal_end.to("ms").magnitude == approx(400.0)
    assert d.window_periods == approx(23.0) and d.used.to("mm").magnitude == approx(197.5)
    assert d.margin.to("mm").magnitude == approx(102.5) and d.late_breach.to("mm").magnitude == approx(97.5)


def test_trajectory_planning_seam_bridge_and_shoulder_swing():
    sh = Scenarios.WarehouseArmShoulderSwing.result
    assert sh.holding_torque.to("N*m").magnitude == approx(20.593965, rel=1e-6)
    assert sh.holding_share == approx(0.236712, rel=1e-5)
    assert sh.effective_inertia.to("kg*m**2").magnitude == approx(1.6535)
    assert sh.accel_up.to("rad/s**2").magnitude == approx(40.160892, rel=1e-6)
    assert sh.accel_down.to("rad/s**2").magnitude == approx(65.070435, rel=1e-6)
    assert sh.binding_acceleration.to("rad/s**2").magnitude == approx(7.5)
    assert sh.bang_bang_time.to("ms").magnitude == approx(516.397779, rel=1e-6)
    assert sh.switch_torque_step.to("N*m").magnitude == approx(24.8025)
    assert sh.jerk_bound.to("rad/s**3").magnitude == approx(604.777744, rel=1e-6)
    assert sh.jerk_limited_time.to("ms").magnitude == approx(528.947916, rel=1e-6)
    assert sh.penalty.to("ms").magnitude == approx(12.550136, rel=1e-5) and sh.reaches_plateau
    b = Scenarios.PandaClassJointSeamBridge.result
    assert b.step_torque.to("N*m").magnitude == approx(36.7)
    assert b.step_torque_rate.to("N*m/s").magnitude == approx(36700.0) and b.step_over_rate_limit == approx(36.7)
    assert b.peak_torque.to("N*m").magnitude == approx(1.101) and b.peak_reduction == approx(100 / 3)
    assert b.peak_jerk.to("rad/s**3").magnitude == approx(480.0)
    assert b.torque_rate.to("N*m/s").magnitude == approx(88.08) and b.within_rate_limit
    a3, a4, a5 = b.coefficients
    assert a3 == approx(80.0) and a4 == approx(-800.0) and abs(a5) < 1e-6
    assert b.loaded_step_torque.to("N*m").magnitude == approx(330.7)
    assert b.shortest_bridge.to("ms").magnitude == approx(14.839137, rel=1e-6)
    assert b.velocity_ceiling.to("rad/s").magnitude == approx(0.692098, rel=1e-5)
    h = Scenarios.PandaClassJointSeamBridgeHalved.result
    assert h.peak_torque.to("N*m").magnitude == approx(2.202) and h.torque_rate.to("N*m/s").magnitude == approx(352.32)
    c = Scenarios.PandaClassJointSeamBridgeCompressed.result
    assert c.peak_acceleration.to("rad/s**2").magnitude == approx(30.0)
    assert c.peak_torque.to("N*m").magnitude == approx(5.505) and c.torque_rate.to("N*m/s").magnitude == approx(2202.0)
    assert not c.within_rate_limit and c.torque_rate_share == approx(2.202)


def test_warehouse_jerk_limited_stop_matches_finished_budget():
    """The trajectory chapter's S-curve stop and the enforcement chapter's finished budget share one jerk bound and one stop."""
    base = Scenarios.WarehouseBaseJerkLimitedStop.result
    budget = Scenarios.WarehouseFinishedStopAtDriveLimit.result
    assert base.jerk.m_as("m/s**3") == pytest.approx(budget.jerk.m_as("m/s**3"))
    assert base.stop_distance.m_as("mm") == pytest.approx(budget.stop_distance.m_as("mm"))
    assert base.jerk_ramp_travel.m_as("mm") == pytest.approx(budget.jerk_ramp_travel.m_as("mm"))
