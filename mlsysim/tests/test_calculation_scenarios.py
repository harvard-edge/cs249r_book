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
