#!/usr/bin/env python3
"""Every number on the Lecture 3 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry) and the calculations (Scenarios.*, each
backed by a solver or a scenario type); the chapter reads the same scenarios.
This file only selects values, converts units, and formats them; it types no
numbers. Each value is recorded with v() under the key the deck prints as
\\val{key}; running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Hardware, Systems, Infrastructure, Scenarios = (mlsysim.Hardware, mlsysim.Systems,
                                                mlsysim.Infrastructure, mlsysim.Scenarios)
H100 = Hardware.Cloud.H100
NDR, HDR = Systems.Fabrics.InfiniBand_NDR, Systems.Fabrics.InfiniBand_HDR
SWITCH = Systems.SwitchFabric
REF = Scenarios.DenseLLMPretrainingReference          # 175B on the 1,024-GPU reference cluster
LADDER = Scenarios.BufferTransfer10GB                 # Lecture 2's staircase
STEP_INTRA, STEP_INTER = Scenarios.ExposedAllReduceIntraNode, Scenarios.ExposedAllReduceInterNode
ECMP = Scenarios.EcmpHashCollision
OPTICS = Scenarios.SpineOpticsTraining10K
GBN = Scenarios.GoBackNResendNdr
AB = Scenarios.AlphaBetaNdrRoce
AB_SMALL = Scenarios.AlphaBetaHdrNdrSmallMessage
NCCL = Scenarios.NcclAllReduceH100NodeFit
CUT4, CUT2 = Scenarios.RingAllReduceAcross4to1Spine, Scenarios.RingAllReduceAcross2to1Spine
K64, K256 = Scenarios.FatTreeNdrSwitch, Scenarios.FatTree102T4Switch
TORUS = Scenarios.TorusVsFatTree4K
RAIL = Scenarios.RailVsLeafSpineHops
INCAST = Scenarios.IncastSuperPodUnit
XCVR = Scenarios.TransceiverDegradationRoce4K
LINKS = Scenarios.DegradedLinksReferenceFabric
SILENT = Scenarios.DegradedTransceiverRun1K
SRIOV = Scenarios.SriovNdrNic
DIAG = Scenarios.HalfRateLinkDiagnosis
SU, META = Systems.Clusters.DgxH100SuperPodSU, Systems.Clusters.Meta_GenAI_H100_24K_RoCE


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


print("== Running case and link rates")
cluster = REF.system
v("params_b", m(REF.workload.parameters, "Gparam"), "{:.0f}", "running model, billions of parameters")
v("grad_gb", m(REF.workload.size_in_bytes(mlsysim.BYTES_FP16), "GB"), "{:.0f}", "16-bit gradients, GB")
v("grad_bytes", m(mlsysim.BYTES_FP16, "byte"), "{:.0f}", "bytes per 16-bit gradient")
v("ref_gpus", cluster.total_accelerators, "{:,.0f}", "reference cluster GPUs")
v("ref_nodes", cluster.count, "{:.0f}", "reference cluster nodes")
v("gpus_per_node", cluster.node.accelerators_per_node, "{:.0f}", "GPUs per DGX H100 node")
v("gpus_per_node_word", cluster.node.accelerators_per_node, "words", "the same, in prose")
v("ndr_gbps", m(NDR.bandwidth, "Gbps"), "{:.0f}", "NDR port, Gb/s per direction")
v("hdr_gbps", m(HDR.bandwidth, "Gbps"), "{:.0f}", "HDR port, Gb/s per direction")
v("ib_gbs", m(NDR.bandwidth, "GB/s"), "{:.0f}", "NDR port, GB/s per direction")
v("hdr_gbs", m(HDR.bandwidth, "GB/s"), "{:.0f}", "HDR port, GB/s per direction")

print("== Recall: the staircase (Lecture 2 transfer scenario)")
t = LADDER.result.times
v("buffer_gb", m(LADDER.payload, "GB"), "{:.0f}", "buffer moved one hop, GB")
v("hbm_gbs", m(LADDER.links["HBM"], "GB/s"), "{:,.0f}", "H100 HBM, GB/s")
v("nvlink_gbs", m(LADDER.links["NVLink"], "GB/s"), "{:.0f}", "NVLink 4 per direction, GB/s")
v("stair_hbm_ms", m(t["HBM"], "ms"), "{:.1f}", "10 GB over HBM, ms")
v("stair_nvlink_ms", m(t["NVLink"], "ms"), "{:.1f}", "10 GB over NVLink, ms")
v("stair_ib_ms", m(t["InfiniBand"], "ms"), "{:.0f}", "10 GB over InfiniBand, ms")

print("== Node cliff")
v("nvlink_bidir_gbs", m(H100.nvlink.bandwidth, "GB/s"), "{:.0f}", "H100 NVLink, both directions, GB/s")
v("nvlink_ib_ratio", ratio(LADDER.links["NVLink"], LADDER.links["InfiniBand"]), "{:.0f}", "NVLink / NDR per direction")

print("== Exposed AllReduce (illustrative step scenarios)")
v("step_fwd_ms", m(STEP_INTER.compute_phases["forward"], "ms"), "{:.0f}", "forward, ms")
v("step_bwd_ms", m(STEP_INTER.compute_phases["backward"], "ms"), "{:.0f}", "backward, ms")
v("ar_intra_ms", m(STEP_INTRA.communication_time, "ms"), "{:.0f}", "AllReduce over NVLink, ms")
v("ar_inter_ms", m(STEP_INTER.communication_time, "ms"), "{:.0f}", "AllReduce across nodes, ms")
v("step_inter_ms", m(STEP_INTER.step_time, "ms"), "{:.0f}", "inter-node step, ms")
v("exposed_pct", 100 * STEP_INTER.exposed_communication_fraction, "{:.0f}", "exposed share, %")
v("step_compute_pct", 100 * STEP_INTER.compute_fraction, "{:.0f}", "compute share of the schedule, %")

print("== ECMP (hash-collision scenario)")
v("ecmp_flows", ECMP.flows, "{:.0f}", "elephant flows")
v("ecmp_paths", ECMP.paths, "{:.0f}", "equal-cost paths")
v("ecmp_even", ECMP.even_load, "{:.0f}", "even split, flows per path (drain units)")
v("ecmp_worst", ECMP.worst_load, "{:.0f}", "worst path, flows (drain units)")
for i, n in enumerate(ECMP.path_loads):
    v(f"ecmp_load_{i}", n, "{:.0f}", f"hashed load on path {i}")

print("== Wire: copper and optics (buildout scenario)")
v("dac_usd", m(Infrastructure.Pricing.Network.DacCableUsd.rate, "USD"), "{:.0f}", "copper DAC, $")
v("aoc_usd", m(Infrastructure.Pricing.Network.AocCableUsd.rate, "USD"), "{:.0f}", "active optical cable, $")
v("aoc_dac_word", ratio(Infrastructure.Pricing.Network.AocCableUsd.rate,
                        Infrastructure.Pricing.Network.DacCableUsd.rate), "words", "optical / copper price, in prose")
v("fec_ns", m(SWITCH.FecLatencyLow, "ns"), "{:.0f}", "FEC per switch, ns")
v("rack_low_kw", Infrastructure.Racks.AI_Standard.power_kw, "{:.0f}", "dense AI rack, low, kW")
v("rack_high_kw", Infrastructure.Racks.AI_High.power_kw, "{:.0f}", "dense AI rack, high, kW")
v("optics_gpus", OPTICS.fleet.total_accelerators, "{:,.0f}", "cluster GPUs")
v("optics_per_gpu", OPTICS.links_per_device, "{:.0f}", "spine links per GPU")
v("optics_per_gpu_word", OPTICS.links_per_device, "words", "the same, in prose")
v("optics_links", OPTICS.n_links, "{:,.0f}", "spine links")
v("optics_link_usd", m(OPTICS.cost_per_link, "USD"), "{:.0f}", "per link, $")
v("optics_link_w", m(OPTICS.power_per_link, "W"), "{:.0f}", "per module, W")
v("optics_musd", m(OPTICS.total_cost, "USD") / 10 ** 6, "{:.0f}", "optics, $M")
v("optics_kw", m(OPTICS.total_power, "kW"), "{:.0f}", "optics module power, kW")

print("== Go-Back-N window")
v("gbn_rtt_us", m(GBN.delay, "microsecond"), "{:.0f}", "loss-detection round trip, us")
v("gbn_kb", m(GBN.product, "KB"), "{:.0f}", "resent window, KB")

print("== Alpha-beta (NDR and RoCE scenario)")
ab = AB.result
v("ndr_alpha_us", m(AB.alphas["NDR"], "microsecond"), "{:.1f}", "NDR alpha, us")
v("roce_alpha_us", m(AB.alphas["RoCE 100G"], "microsecond"), "{:.0f}", "RoCE alpha, us")
v("roce_gbps", m(AB.bandwidths["RoCE 100G"], "Gbps"), "{:.0f}", "RoCE link, Gb/s")
v("roce_beta_gbs", m(AB.bandwidths["RoCE 100G"], "GB/s"), "{:.1f}", "RoCE beta, GB/s")
v("roce400_gbps", m(AB.bandwidths["RoCE 400G"], "Gbps"), "{:.0f}", "faster RoCE link, Gb/s")
v("ctrl_kb", m(AB.messages["control"], "KB"), "{:.0f}", "control message, KB")
v("ctrl_bytes", m(AB.messages["control"], "byte"), "{:,.0f}", "control message, bytes")
v("shard_mb", m(AB.messages["shard"], "MB"), "{:.0f}", "gradient shard, MB")
v("shard_exact_mb", m(REF.workload.size_in_bytes(mlsysim.BYTES_FP16) / cluster.total_accelerators, "MB"),
  "{:.0f}", "one rank's 1/1,024 share, MB")
v("ctrl_ndr_wire_us", m(ab.transmission_times["NDR"]["control"], "microsecond"), "{:.2f}", "4 KB on NDR, n/beta, us")
v("ctrl_roce_wire_us", m(ab.transmission_times["RoCE 100G"]["control"], "microsecond"), "{:.2f}",
  "4 KB on RoCE, n/beta, us")
v("ctrl_ndr_us", m(ab.times["NDR"]["control"], "microsecond"), "{:.2f}", "4 KB on NDR, us")
v("ctrl_roce_us", m(ab.times["RoCE 100G"]["control"], "microsecond"), "{:.2f}", "4 KB on RoCE, us")
v("ctrl_ratio", ratio(ab.times["RoCE 100G"]["control"], ab.times["NDR"]["control"]), "{:.1f}", "RoCE / NDR, 4 KB")
v("ctrl_roce400_us", m(ab.times["RoCE 400G"]["control"], "microsecond"), "{:.2f}", "4 KB on 400G RoCE, us")
v("shard_ndr_ms", m(ab.times["NDR"]["shard"], "ms"), "{:.1f}", "350 MB on NDR, ms")
v("shard_roce_ms", m(ab.times["RoCE 100G"]["shard"], "ms"), "{:.1f}", "350 MB on RoCE, ms")
v("shard_ratio", ratio(ab.times["RoCE 100G"]["shard"], ab.times["NDR"]["shard"]), "{:.0f}", "RoCE / NDR, 350 MB")
v("shard_roce400_ms", m(ab.times["RoCE 400G"]["shard"], "ms"), "{:.1f}", "350 MB on 400G RoCE, ms")
v("cross_ndr_kb", m(ab.crossovers["NDR"], "KB"), "{:.0f}", "NDR crossover, KB")
v("cross_roce_kb", m(ab.crossovers["RoCE 100G"], "KB"), "{:.1f}", "RoCE crossover, KB")

print("== Does 2x bandwidth help a 10 KB message? (HDR and NDR scenario)")
s = AB_SMALL.result
t_hdr, t_ndr = s.times["HDR"]["small"], s.times["NDR"]["small"]
v("bw_doubling", ratio(AB_SMALL.bandwidths["NDR"], AB_SMALL.bandwidths["HDR"]), "{:.0f}", "NDR / HDR line rate")
v("small_kb", m(AB_SMALL.messages["small"], "KB"), "{:.0f}", "message, KB")
v("small_bytes", m(AB_SMALL.messages["small"], "byte"), "{:,.0f}", "message, bytes")
v("small_hdr_wire_us", m(s.transmission_times["HDR"]["small"], "microsecond"), "{:.2f}", "HDR n/beta, us")
v("small_ndr_wire_us", m(s.transmission_times["NDR"]["small"], "microsecond"), "{:.2f}", "NDR n/beta, us")
v("small_hdr_us", m(t_hdr, "microsecond"), "{:.2f}", "HDR, us")
v("small_ndr_us", m(t_ndr, "microsecond"), "{:.2f}", "NDR, us")
v("small_gain_pct", 100 * (1 - ratio(t_ndr, t_hdr)), "{:.1f}", "NDR faster than HDR, %")

print("== Measured NCCL AllReduce (fit scenario)")
fit = NCCL.result
rows = NCCL.rows
excerpt = (rows[0], rows[13], rows[20], rows[24], rows[27], rows[31])  # 8 B, 64 KiB, 8 and 128 MiB, 1 and 16 GiB
for i, r in enumerate(excerpt):
    v(f"nccl_size_{i}", r["size_bytes"], "{:.0f}", f"row {i} size, B")
    v(f"nccl_time_{i}", r["oop_time_us"], "{:.5g}", f"row {i} time, us")
    v(f"nccl_algbw_{i}", r["oop_algbw_GBps"], "{:.2f}", f"row {i} algbw, GB/s")
    v(f"nccl_busbw_{i}", r["oop_busbw_GBps"], "{:.2f}", f"row {i} busbw, GB/s")
v("nccl_rows", fit.row_count, "{:.0f}", "rows in the sweep")
v("nccl_floor_us", m(fit.floor_time, "microsecond"), "{:.0f}", "floor, us")
v("nccl_floor_cut_kb", m(NCCL.floor_cutoff, "KiB"), "{:.0f}", "floor taken up to, KiB")
v("nccl_top_gb", m(rows[-1]["size_bytes"] * mlsysim.core.units.byte, "GiB"), "{:.0f}", "largest message, GiB")
v("nccl_bw_gbs", m(fit.bandwidth, "GB/s"), "{:.0f}", "algbw at the largest message, GB/s")
v("nccl_knee_mb", m(fit.knee, "MB"), "{:.1f}", "model knee, MB")
v("nccl_knee_round_mb", m(fit.knee, "MB"), "{:.0f}", "the chapter's rounding, MB")
v("nccl_mid_lo_mb", m(NCCL.mid_low, "MiB"), "{:.0f}", "middle range from, MiB")
v("nccl_mid_hi_mb", m(NCCL.mid_high, "MiB"), "{:.0f}", "middle range to, MiB")
v("nccl_mid_ratio", fit.worst_mid_ratio, "{:.1f}", "worst measured / model in the middle")
v("nccl_peak_busbw", m(fit.peak_bus_bandwidth, "GB/s"), "{:.0f}", "peak bus bandwidth, GB/s")
v("nccl_mid_busbw_lo", m(fit.mid_bus_bandwidth_low, "GB/s"), "{:.0f}", "middle-range busbw, low, GB/s")
v("nccl_mid_busbw_hi", m(fit.mid_bus_bandwidth_high, "GB/s"), "{:.0f}", "middle-range busbw, high, GB/s")
v("nccl_top_busbw", rows[-1]["oop_busbw_GBps"], "{:.0f}", "busbw at the largest message, GB/s")

print("== Bisection and the 4:1 spine (ring AllReduce scenario)")
b = CUT4.result
f = CUT4.communication_fraction
v("cut_half", b.devices_per_side, "{:.0f}", "GPUs (links) per side")
v("cut_full_tbs", m(b.bisection_full, "TB/s"), "{:.1f}", "1:1 cut, TB/s")
v("cut_over_tbs", m(b.bisection_oversubscribed, "TB/s"), "{:.1f}", "4:1 cut, TB/s")
v("cut_full_gbs", m(b.bisection_full, "GB/s"), "{:,.0f}", "1:1 cut, GB/s")
v("cut_over_gbs", m(b.bisection_oversubscribed, "GB/s"), "{:,.0f}", "4:1 cut, GB/s")
v("oversub", CUT4.oversubscription_ratio, "{:.0f}", "spine oversubscription")
v("cut_slowdown", b.slowdown, "{:.0f}", "crossing-time ratio")
v("os_cost_musd", m(CUT4.cluster_cost, "USD") / 10 ** 6, "{:.0f}", "cluster cost, $M")
v("os_payload_gb", m(CUT4.payload_per_device, "GB"), "{:.0f}", "gradient payload per GPU, GB")
v("os_comm_pct", 100 * f, "{:.0f}", "communicating share of the step, %")
v("os_ring_factor", b.ring_factor, "{:.3f}", "ring factor 2(N-1)/N")
v("os_crossing_gb", m(b.crossing_bytes, "GB"), "{:,.0f}", "bytes crossing the cut, GB")
v("os_t_full_s", m(b.time_full, "second"), "{:.3f}", "crossing time at 1:1, s")
v("os_t_over_s", m(b.time_oversubscribed, "second"), "{:.3f}", "crossing time at 4:1, s")
v("os_noncomm", 1 - f, "{:.2f}", "non-communicating share")
v("os_comm", f, "{:.2f}", "communicating share")
v("os_stretched", f * CUT4.oversubscription_ratio, "{:.2f}", "communicating share x ratio")
v("os_rel", b.relative_throughput, "{:.2f}", "relative throughput")
v("os_loss_pct", 100 * b.throughput_loss, "{:.1f}", "throughput lost, %")
v("os_idle_musd", m(b.idle_cost, "USD") / 10 ** 6, "{:.1f}", "equivalent idle capacity, $M")
v("os2_ratio", CUT2.oversubscription_ratio, "{:.0f}", "milder spine")
v("os2_loss_pct", 100 * CUT2.result.throughput_loss, "{:.1f}", "throughput lost at 2:1, %")

print("== Fat-tree reach")
v("k64", K64.radix, "{:.0f}", "Quantum-2 class radix")
v("k64_two", K64.two_tier_hosts, "{:,.0f}", "two tiers, hosts")
v("k64_three", K64.three_tier_hosts, "{:,.0f}", "three tiers, hosts")
v("k64_switches", K64.three_tier_switches, "{:,.0f}", "three tiers, switches")
v("k256", K256.radix, "{:.0f}", "102.4 Tb/s switch as 400G ports")
v("k256_two", K256.two_tier_hosts, "{:,.0f}", "two tiers, hosts")
v("k256_three", K256.three_tier_hosts, "{:,.0f}", "three tiers, hosts")
v("asic_tbps", m(SWITCH.SwitchAsic102T4, "Tbps"), "{:.1f}", "switch ASIC, Tb/s")
v("hops_pod", RAIL.hops["leaf-spine-leaf"], "{:.0f}", "switch traversals inside a pod")
v("hops_core", TORUS.fat_tree_path_switches, "{:.0f}", "switch traversals between pods")

print("== Rails")
v("rail_gpus", cluster.total_accelerators / cluster.node.accelerators_per_node, "{:.0f}", "GPUs per rail")
v("rail_hops", RAIL.hops["rail"], "{:.0f}", "switch traversals on a rail")
v("hop_us", m(RAIL.hop_latency, "microsecond"), "{:.1f}", "per traversal, us")
v("hops_pod_us", m(RAIL.latencies["leaf-spine-leaf"], "microsecond"), "{:.1f}", "leaf-spine-leaf, us")
v("hops_rail_us", m(RAIL.latencies["rail"], "microsecond"), "{:.1f}", "one rail switch, us")

print("== Torus against fat-tree")
v("torus_n", TORUS.n_endpoints, "{:,.0f}", "endpoints")
v("torus_side", TORUS.side, "{:.0f}", "torus side")
v("torus_cut", TORUS.torus_cut_links, "{:.0f}", "torus cut links")
v("torus_cut_tbs", m(TORUS.torus_bisection, "TB/s"), "{:.1f}", "torus cut, TB/s")
v("ft_cut", TORUS.fat_tree_cut_links, "{:,.0f}", "fat-tree cut links")
v("ft_cut_tbs", m(TORUS.fat_tree_bisection, "TB/s"), "{:.1f}", "fat-tree cut, TB/s")
v("torus_hops", TORUS.torus_diameter, "{:.0f}", "torus worst path, hops")

print("== Incast (SuperPOD scalable-unit scenario)")
inc = INCAST.result
v("incast_senders", INCAST.n_senders, "{:.0f}", "senders")
v("incast_egress_gbs", m(INCAST.egress_bandwidth, "GB/s"), "{:.0f}", "egress, GB/s")
v("incast_offered_tbs", m(inc.offered_load, "TB/s"), "{:.1f}", "offered load, TB/s")
v("incast_offered_gbs", m(inc.offered_load, "GB/s"), "{:.0f}", "offered load, GB/s")
v("incast_ratio", inc.oversubscription, "{:.0f}", "offered / egress")
v("incast_net_tbs", m(inc.net_fill_rate, "TB/s"), "{:.2f}", "net fill rate, TB/s")
v("incast_buffer_mb", m(INCAST.buffer, "MB"), "{:.0f}", "egress buffer, MB")
v("incast_fill_us", m(inc.overflow_time, "microsecond"), "{:.1f}", "buffer fills in, us")
v("incast_drain_us", m(inc.drain_time, "microsecond"), "{:.0f}", "full buffer drains in, us")

print("== Transceivers (exposure scenario)")
v("xcvr_gpus", XCVR.counts["devices"], "{:,.0f}", "RoCE cluster GPUs")
v("xcvr_count", XCVR.n_components, "{:,.0f}", "transceivers")
v("xcvr_p_pct", 100 * XCVR.probability, "{:.3f}", "daily degradation rate per module, %")
v("xcvr_any_pct", 100 * XCVR.p_any, "{:.1f}", "daily chance of at least one, %")

print("== SuperPOD and Grand Teton")
v("su_nodes", SU.count, "{:.0f}", "nodes per scalable unit")
v("su_gpus", SU.total_accelerators, "{:.0f}", "GPUs per scalable unit")
v("su_count", cluster.total_accelerators / SU.total_accelerators, "{:.0f}", "units in the reference cluster")
v("su_count_word", cluster.total_accelerators / SU.total_accelerators, "words", "the same, in prose")
v("meta_gpus", META.total_accelerators, "{:,.0f}", "GPUs per Grand Teton cluster")
v("meta_gbps", m(META.fabric.bandwidth, "Gbps"), "{:.0f}", "endpoint rate, Gb/s")

print("== Silent waste (degraded-transceiver run scenario)")
lost = SILENT.lost_device_hours
cost = {k: m(q, "USD") for k, q in SILENT.lost_cost.items()}
rate = m(SILENT.device_hour_rate, "USD/hour")
v("sw_bw_loss_pct", 100 * SILENT.bandwidth_loss, "{:.0f}", "bandwidth lost, %")
v("sw_low_pct", 100 * SILENT.slowdowns["low"], "{:.0f}", "step slowdown, low, %")
v("sw_high_pct", 100 * SILENT.slowdowns["high"], "{:.0f}", "step slowdown, high, %")
v("sw_days", m(SILENT.run_duration, "day"), "{:.0f}", "run, days")
v("sw_gpuh", SILENT.device_hours, "{:,.0f}", "GPU-hours in the run")
v("sw_lost_low", lost["low"], "{:,.0f}", "GPU-hours lost, low")
v("sw_lost_high", lost["high"], "{:,.0f}", "GPU-hours lost, high")
v("sw_lost_high_round", round(lost["high"], -3), "{:,.0f}", "the same, to the thousand")
v("sw_rate_usd", rate, "{:.0f}", "fleet GPU-hour, $")
v("sw_cost_low", cost["low"], "{:,.0f}", "lost, low, $")
v("sw_cost_high", cost["high"], "{:,.0f}", "lost, high, $")
v("sw_cost_low_book", round(round(lost["low"], -3) * rate, -3), "{:,.0f}", "chapter's rounding, low, $")
v("sw_cost_high_book", round(round(lost["high"], -3) * rate, -3), "{:,.0f}", "chapter's rounding, high, $")
v("sw_no_overlap_pct", 100 * SILENT.no_overlap_slowdown, "{:.1f}", "no-overlap slowdown, %")
v("links_n", LINKS.n_components, "{:,.0f}", "fabric links")
v("links_p_pct", 100 * LINKS.probability, "{:.1f}", "chance one link is degraded, %")
v("links_expected", LINKS.expected_events, "{:.0f}", "degraded links expected")
v("links_any_pct", 100 * LINKS.p_any, "{:.0f}", "chance of at least one, %")

print("== Diagnosis (half-rate link scenario)")
v("diag_slow_pct", 100 * DIAG.slowdown, "{:.0f}", "step slowdown, %")
v("diag_day", DIAG.days_since_launch, "{:.0f}", "day of the run")
v("diag_before_s", m(DIAG.step_before, "second"), "{:.1f}", "step before, s")
v("diag_after_s", m(DIAG.step_after, "second"), "{:.1f}", "step after, s")
v("diag_util_before", 100 * DIAG.utilization_before, "{:.0f}", "utilization before, %")
v("diag_util_after", 100 * DIAG.utilization_after, "{:.0f}", "utilization after, %")
v("diag_expected_gbs", m(DIAG.expected_bandwidth, "GB/s"), "{:.0f}", "healthy pair, GB/s")
v("diag_measured_gbs", m(DIAG.measured_bandwidth, "GB/s"), "{:.0f}", "degraded pair, GB/s")

print("== Sharing a NIC (SR-IOV scenario)")
v("vf_count", SRIOV.n_virtual_functions, "{:.0f}", "virtual functions")
v("vf_gbps", m(SRIOV.slice_rate, "Gbps"), "{:.0f}", "rate cap per function, Gb/s")
v("nic_gbps", m(SRIOV.nic_bandwidth, "Gbps"), "{:.0f}", "NIC line rate, Gb/s")
v("burst_gb", m(SRIOV.burst_size, "GB"), "{:.0f}", "neighbor's checkpoint, GB")
v("burst_s", m(SRIOV.burst_time, "second"), "{:.0f}", "port full for, s")

if __name__ == "__main__":
    write(__file__)
