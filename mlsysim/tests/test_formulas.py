"""
Unit tests for mlsysim.physics — known-answer tests for every formula.

Each test uses hand-computed expected values and pytest.approx for
floating-point comparisons.
"""

import math
import pytest
import pint

from mlsysim.physics import (
    _ensure_unit,
    calc_network_latency_ms,
    calc_alpha_beta_crossover,
    calc_point_to_point_time,
    calc_ring_tree_crossover_size,
    calc_double_binary_tree_allreduce_time,
    calc_ring_allreduce_data_factor,
    calc_ring_collective_data_factor,
    ring_allreduce_data_factor_latex,
    calc_ring_allreduce_latency_steps,
    calc_ring_allreduce_latency_time,
    calc_oversubscription_effect,
    calc_bisection_bandwidth,
    calc_hop_latency,
    dTime,
    calc_amdahls_speedup,
    calc_strong_scaling_speedup,
    calc_bottleneck,
    model_memory,
    calc_ring_allreduce_time,
    calc_tree_allreduce_time,
    calc_all_to_all_time,
    calc_transformer_training_flops,
    calc_activation_memory,
    calc_hierarchical_allreduce_time,
    calc_young_daly_interval,
    calc_mtbf_cluster,
    calc_mtbf_node,
    calc_pipeline_bubble,
    calc_kv_cache_size,
    calc_paged_kv_cache_size,
    calc_queue_latency_mmc,
    calc_failure_probability,
    calc_effective_flops,
    calc_availability_stacked,
    calc_monthly_egress_cost,
    calc_fleet_tco,
    calc_population_stability_index,
    calc_two_proportion_sample_size,
    calc_constraint_propagation_factor,
)
from mlsysim.core.units import ureg, Q_, MB, GB

# ======================================================================
# _ensure_unit
# ======================================================================

class TestEnsureUnit:
    """Guard-rail helper for attaching and verifying Pint units."""

    def test_raw_number_gets_unit(self):
        result = _ensure_unit(42, ureg.meter, "test")
        assert result.magnitude == 42
        assert result.units == ureg.meter

    def test_correct_quantity_passes_through(self):
        q = Q_("10 meter")
        result = _ensure_unit(q, ureg.meter, "test")
        assert result == q

    def test_wrong_dimensionality_raises(self):
        q = Q_("10 second")
        with pytest.raises(pint.DimensionalityError):
            _ensure_unit(q, ureg.meter, "test")

    def test_non_numeric_raises_type_error(self):
        with pytest.raises(TypeError):
            _ensure_unit("hello", ureg.meter, "test")

# ======================================================================
# calc_network_latency_ms
# ======================================================================

class TestNetworkLatency:
    """Round-trip latency based on speed of light in fiber."""

    def test_1000km_round_trip(self):
        # 1000 km one-way, fiber speed = 200,000 km/s
        # RTT = 2 * 1000 / 200_000 = 0.01 s = 10 ms
        result = calc_network_latency_ms(1000)
        assert result == pytest.approx(10.0, rel=1e-6)

    def test_zero_distance(self):
        result = calc_network_latency_ms(0)
        assert result == pytest.approx(0.0)

# ======================================================================
# dTime
# ======================================================================

class TestDTime:
    """Core training time: T = OPs / (N * Peak * eta)."""

    def test_units_cancel_to_seconds(self):
        total_ops = Q_("1e18 flop")
        n_devices = 8
        peak = Q_("312e12 flop/s")
        eta = 0.5
        result = dTime(total_ops, n_devices, peak, eta)
        # 1e18 / (8 * 312e12 * 0.5) = 1e18 / 1.248e15 ≈ 801.28 s
        assert result.units == ureg.second
        assert result.magnitude == pytest.approx(1e18 / (8 * 312e12 * 0.5), rel=1e-4)

# ======================================================================
# calc_amdahls_speedup
# ======================================================================

class TestAmdahlsSpeedup:
    """Amdahl's law: S = 1 / ((1-p) + p/s)."""

    def test_classic_case(self):
        # p=0.9, s=10 => 1 / (0.1 + 0.09) = 1 / 0.19 ≈ 5.2632
        result = calc_amdahls_speedup(0.9, 10)
        assert result == pytest.approx(5.2632, rel=1e-3)

    def test_fully_parallelizable(self):
        # p=1.0, s=10 => speedup = 10
        result = calc_amdahls_speedup(1.0, 10)
        assert result == pytest.approx(10.0)

    def test_no_parallel_portion(self):
        # p=0.0 => speedup = 1.0 regardless of s
        result = calc_amdahls_speedup(0.0, 1000)
        assert result == pytest.approx(1.0)

# ======================================================================
# calc_bottleneck
# ======================================================================

class TestBottleneck:
    """Roofline bottleneck analysis."""

    def test_compute_bound(self):
        # High ops, low model bytes => compute-bound
        ops = Q_("1e15 flop")
        model_bytes = Q_("100 megabyte")
        device_flops = Q_("312e12 flop/s")
        device_bw = Q_("2e12 byte/s")
        result = calc_bottleneck(ops, model_bytes, device_flops, device_bw)
        assert result["bottleneck"] == "Compute"

    def test_memory_bound(self):
        # Low ops, large model => memory-bound
        ops = Q_("1e9 flop")
        model_bytes = Q_("10 gigabyte")
        device_flops = Q_("312e12 flop/s")
        device_bw = Q_("2e12 byte/s")
        result = calc_bottleneck(ops, model_bytes, device_flops, device_bw)
        assert result["bottleneck"] == "Memory"

    def test_intensity_normalizes_scaled_units(self):
        # 1 TFLOP / 1 GB = 1000 FLOP/byte, not 1 by raw Pint magnitude.
        result = calc_bottleneck(
            Q_("1 TFLOP"),
            Q_("1 GB"),
            Q_("1 TFLOP/s"),
            Q_("1 GB/s"),
        )
        assert result["intensity"] == pytest.approx(1000.0)

# ======================================================================
# model_memory
# ======================================================================

class TestModelMemory:
    """Model memory = params * bytes_per_param."""

    def test_resnet50_fp32(self):
        # 25.6M params * 4 bytes = 102.4 MB
        result = model_memory(25.6e6, 4, MB)
        assert result == pytest.approx(102.4, rel=1e-3)

    def test_with_pint_quantities(self):
        params = Q_("25.6e6 param")
        bpp = Q_("4 byte")
        result = model_memory(params, bpp, MB)
        assert result == pytest.approx(102.4, rel=1e-3)

    def test_gpt3_fp16(self):
        # 175e9 params * 2 bytes = 350e9 bytes = 350 GB
        result = model_memory(175e9, 2, GB)
        assert result == pytest.approx(350.0, rel=1e-3)

# ======================================================================
# calc_ring_allreduce_time
# ======================================================================

class TestRingAllreduce:
    """Ring AllReduce: T = 2(N-1)/N * M/beta + 2(N-1) * alpha."""

    def test_known_answer(self):
        # 1 GB on 8 GPUs at 50 GB/s + 500 ns latency
        M = Q_("1e9 byte")          # 1 GB
        N = 8
        beta = Q_("50e9 byte/s")    # 50 GB/s
        alpha = Q_("500 ns")

        # bw_term = 2*7/8 * 1e9/50e9 = 1.75 * 0.02 = 0.035 s
        # lat_term = 2*7 * 500e-9 = 7e-6 s
        # total ≈ 0.035007 s
        result = calc_ring_allreduce_time(M, N, beta, alpha)
        expected = 2 * 7 / 8 * (1e9 / 50e9) + 2 * 7 * 500e-9
        assert result.m_as(ureg.second) == pytest.approx(expected, rel=1e-4)


class TestRingAllreduceFactors:
    """Reusable ring AllReduce factors and latency helpers."""

    def test_data_factor(self):
        assert calc_ring_allreduce_data_factor(1) == 0.0
        assert calc_ring_allreduce_data_factor(8) == pytest.approx(14 / 8)
        assert calc_ring_collective_data_factor(8) == pytest.approx(7 / 8)
        assert ring_allreduce_data_factor_latex() == "2 \\times (N-1)/N"

    def test_latency_steps_and_time(self):
        assert calc_ring_allreduce_latency_steps(1) == 0
        assert calc_ring_allreduce_latency_steps(8) == 14
        assert calc_ring_allreduce_latency_time(8, Q_("500 ns")).to(ureg.second).magnitude == pytest.approx(7e-6)


class TestPointToPointTransfer:
    """Point-to-point transfer with fixed latency and bandwidth: T = α + n/β."""

    def test_known_answer(self):
        payload = Q_("4 KB")
        alpha = Q_("2 us")
        beta = Q_("10 GB/s")
        # pint KB is decimal (1000 bytes), matching the book's prose convention
        expected = alpha.to(ureg.second).magnitude + (4 * 1000) / (10e9)
        result = calc_point_to_point_time(payload, alpha, beta)
        assert result.to(ureg.second).magnitude == pytest.approx(expected, rel=1e-6)


class TestDoubleBinaryTreeAllreduce:
    """Double Binary Tree approximation with empirical latency and bandwidth factors."""

    def test_known_answer(self):
        M = Q_("1e9 byte")
        N = 256
        beta = Q_("25e9 byte/s")
        alpha = Q_("1 us")

        # log2(256) = 8
        # lat_term = 1.2 * 2 * 8 * alpha
        # bw_term = 1.05 * (2 * (N - 1)/N) * M/beta
        expected = 1.2 * (2 * 8 * 1e-6) + 1.05 * (2 * 255 / 256) * (1e9 / 25e9)
        result = calc_double_binary_tree_allreduce_time(M, N, beta, alpha).to(ureg.second)
        assert result.to(ureg.second).m_as(ureg.second) == pytest.approx(expected, rel=1e-4)

    def test_respects_factors(self):
        M = Q_("1e9 byte")
        N = 8
        beta = Q_("50e9 byte/s")
        alpha = Q_("2 us")

        expected = calc_double_binary_tree_allreduce_time(
            M, N, beta, alpha, latency_factor=1.0, bandwidth_factor=1.0
        ).to(ureg.second)
        boosted = calc_double_binary_tree_allreduce_time(
            M, N, beta, alpha, latency_factor=1.5, bandwidth_factor=2.0
        ).to(ureg.second)
        assert boosted > expected


class TestRingTreeCrossoverSize:
    """Crossover estimate M_crossover ≈ N * alpha * beta / log2(N)."""

    def test_known_answer(self):
        N = 64
        alpha = Q_("10 us")
        beta = Q_("10 GB/s")
        result = calc_ring_tree_crossover_size(N, alpha, beta)
        expected = N * 10e-6 * 10e9 / 6  # 64 * 10us * 10GB/s / log2(64)
        assert result.to(ureg.byte).magnitude == pytest.approx(expected, rel=1e-6)


class TestAlphaBetaCrossover:
    """α-β crossover point n* = α·β."""

    def test_known_answer(self):
        alpha = Q_("1.5 us")
        beta = Q_("50 GB/s")
        expected = 1.5e-6 * 50e9
        result = calc_alpha_beta_crossover(alpha, beta)
        assert result.m_as(ureg.byte) == pytest.approx(expected, rel=1e-4)


class TestOversubscriptionEffect:
    """Throughput and loss from oversubscription."""

    def test_2to1_and_30pct_comm(self):
        rel_throughput, loss = calc_oversubscription_effect(0.30, 2)
        assert rel_throughput == pytest.approx(1 / 1.3, rel=1e-6)
        assert loss == pytest.approx(1 - (1 / 1.3), rel=1e-6)


class TestBisectionBandwidth:
    """Bisection bandwidth = N_links * link_bw / oversub_ratio."""

    def test_non_oversubscribed(self):
        result = calc_bisection_bandwidth(512, Q_("50 GB/s"), oversubscription_ratio=1.0)
        assert result.m_as((ureg.byte / ureg.second)) == pytest.approx(512 * 50e9, rel=1e-6)

    def test_oversubscribed(self):
        result = calc_bisection_bandwidth(512, Q_("50 GB/s"), oversubscription_ratio=4.0)
        assert result.m_as(ureg.byte / ureg.second) == pytest.approx(512 * 50e9 / 4, rel=1e-6)


class TestHopLatency:
    """Total path latency from identical hop latencies."""

    def test_linear_scaling(self):
        result = calc_hop_latency(3, Q_("2 us"))
        assert result.to(ureg.microsecond).magnitude == pytest.approx(6.0, rel=1e-6)

# ======================================================================
# calc_tree_allreduce_time
# ======================================================================

class TestTreeAllreduce:
    """Tree AllReduce: T = 2*log2(N)*M/beta + 2*log2(N)*alpha."""

    def test_known_answer(self):
        M = Q_("1e9 byte")
        N = 8
        beta = Q_("50e9 byte/s")
        alpha = Q_("500 ns")

        # log2(8) = 3
        # bw_term = 2*3 * 1e9/50e9 = 6 * 0.02 = 0.12 s
        # lat_term = 2*3 * 500e-9 = 3e-6 s
        # total ≈ 0.120003 s
        result = calc_tree_allreduce_time(M, N, beta, alpha)
        expected = 2 * 3 * (1e9 / 50e9) + 2 * 3 * 500e-9
        assert result.m_as(ureg.second) == pytest.approx(expected, rel=1e-4)

    def test_tree_has_more_bandwidth_cost_than_ring(self):
        """For N=8, tree sends 6x M/beta vs ring's 1.75x — tree is worse for large messages."""
        M = Q_("1e9 byte")
        N = 8
        beta = Q_("50e9 byte/s")
        alpha = Q_("500 ns")
        ring = calc_ring_allreduce_time(M, N, beta, alpha)
        tree = calc_tree_allreduce_time(M, N, beta, alpha)
        assert tree > ring

# ======================================================================
# calc_all_to_all_time
# ======================================================================

class TestAllToAll:
    """All-to-All: T = (N-1)/N * M/beta + (N-1)*alpha."""

    def test_known_answer(self):
        M = Q_("1e9 byte")
        N = 8
        beta = Q_("50e9 byte/s")
        alpha = Q_("500 ns")

        # bw_term = 7/8 * 1e9/50e9 = 0.0175 s
        # lat_term = 7 * 500e-9 = 3.5e-6 s
        expected = 7 / 8 * (1e9 / 50e9) + 7 * 500e-9
        result = calc_all_to_all_time(M, N, beta, alpha)
        assert result.m_as(ureg.second) == pytest.approx(expected, rel=1e-4)

    def test_invalid_gpu_count_raises(self):
        with pytest.raises(ValueError, match="n_gpus"):
            calc_all_to_all_time(Q_("1e9 byte"), 0, Q_("50e9 byte/s"), Q_("500 ns"))

# ======================================================================
# calc_transformer_training_flops
# ======================================================================

class TestTransformerTrainingFlops:
    """6PD rule: T = 6 * P * D."""

    def test_gpt3(self):
        # GPT-3: 175B params, 300B tokens => 6 * 175e9 * 300e9 = 3.15e23
        P = Q_("175e9 param")
        D = Q_("300e9 count")
        result = calc_transformer_training_flops(P, D)
        assert result.m_as(ureg.flop) == pytest.approx(3.15e23, rel=1e-3)

# ======================================================================
# calc_activation_memory
# ======================================================================

class TestActivationMemory:
    """Activation memory, Korthikanti et al. (2023) Sec. 4.1 exact bounds.

    Constants are FP16 bytes (precision_bytes=2 is identity scale):
    none = 34*s*b*h + 5*a*s^2*b ; selective = 34*s*b*h ; full = 2*s*b*h.
    Re-pinned 2026-06-06 (the previous 34/10/2-times-bytes model double-
    counted FP16 width, and its selective=10 matched no published source).
    """

    def test_no_recompute(self):
        # 1 layer, S=1024, B=1, H=768, a=12 heads, FP16 default
        result = calc_activation_memory(1, 1024, 1, 768, n_heads=12, strategy="none")
        expected = 34 * 1024 * 1 * 768 + 5 * 12 * 1024 * 1024 * 1
        assert result.m_as(ureg.byte) == pytest.approx(expected, rel=1e-6)

    def test_no_recompute_requires_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            calc_activation_memory(1, 1024, 1, 768, strategy="none")

    def test_unknown_strategy_rejected(self):
        with pytest.raises(ValueError, match="strategy"):
            calc_activation_memory(1, 1024, 1, 768, strategy="checkpointing")

    def test_selective_recompute(self):
        result = calc_activation_memory(1, 1024, 1, 768, strategy="selective")
        assert result.m_as(ureg.byte) == pytest.approx(34 * 1024 * 1 * 768, rel=1e-6)

    def test_precision_scales_relative_to_fp16(self):
        fp16 = calc_activation_memory(1, 1024, 1, 768, precision_bytes=2)
        fp32 = calc_activation_memory(1, 1024, 1, 768, precision_bytes=4)
        assert fp32.m_as(ureg.byte) == pytest.approx(2 * fp16.m_as(ureg.byte))

    def test_full_recompute(self):
        result = calc_activation_memory(1, 1024, 1, 768, strategy="full")
        assert result.m_as(ureg.byte) == pytest.approx(2 * 1024 * 1 * 768, rel=1e-6)

    def test_scales_with_layers(self):
        single = calc_activation_memory(1, 1024, 1, 768, strategy="selective")
        twelve = calc_activation_memory(12, 1024, 1, 768, strategy="selective")
        assert twelve.m_as(ureg.byte) == pytest.approx(12 * single.m_as(ureg.byte), rel=1e-6)

# ======================================================================
# calc_hierarchical_allreduce_time
# ======================================================================

class TestHierarchicalAllreduce:
    """Hierarchical AllReduce: inter-node uses reduced message size."""

    def test_inter_node_uses_reduced_message(self):
        M = Q_("8e9 byte")          # 8 GB
        n_nodes = 4
        gpus_per_node = 8
        intra_bw = Q_("300e9 byte/s")   # NVLink
        inter_bw = Q_("25e9 byte/s")    # IB
        intra_lat = Q_("500 ns")
        inter_lat = Q_("5 us")

        result = calc_hierarchical_allreduce_time(
            M, n_nodes, gpus_per_node, intra_bw, inter_bw, intra_lat, inter_lat
        )
        # Result should be a valid positive time
        assert result.m_as(ureg.second) > 0

        # The inter-node message should be M / gpus_per_node = 1 GB,
        # not the full 8 GB. Verify by comparing against doing everything
        # with full message on inter-node (which would be much slower).
        slow_result = calc_hierarchical_allreduce_time(
            M, n_nodes, 1, intra_bw, inter_bw, intra_lat, inter_lat
        )
        # With gpus_per_node=1, there's no intra-node reduction benefit
        # and inter-node sends the full message. Should be slower.
        assert result.m_as(ureg.second) < slow_result.m_as(ureg.second)

    def test_known_answer_pins_all_constant_factors(self):
        """2026-06-10 audit: pin the closed form so a constant-factor
        regression cannot pass. The pre-2026-06-06 implementation inflated
        the intra term by (1 + 1/g) and would have passed the qualitative
        assertions above.

        T = 2*(g-1)/g * M/b_intra + 2*(g-1)*a_intra            (RS + AG)
          + 2*(n-1)/n * (M/g)/b_inter + 2*(n-1)*a_inter        (inter ring AR)
        """
        M = 8e9            # bytes
        n, g = 4, 8
        b_intra, b_inter = 300e9, 25e9   # byte/s
        a_intra, a_inter = 500e-9, 5e-6  # s

        expected = (
            2 * (g - 1) / g * M / b_intra + 2 * (g - 1) * a_intra
            + 2 * (n - 1) / n * (M / g) / b_inter + 2 * (n - 1) * a_inter
        )
        result = calc_hierarchical_allreduce_time(
            Q_(M, "byte"), n, g, Q_(b_intra, "byte/s"), Q_(b_inter, "byte/s"),
            Q_(a_intra, "s"), Q_(a_inter, "s"),
        )
        assert result.m_as(ureg.second) == pytest.approx(expected, rel=1e-6)

# ======================================================================
# calc_young_daly_interval
# ======================================================================

class TestYoungDalyInterval:
    """Optimal checkpoint interval: tau = sqrt(2 * delta * M)."""

    def test_known_answer(self):
        # delta = 60 s, MTBF = 50000 hours = 180,000,000 s
        # tau = sqrt(2 * 60 * 180_000_000) = sqrt(21_600_000_000) ≈ 146969.4 s
        delta = Q_("60 s")
        mtbf = Q_("50000 hour")
        result = calc_young_daly_interval(delta, mtbf)
        expected = math.sqrt(2 * 60 * 50000 * 3600)
        assert result.m_as(ureg.second) == pytest.approx(expected, rel=1e-4)

# ======================================================================
# calc_mtbf_cluster
# ======================================================================

class TestMTBFCluster:
    """Cluster MTBF = component MTBF / N."""

    def test_1000_components(self):
        # 50,000 hours / 1000 = 50 hours
        result = calc_mtbf_cluster(50000, 1000)
        assert result.m_as(ureg.hour) == pytest.approx(50.0, rel=1e-6)

    def test_correlation_factor(self):
        # With correlation_factor=0.5 => 25 hours
        result = calc_mtbf_cluster(50000, 1000, correlation_factor=0.5)
        assert result.m_as(ureg.hour) == pytest.approx(25.0, rel=1e-6)


class TestStrongScalingSpeedup:
    """Strong-scaling speedup for communication overhead fraction."""

    def test_no_communication_overhead(self):
        result = calc_strong_scaling_speedup(8, 0.0)
        assert result == pytest.approx(8.0)

    def test_full_communication_overhead(self):
        result = calc_strong_scaling_speedup(8, 1.0)
        assert result == pytest.approx(1.0)

    def test_known_answer(self):
        # N=4, r=0.1 => 4 / (1 + 3*0.1) = 3.0769
        result = calc_strong_scaling_speedup(4, 0.1)
        assert result == pytest.approx(4 / 1.3)

# ======================================================================
# calc_pipeline_bubble
# ======================================================================

class TestPipelineBubble:
    """Bubble fraction = (P-1) / (V*M + P-1)."""

    def test_classic_case(self):
        # P=4, M=8, V=1 => (4-1) / (1*8 + 4-1) = 3/11 ≈ 0.2727
        result = calc_pipeline_bubble(4, 8, v_stages=1)
        assert result == pytest.approx(3 / 11, rel=1e-4)

    def test_interleaved_reduces_bubble(self):
        # P=4, M=8, V=4 => (4-1) / (4*8 + 4-1) = 3/35 ≈ 0.0857
        result = calc_pipeline_bubble(4, 8, v_stages=4)
        assert result == pytest.approx(3 / 35, rel=1e-4)

    def test_more_microbatches_reduces_bubble(self):
        bubble_8 = calc_pipeline_bubble(4, 8)
        bubble_64 = calc_pipeline_bubble(4, 64)
        assert bubble_64 < bubble_8

# ======================================================================
# calc_kv_cache_size
# ======================================================================

class TestKVCacheSize:
    """KV cache = 2 * L * H * D * S * B * bytes."""

    def test_known_answer(self):
        # 2 * 32 * 32 * 128 * 2048 * 1 * 2 = 1,073,741,824 bytes = 1 GiB
        result = calc_kv_cache_size(
            n_layers=32, n_heads=32, head_dim=128,
            seq_len=2048, batch_size=1, bytes_per_elem=2,
        )
        expected = 2 * 32 * 32 * 128 * 2048 * 1 * 2
        assert result.m_as(ureg.byte) == pytest.approx(expected, rel=1e-6)

# ======================================================================
# calc_paged_kv_cache_size
# ======================================================================

class TestPagedKVCacheSize:
    """Paged KV cache with page-aligned sequences."""

    def test_exact_page_boundary(self):
        # seq_len=2048, page_size=16 => padded_seq_len=2048 (exact)
        # Same as non-paged for exact multiples
        size, frag = calc_paged_kv_cache_size(
            n_layers=32, n_heads=32, head_dim=128,
            seq_len=2048, batch_size=1, page_size_tokens=16,
        )
        expected = 2 * 32 * 32 * 128 * 2048 * 1 * 2
        assert size.m_as(ureg.byte) == pytest.approx(expected, rel=1e-6)
        assert frag == pytest.approx(0.0)

    def test_internal_fragmentation(self):
        # seq_len=2050, page_size=16 => padded=2064, frag = 14/2064
        size, frag = calc_paged_kv_cache_size(
            n_layers=32, n_heads=32, head_dim=128,
            seq_len=2050, batch_size=1, page_size_tokens=16,
        )
        assert frag == pytest.approx(14 / 2064, rel=1e-4)

# ======================================================================
# calc_queue_latency_mmc
# ======================================================================

class TestQueueLatencyMMC:
    """M/M/c queueing model for inference serving."""

    def test_stable_queue(self):
        # Low utilization: should have finite wait times
        rho, p50, p99 = calc_queue_latency_mmc(
            arrival_rate_hz=80, service_rate_hz=10, num_servers=10,
        )
        assert 0 < rho < 1
        assert p99.m_as(ureg.second) >= p50.m_as(ureg.second)

    def test_unstable_queue(self):
        # lambda >= c * mu => utilization = 1, infinite waits
        rho, p50, p99 = calc_queue_latency_mmc(
            arrival_rate_hz=100, service_rate_hz=10, num_servers=10,
        )
        assert rho == 1.0
        assert math.isinf(p50.magnitude)

    def test_large_server_count(self):
        # c=500 should not overflow (log-space Erlang C)
        rho, p50, p99 = calc_queue_latency_mmc(
            arrival_rate_hz=400, service_rate_hz=1, num_servers=500,
        )
        assert 0 < rho < 1
        assert p99.m_as(ureg.second) >= 0

    def test_known_answer_erlang_c(self):
        """2026-06-10 audit: pin Erlang-C numerics so a dropped (1-rho), a
        factor-of-2 in the wait rate c*mu*(1-rho), or an off-by-one in the
        series range cannot pass. Reference values from an independent
        direct-summation Erlang-C implementation (Kleinrock Vol. 1):
        lambda=1.5, mu=1, c=2 -> C(2, 1.5) = 0.642857, wait rate = 0.5/s,
        p50 = -ln(0.5/C)/0.5 = 0.502629 s, p99 = -ln(0.01/C)/0.5 = 8.326675 s.
        """
        rho, p50, p99 = calc_queue_latency_mmc(
            arrival_rate_hz=1.5, service_rate_hz=1.0, num_servers=2,
        )
        assert rho == pytest.approx(0.75, rel=1e-9)
        assert p50.m_as(ureg.second) == pytest.approx(0.502629, rel=1e-4)
        assert p99.m_as(ureg.second) == pytest.approx(8.326675, rel=1e-4)

    def test_mm1_reduction(self):
        """M/M/1 sanity: Erlang-C collapses to C(1, rho) = rho. At rho = 0.5,
        p50 never queues (p_wait = 0.5 is not > 0.5... boundary -> 0) and
        p99 = -ln(0.01/0.5)/(mu - lambda) = 7.824046 s. Also pins the -0.0
        normalization at the exact p_wait == quantile boundary."""
        rho, p50, p99 = calc_queue_latency_mmc(
            arrival_rate_hz=0.5, service_rate_hz=1.0, num_servers=1,
        )
        assert rho == pytest.approx(0.5, rel=1e-9)
        assert p50.m_as(ureg.second) == 0.0
        assert not math.copysign(1.0, p50.magnitude) < 0  # no -0.0
        assert p99.m_as(ureg.second) == pytest.approx(7.824046, rel=1e-4)

# ======================================================================
# calc_failure_probability
# ======================================================================

class TestFailureProbability:
    """P(fail) = 1 - exp(-T/MTBF)."""

    def test_job_equals_mtbf(self):
        # When T = MTBF => P = 1 - exp(-1) ≈ 0.6321
        result = calc_failure_probability(
            mtbf=Q_("100 hour"), job_duration=Q_("100 hour"),
        )
        assert result == pytest.approx(1 - math.exp(-1), rel=1e-4)

    def test_raw_numbers(self):
        result = calc_failure_probability(mtbf=100, job_duration=100)
        assert result == pytest.approx(1 - math.exp(-1), rel=1e-4)

    def test_mixed_types_raises(self):
        with pytest.raises(TypeError):
            calc_failure_probability(mtbf=Q_("100 hour"), job_duration=100)

# ======================================================================
# calc_effective_flops
# ======================================================================

class TestEffectiveFlops:
    """Effective = Peak * MFU * scaling_eff * goodput."""

    def test_simple(self):
        peak = Q_("1e15 flop/s")
        result = calc_effective_flops(peak, mfu=0.5, scaling_eff=0.9, goodput_ratio=0.95)
        expected = 1e15 * 0.5 * 0.9 * 0.95
        assert result.m_as(ureg.flop / ureg.second) == pytest.approx(expected, rel=1e-6)

# ======================================================================
# calc_availability_stacked
# ======================================================================

class TestAvailabilityStacked:
    """A_system = 1 - (1 - A)^k."""

    def test_three_nines_triple_replicated(self):
        # 1 - (1-0.999)^3 = 1 - 1e-9 = 0.999999999
        result = calc_availability_stacked(0.999, 3)
        assert result == pytest.approx(0.999999999, rel=1e-6)

    def test_single_replica(self):
        result = calc_availability_stacked(0.99, 1)
        assert result == pytest.approx(0.99)

# ======================================================================
# calc_monthly_egress_cost
# ======================================================================

class TestMonthlyEgressCost:
    """Monthly egress cost = bandwidth * 30 days * $/GB rate."""

    def test_known_answer_raw(self):
        # 1 MB/s * 30 days = 2,592 GB; at $0.09/GB = $233.28
        result = calc_monthly_egress_cost(1e6, 0.09)
        assert result == pytest.approx(233.28, rel=1e-4)

    def test_known_answer_quantity(self):
        result = calc_monthly_egress_cost(
            Q_("1 MB/s"), Q_("0.09 dollar/GB")
        )
        assert result == pytest.approx(233.28, rel=1e-4)

    def test_zero_bandwidth_is_free(self):
        result = calc_monthly_egress_cost(0, 0.09)
        assert result == pytest.approx(0.0)

    def test_scales_linearly_with_bandwidth(self):
        cost_1x = calc_monthly_egress_cost(1e6, 0.09)
        cost_10x = calc_monthly_egress_cost(10e6, 0.09)
        assert cost_10x == pytest.approx(cost_1x * 10, rel=1e-6)

# ======================================================================
# calc_fleet_tco
# ======================================================================

class TestFleetTCO:
    """TCO = capex + opex (energy cost over N years)."""

    def test_known_answer(self):
        # 2026-06-10 audit: pins the 365-day (8,760 h) year convention at
        # rel 1e-6 — the old rel=1e-3 tolerance could not distinguish the
        # 365-day year from Pint's Julian 365.25-day year (a $0.60 gap on
        # this case), and the old comment overstated opex 10x ($8,760; the
        # correct value is $876).
        # 10 units x $1000 = $10,000 capex
        # 100 W * 10 units * 8,760 h * $0.10/kWh = 876 kWh... -> $876 opex
        # total = $10,876
        result = calc_fleet_tco(1000, 100, 10, 1, 0.10)
        capex = 10 * 1000
        energy_kwh = 0.1 * 10 * (1 * 365 * 24)
        opex = energy_kwh * 0.10
        assert opex == pytest.approx(876.0, rel=1e-9)
        assert result == pytest.approx(capex + opex, rel=1e-6)

    def test_zero_quantity(self):
        result = calc_fleet_tco(1000, 500, 0, 3, 0.10)
        assert result == pytest.approx(0.0)

    def test_scales_linearly_with_quantity(self):
        cost_1 = calc_fleet_tco(1000, 500, 1, 3, 0.10)
        cost_100 = calc_fleet_tco(1000, 500, 100, 3, 0.10)
        assert cost_100 == pytest.approx(cost_1 * 100, rel=1e-6)

# ======================================================================
# calc_mtbf_node
# ======================================================================

class TestMTBFNode:
    """Node MTBF from heterogeneous components: 1/MTBF = sum(n_i/MTBF_i)."""

    def test_single_component_type(self):
        # 1 GPU with 10,000 h MTBF => node MTBF = 10,000 h
        result = calc_mtbf_node(10_000, 1, 1e9, 0, 1e9, 0)
        assert result.m_as(ureg.hour) == pytest.approx(10_000.0, rel=1e-4)

    def test_two_identical_gpus_halves_mtbf(self):
        # 2 GPUs each at 10,000 h => failure rate doubles => node MTBF = 5,000 h
        result = calc_mtbf_node(10_000, 2, 1e9, 0, 1e9, 0)
        assert result.m_as(ureg.hour) == pytest.approx(5_000.0, rel=1e-4)

    def test_mixed_components(self):
        # GPU: 10,000 h x4, NIC: 50,000 h x2, PSU: 20,000 h x2
        # rate = 4/10000 + 2/50000 + 2/20000 = 0.0004 + 0.00004 + 0.0001 = 0.00054
        # MTBF = 1/0.00054 ≈ 1851.85 h
        result = calc_mtbf_node(10_000, 4, 50_000, 2, 20_000, 2)
        expected = 1 / (4/10_000 + 2/50_000 + 2/20_000)
        assert result.m_as(ureg.hour) == pytest.approx(expected, rel=1e-4)

def test_calc_binomial_failure_probability():
    from mlsysim.physics.reliability import calc_binomial_failure_probability
    assert calc_binomial_failure_probability(1.0, 100) == 1.0
    assert calc_binomial_failure_probability(0.0, 100) == 0.0
    assert abs(calc_binomial_failure_probability(0.5, 2) - 0.75) < 1e-9

# ======================================================================
# statistics.py — calc_population_stability_index,
# calc_two_proportion_sample_size, calc_constraint_propagation_factor
# (2026-06-10 audit: these previously had ZERO package tests; only book
# render-time check() guards pinned them.)
# ======================================================================

class TestPopulationStabilityIndex:
    """PSI = sum (a_i - e_i) * ln(a_i / e_i) — the Jeffreys divergence."""

    def test_known_answer(self):
        # Independent hand computation:
        # (0.3-0.5)ln(0.3/0.5) + (0.7-0.5)ln(0.7/0.5) = 0.1694596
        psi = calc_population_stability_index([0.5, 0.5], [0.3, 0.7])
        assert psi == pytest.approx(0.16945957207744072, rel=1e-9)

    def test_identical_distributions_zero(self):
        assert calc_population_stability_index([0.25] * 4, [0.25] * 4) == pytest.approx(0.0, abs=1e-12)

    def test_symmetry(self):
        # Jeffreys divergence is symmetric in its arguments
        a, b = [0.1, 0.4, 0.5], [0.2, 0.3, 0.5]
        assert calc_population_stability_index(a, b) == pytest.approx(
            calc_population_stability_index(b, a), rel=1e-12)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            calc_population_stability_index([0.5, 0.5], [1.0])

    def test_empty_bin_uses_epsilon_floor(self):
        # Must not raise on a zero bin
        psi = calc_population_stability_index([0.5, 0.5], [0.0, 1.0])
        assert math.isfinite(psi) and psi > 0


class TestTwoProportionSampleSize:
    """n = 2 (z_a + z_b)^2 p(1-p) / delta^2 (equal-variance approximation)."""

    def test_known_answer_book_scenario(self):
        # p=0.05, delta=0.001, z=1.96/0.84 -> 744,800 exactly
        n = calc_two_proportion_sample_size(0.05, 0.001)
        assert n == pytest.approx(744_800, rel=1e-9)

    def test_quadruples_when_lift_halves(self):
        n1 = calc_two_proportion_sample_size(0.05, 0.002)
        n2 = calc_two_proportion_sample_size(0.05, 0.001)
        assert n2 == pytest.approx(4 * n1, rel=1e-9)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            calc_two_proportion_sample_size(0.05, 0.0)
        with pytest.raises(ValueError):
            calc_two_proportion_sample_size(0.0, 0.001)
        with pytest.raises(ValueError):
            calc_two_proportion_sample_size(1.0, 0.001)


class TestConstraintPropagationFactor:
    """factor = base^(stage_to - stage_from) (Boehm cost-of-delay)."""

    def test_three_stage_gap(self):
        assert calc_constraint_propagation_factor(0, 3) == 8

    def test_equal_stages_is_one(self):
        assert calc_constraint_propagation_factor(2, 2) == 1

    def test_backwards_raises(self):
        with pytest.raises(ValueError):
            calc_constraint_propagation_factor(3, 1)


# ======================================================================
# 2026-06-10 audit: input-contract regressions for reliability functions
# ======================================================================

class TestReliabilityInputContracts:
    def test_failure_probability_rejects_negative_duration(self):
        # Pre-audit: returned -0.105 for a negative duration, violating the
        # documented [0, 1) contract.
        with pytest.raises(ValueError):
            calc_failure_probability(100, -10)

    def test_mtbf_cluster_rejects_zero_components(self):
        # Pre-audit: bare ZeroDivisionError
        with pytest.raises(ValueError):
            calc_mtbf_cluster(50_000, 0)

    def test_young_daly_warns_out_of_regime(self):
        # delta >= 2*MTBF: Young form returns tau < delta (impossible);
        # must warn so a LEGO cell cannot silently render it.
        with pytest.warns(UserWarning, match="out of regime"):
            calc_young_daly_interval(1000, 10)

    def test_young_daly_no_warning_in_valid_regime(self):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            calc_young_daly_interval(60, 50_000 * 3600)

    def test_negative_message_rejected_by_collectives(self):
        # Parity with calc_point_to_point_time (which always validated)
        for fn in (calc_ring_allreduce_time, calc_tree_allreduce_time):
            with pytest.raises(ValueError):
                fn(Q_("-1 GB"), 8, Q_("100 GB/s"), Q_("5 us"))
        with pytest.raises(ValueError):
            calc_all_to_all_time(Q_("-1 GB"), 8, Q_("100 GB/s"), Q_("5 us"))

# ======================================================================
# 2026-06-10 audit: pins for previously-untested decode FLOPs and
# checkpoint size (a 2x YAML edit to DecodeConstant or a default-bytes
# change passed the whole suite before these).
# ======================================================================

class TestTransformerDecodeFlops:
    """2P rule: forward decode ~ 2 FLOPs/param/token (Kaplan 2020 Sec 2.1)."""

    def test_known_answer_8b(self):
        from mlsysim.physics import calc_transformer_decode_flops
        result = calc_transformer_decode_flops(Q_("8e9 param"))
        assert result.m_as(ureg.flop) == pytest.approx(1.6e10, rel=1e-9)

    def test_scales_linearly_with_tokens(self):
        from mlsysim.physics import calc_transformer_decode_flops
        one = calc_transformer_decode_flops(Q_("8e9 param"), n_tokens=1)
        hundred = calc_transformer_decode_flops(Q_("8e9 param"), n_tokens=100)
        assert hundred.m_as(ureg.flop) == pytest.approx(100 * one.m_as(ureg.flop), rel=1e-9)

    def test_decode_constant_pinned(self):
        # Pins Literature.Chinchilla.DecodeConstant = 2.0 itself: an edit of
        # chinchilla.yaml silently changing rendered numbers must fail here.
        from mlsysim.literature.registry import Literature
        assert float(Literature.Chinchilla.DecodeConstant) == 2.0


class TestCheckpointSize:
    """checkpoint = params * bytes_per_param; default 14 B/param is the
    ZeRO mixed-precision convention (2 fp16 w + 4 master + 4 m + 4 v)."""

    def test_default_adam_convention(self):
        from mlsysim.physics import calc_checkpoint_size
        result = calc_checkpoint_size(Q_("8e9 param"))
        assert result.m_as(ureg.GB) == pytest.approx(112.0, rel=1e-9)

    def test_explicit_bytes_per_param(self):
        from mlsysim.physics import calc_checkpoint_size
        result = calc_checkpoint_size(8e9, bytes_per_param=4)
        assert result.m_as(ureg.GB) == pytest.approx(32.0, rel=1e-9)

    def test_calibration_constants_pinned(self):
        from mlsysim.engine import calibration as cal
        assert cal.CHECKPOINT_BYTES_PER_PARAM_ADAM == 14
        assert cal.CHECKPOINT_BYTES_PER_PARAM_SGD == 4
        assert cal.TRAINING_OPTIMIZER_BYTES_ADAM == 12.0


class TestEffectiveFlopsValidation:
    def test_rejects_out_of_range_ratios(self):
        peak = Q_("1e15 flop/s")
        with pytest.raises(ValueError):
            calc_effective_flops(peak, mfu=1.5, scaling_eff=0.9, goodput_ratio=0.95)
        with pytest.raises(ValueError):
            calc_effective_flops(peak, mfu=-0.5, scaling_eff=0.9, goodput_ratio=0.95)

    def test_dtime_rejects_eta_above_one(self):
        with pytest.raises(ValueError):
            dTime(Q_("1e18 flop"), 8, Q_("312e12 flop/s"), 1.5)


class TestKvCacheValidation:
    def test_rejects_negative_dimensions(self):
        with pytest.raises(ValueError):
            calc_kv_cache_size(n_layers=-1, n_heads=8, head_dim=128, seq_len=2048, batch_size=1)
        with pytest.raises(ValueError):
            calc_kv_cache_size(n_layers=32, n_heads=8, head_dim=128, seq_len=-5, batch_size=1)
