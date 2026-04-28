"""
Unit tests for mlsysim.core.formulas — known-answer tests for every formula.

Each test uses hand-computed expected values and pytest.approx for
floating-point comparisons.
"""

import math
import pytest
import pint

from mlsysim.core.formulas import (
    _ensure_unit,
    calc_network_latency_ms,
    dTime,
    calc_amdahls_speedup,
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
)
from mlsysim.core.constants import ureg, Q_, MB, GB


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
    """Activation memory with Korthikanti coefficients (34/10/2)."""

    def test_no_recompute(self):
        # 1 layer, S=1024, B=1, H=768, precision_bytes=1 (default)
        # 34 * 1024 * 1 * 768 * 1 = 26,738,688 bytes per layer
        result = calc_activation_memory(1, 1024, 1, 768, strategy="none")
        assert result.m_as(ureg.byte) == pytest.approx(34 * 1024 * 1 * 768, rel=1e-6)

    def test_selective_recompute(self):
        result = calc_activation_memory(1, 1024, 1, 768, strategy="selective")
        assert result.m_as(ureg.byte) == pytest.approx(10 * 1024 * 1 * 768, rel=1e-6)

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
        # 10 units x $1000 = $10,000 capex
        # 100W * 10 * 1yr * $0.10/kWh = 100*10*8760*0.10/1000 = $8,760 opex
        # total = $18,760
        result = calc_fleet_tco(1000, 100, 10, 1, 0.10)
        capex = 10 * 1000
        energy_kwh = 0.1 * 10 * (1 * 365.25 * 24)
        opex = energy_kwh * 0.10
        assert result == pytest.approx(capex + opex, rel=1e-3)

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
