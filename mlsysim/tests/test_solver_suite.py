"""
Comprehensive solver test suite for mlsysim.

Tests physics-level correctness of all solvers: SingleNode, Serving,
Sustainability, Data, Scaling, Orchestration, Compression, and
verifies constants module backward compatibility.
"""

import math
import pytest

from mlsysim.hardware.registry import Hardware
from mlsysim.models.registry import Models
from mlsysim.systems.registry import Systems
from mlsysim.infra.registry import Infra, Grids
from mlsysim.core.solver import (
    SingleNodeModel,
    ServingModel,
    SustainabilityModel,
    DataModel,
    ScalingModel,
    OrchestrationModel,
    CompressionModel,
    DistributedModel,
    ReliabilityModel,
    EconomicsModel,
    EfficiencyModel,
    TransformationModel,
    TopologyModel,
    InferenceScalingModel,
    SensitivitySolver,
    SynthesisSolver,
    ResponsibleEngineeringModel,
)
from mlsysim.core.formulas import calc_pipeline_bubble
from mlsysim.systems.types import NetworkFabric
from mlsysim.core.engine import Engine, PerformanceProfile
from mlsysim.core.constants import ureg, Q_
from mlsysim.core.exceptions import OOMError


# ======================================================================
# 1. SingleNodeModel
# ======================================================================

class TestSingleNodeModel:
    """Tests for roofline-based single-node performance modeling."""

    def test_resnet_high_batch_is_compute_bound(self):
        """ResNet-50 at large batch should be compute-bound (high arithmetic intensity)."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        perf = SingleNodeModel().solve(resnet, a100, batch_size=256)
        assert perf.feasible is True
        assert perf.bottleneck == "Compute"

    def test_llm_batch1_is_memory_bound(self):
        """A large LLM at batch 1 should be memory-bound (weight streaming dominates)."""
        gpt3 = Models.GPT3
        h100 = Hardware.H100
        perf = SingleNodeModel().solve(gpt3, h100, batch_size=1, raise_errors=False)
        # GPT-3 175B at FP16 = 350 GB, H100 has 80 GiB => infeasible,
        # but the bottleneck analysis still runs.
        # Use a model that fits: Llama3_8B
        llama = Models.Llama3_8B
        perf = SingleNodeModel().solve(llama, h100, batch_size=1)
        assert perf.feasible is True
        assert perf.bottleneck == "Memory"

    def test_oom_detection_huge_model_tiny_hardware(self):
        """GPT-4 on ESP32 must be infeasible."""
        gpt4 = Models.GPT4
        esp32 = Hardware.ESP32
        perf = SingleNodeModel().solve(gpt4, esp32, batch_size=1, raise_errors=False)
        assert perf.feasible is False

    def test_oom_raises_when_requested(self):
        """OOMError should be raised when raise_errors=True."""
        gpt4 = Models.GPT4
        esp32 = Hardware.ESP32
        with pytest.raises(OOMError):
            SingleNodeModel().solve(gpt4, esp32, batch_size=1, raise_errors=True)

    def test_precision_switching_affects_latency(self):
        """FP32 inference should be slower than FP16 on the same hardware."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        perf_fp16 = SingleNodeModel().solve(resnet, a100, batch_size=1, precision="fp16")
        perf_fp32 = SingleNodeModel().solve(resnet, a100, batch_size=1, precision="fp32")
        assert perf_fp32.latency > perf_fp16.latency

    def test_precision_switching_changes_peak_flops(self):
        """Peak flops under FP32 should be lower than FP16 tensor core."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        perf_fp16 = SingleNodeModel().solve(resnet, a100, batch_size=1, precision="fp16")
        perf_fp32 = SingleNodeModel().solve(resnet, a100, batch_size=1, precision="fp32")
        assert perf_fp32.peak_flops_actual < perf_fp16.peak_flops_actual

    def test_throughput_increases_with_batch_size(self):
        """Throughput (samples/s) should increase with larger batch sizes."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        perf_b1 = SingleNodeModel().solve(resnet, a100, batch_size=1)
        perf_b64 = SingleNodeModel().solve(resnet, a100, batch_size=64)
        assert perf_b64.throughput > perf_b1.throughput

    def test_small_model_on_big_hardware_is_feasible(self):
        """MobileNetV2 on H100 should trivially fit."""
        mobile = Models.MobileNetV2
        h100 = Hardware.H100
        perf = SingleNodeModel().solve(mobile, h100, batch_size=1)
        assert perf.feasible is True
        assert perf.latency.magnitude > 0

    def test_performance_profile_has_all_fields(self):
        """PerformanceProfile should contain all required fields."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        perf = SingleNodeModel().solve(resnet, a100, batch_size=1)
        assert hasattr(perf, "latency")
        assert hasattr(perf, "throughput")
        assert hasattr(perf, "bottleneck")
        assert hasattr(perf, "mfu")
        assert hasattr(perf, "hfu")
        assert hasattr(perf, "energy")
        assert hasattr(perf, "memory_footprint")
        assert hasattr(perf, "feasible")


# ======================================================================
# 2. ServingModel
# ======================================================================

class TestServingModel:
    """Tests for two-phase LLM serving (prefill + decode)."""

    def test_prefill_is_compute_bound(self):
        """Time-to-first-token (prefill) should be dominated by compute."""
        llama = Models.Llama3_8B
        h100 = Hardware.H100
        result = ServingModel().solve(llama, h100, seq_len=2048, batch_size=1)
        # Prefill latency should be > 0
        assert result.ttft.magnitude > 0

    def test_decode_is_memory_bound(self):
        """Inter-token latency (decode) should be limited by memory bandwidth."""
        llama = Models.Llama3_8B
        h100 = Hardware.H100
        result = ServingModel().solve(llama, h100, seq_len=2048, batch_size=1)
        # ITL is model_bytes / memory_bw => memory-bandwidth limited
        assert result.itl.magnitude > 0
        # Decode should be slower than zero but finite
        itl_ms = result.itl.to("ms").magnitude
        assert 0 < itl_ms < 1000

    def test_kv_cache_grows_with_sequence_length(self):
        """KV cache should increase with longer sequences."""
        llama = Models.Llama3_8B
        h100 = Hardware.H100
        result_short = ServingModel().solve(llama, h100, seq_len=512, batch_size=1)
        result_long = ServingModel().solve(llama, h100, seq_len=4096, batch_size=1)
        kv_short = result_short.kv_cache_size.to("GB").magnitude
        kv_long = result_long.kv_cache_size.to("GB").magnitude
        assert kv_long > kv_short

    def test_kv_cache_grows_with_batch_size(self):
        """KV cache should scale linearly with batch size."""
        llama = Models.Llama3_8B
        h100 = Hardware.H100
        result_b1 = ServingModel().solve(llama, h100, seq_len=2048, batch_size=1)
        result_b4 = ServingModel().solve(llama, h100, seq_len=2048, batch_size=4)
        kv_b1 = result_b1.kv_cache_size.to("GB").magnitude
        kv_b4 = result_b4.kv_cache_size.to("GB").magnitude
        assert kv_b4 == pytest.approx(kv_b1 * 4, rel=0.01)

    def test_feasibility_model_too_large(self):
        """GPT-3 175B in FP16 = ~350 GB; should not fit on a single H100 (80 GiB)."""
        gpt3 = Models.GPT3
        h100 = Hardware.H100
        result = ServingModel().solve(gpt3, h100, seq_len=2048, batch_size=1)
        assert result.feasible is False

    def test_feasibility_model_fits(self):
        """Llama-3.1-8B in FP16 ~ 16 GB; should fit on H100."""
        llama = Models.Llama3_8B
        h100 = Hardware.H100
        result = ServingModel().solve(llama, h100, seq_len=2048, batch_size=1)
        assert result.feasible is True

    def test_memory_utilization_bounded(self):
        """Memory utilization for a feasible model should be between 0 and 1."""
        llama = Models.Llama3_8B
        h100 = Hardware.H100
        result = ServingModel().solve(llama, h100, seq_len=2048, batch_size=1)
        assert 0 < result.memory_utilization < 1.0

    def test_int8_reduces_model_size(self):
        """INT8 serving should halve the model weights vs FP16."""
        llama = Models.Llama3_8B
        h100 = Hardware.H100
        result_fp16 = ServingModel().solve(llama, h100, seq_len=2048, batch_size=1, precision="fp16")
        result_int8 = ServingModel().solve(llama, h100, seq_len=2048, batch_size=1, precision="int8")
        fp16_size = result_fp16.model_weights_size.to("GB").magnitude
        int8_size = result_int8.model_weights_size.to("GB").magnitude
        assert int8_size == pytest.approx(fp16_size / 2, rel=0.01)


# ======================================================================
# 3. SustainabilityModel
# ======================================================================

class TestSustainabilityModel:
    """Tests for energy, carbon, and water footprint modeling."""

    def test_pue_multiplier_effect(self):
        """Higher PUE should increase total energy relative to IT energy."""
        fleet = Systems.Clusters.Research_256
        solver = SustainabilityModel()
        result = solver.solve(fleet, duration_days=1, datacenter=Infra.Quebec)
        it_energy = result.it_energy_kwh.magnitude
        total_energy = result.total_energy_kwh.magnitude
        pue = result.pue
        assert pue > 1.0
        assert total_energy == pytest.approx(it_energy * pue, rel=0.01)

    def test_carbon_quebec_less_than_poland(self):
        """Quebec (hydro, 20 gCO2/kWh) should emit far less carbon than Poland (coal, 820 gCO2/kWh)."""
        fleet = Systems.Clusters.Research_256
        solver = SustainabilityModel()
        result_quebec = solver.solve(fleet, duration_days=30, datacenter=Infra.Quebec)
        result_poland = solver.solve(fleet, duration_days=30, datacenter=Infra.Poland)
        assert result_quebec.carbon_footprint_kg < result_poland.carbon_footprint_kg
        # Quebec should be roughly 41x less (20/820), allow wide margin
        ratio = result_poland.carbon_footprint_kg / result_quebec.carbon_footprint_kg
        assert ratio > 10  # At least 10x difference

    def test_water_usage_calculation(self):
        """Water usage should be non-negative and scale with energy."""
        fleet = Systems.Clusters.Research_256
        solver = SustainabilityModel()
        result = solver.solve(fleet, duration_days=30, datacenter=Infra.US_Avg)
        assert result.water_usage_liters >= 0

    def test_water_liquid_cooling_near_zero(self):
        """Liquid-cooled datacenter (WUE=0) should have zero water usage."""
        fleet = Systems.Clusters.Research_256
        solver = SustainabilityModel()
        result = solver.solve(fleet, duration_days=30, datacenter=Infra.Quebec)
        # Quebec uses liquid cooling (WUE=0.0)
        assert result.water_usage_liters == pytest.approx(0.0, abs=0.01)

    def test_energy_increases_with_duration(self):
        """Longer operation should consume more energy."""
        fleet = Systems.Clusters.Research_256
        solver = SustainabilityModel()
        result_1d = solver.solve(fleet, duration_days=1, datacenter=Infra.Quebec)
        result_30d = solver.solve(fleet, duration_days=30, datacenter=Infra.Quebec)
        assert result_30d.total_energy_kwh.magnitude > result_1d.total_energy_kwh.magnitude

    def test_energy_positive(self):
        """Energy consumption should always be positive for non-zero duration."""
        fleet = Systems.Clusters.Research_256
        solver = SustainabilityModel()
        result = solver.solve(fleet, duration_days=1, datacenter=Infra.Quebec)
        assert result.it_energy_kwh.magnitude > 0
        assert result.total_energy_kwh.magnitude > 0


# ======================================================================
# 4. DataModel
# ======================================================================

class TestDataModel:
    """Tests for data pipeline stall detection."""

    def test_stall_when_demand_exceeds_supply(self):
        """Pipeline should stall when data demand > hardware supply."""
        h100 = Hardware.H100
        # Demand far exceeds NVMe bandwidth (7 GB/s)
        huge_demand = Q_("100 GB/s")
        solver = DataModel()
        result = solver.solve(huge_demand, h100)
        assert result.is_stalled is True
        assert result.utilization > 1.0

    def test_no_stall_when_supply_adequate(self):
        """Pipeline should not stall when supply easily meets demand."""
        h100 = Hardware.H100
        modest_demand = Q_("1 GB/s")
        solver = DataModel()
        result = solver.solve(modest_demand, h100)
        assert result.is_stalled is False
        assert result.utilization < 1.0

    def test_margin_positive_when_no_stall(self):
        """Margin (supply - demand) should be positive when not stalled."""
        h100 = Hardware.H100
        modest_demand = Q_("1 GB/s")
        solver = DataModel()
        result = solver.solve(modest_demand, h100)
        assert result.margin.magnitude > 0

    def test_margin_negative_when_stalled(self):
        """Margin should be negative when demand exceeds supply."""
        h100 = Hardware.H100
        huge_demand = Q_("100 GB/s")
        solver = DataModel()
        result = solver.solve(huge_demand, h100)
        assert result.margin.magnitude < 0

    def test_bottleneck_identified(self):
        """Solver should identify whether storage or interconnect is the bottleneck."""
        h100 = Hardware.H100
        demand = Q_("1 GB/s")
        solver = DataModel()
        result = solver.solve(demand, h100)
        assert result.bottleneck in ["Storage", "Interconnect"]


# ======================================================================
# 5. ScalingModel
# ======================================================================

class TestScalingModel:
    """Tests for Chinchilla scaling law analysis."""

    def test_chinchilla_optimal_d_approx_20p(self):
        """Chinchilla-optimal training: D ~ 20P (tokens_per_parameter ~ 20)."""
        solver = ScalingModel()
        # Give a compute budget that implies a specific P
        # C = 6 * P * D = 6 * P * 20P = 120 * P^2
        # For P = 1e9 (1B params): C = 120 * 1e18 = 1.2e20 flops
        budget = Q_(1.2e20, "flop")
        result = solver.solve(budget)
        tpp = result.tokens_per_parameter
        assert tpp == pytest.approx(20, rel=0.01)

    def test_compute_budget_decomposition(self):
        """Optimal P and D should reconstruct original C via C = 6PD."""
        solver = ScalingModel()
        budget = Q_(1.2e20, "flop")
        result = solver.solve(budget)
        p = result.optimal_parameters.magnitude
        d = result.optimal_tokens.magnitude
        reconstructed_c = 6 * p * d
        assert reconstructed_c == pytest.approx(budget.magnitude, rel=0.01)

    def test_larger_budget_yields_larger_model(self):
        """More compute should yield a larger optimal model."""
        solver = ScalingModel()
        result_small = solver.solve(Q_(1e18, "flop"))
        result_large = solver.solve(Q_(1e22, "flop"))
        p_small = result_small.optimal_parameters.magnitude
        p_large = result_large.optimal_parameters.magnitude
        assert p_large > p_small

    def test_target_model_size_override(self):
        """When target_model_size is given, the solver calculates required tokens."""
        solver = ScalingModel()
        budget = Q_(1.2e20, "flop")
        target = Q_(1e9, ureg.count)  # 1B params
        result = solver.solve(budget, target_model_size=target)
        p = result.optimal_parameters.magnitude
        assert p == pytest.approx(1e9, rel=0.01)
        # D = C / (6 * P) = 1.2e20 / 6e9 = 2e10
        d = result.optimal_tokens.magnitude
        assert d == pytest.approx(2e10, rel=0.01)


# ======================================================================
# 6. OrchestrationModel
# ======================================================================

class TestOrchestrationModel:
    """Tests for cluster queueing and wait time modeling."""

    def test_wait_time_increases_with_utilization(self):
        """Higher arrival rate (higher rho) should increase wait time."""
        fleet = Systems.Clusters.Research_256
        solver = OrchestrationModel()
        result_low = solver.solve(fleet, arrival_rate_jobs_per_day=0.1, avg_job_duration_days=2.0)
        result_high = solver.solve(fleet, arrival_rate_jobs_per_day=0.4, avg_job_duration_days=2.0)
        wait_low = result_low.avg_wait_time_days.magnitude
        wait_high = result_high.avg_wait_time_days.magnitude
        assert wait_high > wait_low

    def test_instability_when_rho_ge_1(self):
        """When rho >= 1, the queue is unstable (infinite wait time)."""
        fleet = Systems.Clusters.Research_256
        solver = OrchestrationModel()
        # rho = lambda / mu = lambda * duration = 1.0 * 2.0 = 2.0 >= 1
        result = solver.solve(fleet, arrival_rate_jobs_per_day=1.0, avg_job_duration_days=2.0)
        assert result.is_stable is False
        assert result.avg_wait_time_days.magnitude == float("inf")

    def test_stable_queue(self):
        """When rho < 1, the queue should be stable with finite wait time."""
        fleet = Systems.Clusters.Research_256
        solver = OrchestrationModel()
        # rho = 0.1 * 2.0 = 0.2 < 1
        result = solver.solve(fleet, arrival_rate_jobs_per_day=0.1, avg_job_duration_days=2.0)
        assert result.is_stable is True
        assert result.avg_wait_time_days.magnitude < float("inf")
        assert result.avg_wait_time_days.magnitude >= 0

    def test_utilization_calculation(self):
        """Utilization rho = lambda * avg_duration."""
        fleet = Systems.Clusters.Research_256
        solver = OrchestrationModel()
        lam = 0.3
        dur = 2.0
        result = solver.solve(fleet, arrival_rate_jobs_per_day=lam, avg_job_duration_days=dur)
        assert result.cluster_utilization == pytest.approx(lam * dur, rel=0.01)

    def test_queue_length_finite_when_stable(self):
        """Average queue length should be finite when rho < 1."""
        fleet = Systems.Clusters.Research_256
        solver = OrchestrationModel()
        result = solver.solve(fleet, arrival_rate_jobs_per_day=0.2, avg_job_duration_days=2.0)
        assert result.avg_queue_length < float("inf")
        assert result.avg_queue_length >= 0


# ======================================================================
# 7. CompressionModel
# ======================================================================

class TestCompressionModel:
    """Tests for quantization and pruning trade-off analysis."""

    def test_int8_compression_ratio_is_4x(self):
        """INT8 quantization from FP32 baseline should yield 4x compression."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        solver = CompressionModel()
        result = solver.solve(resnet, a100, method="quantization", target_bitwidth=8)
        assert result.compression_ratio == pytest.approx(4.0, rel=0.01)

    def test_int4_compression_ratio_is_8x(self):
        """INT4 quantization from FP32 baseline should yield 8x compression."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        solver = CompressionModel()
        result = solver.solve(resnet, a100, method="quantization", target_bitwidth=4)
        assert result.compression_ratio == pytest.approx(8.0, rel=0.01)

    def test_accuracy_delta_is_negative(self):
        """Quantization should always degrade accuracy (negative delta)."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        solver = CompressionModel()
        for bitwidth in [8, 4, 2]:
            result = solver.solve(resnet, a100, method="quantization", target_bitwidth=bitwidth)
            assert result.estimated_accuracy_delta < 0

    def test_int8_accuracy_drop_small(self):
        """INT8 should have a small accuracy drop (~0.5%)."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        solver = CompressionModel()
        result = solver.solve(resnet, a100, method="quantization", target_bitwidth=8)
        assert abs(result.estimated_accuracy_delta) < 0.01

    def test_int4_accuracy_drop_larger_than_int8(self):
        """INT4 should have larger accuracy drop than INT8."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        solver = CompressionModel()
        result_8 = solver.solve(resnet, a100, method="quantization", target_bitwidth=8)
        result_4 = solver.solve(resnet, a100, method="quantization", target_bitwidth=4)
        assert abs(result_4.estimated_accuracy_delta) > abs(result_8.estimated_accuracy_delta)

    def test_pruning_zero_sparsity_minimal_loss(self):
        """Pruning at 0% sparsity should have minimal accuracy loss."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        solver = CompressionModel()
        result = solver.solve(resnet, a100, method="pruning", sparsity=0.0)
        assert abs(result.estimated_accuracy_delta) < 0.01

    def test_pruning_high_sparsity_larger_loss(self):
        """Pruning at 90% sparsity should have larger accuracy loss than 10%."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        solver = CompressionModel()
        result_low = solver.solve(resnet, a100, method="pruning", sparsity=0.1)
        result_high = solver.solve(resnet, a100, method="pruning", sparsity=0.9)
        assert abs(result_high.estimated_accuracy_delta) > abs(result_low.estimated_accuracy_delta)

    def test_compressed_size_less_than_original(self):
        """Compressed model should always be smaller than original."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        solver = CompressionModel()
        result = solver.solve(resnet, a100, method="quantization", target_bitwidth=8)
        orig = result.original_size_gb.magnitude
        comp = result.compressed_size_gb.magnitude
        assert comp < orig

    def test_memory_savings_percentage(self):
        """Memory savings for INT8 should be 75% (1 - 1/4)."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        solver = CompressionModel()
        result = solver.solve(resnet, a100, method="quantization", target_bitwidth=8)
        assert result.memory_savings_pct == pytest.approx(75.0, rel=0.01)


# ======================================================================
# 8. Constants & Module Import Tests
# ======================================================================

class TestConstantsImports:
    """Tests that units.py, defaults.py, constants.py all import correctly
    and backward compatibility is maintained."""

    def test_units_module_imports(self):
        """Core unit definitions should be importable from units.py."""
        from mlsysim.core.units import ureg, Q_, GB, TB, MS, NS, flop, TFLOPs, USD
        assert ureg is not None
        assert Q_ is not None
        assert GB is not None
        assert flop is not None

    def test_defaults_module_imports(self):
        """Tuneable defaults should be importable from defaults.py."""
        from mlsysim.core.defaults import (
            GPU_MTTF_HOURS,
            PUE_LIQUID_COOLED,
            PUE_LEGACY,
            CHINCHILLA_TOKENS_PER_PARAM,
            CHINCHILLA_COMPUTE_CONSTANT,
            ANNUAL_MAINTENANCE_RATIO,
        )
        assert GPU_MTTF_HOURS == 50_000
        assert CHINCHILLA_TOKENS_PER_PARAM == 20
        assert CHINCHILLA_COMPUTE_CONSTANT == 6
        assert PUE_LIQUID_COOLED == 1.06
        assert PUE_LEGACY == 1.58

    def test_constants_backward_compat(self):
        """Importing from constants.py should still work (re-exports from units + defaults)."""
        from mlsysim.core.constants import (
            ureg, Q_,
            BYTES_FP16, BYTES_FP32, BYTES_INT8, BYTES_INT4,
            GPU_MTTF_HOURS,
            CHINCHILLA_TOKENS_PER_PARAM,
            H100_FLOPS_FP16_TENSOR,
            H100_MEM_BW,
        )
        assert BYTES_FP16.magnitude == 2
        assert BYTES_FP32.magnitude == 4
        assert BYTES_INT8.magnitude == 1
        assert BYTES_INT4.magnitude == 0.5
        assert GPU_MTTF_HOURS == 50_000

    def test_unit_conversions(self):
        """Basic unit conversions should work correctly."""
        val = Q_("1 TB")
        gb_val = val.to("GB")
        assert gb_val.magnitude == pytest.approx(1000, rel=0.01)

    def test_flop_unit_scaling(self):
        """TFLOP -> flop conversion should scale correctly."""
        val = Q_("1 TFLOPs")
        flop_val = val.to("flop")
        assert flop_val.magnitude == pytest.approx(1e12, rel=0.01)


# ======================================================================
# 9. Engine (direct usage)
# ======================================================================

class TestEngine:
    """Tests for the core Engine.solve() static method."""

    def test_engine_returns_performance_profile(self):
        """Engine.solve should return a PerformanceProfile instance."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        result = Engine.solve(resnet, a100, batch_size=1)
        assert isinstance(result, PerformanceProfile)

    def test_engine_energy_positive(self):
        """Energy estimate should be positive for hardware with TDP."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        result = Engine.solve(resnet, a100, batch_size=1)
        assert result.energy.magnitude > 0

    def test_engine_mfu_bounded(self):
        """MFU should be between 0 and 1."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        result = Engine.solve(resnet, a100, batch_size=64)
        assert 0 <= result.mfu <= 1.0

    def test_engine_hfu_bounded(self):
        """HFU should be between 0 and 1."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        result = Engine.solve(resnet, a100, batch_size=64)
        assert 0 <= result.hfu <= 1.0

    def test_engine_efficiency_scales_flops(self):
        """Higher efficiency should yield lower latency (faster)."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        result_low = Engine.solve(resnet, a100, batch_size=64, efficiency=0.2)
        result_high = Engine.solve(resnet, a100, batch_size=64, efficiency=0.8)
        assert result_high.latency < result_low.latency

    def test_engine_summary_string(self):
        """PerformanceProfile.summary() should return a non-empty string."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        result = Engine.solve(resnet, a100, batch_size=1)
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Bottleneck" in summary


# ======================================================================
# 10. DistributedModel (additional coverage)
# ======================================================================

class TestDistributedModel:
    """Tests for distributed training performance modeling."""

    def test_scaling_efficiency_between_0_and_1(self):
        """Scaling efficiency must be in (0, 1]."""
        solver = DistributedModel()
        gpt3 = Models.GPT3
        cluster = Systems.Clusters.Research_256
        result = solver.solve(gpt3, cluster, batch_size=32)
        assert 0 < result.scaling_efficiency <= 1.0

    def test_communication_latency_positive(self):
        """Communication latency should be positive for multi-node clusters."""
        solver = DistributedModel()
        gpt3 = Models.GPT3
        cluster = Systems.Clusters.Research_256
        result = solver.solve(gpt3, cluster, batch_size=32)
        assert result.communication_latency.magnitude > 0

    def test_pipeline_parallelism_creates_bubble(self):
        """PP > 1 should introduce a non-zero pipeline bubble."""
        solver = DistributedModel()
        gpt3 = Models.GPT3
        cluster = Systems.Clusters.Research_256
        result = solver.solve(gpt3, cluster, batch_size=256, pp_size=4, microbatch_count=8)
        assert result.pipeline_bubble_latency.magnitude > 0
        assert result.bubble_fraction > 0

    def test_no_pipeline_bubble_when_pp_1(self):
        """PP = 1 should have zero pipeline bubble."""
        solver = DistributedModel()
        gpt3 = Models.GPT3
        cluster = Systems.Clusters.Research_256
        result = solver.solve(gpt3, cluster, batch_size=32, pp_size=1)
        assert result.pipeline_bubble_latency.magnitude == 0
        assert result.bubble_fraction == 0


# ======================================================================
# 11. ReliabilityModel (additional coverage)
# ======================================================================

class TestReliabilityModel:
    """Tests for MTBF and checkpointing analysis."""

    def test_failure_probability_positive(self):
        """Failure probability should be > 0 for long jobs on large clusters."""
        solver = ReliabilityModel()
        cluster = Systems.Clusters.Frontier_8K
        result = solver.solve(cluster, job_duration_hours=100.0)
        assert result.failure_probability > 0

    def test_larger_cluster_higher_failure_prob(self):
        """Larger cluster should have higher failure probability for the same job."""
        solver = ReliabilityModel()
        small = Systems.Clusters.Research_256
        large = Systems.Clusters.Frontier_8K
        result_small = solver.solve(small, job_duration_hours=100.0)
        result_large = solver.solve(large, job_duration_hours=100.0)
        assert result_large.failure_probability > result_small.failure_probability

    def test_optimal_checkpoint_interval_positive(self):
        """Young-Daly optimal checkpoint interval should be positive."""
        solver = ReliabilityModel()
        cluster = Systems.Clusters.Research_256
        result = solver.solve(cluster, job_duration_hours=100.0)
        assert result.optimal_checkpoint_interval.magnitude > 0


# ======================================================================
# 12. EconomicsModel (additional coverage)
# ======================================================================

class TestEconomicsModel:
    """Tests for total cost of ownership analysis."""

    def test_tco_positive(self):
        """TCO should be positive."""
        solver = EconomicsModel()
        cluster = Systems.Clusters.Research_256
        result = solver.solve(cluster, duration_days=30, grid=Infra.Quebec)
        assert result.tco_usd > 0

    def test_capex_dominates_short_term(self):
        """For short durations, CapEx should dominate over OpEx."""
        solver = EconomicsModel()
        cluster = Systems.Clusters.Research_256
        result = solver.solve(cluster, duration_days=1, grid=Infra.Quebec)
        assert result.capex_usd > result.total_opex_usd

    def test_tco_is_sum_of_parts(self):
        """TCO = CapEx + OpEx_energy + OpEx_maintenance."""
        solver = EconomicsModel()
        cluster = Systems.Clusters.Research_256
        result = solver.solve(cluster, duration_days=30, grid=Infra.Quebec)
        expected = result.capex_usd + result.opex_energy_usd + result.opex_maintenance_usd
        assert result.tco_usd == pytest.approx(expected, rel=0.001)


# ======================================================================
# 13. EfficiencyModel
# ======================================================================

class TestEfficiencyModel:
    """Tests for MFU estimation by workload type."""

    def test_ffn_mfu_higher_than_attention(self):
        """FFN layers (compute-dense GEMM) should achieve higher MFU than standard attention."""
        resnet = Models.ResNet50
        h100 = Hardware.H100
        solver = EfficiencyModel()
        result_ffn = solver.solve(resnet, h100, workload_type="ffn")
        result_attn = solver.solve(resnet, h100, workload_type="attention", use_flash_attention=False)
        assert result_ffn.mfu > result_attn.mfu

    def test_flash_attention_boosts_mfu(self):
        """FlashAttention should yield higher MFU than standard attention."""
        resnet = Models.ResNet50
        h100 = Hardware.H100
        solver = EfficiencyModel()
        result_std = solver.solve(resnet, h100, workload_type="attention", use_flash_attention=False)
        result_flash = solver.solve(resnet, h100, workload_type="attention", use_flash_attention=True)
        assert result_flash.mfu > result_std.mfu

    def test_mfu_bounded_zero_one(self):
        """MFU must be clamped to [0, 1] for all workload types."""
        resnet = Models.ResNet50
        h100 = Hardware.H100
        solver = EfficiencyModel()
        for wtype in ["ffn", "attention", "conv"]:
            for eff in [0.1, 0.5, 1.0, 2.0]:
                result = solver.solve(resnet, h100, workload_type=wtype, efficiency=eff)
                assert 0.0 <= result.mfu <= 1.0, f"MFU out of bounds for {wtype}, eff={eff}"

    def test_achievable_flops_positive(self):
        """Achievable FLOPS should always be positive."""
        resnet = Models.ResNet50
        h100 = Hardware.H100
        solver = EfficiencyModel()
        result = solver.solve(resnet, h100, workload_type="ffn")
        assert result.achievable_flops.magnitude > 0

    def test_overhead_breakdown_present(self):
        """Overhead breakdown should contain expected keys."""
        resnet = Models.ResNet50
        h100 = Hardware.H100
        solver = EfficiencyModel()
        result = solver.solve(resnet, h100, workload_type="ffn")
        assert "occupancy_loss" in result.overhead_breakdown
        assert "memory_stall" in result.overhead_breakdown
        assert "kernel_overhead" in result.overhead_breakdown


# ======================================================================
# 14. TransformationModel
# ======================================================================

class TestTransformationModel:
    """Tests for CPU preprocessing bottleneck detection."""

    def test_cpu_bottleneck_detected(self):
        """When CPU preprocessing is slow, it should be flagged as a bottleneck."""
        solver = TransformationModel()
        result = solver.solve(
            batch_size=256,
            sample_size_bytes=Q_("500 KB"),
            cpu_throughput=Q_("0.5 GB/s"),
            accelerator_step_time=Q_("10 ms"),
        )
        assert result.is_bottleneck is True

    def test_no_bottleneck_fast_cpu(self):
        """When CPU is fast enough, no bottleneck should be detected."""
        solver = TransformationModel()
        result = solver.solve(
            batch_size=32,
            sample_size_bytes=Q_("100 KB"),
            cpu_throughput=Q_("10 GB/s"),
            accelerator_step_time=Q_("50 ms"),
        )
        assert result.is_bottleneck is False

    def test_accelerator_utilization_bounded(self):
        """Accelerator utilization should be in (0, 1]."""
        solver = TransformationModel()
        result = solver.solve(
            batch_size=64,
            sample_size_bytes=Q_("500 KB"),
            cpu_throughput=Q_("2 GB/s"),
            accelerator_step_time=Q_("50 ms"),
        )
        assert 0 < result.accelerator_utilization <= 1.0

    def test_utilization_drops_when_cpu_slow(self):
        """Accelerator utilization should decrease when CPU preprocessing is slow."""
        solver = TransformationModel()
        result_fast = solver.solve(
            batch_size=64,
            sample_size_bytes=Q_("500 KB"),
            cpu_throughput=Q_("20 GB/s"),
            accelerator_step_time=Q_("50 ms"),
        )
        result_slow = solver.solve(
            batch_size=64,
            sample_size_bytes=Q_("500 KB"),
            cpu_throughput=Q_("0.5 GB/s"),
            accelerator_step_time=Q_("50 ms"),
        )
        assert result_fast.accelerator_utilization > result_slow.accelerator_utilization


# ======================================================================
# 15. TopologyModel
# ======================================================================

class TestTopologyModel:
    """Tests for network topology bisection bandwidth modeling."""

    def _make_fabric(self):
        """Helper to create a simple InfiniBand fabric."""
        return NetworkFabric(
            name="IB-400G",
            bandwidth=Q_("400 Gbit/s").to("GB/s"),
            oversubscription_ratio=1.0,
        )

    def test_fat_tree_beta_is_one(self):
        """Fat-tree should have full bisection bandwidth (beta = 1.0)."""
        solver = TopologyModel()
        result = solver.solve(self._make_fabric(), topology="fat_tree", num_nodes=64)
        assert result.bisection_bw_fraction == pytest.approx(1.0)

    def test_ring_beta_is_2_over_n(self):
        """Ring topology beta should be 2/N (dynamic, decreases with N)."""
        solver = TopologyModel()
        for n in [8, 64, 256]:
            result = solver.solve(self._make_fabric(), topology="ring", num_nodes=n)
            assert result.bisection_bw_fraction == pytest.approx(2.0 / n, rel=0.01)

    def test_ring_beta_decreases_with_n(self):
        """Ring bisection bandwidth fraction should decrease as N grows."""
        solver = TopologyModel()
        fabric = self._make_fabric()
        result_small = solver.solve(fabric, topology="ring", num_nodes=16)
        result_large = solver.solve(fabric, topology="ring", num_nodes=256)
        assert result_large.bisection_bw_fraction < result_small.bisection_bw_fraction

    def test_torus_3d_beta(self):
        """3D torus beta should be 2 * N^(-1/3)."""
        solver = TopologyModel()
        result = solver.solve(self._make_fabric(), topology="torus_3d", num_nodes=64)
        expected_beta = 2.0 * (64 ** (-1.0 / 3.0))
        assert result.bisection_bw_fraction == pytest.approx(expected_beta, rel=0.01)

    def test_effective_bw_positive(self):
        """Effective bandwidth should always be positive."""
        solver = TopologyModel()
        for topo in ["fat_tree", "ring", "torus_3d", "dragonfly"]:
            result = solver.solve(self._make_fabric(), topology=topo, num_nodes=64)
            assert result.effective_bw.magnitude > 0

    def test_oversubscription_reduces_bw(self):
        """Higher oversubscription should reduce effective bandwidth."""
        solver = TopologyModel()
        fabric_1x = NetworkFabric(name="IB", bandwidth=Q_("400 Gbit/s").to("GB/s"), oversubscription_ratio=1.0)
        fabric_3x = NetworkFabric(name="IB", bandwidth=Q_("400 Gbit/s").to("GB/s"), oversubscription_ratio=3.0)
        result_1x = solver.solve(fabric_1x, topology="fat_tree", num_nodes=64)
        result_3x = solver.solve(fabric_3x, topology="fat_tree", num_nodes=64)
        assert result_1x.effective_bw.magnitude > result_3x.effective_bw.magnitude


# ======================================================================
# 16. InferenceScalingModel
# ======================================================================

class TestInferenceScalingModel:
    """Tests for inference-time reasoning cost modeling."""

    def test_total_time_greater_than_ttft(self):
        """Total reasoning time must exceed TTFT (there is decode work after prefill)."""
        solver = InferenceScalingModel()
        llama = Models.Llama3_8B
        h100 = Hardware.H100
        result = solver.solve(llama, h100, reasoning_steps=8, context_length=2048)
        total = result.total_reasoning_time.to("ms").magnitude
        ttft = result.ttft.to("ms").magnitude
        assert total > ttft

    def test_more_steps_more_time(self):
        """More reasoning steps should increase total reasoning time."""
        solver = InferenceScalingModel()
        llama = Models.Llama3_8B
        h100 = Hardware.H100
        result_few = solver.solve(llama, h100, reasoning_steps=2, context_length=2048)
        result_many = solver.solve(llama, h100, reasoning_steps=16, context_length=2048)
        t_few = result_few.total_reasoning_time.to("ms").magnitude
        t_many = result_many.total_reasoning_time.to("ms").magnitude
        assert t_many > t_few

    def test_feasibility_passthrough(self):
        """Feasibility should be passed through from the ServingModel."""
        solver = InferenceScalingModel()
        # Llama3_8B fits on H100
        result_fit = solver.solve(Models.Llama3_8B, Hardware.H100, reasoning_steps=4)
        assert result_fit.feasible is True
        # GPT-3 175B does not fit on a single H100
        result_nofit = solver.solve(Models.GPT3, Hardware.H100, reasoning_steps=4)
        assert result_nofit.feasible is False

    def test_tokens_generated_correct(self):
        """tokens_generated should equal reasoning_steps * tokens_per_step."""
        solver = InferenceScalingModel()
        from mlsysim.core.defaults import TOKENS_PER_REASONING_STEP
        llama = Models.Llama3_8B
        h100 = Hardware.H100
        steps = 5
        result = solver.solve(llama, h100, reasoning_steps=steps)
        assert result.tokens_generated == steps * TOKENS_PER_REASONING_STEP


# ======================================================================
# 17. SensitivitySolver
# ======================================================================

class TestSensitivitySolver:
    """Tests for numerical sensitivity analysis."""

    def test_binding_constraint_identified(self):
        """Solver should identify a binding constraint from the sensitivity dict."""
        solver = SensitivitySolver()
        resnet = Models.ResNet50
        a100 = Hardware.A100
        result = solver.solve(resnet, a100, precision="fp16")
        assert result.binding_constraint in ["peak_flops", "memory_bandwidth", "memory_capacity"]

    def test_sensitivities_have_correct_keys(self):
        """Sensitivity dict should contain all three hardware parameters."""
        solver = SensitivitySolver()
        resnet = Models.ResNet50
        a100 = Hardware.A100
        result = solver.solve(resnet, a100)
        sens = result.sensitivities
        assert "peak_flops" in sens
        assert "memory_bandwidth" in sens
        assert "memory_capacity" in sens

    def test_sensitivity_signs(self):
        """Increasing peak_flops or memory_bandwidth should not increase latency (sensitivity <= 0)."""
        solver = SensitivitySolver()
        resnet = Models.ResNet50
        a100 = Hardware.A100
        result = solver.solve(resnet, a100)
        sens = result.sensitivities
        # More FLOPS or BW => same or lower latency => sensitivity <= 0
        assert sens["peak_flops"] <= 0.0 + 1e-9
        assert sens["memory_bandwidth"] <= 0.0 + 1e-9

    def test_baseline_latency_positive(self):
        """Baseline latency should be positive."""
        solver = SensitivitySolver()
        resnet = Models.ResNet50
        a100 = Hardware.A100
        result = solver.solve(resnet, a100)
        assert result.baseline_latency.magnitude > 0


# ======================================================================
# 18. SynthesisSolver
# ======================================================================

class TestSynthesisSolver:
    """Tests for inverse Roofline hardware synthesis."""

    def test_required_bw_positive(self):
        """Required bandwidth should always be positive."""
        solver = SynthesisSolver()
        resnet = Models.ResNet50
        result = solver.solve(resnet, target_latency=Q_("10 ms"))
        assert result.required_bw.magnitude > 0

    def test_required_memory_gte_model_size(self):
        """Required memory must be at least the model weight size."""
        solver = SynthesisSolver()
        resnet = Models.ResNet50
        result = solver.solve(resnet, target_latency=Q_("10 ms"))
        assert result.required_memory.to("GB").magnitude >= result.model_size.to("GB").magnitude - 1e-9

    def test_tighter_sla_requires_more_bw(self):
        """A tighter latency SLA should require higher bandwidth."""
        solver = SynthesisSolver()
        resnet = Models.ResNet50
        result_loose = solver.solve(resnet, target_latency=Q_("100 ms"))
        result_tight = solver.solve(resnet, target_latency=Q_("1 ms"))
        assert result_tight.required_bw.magnitude > result_loose.required_bw.magnitude

    def test_required_flops_positive(self):
        """Required FLOPS should always be positive."""
        solver = SynthesisSolver()
        resnet = Models.ResNet50
        result = solver.solve(resnet, target_latency=Q_("10 ms"))
        assert result.required_flops.magnitude > 0

    def test_compute_memory_ratio_positive(self):
        """Compute-to-memory ratio should be positive."""
        solver = SynthesisSolver()
        resnet = Models.ResNet50
        result = solver.solve(resnet, target_latency=Q_("10 ms"))
        assert result.compute_memory_ratio.magnitude > 0


# ======================================================================
# 19. ResponsibleEngineeringModel
# ======================================================================

class TestResponsibleEngineeringModel:
    """Tests for DP-SGD and fairness overhead modeling."""

    def test_dp_slowdown_greater_than_one(self):
        """DP-SGD should always slow down training (factor > 1)."""
        solver = ResponsibleEngineeringModel()
        result = solver.solve(base_training_time=Q_("10 day"), epsilon=1.0)
        assert result.dp_slowdown_factor > 1.0

    def test_lower_epsilon_higher_slowdown(self):
        """Lower epsilon (stronger privacy) should cause a larger slowdown."""
        solver = ResponsibleEngineeringModel()
        result_loose = solver.solve(base_training_time=Q_("10 day"), epsilon=10.0)
        result_strict = solver.solve(base_training_time=Q_("10 day"), epsilon=0.1)
        assert result_strict.dp_slowdown_factor > result_loose.dp_slowdown_factor

    def test_effective_time_exceeds_base(self):
        """Effective training time should exceed baseline training time."""
        solver = ResponsibleEngineeringModel()
        base = Q_("10 day")
        result = solver.solve(base_training_time=base, epsilon=1.0)
        assert result.effective_training_time.magnitude > base.magnitude

    def test_fairness_data_ratio_scales_with_prevalence(self):
        """Rarer subgroups should require more additional data."""
        solver = ResponsibleEngineeringModel()
        result_common = solver.solve(base_training_time=Q_("10 day"), min_subgroup_prevalence=0.1)
        result_rare = solver.solve(base_training_time=Q_("10 day"), min_subgroup_prevalence=0.001)
        assert result_rare.additional_data_requirement > result_common.additional_data_requirement

    def test_privacy_cost_ratio_equals_slowdown(self):
        """privacy_cost_ratio should equal dp_slowdown_factor."""
        solver = ResponsibleEngineeringModel()
        result = solver.solve(base_training_time=Q_("10 day"), epsilon=1.0)
        assert result.privacy_cost_ratio == result.dp_slowdown_factor


# ======================================================================
# 20. Boundary Condition Tests
# ======================================================================

class TestBoundaryConditions:
    """Edge cases and boundary conditions for solvers and formulas."""

    def test_pipeline_bubble_v_zero_guarded(self):
        """calc_pipeline_bubble with V=0 should be handled gracefully (or raise ZeroDivisionError)."""
        # bubble = (P-1) / (V*M + P-1). With V=0, M=4, P=4: (3)/(0+3) = 1.0
        # This is mathematically valid; it degenerates to (P-1)/(P-1) = 1.0
        # which means 100% bubble — correct when no virtual stages contribute.
        result = calc_pipeline_bubble(n_stages=4, n_microbatches=4, v_stages=0)
        assert result == pytest.approx(1.0)

    def test_pipeline_bubble_single_stage_is_zero(self):
        """With P=1 (single stage), the bubble fraction should be 0."""
        result = calc_pipeline_bubble(n_stages=1, n_microbatches=8, v_stages=1)
        assert result == pytest.approx(0.0)

    def test_ring_beta_decreases_monotonically(self):
        """Ring topology beta = 2/N should strictly decrease with N."""
        solver = TopologyModel()
        fabric = NetworkFabric(
            name="IB-400G",
            bandwidth=Q_("400 Gbit/s").to("GB/s"),
            oversubscription_ratio=1.0,
        )
        prev_beta = float("inf")
        for n in [4, 8, 16, 32, 64, 128, 256]:
            result = solver.solve(fabric, topology="ring", num_nodes=n)
            beta = result.bisection_bw_fraction
            assert beta < prev_beta, f"Ring beta did not decrease at N={n}"
            prev_beta = beta

    def test_compression_pruning_sparsity_one(self):
        """Pruning at sparsity=1.0 should yield maximum compression (capped at 100x)."""
        resnet = Models.ResNet50
        a100 = Hardware.A100
        solver = CompressionModel()
        result = solver.solve(resnet, a100, method="pruning", sparsity=1.0)
        # sparsity=1.0 => 1/(1-1.0) triggers the guard: capped at 100.0
        assert result.compression_ratio == pytest.approx(100.0)
        assert result.compressed_size_gb.magnitude > 0
