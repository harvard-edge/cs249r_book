"""Regression and invariant tests for the MLSysIM first-principles and unit-safety audit."""

from __future__ import annotations

import io
import json
import math
import ssl
import urllib.error
import warnings
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import mlsysim
from mlsysim import Hardware, Infrastructure, Models, Sensors, Systems
from mlsysim.cli.main import app
from mlsysim.cli.schemas import MlsysPlanSchema, _resolve_hardware
from mlsysim.core.optimization.exhaustive_backend import ExhaustiveBackend
from mlsysim.core.optimization.scipy_backend import ScipyBackend
from mlsysim.core.types import require_unit_family
from mlsysim.core.units import Q_, ureg
from mlsysim.engine import results as engine_results
from mlsysim.engine import solvers as engine_solvers
from mlsysim.fmt import (
    fmt_eur,
    fmt_inertia,
    fmt_memory,
    fmt_params,
    fmt_power_rate,
    fmt_pp,
    fmt_specific_heat,
    fmt_stiffness,
    fmt_tokens,
    fmt_torque,
    fmt_usd_range,
    fmt_volume,
)
from mlsysim.models.importer import import_hf_model
from mlsysim.models.types import SparseTransformerWorkload, TransformerWorkload
from mlsysim.physics import (
    calc_activation_memory,
    calc_binomial_tail,
    calc_bottleneck,
    calc_coolant_mass_flow,
    calc_fleet_tco,
    calc_float_spacing,
    calc_littles_law_occupancy,
    calc_poisson_tail,
    calc_queue_latency_mmc,
    calc_ring_attention_min_chunk,
    calc_sigmoid,
    calc_thermal_expansion,
)
from mlsysim.solvers import SamplingModel, StoppingBudgetModel


# ---------------------------------------------------------------------------
# 1. Core Physics & Mathematical First Principles
# ---------------------------------------------------------------------------


def test_ring_attention_min_chunk_byte_vs_bit_bandwidth():
    bw_bits = Q_(800, "Gbps")
    bw_bytes = Q_(100, "GB/s")
    c_bits = calc_ring_attention_min_chunk(
        element_bytes=2,
        sustained_flops=Q_(400, "TFLOPs/s"),
        link_bandwidth=bw_bits,
    )
    c_bytes = calc_ring_attention_min_chunk(
        element_bytes=2,
        sustained_flops=Q_(400, "TFLOPs/s"),
        link_bandwidth=bw_bytes,
    )
    assert math.isclose(c_bits, c_bytes, rel_tol=1e-12)
    # Exact check: s * R / (2 * B) = 2 * 400e12 / (2 * 100e9) = 4000.0 tokens
    assert math.isclose(c_bytes, 4000.0, rel_tol=1e-9)


def test_thermal_rejects_offset_temperature_units():
    with pytest.raises(ValueError, match="kelvin|delta"):
        calc_coolant_mass_flow(Q_(1000, "W"), Q_(4184, "J/(kg*K)"), Q_(15, "degC"))
    with pytest.raises(ValueError, match="kelvin|delta"):
        calc_thermal_expansion(Q_(12e-6, "1/K"), Q_(0.5, "m"), Q_(25, "degC"))

    flow = calc_coolant_mass_flow(Q_(4184, "W"), Q_(4184, "J/(kg*K)"), Q_(10, "kelvin"))
    assert math.isclose(flow.m_as("kg/s"), 0.1, rel_tol=1e-9)


def test_activation_memory_preserves_single_byte_dimension():
    mem_int = calc_activation_memory(n_layers=32, seq_len=1024, batch_size=4, hidden_dim=4096, precision_bytes=2)
    mem_qty = calc_activation_memory(
        n_layers=32, seq_len=1024, batch_size=4, hidden_dim=4096, precision_bytes=Q_(2, "byte")
    )
    assert mem_int.units == ureg.byte
    assert mem_qty.units == ureg.byte
    assert math.isclose(mem_int.m_as("byte"), mem_qty.m_as("byte"), rel_tol=1e-12)


def test_fleet_tco_handles_non_year_duration_quantity():
    tco_half_yr = calc_fleet_tco(
        unit_cost=Q_(25_000, "USD"),
        power_w=Q_(1_000, "W"),
        quantity=4,
        years=0.5,
        kwh_price=0.10,
    )
    tco_days = calc_fleet_tco(
        unit_cost=Q_(25_000, "USD"),
        power_w=Q_(1_000, "W"),
        quantity=4,
        years=Q_(182.5, "day"),
        kwh_price=0.10,
    )
    val_half = tco_half_yr.m_as("USD") if hasattr(tco_half_yr, "m_as") else float(tco_half_yr)
    val_days = tco_days.m_as("USD") if hasattr(tco_days, "m_as") else float(tco_days)
    # 182.5 days == 0.5 * 365 days (~0.49966 Julian years), within 0.01% rather than 365x off
    assert math.isclose(val_half, val_days, rel_tol=2e-4)


def test_numerical_stability_sigmoid_spacing_binomial_and_poisson():
    assert calc_sigmoid(-1000.0) == 0.0
    assert calc_sigmoid(1000.0) == 1.0
    assert math.isclose(calc_sigmoid(0.0), 0.5, rel_tol=1e-12)

    assert math.isclose(calc_float_spacing(0.0, 8, 23), math.ldexp(1.0, -149), rel_tol=1e-12)
    assert math.isclose(calc_float_spacing(0.0, 5, 10), math.ldexp(1.0, -24), rel_tol=1e-12)
    assert math.isclose(calc_float_spacing(0.0, 8, 7), math.ldexp(1.0, -133), rel_tol=1e-12)
    with pytest.raises(ValueError):
        calc_float_spacing(float("nan"), 8, 23)
    with pytest.raises(ValueError):
        calc_float_spacing(float("inf"), 8, 23)

    # Exact small-n and overflow-free large-n binomial tail
    assert calc_binomial_tail(2, 0.5, 1) == 0.75
    large_tail = calc_binomial_tail(2000, 0.5, 1000)
    assert 0.5 < large_tail < 0.52

    # Regularized gamma handles events >= 745 without exp(-mean) underflow/overflow
    p_tail = calc_poisson_tail(800.0, 810)
    assert 0.30 < p_tail < 0.40


def test_bottleneck_and_littles_law_and_mmc_zero_arrival():
    res_bn = calc_bottleneck(
        ops=Q_(1e12, "flop"),
        model_bytes=Q_(1e9, "byte"),
        device_flops=Q_(float("inf"), "flop/s"),
        device_bw=Q_(1e12, "byte/s"),
    )
    assert math.isclose(res_bn["memory_ms"], 1.0, rel_tol=1e-9)
    assert res_bn["compute_ms"] == 0.0
    assert res_bn["bottleneck"] == "Memory"

    occ = calc_littles_law_occupancy(Q_(200, "1/s"), Q_(25, "ms"))
    occ_val = float(occ.to_base_units().magnitude if hasattr(occ, "magnitude") else occ)
    assert math.isclose(occ_val, 5.0, rel_tol=1e-9)

    rho0, p50_0, p99_0 = calc_queue_latency_mmc(0.0, 10.0, num_servers=4)
    assert rho0 == 0.0
    assert p50_0.m_as("s") == 0.0
    assert p99_0.m_as("s") == 0.0


def test_public_exports_complete_across_physics_and_engine():
    assert "calc_ring_attention_min_chunk" in mlsysim.physics.__all__
    assert "calc_safe_stopping_distance" in mlsysim.physics.robotics.__all__
    assert len(mlsysim.physics.__all__) == len(set(mlsysim.physics.__all__))
    assert len(engine_results.__all__) == len(set(engine_results.__all__))
    assert len(engine_solvers.__all__) == len(set(engine_solvers.__all__))


# ---------------------------------------------------------------------------
# 2. Engine Solvers
# ---------------------------------------------------------------------------


def test_stopping_budget_preserves_negative_delay_spare_when_infeasible():
    res = StoppingBudgetModel().solve(
        speed=Q_(20.0, "m/s"),
        delay=Q_(0.1, "s"),
        deceleration=Q_(5.0, "m/s**2"),
        clear_distance=Q_(30.0, "m"),  # braking distance alone is v^2/(2a) = 40 m
        physical_margin=Q_(2.0, "m"),
    )
    assert res.fits is False
    assert res.delay_spare.m_as("s") < 0.0


def test_sampling_model_extreme_logit_gap_no_overflow():
    res = SamplingModel().solve(
        logits=[1000.0, 0.0, -5.0],
        base_temperature=1.0,
        sharp_temperature=0.5,
        top_p=0.9,
    )
    assert math.isclose(res.base_probabilities[0], 1.0, rel_tol=1e-12)
    assert math.isclose(res.sharp_probabilities[0], 1.0, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 3. Units & Formatting Guards
# ---------------------------------------------------------------------------


def test_ttoken_and_formatter_unit_guards():
    q_ttok = Q_(15, "Ttoken")
    validated = require_unit_family(q_ttok, ureg.count, "dataset_tokens", "count")
    assert math.isclose(validated.m_as("token"), 15e12, rel_tol=1e-9)

    with pytest.raises(ValueError, match="EUR"):
        fmt_eur(Q_(100, "USD"))

    assert fmt_pp(Q_(5.2, "percent"), precision=1) == "5.2 percentage points"

    with pytest.raises(ValueError):
        fmt_params(Q_(80, "watt"))
    with pytest.raises(ValueError):
        fmt_tokens(Q_(80, "byte"))
    with pytest.raises(ValueError):
        fmt_memory(Q_(10, "second"))
    with pytest.raises(ValueError):
        fmt_memory(Q_(7, "Bparam"))
    assert fmt_memory(Q_(7, "Bparam") * Q_(2, "byte")) == "14 GB"

    rng = fmt_usd_range(Q_(2, "kUSD/year"), Q_(4, "kUSD/year"), per="year")
    assert "2,000" in rng and "4,000" in rng


def test_fmt_domains_respect_explicit_unit_overrides():
    assert "mL" in fmt_volume(Q_(250, "mL"), unit="mL", precision=0)
    assert "mN" in fmt_torque(Q_(50, "mN * m"), unit="mN*m", precision=0)
    assert "kJ" in fmt_specific_heat(Q_(4.184, "kJ / (kg * K)"), unit="kJ/(kg*K)", precision=2)
    assert "cm" in fmt_inertia(Q_(12, "kg * cm**2"), unit="kg*cm**2", precision=0)
    assert "nm" in fmt_stiffness(Q_(2.5, "N / nm"), unit="N/nm", precision=1)
    assert "MW/min" in fmt_power_rate(Q_(5, "MW / min"), unit="MW/min", precision=0)


# ---------------------------------------------------------------------------
# 4. Optimization, Hardware, Models, Sensors & Actuators
# ---------------------------------------------------------------------------


def test_exhaustive_and_scipy_optimization_backends():
    ex = ExhaustiveBackend().compile(
        lambda x: float("nan") if x[0] < 1.5 else (x[0] - 3.0) ** 2,
        ranges=[(0.0, 4.0)],
        grid_size=5,
    )
    res_nan = ex.solve()
    assert res_nan.feasible is True
    assert math.isclose(res_nan.best_configuration["optimal_variable"], 3.0, rel_tol=1e-9)

    ex_all_nan = ExhaustiveBackend().compile(
        lambda x: float("nan"),
        ranges=[(0.0, 4.0)],
        grid_size=4,
    )
    assert ex_all_nan.solve().feasible is False

    sc = ScipyBackend().compile(
        lambda x: (x - 1.0) ** 2,
        bounds=(0.0, 5.0),
        constraints=[{"type": "ineq", "fun": lambda x: x - 2.5}],
        is_scalar=True,
    )
    res_sc = sc.solve()
    assert res_sc.feasible is True
    assert res_sc.best_configuration["optimal_variable"] >= 2.499


def test_hardware_tech_frozen_and_ridge_point_precision():
    with pytest.raises(ValidationError):
        Hardware.Tech.Memory.HBM3.bandwidth = Q_(1.0, "TB/s")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rp_h100 = Hardware.Cloud.H100.ridge_point("fp16")
    assert len(caught) == 0
    assert rp_h100.m_as("flop/byte") > 250.0

    # JetsonAGXOrin has peak_flops=275 TOPS (INT8 sparse) and precision_flops['fp16']=43.17 TFLOP/s;
    # bf16 must match fp16 (~210.8 flop/byte), not peak_flops (~1348 flop/byte)
    rp_orin_bf16 = Hardware.Edge.JetsonAGXOrin.ridge_point("bf16")
    rp_orin_fp16 = Hardware.Edge.JetsonAGXOrin.ridge_point("fp16")
    assert math.isclose(rp_orin_bf16.m_as("flop/byte"), rp_orin_fp16.m_as("flop/byte"), rel_tol=1e-9)
    assert rp_orin_bf16.m_as("flop/byte") < 300.0

    assert hasattr(Sensors, "SafetyLightCurtain")
    assert "SafetyLightCurtain" in mlsysim.sensors.__all__
    assert "SafetyLaserScanner" in mlsysim.sensors.__all__
    assert all(k in mlsysim.actuators.__all__ for k in ("FramelessMotor", "SafetyRelay", "WheelHubMotor"))


def _mock_urlopen_json(payload: dict):
    body = json.dumps(payload).encode("utf-8")

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    return _Resp(body)


def test_hf_importer_gpt2_moe_head_dim_and_tls_safety():
    # 1. GPT-2 style 2-matrix GELU FFN + tied embeddings
    gpt2_cfg = {
        "model_type": "gpt2",
        "n_layer": 12,
        "n_embd": 768,
        "n_head": 12,
        "n_inner": 3072,
        "vocab_size": 50257,
        "n_positions": 1024,
        "activation_function": "gelu_new",
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen_json(gpt2_cfg)):
        w_gpt2 = import_hf_model("openai-community/gpt2")
    assert isinstance(w_gpt2, TransformerWorkload)
    assert w_gpt2.vocab_size == 50257
    assert w_gpt2.context_length == 1024
    expected_gpt2 = 12 * (4 * 768**2 + 2 * 768 * 3072) + 50257 * 768
    assert math.isclose(w_gpt2.parameters.m_as("count"), expected_gpt2, rel_tol=1e-9)

    # 2. MoE config -> SparseTransformerWorkload
    mixtral_cfg = {
        "model_type": "mixtral",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "vocab_size": 32000,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen_json(mixtral_cfg)):
        w_moe = import_hf_model("mistralai/Mixtral-8x7B-v0.1")
    assert isinstance(w_moe, SparseTransformerWorkload)
    assert w_moe.experts == 8
    assert w_moe.active_experts_per_token == 2
    assert w_moe.active_parameters.m_as("count") < w_moe.parameters.m_as("count")

    # 3. TLS certificate verification failure must raise ValueError, never disable SSL verification
    cert_err = urllib.error.URLError(ssl.SSLCertVerificationError(1, "CERTIFICATE_VERIFY_FAILED"))
    with patch("urllib.request.urlopen", side_effect=cert_err) as mocked:
        with pytest.raises(ValueError, match="TLS certificate verification failed"):
            import_hf_model("meta-llama/Llama-3-8B")
        assert mocked.call_count == 1


# ---------------------------------------------------------------------------
# 5. CLI, GB200_NVL72 Decomposition, Region Wiring & Visualization
# ---------------------------------------------------------------------------


def test_cli_rejects_tech_tiers_decomposes_nvl72_and_wires_region():
    with pytest.raises(ValueError, match="not found"):
        _resolve_hardware("HBM3")

    plan = MlsysPlanSchema.model_validate(
        {
            "version": "1.0",
            "name": "NVL72 rack plan",
            "workload": {"name": "Llama3_70B", "batch_size": 64},
            "hardware": {"name": "GB200_NVL72", "accelerators": 72},
            "ops": {"region": "Quebec", "duration_days": 7.0},
        }
    )
    assert plan.total_accelerators == 72
    assert plan.fleet_obj.total_accelerators == 72
    assert plan.fleet_obj.count == 1
    assert plan.fleet_obj.node.accelerators_per_node == 72
    assert math.isclose(
        plan.fleet_obj.node.accelerator.compute.peak_flops.m_as("PFLOPs/s"), 2.5, rel_tol=1e-9
    )
    assert math.isclose(plan.fleet_obj.node.accelerator.tdp.m_as("kW"), 120.0 / 72.0, rel_tol=1e-9)
    assert plan.fleet_obj.region == Infrastructure.Grids.Quebec

    with pytest.raises(ValidationError, match="Unknown ops.region"):
        MlsysPlanSchema.model_validate(
            {
                "version": "1.0",
                "name": "Bad region",
                "workload": {"name": "Llama3_8B"},
                "hardware": {"name": "H100"},
                "ops": {"region": "Atlantis", "duration_days": 1.0},
            }
        )


def test_cli_eval_and_serve_exit_physics_fail_on_oom():
    runner = CliRunner()
    res_eval = runner.invoke(app, ["eval", "Llama3_405B", "T4", "-o", "json"])
    assert res_eval.exit_code == 2
    payload = json.loads(res_eval.stdout)
    assert payload["f_status"] == "FAIL"

    res_serve = runner.invoke(app, ["serve", "Llama3_405B", "T4", "-o", "json"])
    assert res_serve.exit_code == 2
    serve_payload = json.loads(res_serve.stdout)
    assert serve_payload["feasible"] is False


def test_viz_timing_strip_ax_and_distributed_roofline():
    import matplotlib.pyplot as plt
    from mlsysim.viz import plot_distributed_roofline, plot_roofline
    from mlsysim.viz.timing_strip import Span, TimingStrip, plot_timing_strip

    fig_in, ax_in = plt.subplots()
    strip = TimingStrip(
        lanes=["Sensor", "Policy"],
        spans=[Span(0, 0.0, 5.0, "capture"), Span(1, 5.0, 12.0, "infer")],
    )
    fig_out, ax_out, axd_out = plot_timing_strip(strip, ax=ax_in)
    assert fig_out is fig_in
    assert ax_out is ax_in
    assert axd_out is None
    plt.close(fig_in)

    fig_fp32, ax_fp32 = plot_roofline(Hardware.Cloud.H100, precision="fp32")
    texts_fp32 = " ".join(t.get_text() for t in ax_fp32.texts)
    assert "67 TFLOP/s" in texts_fp32
    plt.close(fig_fp32)

    fleet = Systems.Clusters.Frontier_8K
    fig_dist, ax_dist = plot_distributed_roofline(fleet, workloads=[Models.Language.Llama3_70B])
    assert "8192x" in ax_dist.get_title()
    texts_dist = " ".join(t.get_text() for t in ax_dist.texts)
    assert "204800 GB/s" in texts_dist
    plt.close(fig_dist)


# ---------------------------------------------------------------------------
# 5. Phase 2 First-Principles Solver & Heuristic Audit (A1-A6, B1-B5)
# ---------------------------------------------------------------------------


def test_moe_active_compute_and_ep_sharding_first_principles():
    from mlsysim.engine.solvers import (
        ContinuousBatchingModel,
        DecodeBoundModel,
        DistributedModel,
        ServingModel,
        TrainingMemoryModel,
        WeightStreamingModel,
    )

    mixtral = Models.Language.Mixtral_8x7B
    h100 = Hardware.Cloud.H100
    assert mixtral.effective_compute_parameters.m_as("Bparam") == pytest.approx(12.9)
    assert Models.Language.Llama3_405B.context_length == 131072

    # A1: DecodeBoundModel arithmetic intensity uses active parameters (12.9B) over total weight bytes (46.7B * 2)
    db = DecodeBoundModel().solve(mixtral, h100, precision="fp16", batch_size=1)
    assert db.flops_per_step.m_as("GFLOP") == pytest.approx(2.0 * 12.9)
    assert db.intensity.m_as("flop/byte") == pytest.approx(12.9 / 46.7)

    # ServingModel, ContinuousBatchingModel, and WeightStreamingModel use active parameters for compute
    dense_equiv = Models.Language.Llama3_8B.model_copy(
        update={
            "parameters": mixtral.active_parameters,
            "layers": mixtral.layers,
            "heads": mixtral.heads,
            "kv_heads": mixtral.kv_heads,
            "hidden_dim": mixtral.hidden_dim,
            "head_dim": mixtral.head_dim,
            "inference_flops": None,
        }
    )
    srv_moe = ServingModel().solve(mixtral, h100, seq_len=512, batch_size=1, efficiency=0.5)
    srv_dense = ServingModel().solve(dense_equiv, h100, seq_len=512, batch_size=1, efficiency=0.5)
    assert srv_moe.ttft.m_as("ms") == pytest.approx(srv_dense.ttft.m_as("ms"))
    assert srv_moe.model_weights_size.m_as("GB") == pytest.approx(46.7 * 2.0)

    cb_moe = ContinuousBatchingModel().solve(mixtral, h100, max_seq_len=512, max_batch_size=1, efficiency=0.5)
    assert cb_moe.ttft.m_as("ms") == pytest.approx(srv_dense.ttft.m_as("ms"))

    ws_moe = WeightStreamingModel().solve(
        mixtral, Hardware.Cloud.Cerebras_CS3, seq_len=128, batch_size=1, phase="prefill"
    )
    ws_dense = WeightStreamingModel().solve(
        dense_equiv, Hardware.Cloud.Cerebras_CS3, seq_len=128, batch_size=1, phase="prefill"
    )
    assert ws_moe.layer_compute_time.m_as("ms") == pytest.approx(ws_dense.layer_compute_time.m_as("ms"))

    # A2: Shared vs routed parameter separation under EP
    p_shared = (8 * 12.9e9 - 2 * 46.7e9) / 6.0
    p_routed = 46.7e9 - p_shared
    sharded_ep8 = mixtral.sharded_parameter_count(tp_size=1, pp_size=1, ep_size=8).m_as("param")
    assert sharded_ep8 == pytest.approx(p_shared + p_routed / 8.0)

    tm_ep8 = TrainingMemoryModel().solve(mixtral, h100, batch_size=1, seq_len=512, ep_size=8)
    assert tm_ep8.weights.m_as("byte") == pytest.approx(sharded_ep8 * 2.0)

    # Dense workload rejects ep_size > 1 in TrainingMemoryModel and DistributedModel
    with pytest.raises(ValueError, match="ep_size > 1"):
        TrainingMemoryModel().solve(Models.Language.Llama3_8B, h100, batch_size=1, ep_size=2)
    with pytest.raises(ValueError, match="ep_size > 1"):
        DistributedModel().solve(
            Models.Language.Llama3_8B, Systems.Clusters.Research_256, batch_size=256, ep_size=2
        )


def test_young_daly_rework_wue_and_causal_chunked_prefill():
    from mlsysim.engine.solvers import ReliabilityModel, ServingModel, SustainabilityModel, WaterUseModel

    # A3: ReliabilityModel includes tau / (2 * M) rework loss
    fleet = Systems.Clusters.Research_256
    rel = ReliabilityModel().solve(
        fleet, job_duration_hours=24.0, checkpoint_time_s=60.0, avg_recovery_time_s=600.0
    )
    mtbf_s = rel.fleet_mtbf.m_as("s")
    tau_s = rel.optimal_checkpoint_interval.m_as("s")
    expected_goodput = max(0.0, 1.0 - 60.0 / tau_s - tau_s / (2.0 * mtbf_s) - 600.0 / mtbf_s)
    rework_omitted = 1.0 - 60.0 / tau_s - 600.0 / mtbf_s
    assert rel.goodput_ratio == pytest.approx(expected_goodput)
    assert rel.goodput_ratio < rework_omitted

    # A4: SustainabilityModel water matches WaterUseModel (E_IT * WUE, no PUE double-count)
    from mlsysim import Infrastructure

    sus = SustainabilityModel().solve(fleet, duration_days=1.0, mfu=0.5)
    wue_val = fleet.datacenter.grid.wue if (fleet.datacenter and fleet.datacenter.grid) else Infrastructure.Grids.US_Avg.wue
    wu = WaterUseModel().solve(sus.it_energy_kwh, {"site": Q_(wue_val, "liter / kWh")})
    assert sus.water_usage_liters == pytest.approx(wu.water["site"].m_as("liter"))

    # A5: Chunked causal prefill sums exact causal tiles (no quadratic underestimation)
    zero_tax_h100 = Hardware.Cloud.H100.model_copy(update={"dispatch_tax": Q_(0.0, "s")})
    full = ServingModel().solve(
        Models.Language.Llama3_8B, zero_tax_h100, seq_len=1024, batch_size=1, prefill_chunk_tokens=None
    )
    chunked = ServingModel().solve(
        Models.Language.Llama3_8B, zero_tax_h100, seq_len=1024, batch_size=1, prefill_chunk_tokens=256
    )
    assert chunked.ttft.m_as("ms") == pytest.approx(full.ttft.m_as("ms"), rel=1e-9)


def test_efficiency_dao_io_attention_flops_and_continuous_offload():
    from mlsysim.engine.engine import Engine
    from mlsysim.engine.solvers import EfficiencyModel, ResponsibleEngineeringModel
    from mlsysim.physics.transformer import calc_attention_matmul_flops

    # B1: CNN feature-map activation traffic and Transformer residual-stream traffic
    assert Models.Vision.ResNet50.activation_traffic_bytes().m_as("byte") == pytest.approx(
        2.0 * 11468264.0 * 2.0
    )
    llama8 = Models.Language.Llama3_8B
    assert llama8.activation_traffic_bytes(tokens_per_sample=512).m_as("byte") == pytest.approx(
        4.0 * llama8.layers * 512 * llama8.hidden_dim * 2.0
    )

    # B2: EfficiencyModel Roofline + Dao IO-complexity + unity partition
    eff_flash = EfficiencyModel().solve(
        llama8, Hardware.Cloud.H100, workload_type="attention", use_flash_attention=True, seq_len=8192
    )
    eff_unfused = EfficiencyModel().solve(
        llama8, Hardware.Cloud.H100, workload_type="attention", use_flash_attention=False, seq_len=8192
    )
    assert eff_flash.mfu > eff_unfused.mfu
    assert (
        eff_flash.mfu
        + eff_flash.overhead_breakdown["occupancy_loss"]
        + eff_flash.overhead_breakdown["memory_stall"]
    ) == pytest.approx(1.0)
    assert (
        eff_unfused.mfu
        + eff_unfused.overhead_breakdown["occupancy_loss"]
        + eff_unfused.overhead_breakdown["memory_stall"]
    ) == pytest.approx(1.0)

    # B2: ResponsibleEngineeringModel Gaussian mechanism noise multiplier
    dp = ResponsibleEngineeringModel().solve(
        base_training_time=Q_("5 day"), epsilon=2.0, delta=1e-5, sample_rate=0.01, steps=10000
    )
    expected_sigma = 0.01 * math.sqrt(2.0 * 10000.0 * math.log(1.25 / 1e-5)) / 2.0
    assert dp.noise_multiplier == pytest.approx(expected_sigma)

    # B4: Optional quadratic attention FLOPs in Engine.solve
    base_p = Engine.solve(llama8, Hardware.Cloud.H100, batch_size=4, seq_len=4096, is_training=True)
    attn_p = Engine.solve(
        llama8, Hardware.Cloud.H100, batch_size=4, seq_len=4096, is_training=True, include_attention_flops=True
    )
    expected_extra = (
        4 * 3 * (calc_attention_matmul_flops(llama8.layers, llama8.hidden_dim, 4096, 4096, 1).m_as("flop") / 2.0)
    )
    assert attn_p.latency_compute.m_as("s") - base_p.latency_compute.m_as("s") == pytest.approx(
        expected_extra / (Hardware.Cloud.H100.compute.peak_flops.m_as("flop/s") * 0.5),
        rel=1e-9,
    )

    # B5: Continuous spilled-bytes offload bandwidth (no cliff jump at tiny spill)
    small_hbm = Hardware.Cloud.H100.model_copy(
        update={
            "memory": Hardware.Cloud.H100.memory.model_copy(
                update={"capacity": Q_(16.05, "GB")}  # Llama3_8B weights + tiny act = ~16.060066 GB
            )
        }
    )
    off = Engine.solve(llama8, small_hbm, batch_size=1, seq_len=1)
    assert off.feasible is False
    assert off.offload_spill_bytes is not None and off.offload_spill_bytes.m_as("GB") < 0.02
    # Effective bandwidth stays close to HBM (> 2000 GB/s), NOT collapsing to link bw (~450 GB/s)!
    assert off.offload_effective_bw.m_as("GB/s") > 2000.0

