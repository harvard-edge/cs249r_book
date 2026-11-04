"""Unit representations and impossible-event boundaries are physical contracts."""

import itertools
import math

import pint
import pytest

from mlsysim import Q_
from mlsysim.physics.coordination import calc_disjoint_writes
from mlsysim.solvers import ScalingModel


@pytest.mark.parametrize("unit", ["flop", "kiloflop", "MFLOP", "GFLOP", "TFLOP", "PFLOP"])
@pytest.mark.parametrize("fixed_params", [None, Q_(100000, "count")])
def test_scaling_budget_is_invariant_to_flop_prefix(unit, fixed_params):
    budget = Q_(120, "TFLOP").to(unit)
    result = ScalingModel().solve(budget, target_model_size=fixed_params)
    # Independently solve C=6PD and D=20P under the declared Chinchilla
    # convention. The prefix conversion cannot change the physical budget.
    work = 120 * 10**12
    expected_params = math.sqrt(work / (6 * 20)) if fixed_params is None else 100000
    expected_tokens = work / (6 * expected_params)
    assert result.optimal_parameters.m_as("count") == pytest.approx(expected_params)
    assert result.optimal_tokens.m_as("count") == pytest.approx(expected_tokens)
    assert result.compute_budget_flops.m_as("flop") == pytest.approx(work)


def test_scaling_duration_units_preserve_reference_hardware_conversion():
    one_minute = ScalingModel().solve(Q_(1, "minute"))
    seconds = ScalingModel().solve(Q_(60, "second"))
    assert one_minute == seconds
    work_form = ScalingModel().solve(one_minute.compute_budget_flops)
    assert work_form == one_minute


def test_scaling_rejects_an_unrelated_budget_dimension():
    with pytest.raises(pint.DimensionalityError):
        ScalingModel().solve(Q_(120, "GB"))


@pytest.mark.parametrize("workers,files,writes", [(1, 4, 2), (2, 4, 0), (2, 4, 2), (3, 4, 1), (4, 3, 2)])
def test_disjoint_write_probability_matches_exhaustive_outcomes(workers, files, writes):
    choices = list(itertools.combinations(range(files), writes))
    outcomes = list(itertools.product(choices, repeat=workers))
    disjoint = sum(len(set(itertools.chain.from_iterable(outcome))) == workers * writes for outcome in outcomes)
    assert calc_disjoint_writes(workers, files, writes) == pytest.approx(disjoint / len(outcomes))


@pytest.mark.parametrize(
    "workers,files,writes",
    [
        (0, 4, 1),
        (-1, 4, 1),
        (2, 0, 0),
        (2, -1, 0),
        (2, 4, -1),
        (2, 4, 5),
        (2.5, 4, 1),
        (2, 4.5, 1),
        (2, 4, 1.5),
        (True, 4, 1),
    ],
)
def test_disjoint_write_probability_rejects_invalid_cardinalities(workers, files, writes):
    with pytest.raises(ValueError):
        calc_disjoint_writes(workers, files, writes)


def test_weight_streaming_gqa_reduces_cache_not_query_attention_work():
    from mlsysim import Hardware, Models
    from mlsysim.solvers import WeightStreamingModel

    # Construct a compute-bound device so a query-head accounting error cannot
    # hide under weight-injection time or dispatch overhead.
    template = Hardware.Cloud.Cerebras_CS3
    hardware = template.model_copy(
        update={
            "compute": template.compute.model_copy(
                update={
                    "peak_flops": Q_(1, "MFLOP/s"),
                    "precision_flops": {},
                }
            ),
            "dispatch_tax": Q_(0, "second"),
            "interconnect": template.interconnect.model_copy(update={"bandwidth": Q_(1, "TB/s")}),
        }
    )
    model = Models.Language.Llama3_8B.model_copy(
        update={
            "parameters": Q_(1000, "param"),
            "layers": 2,
            "heads": 4,
            "hidden_dim": 32,
            "kv_heads": 1,
        }
    )
    seq, batch, efficiency = 8, 2, 0.5
    gqa = WeightStreamingModel().solve(model, hardware, seq, batch, efficiency=efficiency, phase="prefill")
    mha = WeightStreamingModel().solve(
        model.model_copy(update={"kv_heads": model.heads}),
        hardware,
        seq,
        batch,
        efficiency=efficiency,
        phase="prefill",
    )
    linear_ops = 2 * 1000 * seq * batch
    # Causal half-square convention: two matrix products, two FLOPs/MAC,
    # and half the square visited for each layer/query head.
    attention_ops = 2 * model.layers * model.heads * (model.hidden_dim // model.heads) * seq**2 * batch
    expected = Q_(linear_ops + attention_ops, "flop") / (hardware.compute.peak_flops * efficiency)
    assert (gqa.layer_compute_time * model.layers).m_as("second") == pytest.approx(expected.m_as("second"))
    assert gqa.layer_compute_time == mha.layer_compute_time
    assert gqa.wafer_memory_utilization < mha.wafer_memory_utilization


@pytest.mark.parametrize("explicit_override", [False, True])
def test_vla_solvers_resolve_canonical_fp16_peak(explicit_override):
    from mlsysim import Hardware, Models
    from mlsysim.solvers import DecoderFamiliesModel, VlaEvaluationModel

    processor = Hardware.Cloud.H100
    expected_peak = processor.compute.peak_flops
    if explicit_override:
        expected_peak /= 2
        processor = processor.model_copy(
            update={
                "compute": processor.compute.model_copy(
                    update={
                        "precision_flops": {"fp16": expected_peak},
                    }
                )
            }
        )
    utilization = 0.5
    evaluation = VlaEvaluationModel().solve(
        Models.Language.Llama3_8B,
        processor,
        Q_(2, "byte/param"),
        8,
        1,
        0.5,
        utilization,
    )
    assert evaluation.achieved_rate == expected_peak * utilization
    assert evaluation.decode_steps == 0
    families = DecoderFamiliesModel().solve(
        Models.Language.Llama3_8B,
        processor,
        Q_(2, "byte/param"),
        8,
        0.5,
        utilization,
        2,
        3,
        4,
        Q_(1000, "param"),
        2,
        Q_(2, "byte/param"),
        Q_(2, "byte"),
        Q_(1, "second"),
    )
    expected = Q_(2 * 1000 * 2, "flop") / (expected_peak * utilization)
    assert families.t_expert_step_compute.m_as("second") == pytest.approx(expected.m_as("second"))


@pytest.mark.parametrize("magnitude", [0, -1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("unit", ["TFLOP", "second"])
def test_scaling_rejects_nonpositive_or_nonfinite_budgets(magnitude, unit):
    with pytest.raises(ValueError, match="compute_budget"):
        ScalingModel().solve(Q_(magnitude, unit))


@pytest.mark.parametrize("magnitude", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_scaling_rejects_nonpositive_or_nonfinite_target_counts(magnitude):
    with pytest.raises(ValueError, match="target_model_size"):
        ScalingModel().solve(Q_(120, "TFLOP"), target_model_size=Q_(magnitude, "param"))


@pytest.mark.parametrize("explicit_width", [8, 16])
def test_prefill_solvers_use_explicit_head_geometry(explicit_width):
    from mlsysim import Hardware, Models
    from mlsysim.solvers import ContinuousBatchingModel, ServingModel, WeightStreamingModel

    template = Hardware.Cloud.Cerebras_CS3
    hardware = template.model_copy(
        update={
            "compute": template.compute.model_copy(
                update={
                    "peak_flops": Q_(1, "MFLOP/s"),
                    "precision_flops": {},
                }
            ),
            "dispatch_tax": Q_(0, "second"),
            "interconnect": template.interconnect.model_copy(update={"bandwidth": Q_(1, "TB/s")}),
        }
    )
    layers, heads, kv_heads, params = 2, 4, 1, 1000
    model = Models.Language.Llama3_8B.model_copy(
        update={
            "parameters": Q_(params, "param"),
            "layers": layers,
            "heads": heads,
            "hidden_dim": 32,
            "head_dim": explicit_width,
            "kv_heads": kv_heads,
            "inference_flops": None,
        }
    )
    seq, batch, efficiency = 8, 2, 0.5
    serving = ServingModel().solve(model, hardware, seq, batch, efficiency=efficiency)
    batching = ContinuousBatchingModel().solve(
        model,
        hardware,
        max_seq_len=seq,
        max_batch_size=batch,
        efficiency=efficiency,
    )
    streaming = WeightStreamingModel().solve(
        model,
        hardware,
        seq,
        batch,
        efficiency=efficiency,
        phase="prefill",
    )
    linear = 2 * params * seq * batch
    rectangular_attention = 4 * layers * heads * explicit_width * seq**2 * batch
    expected_serving = Q_(linear + rectangular_attention, "flop") / (hardware.compute.peak_flops * efficiency)
    expected_streaming = Q_(linear + rectangular_attention / 2, "flop") / (hardware.compute.peak_flops * efficiency)
    assert serving.ttft.m_as("second") == pytest.approx(expected_serving.m_as("second"))
    assert batching.ttft.m_as("second") == pytest.approx(expected_serving.m_as("second"))
    assert (streaming.layer_compute_time * layers).m_as("second") == pytest.approx(expected_streaming.m_as("second"))
    kv_bytes = 2 * layers * kv_heads * explicit_width * seq * batch * 2
    expected_utilization = (Q_(kv_bytes, "byte") * 1.1 / hardware.memory.capacity).m_as("")
    assert streaming.wafer_memory_utilization == pytest.approx(expected_utilization)


def test_weight_streaming_palm_registry_cache_uses_explicit_head_width():
    from mlsysim import Hardware, Models
    from mlsysim.solvers import WeightStreamingModel

    model = Models.Language.PaLM_540B
    hardware = Hardware.Cloud.Cerebras_CS3
    assert model.head_dim is not None
    assert model.head_dim != model.hidden_dim // model.heads
    context, batch = 128, 2
    result = WeightStreamingModel().solve(model, hardware, context, batch)
    kv_bytes = 2 * model.layers * (model.kv_heads or model.heads) * model.head_dim * context * batch * 2
    expected = (Q_(kv_bytes, "byte") * 1.1 / hardware.memory.capacity).m_as("")
    assert result.wafer_memory_utilization == pytest.approx(expected)


@pytest.mark.parametrize("explicit_override", [False, True])
def test_mobile_manipulator_intent_resolves_canonical_fp16_peak(explicit_override):
    from mlsysim import Hardware
    from mlsysim.embodied.registry import Embodied
    from mlsysim.embodied.types import MobileManipulatorPlatform

    processor = Hardware.Cloud.H100
    expected_peak = processor.compute.peak_flops
    if explicit_override:
        expected_peak /= 2
        processor = processor.model_copy(
            update={
                "compute": processor.compute.model_copy(
                    update={
                        "precision_flops": {"fp16": expected_peak},
                    }
                )
            }
        )
    base = Embodied.MobileManipulator.WarehouseMobileManipulator
    # Use the public validated constructor: its intent-rate validator evaluates
    # this property too, so unsupported precision lookup would reject the device.
    machine = MobileManipulatorPlatform(**{**base.model_dump(), "brain_soc": processor})
    breakdown = machine.intent_latency_breakdown
    params = machine.intent_model.parameters.m_as("param")
    tokens = machine.intent_input_tokens
    if machine.intent_parallel_decode:
        tokens += machine.intent_model.action_dim
    linear_ops = 2 * params * tokens
    # Parallel action prefill is bidirectional; the ordinary decoder uses
    # the documented causal half-square approximation.
    attention_factor = 4 if machine.intent_parallel_decode else 2
    attention_ops = attention_factor * machine.intent_model.layers * machine.intent_model.hidden_dim * tokens**2
    expected_compute = Q_(linear_ops + attention_ops, "flop") / (expected_peak * machine.intent_compute_utilization)
    assert breakdown["t_prefill_compute"].m_as("second") == pytest.approx(expected_compute.m_as("second"))
    assert machine.intent_inference_latency.m_as("second") > 0
