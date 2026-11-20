from __future__ import annotations

import doctest

import pytest
from pydantic import ValidationError

from mlsysim.core.provenance import Provenance, ProvenanceKind
from mlsysim.engine import accuracy_curve as accuracy_curve_module
from mlsysim.engine import calibration as cal
from mlsysim.engine.accuracy_curve import AccuracyCurve, shipped_curve, shipped_curve_names
from mlsysim.hardware.registry import Hardware
from mlsysim.models.registry import Models
from mlsysim.solvers import CompressionModel

LLM = Models.Language.Llama3_8B
CNN = Models.Vision.ResNet50
H100 = Hardware.Cloud.H100


def _curve(axis="bitwidth", points=((4, 0.70), (8, 0.75), (16, 0.76)), baseline=0.76, **overrides):
    fields = dict(
        name="test-curve",
        axis=axis,
        measured_model="a model",
        method="a method",
        metric="top-1 accuracy",
        dataset="a dataset",
        retrained=False,
        baseline_accuracy=baseline,
        points=points,
        provenance=Provenance(kind=ProvenanceKind.ILLUSTRATIVE, ref="test fixture", verified="2026-11-16"),
    )
    fields.update(overrides)
    return AccuracyCurve(**fields)


# --- the curve itself ---------------------------------------------------------


def test_measured_points_are_returned_exactly_and_sorted():
    curve = _curve(points=((16, 0.76), (4, 0.70), (8, 0.75)))
    assert curve.points == ((4.0, 0.70), (8.0, 0.75), (16.0, 0.76))
    assert curve.measured_range == (4.0, 16.0)
    assert curve.accuracy_at(8) == 0.75
    assert curve.is_measured(8) and not curve.is_measured(6)
    assert curve.delta_at(4) == pytest.approx(-0.06)
    assert curve.delta_at(16) == 0.0


def test_interpolation_is_linear_between_adjacent_points_only():
    curve = _curve()
    # Halfway between 4 bits (0.70) and 8 bits (0.75).
    assert curve.accuracy_at(6) == pytest.approx(0.725)
    # A quarter of the way from 8 bits (0.75) to 16 bits (0.76).
    assert curve.accuracy_at(10) == pytest.approx(0.7525)


def test_levels_within_rounding_of_a_point_count_as_that_point():
    curve = _curve(axis="sparsity", points=((0.0, 0.76), (0.7, 0.75), (0.8, 0.60)))
    # 0.1 * 7 is 0.7000000000000001 in binary floating point.
    assert curve.is_measured(0.1 * 7)
    assert curve.accuracy_at(0.1 * 7) == 0.75
    # The ends of the range get the same allowance.
    assert curve.accuracy_at(0.8 + 1e-12) == 0.60


@pytest.mark.parametrize("level", [3.9, 2, 17, 32])
def test_curve_refuses_to_extrapolate(level):
    with pytest.raises(ValueError, match="does not extrapolate"):
        _curve().accuracy_at(level)


@pytest.mark.parametrize("level", [float("nan"), float("inf")])
def test_curve_rejects_non_finite_levels(level):
    with pytest.raises(ValueError, match="must be finite"):
        _curve().accuracy_at(level)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"points": ()}, "at least one measured point"),
        ({"points": ((4, 0.7), (4, 0.6))}, "distinct compression levels"),
        ({"points": ((4, 0.7), (4 + 1e-12, 0.6))}, "distinct compression levels"),
        ({"points": ((4, 1.2),)}, r"fraction in \[0, 1\]"),
        ({"points": ((0.5, 0.7),)}, "at least one bit per weight"),
        ({"points": ((4, float("nan")),)}, "finite"),
        ({"baseline": 0.0}, r"fraction in \(0, 1\]"),
        ({"baseline": 76.1}, r"fraction in \(0, 1\]"),
        ({"axis": "sparsity", "points": ((1.5, 0.7),)}, r"every level in \[0, 1\]"),
    ],
)
def test_invalid_curves_are_rejected(overrides, message):
    with pytest.raises(ValidationError, match=message):
        _curve(**overrides)


def test_curve_is_immutable_and_rejects_unknown_fields():
    curve = _curve()
    with pytest.raises(ValidationError):
        curve.baseline_accuracy = 0.5
    with pytest.raises(ValidationError):
        _curve(perplexity=5.6)


def test_curves_are_hashable():
    # Equal curves hash equally, so a curve can key a dict or a cache.
    assert len({_curve(), _curve(), _curve(name="other")}) == 2
    assert {shipped_curve(name): name for name in shipped_curve_names()}


# --- the example curves that ship with the package ---------------------------------

# Every shipped number, as (baseline, points), so a change to the data file is deliberate.
SHIPPED = {
    # Jin et al. (2024), Table 2, Qwen-7B-Chat, Average Accuracy.
    "qwen-7b-chat-gptq": (0.5710, ((2, 0.1652), (3, 0.5142), (4, 0.5486), (8, 0.5721), (16, 0.5710))),
    "qwen-7b-chat-spqr": (0.5710, ((2, 0.5318), (3, 0.5545), (4, 0.5641), (8, 0.5649), (16, 0.5710))),
    # Gale et al. (2019): dense accuracy from Section 5, pruned accuracies from the authors' README.
    "resnet50-magnitude-pruning-during-training": (
        0.7669,
        ((0.0, 0.7669), (0.5, 0.7653), (0.7, 0.7638), (0.8, 0.7558), (0.9, 0.7391), (0.95, 0.7059), (0.98, 0.579)),
    ),
    # Singh and Alistarh (2020), Table 7, empirical WoodFisher.
    "resnet50-woodfisher-one-shot": (
        0.7613,
        ((0.0, 0.7613), (0.2, 0.7610), (0.4, 0.7522), (0.6, 0.6921), (0.7, 0.4835), (0.8, 0.0548)),
    ),
}


def test_shipped_curves_load_with_sourced_provenance():
    names = shipped_curve_names()
    assert len(names) == len(set(names)) >= 4
    for name in names:
        curve = shipped_curve(name)
        assert curve.name == name
        assert curve.provenance.kind == ProvenanceKind.LITERATURE
        assert curve.provenance.url.startswith("https://")
        # Blalock et al. (2020) ask for at least five operating points per curve.
        assert len(curve.points) >= 5, name
        # Every shipped curve includes its own uncompressed operating point.
        assert any(accuracy == curve.baseline_accuracy for _, accuracy in curve.points), name
    with pytest.raises(KeyError, match="no shipped accuracy curve"):
        shipped_curve("resnet50-int8")


def test_shipped_values_match_the_cited_tables():
    assert set(shipped_curve_names()) == set(SHIPPED)
    for name, (baseline, points) in SHIPPED.items():
        curve = shipped_curve(name)
        assert curve.baseline_accuracy == baseline, name
        assert curve.points == points, name


def test_module_examples_run():
    results = doctest.testmod(accuracy_curve_module)
    assert results.attempted > 0 and results.failed == 0


# --- CompressionModel with and without a curve ------------------------------------


def test_defaults_are_unchanged_without_a_curve():
    solver = CompressionModel()
    for bits, expected in ((16, 0.0), (8, cal.QUANT_ACCURACY_DELTA_FP8), (4, cal.QUANT_ACCURACY_DELTA_INT4),
                           (2, cal.QUANT_ACCURACY_DELTA_SUB_INT4)):
        result = solver.solve(LLM, H100, target_bitwidth=bits, baseline_precision="fp16")
        assert result.estimated_accuracy_delta == expected
        # The solver sets no curve field, so a dump of the set fields is as before.
        assert "constraint_trace" not in result.model_fields_set
        assert "constraint_trace" not in result.model_dump(exclude_unset=True)
    pruned = solver.solve(CNN, H100, method="pruning", sparsity=0.4)
    assert pruned.estimated_accuracy_delta == cal.PRUNING_MILD_DELTA
    assert "constraint_trace" not in pruned.model_fields_set
    candidate = solver.candidate(LLM, H100, target_bitwidth=4, baseline_precision="fp16")
    assert not any(item.startswith("accuracy_curve=") for item in candidate.source_trace)


def test_curve_replaces_only_the_accuracy_estimate():
    solver = CompressionModel()
    curve = shipped_curve("qwen-7b-chat-gptq")
    default = solver.solve(LLM, H100, target_bitwidth=4, baseline_precision="fp16")
    measured = solver.solve(LLM, H100, target_bitwidth=4, baseline_precision="fp16", accuracy_curve=curve)

    assert measured.estimated_accuracy_delta == pytest.approx(0.5486 - 0.5710)
    assert measured.constraint_trace == ["accuracy from curve qwen-7b-chat-gptq: measured point at bitwidth 4"]
    changed = {"estimated_accuracy_delta", "constraint_trace"}
    assert {k: v for k, v in measured.model_dump().items() if k not in changed} == {
        k: v for k, v in default.model_dump().items() if k not in changed
    }


def test_interpolated_levels_are_labelled_in_the_trace():
    result = CompressionModel().solve(
        LLM, H100, target_bitwidth=6, baseline_precision="fp16", accuracy_curve=shipped_curve("qwen-7b-chat-gptq")
    )
    assert result.estimated_accuracy_delta == pytest.approx((0.5486 + 0.5721) / 2 - 0.5710)
    assert result.constraint_trace == [
        "accuracy from curve qwen-7b-chat-gptq: interpolated between measured points at bitwidth 6"
    ]


def test_sparsity_curve_drives_the_pruning_branch():
    solver = CompressionModel()
    one_shot = shipped_curve("resnet50-woodfisher-one-shot")
    trained = shipped_curve("resnet50-magnitude-pruning-during-training")
    # Same architecture and sparsity, two methods: 70.65 points lost against 1.11.
    assert solver.solve(CNN, H100, method="pruning", sparsity=0.8, accuracy_curve=one_shot
                        ).estimated_accuracy_delta == pytest.approx(0.0548 - 0.7613)
    assert solver.solve(CNN, H100, method="pruning", sparsity=0.8, accuracy_curve=trained
                        ).estimated_accuracy_delta == pytest.approx(0.7558 - 0.7669)


def test_curve_must_match_the_method_and_cover_the_level():
    solver = CompressionModel()
    bits, sparsity = shipped_curve("qwen-7b-chat-gptq"), shipped_curve("resnet50-woodfisher-one-shot")
    with pytest.raises(ValueError, match="pruning needs a sparsity curve"):
        solver.solve(CNN, H100, method="pruning", sparsity=0.5, accuracy_curve=bits)
    with pytest.raises(ValueError, match="quantization needs a bitwidth curve"):
        solver.solve(CNN, H100, target_bitwidth=8, accuracy_curve=sparsity)
    with pytest.raises(ValueError, match="applies to quantization or pruning"):
        solver.solve(CNN, H100, method="distillation", accuracy_curve=bits)
    # The one-shot curve stops at 80% sparsity, just after its collapse.
    with pytest.raises(ValueError, match="does not extrapolate"):
        solver.solve(CNN, H100, method="pruning", sparsity=0.9, accuracy_curve=sparsity)
    with pytest.raises(ValueError, match="does not extrapolate"):
        solver.solve(LLM, H100, target_bitwidth=1, accuracy_curve=bits)


def test_a_curve_name_is_refused_with_a_hint():
    with pytest.raises(TypeError, match=r"must be an AccuracyCurve.*shipped_curve\(name\)"):
        CompressionModel().solve(LLM, H100, target_bitwidth=4, accuracy_curve="qwen-7b-chat-gptq")


def test_candidate_gate_uses_the_curve_and_records_it():
    solver = CompressionModel()
    gptq, spqr = shipped_curve("qwen-7b-chat-gptq"), shipped_curve("qwen-7b-chat-spqr")
    # A 99%-of-baseline quality target, the form MLPerf Inference uses (Reddi et al., 2020).
    target = 0.01 * gptq.baseline_accuracy
    common = dict(target_bitwidth=3, baseline_precision="fp16", max_accuracy_drop=target)

    without = solver.candidate(LLM, H100, **common)
    with_gptq = solver.candidate(LLM, H100, accuracy_curve=gptq, **common)
    with_spqr = solver.candidate(LLM, H100, accuracy_curve=spqr, **common)

    assert "accuracy_curve=qwen-7b-chat-gptq" in with_gptq.source_trace
    assert not any(item.startswith("accuracy_curve=") for item in without.source_trace)
    assert with_gptq.estimated_accuracy_delta == pytest.approx(-0.0568)
    assert with_spqr.estimated_accuracy_delta == pytest.approx(-0.0165)
    assert not with_gptq.feasible and with_gptq.binding_constraint == "quality"
    assert not with_spqr.feasible  # 1.65 points is still above a 0.571-point target
    assert solver.candidate(LLM, H100, accuracy_curve=spqr, **{**common, "target_bitwidth": 4,
                                                                 "max_accuracy_drop": 0.01}).feasible


def test_candidate_gate_uses_a_sparsity_curve():
    solver = CompressionModel()
    trained = shipped_curve("resnet50-magnitude-pruning-during-training")
    common = dict(method="pruning", sparsity=0.8, max_accuracy_drop=0.02)
    # 1.11 points lost when pruned during training, 70.65 when pruned in one shot.
    assert solver.candidate(CNN, H100, accuracy_curve=trained, **common).feasible
    one_shot = solver.candidate(CNN, H100, accuracy_curve=shipped_curve("resnet50-woodfisher-one-shot"), **common)
    assert not one_shot.feasible and one_shot.binding_constraint == "quality"
    assert "accuracy_curve=resnet50-woodfisher-one-shot" in one_shot.source_trace


def test_a_measured_gain_does_not_violate_the_accuracy_gate():
    solver = CompressionModel()
    # Jin et al. measured Qwen-7B-Chat 0.11 points above its baseline at 8 bits under GPTQ.
    gain = solver.candidate(LLM, H100, target_bitwidth=8, baseline_precision="fp16", max_accuracy_drop=0.0005,
                            accuracy_curve=shipped_curve("qwen-7b-chat-gptq"))
    assert gain.estimated_accuracy_delta == pytest.approx(0.0011)
    assert gain.feasible and gain.binding_constraint == "none"
    # Without a curve the default is a drop, so the same gate still binds.
    default = solver.candidate(LLM, H100, target_bitwidth=8, baseline_precision="fp16", max_accuracy_drop=0.0005)
    assert not default.feasible and default.binding_constraint == "quality"


def test_sweep_accepts_a_curve_per_candidate():
    solver = CompressionModel()
    gptq = shipped_curve("qwen-7b-chat-gptq")
    configs = [{"label": f"{bits}-bit", "target_bitwidth": bits, "accuracy_curve": gptq} for bits in (8, 4, 3, 2)]
    sweep = solver.sweep(LLM, H100, configs, max_accuracy_drop=0.03, baseline_precision="fp16")

    by_label = {c.label: c for c in sweep.candidates}
    assert [by_label[k].feasible for k in ("8-bit", "4-bit", "3-bit", "2-bit")] == [True, True, False, False]
    assert by_label["2-bit"].estimated_accuracy_delta == pytest.approx(0.1652 - 0.5710)
    # The measured 8-bit accuracy is above the baseline, so it has no drop and ranks first.
    assert sweep.best_candidate_label == "8-bit"
    assert all("accuracy_curve=qwen-7b-chat-gptq" in c.source_trace for c in sweep.candidates)


def test_sweep_ranks_a_large_gain_ahead_of_a_small_drop():
    # A 3-point gain at 8 bits and a 0.5-point drop at 4 bits. Neither dominates the other,
    # and the selection ranks by drop, so the gain wins even though its size is larger.
    curve = _curve(points=((4, 0.755), (8, 0.79), (16, 0.76)))
    configs = [{"label": f"{bits}-bit", "target_bitwidth": bits, "accuracy_curve": curve} for bits in (4, 8)]
    sweep = CompressionModel().sweep(LLM, H100, configs, max_accuracy_drop=0.02, baseline_precision="fp16")
    assert set(sweep.frontier_labels) == {"4-bit", "8-bit"}
    assert sweep.best_candidate_label == "8-bit"


def test_sweep_compares_pruning_methods_at_one_sparsity():
    trained = shipped_curve("resnet50-magnitude-pruning-during-training")
    one_shot = shipped_curve("resnet50-woodfisher-one-shot")
    configs = [
        {"label": "one-shot 70%", "method": "pruning", "sparsity": 0.7, "accuracy_curve": one_shot},
        {"label": "during training 70%", "method": "pruning", "sparsity": 0.7, "accuracy_curve": trained},
        {"label": "during training 90%", "method": "pruning", "sparsity": 0.9, "accuracy_curve": trained},
    ]
    sweep = CompressionModel().sweep(CNN, H100, configs, max_accuracy_drop=0.01)
    by_label = {c.label: c for c in sweep.candidates}
    # Same size and speed at 70%, so the curve with the smaller drop dominates.
    assert by_label["one-shot 70%"].pareto_status == "dominated"
    assert [by_label[k].feasible for k in ("during training 70%", "during training 90%")] == [True, False]
    assert sweep.best_candidate_label == "during training 70%"
