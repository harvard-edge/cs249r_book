from __future__ import annotations

import pytest

from mlsysim.engine.results import CompressionCandidate, CompressionSweepResult
from mlsysim.hardware.registry import Hardware
from mlsysim.models.registry import Models
from mlsysim.solvers import CompressionModel


def test_candidate_records_feasible_source_traced_int8_oura():
    solver = CompressionModel()
    candidate = solver.candidate(
        Models.Tiny.DS_CNN,
        Hardware.Tiny.OuraRing,
        label="Oura INT8",
        method="quantization",
        target_bitwidth=8,
        size_limit=Hardware.Tiny.OuraRing.memory.flash_capacity,
        max_accuracy_drop=0.01,
        require_hardware_support=True,
        # TinyML CNNs ship from FP32 training, so INT8 is quoted as 4x.
        baseline_precision="fp32",
    )

    assert isinstance(candidate, CompressionCandidate)
    assert candidate.label == "Oura INT8"
    assert candidate.feasible
    assert candidate.hardware_supported
    assert candidate.binding_constraint == "none"
    assert candidate.baseline_precision == "fp32"
    assert candidate.compression_ratio == pytest.approx(4.0)
    assert "CompressionModel.solve" in candidate.source_trace
    assert "baseline_precision=fp32" in candidate.source_trace
    assert any("hardware=Oura Ring" in item for item in candidate.source_trace)


def test_sweep_passes_baseline_precision_to_every_candidate():
    solver = CompressionModel()
    configs = [
        {"label": "INT8 weights", "method": "quantization", "target_bitwidth": 8},
        {"label": "INT4 weights", "method": "quantization", "target_bitwidth": 4},
    ]
    default = solver.sweep(Models.Language.Llama3_8B, Hardware.Cloud.H100, configs)
    fp32 = solver.sweep(
        Models.Language.Llama3_8B, Hardware.Cloud.H100, configs, baseline_precision="fp32"
    )

    assert [c.baseline_precision for c in default.candidates] == ["fp16", "fp16"]
    assert [c.compression_ratio for c in default.candidates] == pytest.approx([2.0, 4.0])
    assert [c.baseline_precision for c in fp32.candidates] == ["fp32", "fp32"]
    assert [c.compression_ratio for c in fp32.candidates] == pytest.approx([4.0, 8.0])


def test_unstructured_pruning_marks_missing_fast_path_when_required():
    solver = CompressionModel()
    candidate = solver.candidate(
        Models.Vision.ResNet50,
        Hardware.Cloud.H100,
        method="pruning",
        sparsity=0.9,
        sparsity_type="unstructured",
        require_hardware_support=True,
    )

    assert not candidate.hardware_supported
    assert not candidate.feasible
    assert candidate.binding_constraint == "hardware_support"
    assert candidate.inference_speedup == pytest.approx(1.0)
    assert candidate.guardrail_violations == [
        "hardware_support: no explicit fast path in the hardware profile"
    ]


def test_sweep_marks_dominated_and_frontier_candidates():
    solver = CompressionModel()
    sweep = solver.sweep(
        Models.Vision.ResNet50,
        Hardware.Cloud.H100,
        [
            {"label": "FP32 baseline", "method": "quantization", "target_bitwidth": 32},
            {"label": "FP16 weights", "method": "quantization", "target_bitwidth": 16},
            {"label": "INT8 weights", "method": "quantization", "target_bitwidth": 8},
        ],
    )

    assert isinstance(sweep, CompressionSweepResult)
    assert "FP32 baseline" in sweep.dominated_labels
    assert "FP16 weights" in sweep.frontier_labels
    assert sweep.best_candidate_label == "FP16 weights"
    statuses = {candidate.label: candidate.pareto_status for candidate in sweep.candidates}
    assert statuses["FP32 baseline"] == "dominated"
    assert statuses["FP16 weights"] == "frontier"
