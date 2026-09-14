from __future__ import annotations

import pytest
import mlsysim
from mlsysbook_labs.rationale import (
    RationaleChallenge,
    evaluate_rationale,
    render_interactive_roofline,
    render_latency_breakdown,
)


def test_rationale_evaluation_verified():
    challenge = RationaleChallenge(
        question="If batch size increases from 1 to 4 on ResNet-50, what happens?",
        metric_label="Throughput",
        options={"4x": "Throughput ~4x", "1x": "Throughput flat"},
        mechanisms={
            "compute": "Crosses ridge point into compute-bound regime",
            "memory": "Remains memory-bound",
        },
        correct_option="4x",
        correct_mechanism="compute",
        concept_title="The Iron Law & Ridge Point",
        chapter_reference="Chapter 2: Machine Learning Systems",
        literature_source="Williams et al. (2009)",
        fallacy_explanation="Small batch sizes underutilize tensor cores due to memory bandwidth limits.",
    )

    baseline = mlsysim.Engine.solve(
        mlsysim.Models.Vision.ResNet50,
        mlsysim.Hardware.Cloud.H100,
        batch_size=1,
    )
    proposal = mlsysim.Engine.solve(
        mlsysim.Models.Vision.ResNet50,
        mlsysim.Hardware.Cloud.H100,
        batch_size=4,
    )

    eval_result = evaluate_rationale(
        challenge=challenge,
        student_prediction="4x",
        student_mechanism="compute",
        baseline_profile=baseline,
        proposal_profile=proposal,
    )

    assert eval_result.prediction_correct is True
    assert eval_result.mechanism_correct is True
    assert eval_result.regime_shifted is True
    assert "Verified First-Principles Prediction" in eval_result.critique_markdown
    assert "Williams et al. (2009)" in eval_result.critique_markdown


def test_rationale_evaluation_intuition_trap():
    challenge = RationaleChallenge(
        question="If batch size increases from 1 to 32 on LLaMA-3 8B decode, what happens?",
        metric_label="Throughput",
        options={"32x": "32x Throughput increase", "moderate": "Moderate ~10x increase"},
        mechanisms={
            "compute": "Compute-bound execution saturates tensor cores",
            "memory": "Weight memory bandwidth remains the bottleneck",
        },
        correct_option="moderate",
        correct_mechanism="memory",
        concept_title="Decode Regime",
        chapter_reference="Chapter 2",
        literature_source="Pope et al. (2023)",
        fallacy_explanation="Decode is memory-bandwidth bound at small batch.",
    )

    baseline = mlsysim.Engine.solve(
        mlsysim.Models.Language.Llama3_8B,
        mlsysim.Hardware.Cloud.H100,
        batch_size=1,
    )
    proposal = mlsysim.Engine.solve(
        mlsysim.Models.Language.Llama3_8B,
        mlsysim.Hardware.Cloud.H100,
        batch_size=32,
    )

    # Student made the mistake of choosing "compute"
    eval_result = evaluate_rationale(
        challenge=challenge,
        student_prediction="32x",
        student_mechanism="compute",
        baseline_profile=baseline,
        proposal_profile=proposal,
    )

    assert eval_result.prediction_correct is False
    assert eval_result.mechanism_correct is False
    assert "Intuition Trap" in eval_result.critique_markdown


def test_roofline_and_latency_renderers():
    h100 = mlsysim.Hardware.Cloud.H100
    model = mlsysim.Models.Vision.ResNet50

    baseline = mlsysim.Engine.solve(model, h100, batch_size=1)
    proposal = mlsysim.Engine.solve(model, h100, batch_size=64)

    b_ai = baseline.arithmetic_intensity.magnitude
    b_perf = baseline.throughput.to("1/s").magnitude * model.inference_flops.to("GFLOP").magnitude
    p_ai = proposal.arithmetic_intensity.magnitude
    p_perf = proposal.throughput.to("1/s").magnitude * model.inference_flops.to("GFLOP").magnitude

    roofline_fig = render_interactive_roofline(
        hardware=h100,
        points=[
            ("Batch 1", b_ai, b_perf, "#006395"),
            ("Batch 64", p_ai, p_perf, "#CB202D"),
        ],
    )
    assert roofline_fig is not None
    assert "Roofline" in roofline_fig.layout.title.text

    waterfall_fig = render_latency_breakdown(baseline, proposal)
    assert waterfall_fig is not None
    assert "Latency" in waterfall_fig.layout.title.text
