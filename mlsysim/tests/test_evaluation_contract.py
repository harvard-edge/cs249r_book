from mlsysim.core.evaluation import SystemEvaluator
from mlsysim.hardware.registry import Hardware
from mlsysim.models.registry import Models
from mlsysim import Applications, plot_evaluation_scorecard
from mlsysim.systems.registry import Systems


def test_system_evaluation_json_contract_includes_tco():
    """Machine-readable scorecards expose flat m_* economics keys."""
    evaluation = SystemEvaluator.evaluate(
        scenario_name="contract",
        model_obj=Models.Llama3_8B,
        hardware_obj=Hardware.H100,
        batch_size=8,
        precision="fp16",
        efficiency=0.4,
        fleet_obj=Systems.Clusters.Research_256,
        nodes=256,
        duration_days=1.0,
    )

    payload = evaluation.to_dict()
    assert payload["m_status"] == "PASS"
    assert payload["m_tco_usd"] > 0
    assert "macro" not in payload


def test_single_node_evaluation_passed_all_with_skipped_macro():
    evaluation = SystemEvaluator.evaluate(
        scenario_name="single-node",
        model_obj=Models.ResNet50,
        hardware_obj=Hardware.A100,
        batch_size=1,
        precision="fp16",
        efficiency=0.5,
    )
    assert evaluation.macro.status == "SKIPPED"
    assert evaluation.passed_all is True


def test_infeasible_single_node_marks_performance_failed():
    evaluation = SystemEvaluator.evaluate(
        scenario_name="oom",
        model_obj=Models.GPT4,
        hardware_obj=Hardware.ESP32,
        batch_size=1,
        precision="fp16",
        efficiency=0.5,
    )
    assert evaluation.feasibility.status == "FAIL"
    assert evaluation.performance.status == "FAIL"


def test_scorecard_plot_accepts_scenario_evaluation_quantities():
    """Scenario evaluations expose Pint quantities; plots normalize them."""
    evaluation = Applications.Doorbell.evaluate()
    fig, ax = plot_evaluation_scorecard(evaluation)
    try:
        assert len(ax.patches) == 2
        assert all(patch.get_width() > 0 for patch in ax.patches)
    finally:
        fig.clf()
