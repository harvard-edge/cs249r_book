import pytest

from mlperf.runners.tiny import (
    _anomaly_quality_gates,
    _classwise_anomaly_metrics,
)


def _controls() -> dict[str, dict]:
    return {
        "zero_reconstructor": {"class_aurocs": {"3": 0.58, "8": 0.64, "9": 0.46}},
        "normal_centroid": {"class_aurocs": {"3": 0.66, "8": 0.69, "9": 0.63}},
        "random_init_autoencoder": {"class_aurocs": {"3": 0.59, "8": 0.65, "9": 0.47}},
    }


def _gate(learned: dict[str, float]) -> dict:
    return _anomaly_quality_gates(
        learned,
        _controls(),
        macro_auroc_target=0.93,
        worst_class_auroc_target=0.90,
        control_margin_target=0.20,
    )


def test_trained_anomaly_candidate_passes_all_independent_gates():
    result = _gate({"3": 0.918, "8": 0.954, "9": 0.949})

    assert result["passed"] is True
    assert result["macro_auroc"]["met"] is True
    assert result["worst_class_auroc"]["met"] is True
    assert result["control_margin"]["met"] is True
    assert result["strongest_controls"]["3"]["name"] == "normal_centroid"


def test_centroid_output_cannot_qualify_even_if_absolute_auroc_is_high():
    controls = _controls()
    controls["normal_centroid"]["class_aurocs"] = {
        "3": 0.94,
        "8": 0.96,
        "9": 0.95,
    }
    result = _anomaly_quality_gates(
        {"3": 0.94, "8": 0.96, "9": 0.95},
        controls,
        macro_auroc_target=0.93,
        worst_class_auroc_target=0.90,
        control_margin_target=0.20,
    )

    assert result["macro_auroc"]["met"] is True
    assert result["worst_class_auroc"]["met"] is True
    assert result["control_margin"]["value"] == pytest.approx(0.0)
    assert result["control_margin"]["met"] is False
    assert result["passed"] is False


def test_no_training_candidate_cannot_hide_a_weak_anomaly_class_in_macro_score():
    result = _gate({"3": 0.89, "8": 0.99, "9": 0.99})

    assert result["macro_auroc"]["met"] is True
    assert result["worst_class_auroc"]["met"] is False
    assert result["passed"] is False


def test_control_class_coverage_must_match_learned_class_coverage():
    controls = _controls()
    del controls["zero_reconstructor"]["class_aurocs"]["9"]

    with pytest.raises(ValueError, match="same anomaly classes"):
        _anomaly_quality_gates(
            {"3": 0.94, "8": 0.95, "9": 0.96},
            controls,
            macro_auroc_target=0.93,
            worst_class_auroc_target=0.90,
            control_margin_target=0.20,
        )


def test_classwise_metrics_reject_missing_hard_class():
    with pytest.raises(ValueError, match="anomaly class 9"):
        _classwise_anomaly_metrics(
            [0.1, 0.2, 0.3, 0.4],
            [5, 3, 5, 8],
            normal_class=5,
            anomaly_classes=(3, 8, 9),
        )
