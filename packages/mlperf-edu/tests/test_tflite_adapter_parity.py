import numpy as np

from tools.audit_tflite_adapter_parity import AUDIT_SCHEMA, comparison_summary


def test_adapter_audit_schema_is_portable_version():
    assert AUDIT_SCHEMA == "mlperf-edu-tflite-adapter-audit/0.2"


def test_adapter_comparison_requires_exact_predictions_and_quality():
    pytorch = np.asarray([[0.9, 0.1], [0.4, 0.6]], dtype=np.float32)
    tflite = np.asarray([[0.8, 0.2], [0.7, 0.3]], dtype=np.float32)
    labels = np.asarray([0, 1])

    result = comparison_summary(pytorch, tflite, labels, quality_target=0.5)

    assert result["sample_count"] == 2
    assert result["prediction_disagreement_count"] == 1
    assert result["prediction_disagreement_indices"] == [1]
    assert result["exact_prediction_parity"] is False
    assert result["pytorch_accuracy"] == 1.0
    assert result["tflite_accuracy"] == 0.5
    assert result["pytorch_quality_pass"] is True
    assert result["tflite_quality_pass"] is True


def test_adapter_comparison_passes_exact_prediction_parity():
    pytorch = np.asarray([[0.9, 0.1], [0.4, 0.6]], dtype=np.float32)
    tflite = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32)
    labels = np.asarray([0, 1])

    result = comparison_summary(pytorch, tflite, labels, quality_target=0.75)

    assert result["exact_prediction_parity"] is True
    assert result["pytorch_quality_pass"] is True
    assert result["tflite_quality_pass"] is True
