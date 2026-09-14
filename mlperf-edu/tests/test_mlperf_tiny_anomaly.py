from __future__ import annotations

import torch

from mlperf.reference.tiny.mlperf_tiny_anomaly import MLPerfTinyAnomalyAutoencoder


def test_mlperf_tiny_anomaly_preserves_official_shape_and_parameter_count():
    model = MLPerfTinyAnomalyAutoencoder().eval()
    inputs = torch.zeros(2, 640)

    with torch.inference_mode():
        outputs = model(inputs)

    assert outputs.shape == (2, 640)
    assert sum(parameter.numel() for parameter in model.parameters()) == 265_864
    assert [(layer.in_features, layer.out_features) for layer in model.layers] == [
        (640, 128),
        (128, 128),
        (128, 128),
        (128, 128),
        (128, 8),
        (8, 128),
        (128, 128),
        (128, 128),
        (128, 128),
        (128, 640),
    ]
