from __future__ import annotations

import torch

from mlperf.reference.tiny.mlperf_tiny_vww import MLPerfTinyVWW


def test_mlperf_tiny_vww_preserves_official_shape_and_parameter_count():
    model = MLPerfTinyVWW().eval()
    inputs = torch.zeros(2, 3, 96, 96)

    with torch.inference_mode():
        outputs = model(inputs)

    assert outputs.shape == (2, 2)
    assert torch.allclose(outputs.sum(dim=1), torch.ones(2))
    assert sum(parameter.numel() for parameter in model.parameters()) == 210_850
    assert [layer.stride for layer in model.depthwise] == [
        (1, 1),
        (2, 2),
        (1, 1),
        (2, 2),
        (1, 1),
        (2, 2),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (2, 2),
        (1, 1),
    ]
