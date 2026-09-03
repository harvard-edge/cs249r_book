"""PyTorch adapter for the pinned MLPerf Tiny anomaly-detection model.

The adapter reads the fused dense weights from the official float32 TFLite
artifact. It preserves the complete 640-128-128-128-128-8-128-128-128-128-640
autoencoder and does not retrain, redesign, or approximate the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mlperf.assets import MLPERF_TINY_ANOMALY_FLOAT_MODEL_SHA256, sha256_file


class MLPerfTinyAnomalyAutoencoder(nn.Module):
    """Fused inference topology of the official ToyCar autoencoder."""

    _DIMENSIONS = (640, 128, 128, 128, 128, 8, 128, 128, 128, 128, 640)

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(input_size, output_size)
            for input_size, output_size in zip(
                self._DIMENSIONS[:-1], self._DIMENSIONS[1:], strict=True
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


def load_mlperf_tiny_anomaly(path: Path) -> MLPerfTinyAnomalyAutoencoder:
    """Load and verify official fused TFLite weights in the PyTorch graph."""
    if sha256_file(path) != MLPERF_TINY_ANOMALY_FLOAT_MODEL_SHA256:
        raise ValueError(
            "MLPerf Tiny anomaly float model does not match the pinned SHA-256"
        )

    import tflite

    graph = tflite.Model.GetRootAsModel(path.read_bytes(), 0)
    subgraph = graph.Subgraphs(0)
    input_shape = tuple(
        int(value) for value in subgraph.Tensors(subgraph.Inputs(0)).ShapeAsNumpy()
    )
    output_shape = tuple(
        int(value) for value in subgraph.Tensors(subgraph.Outputs(0)).ShapeAsNumpy()
    )
    if input_shape != (1, 640) or output_shape != (1, 640):
        raise ValueError("MLPerf Tiny anomaly model has an unexpected I/O contract")
    if subgraph.OperatorsLength() != 10:
        raise ValueError("MLPerf Tiny anomaly model has an unexpected operator count")

    model = MLPerfTinyAnomalyAutoencoder()
    with torch.no_grad():
        for layer, weight_index, bias_index in zip(
            model.layers, range(11, 21), range(1, 11), strict=True
        ):
            layer.weight.copy_(
                torch.from_numpy(
                    _tensor_array(
                        graph,
                        subgraph,
                        weight_index,
                        (layer.out_features, layer.in_features),
                    ).copy()
                )
            )
            layer.bias.copy_(
                torch.from_numpy(
                    _tensor_array(
                        graph, subgraph, bias_index, (layer.out_features,)
                    ).copy()
                )
            )
    return model.eval()


def _tensor_array(
    graph, subgraph, index: int, expected_shape: tuple[int, ...]
) -> np.ndarray:
    tensor = subgraph.Tensors(index)
    shape = tuple(int(value) for value in tensor.ShapeAsNumpy())
    if shape != expected_shape:
        raise ValueError(
            f"unexpected MLPerf Tiny anomaly tensor {index} shape {shape}, "
            f"expected {expected_shape}"
        )
    buffer = graph.Buffers(tensor.Buffer()).DataAsNumpy()
    values = np.frombuffer(buffer.tobytes(), dtype=np.float32)
    return values.reshape(shape)
