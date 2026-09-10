"""PyTorch adapter for the pinned MLPerf Tiny keyword-spotting model.

The adapter reads fused weights from the official float32 TFLite artifact. It
does not retrain, redesign, or approximate the model topology. The source model
is pinned to MLCommons Tiny commit
``1afd2c9820f795965a6134facd0b4dfae41ef23f``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mlperf.assets import (
    MLPERF_TINY_KWS_FLOAT_MODEL_SHA256,
    MLPERF_TINY_KWS_INT8_MODEL_SHA256,
    sha256_file,
)


class MLPerfTinyKWS(nn.Module):
    """Fused inference topology of the official 49x10 MFCC DS-CNN."""

    def __init__(self):
        super().__init__()
        self.initial = nn.Conv2d(1, 64, (10, 4), stride=2, bias=True)
        self.depthwise = nn.ModuleList(
            [nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=True) for _ in range(4)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv2d(64, 64, 1, bias=True) for _ in range(4)]
        )
        self.classifier = nn.Linear(64, 12)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # TensorFlow SAME padding is asymmetric for the even 10x4 kernel.
        x = F.relu(self.initial(F.pad(inputs, (1, 1, 4, 5))))
        for depthwise, pointwise in zip(self.depthwise, self.pointwise, strict=True):
            x = F.relu(depthwise(x))
            x = F.relu(pointwise(x))
        # The upstream Keras model fixes pool_size=(int(49/2), int(10/2)).
        x = F.avg_pool2d(x, kernel_size=(24, 5))
        return self.classifier(torch.flatten(x, 1))


def load_mlperf_tiny_kws(
    float_model_path: Path, int8_model_path: Path
) -> tuple[MLPerfTinyKWS, dict[str, float | int | str]]:
    """Load official fused TFLite weights into the exact PyTorch topology."""
    if sha256_file(float_model_path) != MLPERF_TINY_KWS_FLOAT_MODEL_SHA256:
        raise ValueError("MLPerf Tiny KWS float model does not match the pinned SHA-256")
    if sha256_file(int8_model_path) != MLPERF_TINY_KWS_INT8_MODEL_SHA256:
        raise ValueError("MLPerf Tiny KWS INT8 model does not match the pinned SHA-256")

    import tflite

    raw = float_model_path.read_bytes()
    graph = tflite.Model.GetRootAsModel(raw, 0)
    subgraph = graph.Subgraphs(0)

    model = MLPerfTinyKWS()
    pairs = (
        (model.initial, 17, 3, "conv"),
        (model.depthwise[0], 5, 4, "depthwise"),
        (model.pointwise[0], 18, 6, "conv"),
        (model.depthwise[1], 8, 7, "depthwise"),
        (model.pointwise[1], 19, 9, "conv"),
        (model.depthwise[2], 11, 10, "depthwise"),
        (model.pointwise[2], 20, 12, "conv"),
        (model.depthwise[3], 14, 13, "depthwise"),
        (model.pointwise[3], 21, 15, "conv"),
    )
    with torch.no_grad():
        for layer, weight_index, bias_index, kind in pairs:
            if subgraph.Tensors(weight_index).Type() == 9:
                weight = _dequantized_tensor(graph, subgraph, weight_index)
            else:
                weight = _tensor_array(graph, subgraph, weight_index)
            if kind == "depthwise":
                weight = weight.transpose(3, 0, 1, 2)
            else:
                weight = weight.transpose(0, 3, 1, 2)
            layer.weight.copy_(torch.from_numpy(weight.copy()))
            layer.bias.copy_(
                torch.from_numpy(_tensor_array(graph, subgraph, bias_index).copy())
            )
        model.classifier.weight.copy_(
            torch.from_numpy(_tensor_array(graph, subgraph, 16).copy())
        )
        model.classifier.bias.copy_(
            torch.from_numpy(_tensor_array(graph, subgraph, 1).copy())
        )

    int8_raw = int8_model_path.read_bytes()
    int8_graph = tflite.Model.GetRootAsModel(int8_raw, 0)
    int8_subgraph = int8_graph.Subgraphs(0)
    input_tensor = int8_subgraph.Tensors(int8_subgraph.Inputs(0))
    quantization = input_tensor.Quantization()
    scale = float(quantization.Scale(0))
    zero_point = int(quantization.ZeroPoint(0))
    if not np.isclose(scale, 0.5847029089927673) or zero_point != 83:
        raise ValueError("MLPerf Tiny KWS input quantization does not match the pinned contract")

    model.eval()
    metadata: dict[str, float | int | str] = {
        "float_model_sha256": f"sha256:{MLPERF_TINY_KWS_FLOAT_MODEL_SHA256}",
        "int8_model_sha256": f"sha256:{MLPERF_TINY_KWS_INT8_MODEL_SHA256}",
        "input_scale": scale,
        "input_zero_point": zero_point,
        "adapter": "fused-tflite-weights-to-pytorch-v1",
    }
    return model, metadata


def _tensor_array(graph, subgraph, index: int) -> np.ndarray:
    tensor = subgraph.Tensors(index)
    shape = tuple(tensor.Shape(i) for i in range(tensor.ShapeLength()))
    buffer = graph.Buffers(tensor.Buffer()).DataAsNumpy()
    return np.frombuffer(buffer.tobytes(), dtype=np.float32).reshape(shape)


def _dequantized_tensor(graph, subgraph, index: int) -> np.ndarray:
    tensor = subgraph.Tensors(index)
    shape = tuple(tensor.Shape(i) for i in range(tensor.ShapeLength()))
    buffer = graph.Buffers(tensor.Buffer()).DataAsNumpy()
    values = np.frombuffer(buffer.tobytes(), dtype=np.int8).reshape(shape)
    quantization = tensor.Quantization()
    scales = quantization.ScaleAsNumpy().astype(np.float32)
    zero_points = quantization.ZeroPointAsNumpy().astype(np.float32)
    axis = quantization.QuantizedDimension()
    broadcast_shape = [1] * values.ndim
    broadcast_shape[axis] = len(scales)
    return (values.astype(np.float32) - zero_points.reshape(broadcast_shape)) * scales.reshape(
        broadcast_shape
    )
