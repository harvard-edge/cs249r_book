"""PyTorch adapter for the pinned MLPerf Tiny visual-wake-words model.

The module preserves the official MobileNetV1 0.25 topology and reads fused
weights from the MLCommons float32 TFLite artifact. It does not retrain,
replace, or approximate the reference model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mlperf.assets import MLPERF_TINY_VWW_FLOAT_MODEL_SHA256, sha256_file


class _TensorFlowSameConv2d(nn.Conv2d):
    """Conv2d with TensorFlow SAME padding for arbitrary input dimensions."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        stride_height, stride_width = self.stride
        kernel_height, kernel_width = self.kernel_size
        output_height = (height + stride_height - 1) // stride_height
        output_width = (width + stride_width - 1) // stride_width
        pad_height = max(
            (output_height - 1) * stride_height + kernel_height - height, 0
        )
        pad_width = max((output_width - 1) * stride_width + kernel_width - width, 0)
        inputs = F.pad(
            inputs,
            (
                pad_width // 2,
                pad_width - pad_width // 2,
                pad_height // 2,
                pad_height - pad_height // 2,
            ),
        )
        return F.conv2d(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )


class MLPerfTinyVWW(nn.Module):
    """Fused inference topology of the official 96x96 MobileNetV1 0.25."""

    _DEPTHWISE_STRIDES = (1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1)
    _POINTWISE_CHANNELS = (16, 32, 32, 64, 64, 128, 128, 128, 128, 128, 128, 256, 256)

    def __init__(self) -> None:
        super().__init__()
        self.initial = _TensorFlowSameConv2d(3, 8, 3, stride=2, bias=True)
        depthwise: list[nn.Module] = []
        pointwise: list[nn.Module] = []
        channels = 8
        for stride, output_channels in zip(
            self._DEPTHWISE_STRIDES, self._POINTWISE_CHANNELS, strict=True
        ):
            depthwise.append(
                _TensorFlowSameConv2d(
                    channels,
                    channels,
                    3,
                    stride=stride,
                    groups=channels,
                    bias=True,
                )
            )
            pointwise.append(nn.Conv2d(channels, output_channels, 1, bias=True))
            channels = output_channels
        self.depthwise = nn.ModuleList(depthwise)
        self.pointwise = nn.ModuleList(pointwise)
        self.classifier = nn.Linear(256, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.initial(inputs))
        for depthwise, pointwise in zip(self.depthwise, self.pointwise, strict=True):
            x = F.relu(depthwise(x))
            x = F.relu(pointwise(x))
        x = x.mean(dim=(-2, -1))
        return torch.softmax(self.classifier(x), dim=1)


def load_mlperf_tiny_vww(path: Path) -> MLPerfTinyVWW:
    """Load and verify official fused TFLite weights into the PyTorch graph."""
    if sha256_file(path) != MLPERF_TINY_VWW_FLOAT_MODEL_SHA256:
        raise ValueError(
            "MLPerf Tiny VWW float model does not match the pinned SHA-256"
        )

    import tflite

    graph = tflite.Model.GetRootAsModel(path.read_bytes(), 0)
    subgraph = graph.Subgraphs(0)
    if tuple(subgraph.Tensors(subgraph.Inputs(0)).ShapeAsNumpy()) != (1, 96, 96, 3):
        raise ValueError("MLPerf Tiny VWW model has an unexpected input shape")
    if tuple(subgraph.Tensors(subgraph.Outputs(0)).ShapeAsNumpy()) != (1, 2):
        raise ValueError("MLPerf Tiny VWW model has an unexpected output shape")

    model = MLPerfTinyVWW()
    depthwise_weight_indices = (5, 33, 36, 39, 42, 8, 11, 14, 17, 20, 24, 27, 30)
    depthwise_bias_indices = (4, 32, 35, 38, 41, 7, 10, 13, 16, 19, 23, 26, 29)
    pointwise_weight_indices = (45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57)
    pointwise_bias_indices = (21, 34, 37, 40, 6, 9, 12, 15, 18, 22, 25, 28, 31)

    with torch.no_grad():
        model.initial.weight.copy_(
            torch.from_numpy(
                _tensor_array(graph, subgraph, 44, (8, 3, 3, 3))
                .transpose(0, 3, 1, 2)
                .copy()
            )
        )
        model.initial.bias.copy_(
            torch.from_numpy(_tensor_array(graph, subgraph, 3, (8,)).copy())
        )
        for position, (depthwise, pointwise) in enumerate(
            zip(model.depthwise, model.pointwise, strict=True)
        ):
            channels = depthwise.in_channels
            output_channels = pointwise.out_channels
            depthwise.weight.copy_(
                torch.from_numpy(
                    _tensor_array(
                        graph,
                        subgraph,
                        depthwise_weight_indices[position],
                        (1, 3, 3, channels),
                    )
                    .transpose(3, 0, 1, 2)
                    .copy()
                )
            )
            depthwise.bias.copy_(
                torch.from_numpy(
                    _tensor_array(
                        graph,
                        subgraph,
                        depthwise_bias_indices[position],
                        (channels,),
                    ).copy()
                )
            )
            pointwise.weight.copy_(
                torch.from_numpy(
                    _tensor_array(
                        graph,
                        subgraph,
                        pointwise_weight_indices[position],
                        (output_channels, 1, 1, channels),
                    )
                    .transpose(0, 3, 1, 2)
                    .copy()
                )
            )
            pointwise.bias.copy_(
                torch.from_numpy(
                    _tensor_array(
                        graph,
                        subgraph,
                        pointwise_bias_indices[position],
                        (output_channels,),
                    ).copy()
                )
            )
        model.classifier.weight.copy_(
            torch.from_numpy(_tensor_array(graph, subgraph, 43, (2, 256)).copy())
        )
        model.classifier.bias.copy_(
            torch.from_numpy(_tensor_array(graph, subgraph, 1, (2,)).copy())
        )
    return model.eval()


def _tensor_array(
    graph, subgraph, index: int, expected_shape: tuple[int, ...]
) -> np.ndarray:
    tensor = subgraph.Tensors(index)
    shape = tuple(int(value) for value in tensor.ShapeAsNumpy())
    if shape != expected_shape:
        raise ValueError(
            f"unexpected MLPerf Tiny VWW tensor {index} shape {shape}, "
            f"expected {expected_shape}"
        )
    buffer = graph.Buffers(tensor.Buffer()).DataAsNumpy()
    values = np.frombuffer(buffer.tobytes(), dtype=np.float32)
    return values.reshape(shape)
