"""PyTorch adapters for the MLPerf Tiny ResNet8 CIFAR-10 model.

The architecture is transcribed from the official MLCommons Tiny repository at
commit ``1afd2c9820f795965a6134facd0b4dfae41ef23f``. The pinned upstream
``utils/model.py`` SHA-256 is
``4eb6abbe58eaf4b4c6c5cb2f2988636b172a206652d151282daf8041e8bc8d6b``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MLPERF_TINY_FLOAT_MODEL_SHA256 = (
    "b5c0046d6e0328b4956afd6baa29555a29b1f1c65bdd45aaed75b7cd484d9f79"
)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.block(inputs) + self.residual(inputs))


class MLPerfTinyResNet8(nn.Module):
    """Exact layer topology of the official MLPerf Tiny PyTorch ResNet8."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.first_stack = ResNetBlock(16, 16, stride=1)
        self.second_stack = ResNetBlock(16, 32, stride=2)
        self.third_stack = ResNetBlock(32, 64, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.stem(inputs)
        x = self.first_stack(x)
        x = self.second_stack(x)
        x = self.third_stack(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


class _KerasSameConv2d(nn.Conv2d):
    """Conv2d with the asymmetric stride-two padding used by TensorFlow SAME."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.stride == (2, 2) and self.kernel_size == (3, 3):
            inputs = F.pad(inputs, (0, 1, 0, 1))
        return F.conv2d(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class MLPerfTinyFusedResNet8(nn.Module):
    """Float inference graph represented by the official fused TFLite model."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = _KerasSameConv2d(3, 16, 3, padding=1)
        self.block1_conv1 = _KerasSameConv2d(16, 16, 3, padding=1)
        self.block1_conv2 = _KerasSameConv2d(16, 16, 3, padding=1)
        self.block2_conv1 = _KerasSameConv2d(16, 32, 3, stride=2, padding=0)
        self.block2_conv2 = _KerasSameConv2d(32, 32, 3, padding=1)
        self.block2_skip = _KerasSameConv2d(16, 32, 1, stride=2)
        self.block3_conv1 = _KerasSameConv2d(32, 64, 3, stride=2, padding=0)
        self.block3_conv2 = _KerasSameConv2d(64, 64, 3, padding=1)
        self.block3_skip = _KerasSameConv2d(32, 64, 1, stride=2)
        self.fc = nn.Linear(64, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(inputs))
        x = F.relu(self.block1_conv2(F.relu(self.block1_conv1(x))) + x)
        x = F.relu(
            self.block2_conv2(F.relu(self.block2_conv1(x))) + self.block2_skip(x)
        )
        x = F.relu(
            self.block3_conv2(F.relu(self.block3_conv1(x))) + self.block3_skip(x)
        )
        return self.fc(x.mean(dim=(-2, -1)))


def load_mlperf_tiny_float_resnet(path: Path) -> MLPerfTinyFusedResNet8:
    """Load and verify the exact official float TFLite weights into PyTorch."""
    import tflite

    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != MLPERF_TINY_FLOAT_MODEL_SHA256:
        raise ValueError("MLPerf Tiny image model failed SHA-256 verification")
    tflite_model = tflite.Model.GetRootAsModel(payload, 0)
    subgraph = tflite_model.Subgraphs(0)

    def tensor_array(index: int, expected_shape: tuple[int, ...]) -> np.ndarray:
        tensor = subgraph.Tensors(index)
        shape = tuple(tensor.Shape(i) for i in range(tensor.ShapeLength()))
        if shape != expected_shape:
            raise ValueError(
                f"unexpected MLPerf Tiny TFLite tensor {index} shape: {shape}"
            )
        buffer = tflite_model.Buffers(tensor.Buffer())
        values = np.frombuffer(bytes(buffer.DataAsNumpy()), dtype=np.float32)
        return values.reshape(shape).copy()

    model = MLPerfTinyFusedResNet8()
    convolution_tensors = (
        (model.stem, 8, 3),
        (model.block1_conv1, 9, 4),
        (model.block1_conv2, 10, 17),
        (model.block2_conv1, 11, 5),
        (model.block2_conv2, 12, 18),
        (model.block2_skip, 13, 19),
        (model.block3_conv1, 14, 6),
        (model.block3_conv2, 15, 20),
        (model.block3_skip, 16, 21),
    )
    with torch.no_grad():
        for layer, weight_index, bias_index in convolution_tensors:
            out_channels, in_channels, height, width = layer.weight.shape
            weight = tensor_array(
                weight_index, (out_channels, height, width, in_channels)
            ).transpose(0, 3, 1, 2)
            bias = tensor_array(bias_index, (out_channels,))
            layer.weight.copy_(torch.from_numpy(weight.copy()))
            layer.bias.copy_(torch.from_numpy(bias))
        model.fc.weight.copy_(torch.from_numpy(tensor_array(7, (10, 64))))
        model.fc.bias.copy_(torch.from_numpy(tensor_array(1, (10,))))
    return model.eval()
