"""Forward-pass time of a CNN served by different engines, from the iron law's additive terms.

One function, ``resnet50_h100_profile``, is the single source for the ResNet-50
serving profile on the reference H100. Every chapter that quotes a ResNet-50
H100 inference latency, throughput, or engine/precision comparison calls it, so
the numbers cannot drift between chapters.

Model (per forward pass of ``batch`` images):

    time = launches x host cost per launch          (Python dispatch + driver launch)
         + kernels x per-kernel device floor        (least device time one kernel takes)
         + weight bytes / HBM bandwidth             (one weight read per batch)
         + batch x per-image device time

Per-image FP16/TF32 device time uses the sustained fraction of peak implied by
the registry's ResNet-50 training rate on an A100
(``Benchmarks.ResNet50A100TrainThroughput``, the anchor the data engineering and
training chapters share). Per-image INT8 time is the MLPerf Inference v5.0
measured Offline rate per H100. Kernel counts and the per-kernel floor are
stated scenario assumptions.
"""
from dataclasses import dataclass

from ..core.units import ureg, BYTES_FP16, BYTES_FP32, BYTES_INT8
from ..hardware.registry import Hardware
from ..models.registry import Models
from ..ops.runtime import RuntimeOverheads
from ..literature.registry import Benchmarks

second = ureg.second
millisecond = ureg.millisecond
microsecond = ureg.microsecond
flop = ureg.flop

# Shared anchor (2026-10-13: was a hand-typed 3000 that disagreed with the
# registry's 3,200 (NVIDIA-measured); the chapters now all read the registry).
A100_SUSTAINED_TRAIN_IMG_S = float(Benchmarks.ResNet50A100TrainThroughput)

# Scenario assumptions (stated, not measured)
EAGER_KERNELS = 53 + 53 + 49 + 16 + 2 + 1   # conv, batch norm, ReLU, residual add, 2 pools, FC
FUSED_KERNELS = 53 + 2 + 1                  # BN, ReLU, add folded into each conv; 2 pools; FC
KERNEL_FLOOR = 5 * microsecond              # least device time one kernel takes at batch one


@dataclass(frozen=True)
class ResNet50H100Profile:
    sustained_fraction: float
    per_image_fp16: object
    per_image_tf32: object
    per_image_int8: object
    weight_read_fp16: object
    weight_read_fp32: object
    weight_read_int8: object
    python_dispatch: object
    kernel_launch: object
    kernel_floor: object
    eager_kernels: int
    fused_kernels: int
    int8_offline_img_s: float

    def engine_time(self, n_launch, host_per_launch, n_kernels, weight_read, per_image, batch=1):
        t = n_launch * host_per_launch + n_kernels * self.kernel_floor + weight_read + batch * per_image
        return t.to(millisecond)

    # Engines at a given batch size
    def eager_fp16(self, batch=1):
        return self.engine_time(self.eager_kernels, self.python_dispatch + self.kernel_launch,
                                self.eager_kernels, self.weight_read_fp16, self.per_image_fp16, batch)

    def torchscript_fp16(self, batch=1):
        return self.engine_time(self.eager_kernels, self.kernel_launch, self.eager_kernels,
                                self.weight_read_fp16, self.per_image_fp16, batch)

    def onnx_fp16(self, batch=1):
        return self.engine_time(self.fused_kernels, self.kernel_launch, self.fused_kernels,
                                self.weight_read_fp16, self.per_image_fp16, batch)

    def tensorrt(self, precision="fp16", batch=1):
        per_image, weights = {
            "fp32": (self.per_image_tf32, self.weight_read_fp32),   # FP32 engines run on TF32 Tensor Cores
            "fp16": (self.per_image_fp16, self.weight_read_fp16),
            "int8": (self.per_image_int8, self.weight_read_int8),
        }[precision]
        return self.engine_time(1, self.kernel_launch, self.fused_kernels, weights, per_image, batch)

    def large_batch_throughput(self, precision="fp16"):
        """Images/s at large batch: modeled for FP32 (TF32) and FP16, measured (MLPerf) for INT8."""
        if precision == "int8":
            return self.int8_offline_img_s
        per_image = {"fp32": self.per_image_tf32, "fp16": self.per_image_fp16}[precision]
        return (1 / per_image).to(1 / second).magnitude


def resnet50_h100_profile() -> ResNet50H100Profile:
    resnet = Models.Vision.ResNet50
    h100 = Hardware.Cloud.H100
    a100 = Hardware.Cloud.A100
    flops = resnet.inference_flops.to(flop)
    sustained = (A100_SUSTAINED_TRAIN_IMG_S * 3 * flops.magnitude) / a100.compute.peak_flops.to(flop / second).magnitude
    int8_img_s = Benchmarks.ResNet50H100InferenceOfflinePerGpu
    int8_img_s = float(getattr(int8_img_s, "value", int8_img_s))
    bw = h100.memory.bandwidth
    return ResNet50H100Profile(
        sustained_fraction=sustained,
        per_image_fp16=(flops / (h100.compute.precision_flops["fp16_dense"] * sustained)).to(microsecond),
        per_image_tf32=(flops / (h100.compute.precision_flops["tf32"] * sustained)).to(microsecond),
        per_image_int8=(1 / (int8_img_s / second)).to(microsecond),
        weight_read_fp16=(resnet.size_in_bytes(BYTES_FP16) / bw).to(microsecond),
        weight_read_fp32=(resnet.size_in_bytes(BYTES_FP32) / bw).to(microsecond),
        weight_read_int8=(resnet.size_in_bytes(BYTES_INT8) / bw).to(microsecond),
        python_dispatch=RuntimeOverheads.PythonDispatch,
        kernel_launch=RuntimeOverheads.KernelLaunch,
        kernel_floor=KERNEL_FLOOR,
        eager_kernels=EAGER_KERNELS,
        fused_kernels=FUSED_KERNELS,
        int8_offline_img_s=int8_img_s,
    )
