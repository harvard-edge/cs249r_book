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
        """Add host dispatch, kernel floors, weight reads, and image work.

        Parameters
        ----------
        n_launch : int
            Nonnegative number of host launches per forward pass.
        host_per_launch : Quantity
            Nonnegative host overhead per launch in time units.
        n_kernels : int
            Nonnegative number of device kernels per forward pass.
        weight_read : Quantity
            Nonnegative total weight-read time per batch.
        per_image : Quantity
            Nonnegative device execution time per image.
        batch : int, optional
            Positive number of images per forward pass; default one.

        Returns
        -------
        Quantity
            Modeled elapsed time for the entire batch, in milliseconds.

        Notes
        -----
        Equation: T = n_launch * host_per_launch + n_kernels * kernel_floor + weight_read +
        batch * per_image.

        Assumptions: The four components execute serially and all images share one weight read.
        Counts and times are supplied scenario assumptions.

        Excludes: Host/device overlap, queueing, intermediate activation transfers, and
        batch-dependent efficiency.

        References
        ----------
        Derivation: Summing nonoverlapping stage times gives elapsed time; the repeated
        per-image component scales with the number of images.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> p = resnet50_h100_profile()
        >>> p.engine_time(0, Q_(0,"s"), 0, Q_(0,"s"), Q_(1,"ms"), 2) == Q_(2,"ms")
        True
        """
        t = n_launch * host_per_launch + n_kernels * self.kernel_floor + weight_read + batch * per_image
        return t.to(millisecond)

    # Engines at a given batch size
    def eager_fp16(self, batch=1):
        """Estimate an eager FP16 forward pass with Python dispatch per kernel.

        Parameters
        ----------
        batch : int, optional
            Positive images per pass; default one.

        Returns
        -------
        Quantity
            Modeled full-batch latency in milliseconds.

        See Also
        --------
        ResNet50H100Profile.engine_time : Add the forward-pass timing components.

        Notes
        -----
        Equation: T = eager_kernels * (python_dispatch + kernel_launch + kernel_floor) +
        weight_read_fp16 + batch * per_image_fp16.

        Assumptions: One Python dispatch and driver launch per eager kernel, with no fusion.
        Per-image FP16 execution is extrapolated from the training-derived sustained fraction.

        Excludes: Actual framework tracing, asynchronous overlap, activation traffic, and
        measured eager-engine performance.

        References
        ----------
        Derivation: Substituting eager kernel and launch counts into engine_time retains all
        per-kernel host overhead and one batch weight read.

        Examples
        --------
        >>> p = resnet50_h100_profile()
        >>> p.eager_fp16(2) > p.eager_fp16(1)
        True
        """
        return self.engine_time(self.eager_kernels, self.python_dispatch + self.kernel_launch,
                                self.eager_kernels, self.weight_read_fp16, self.per_image_fp16, batch)

    def torchscript_fp16(self, batch=1):
        """Estimate FP16 execution without per-kernel Python dispatch.

        Parameters
        ----------
        batch : int, optional
            Positive images per pass; default one.

        Returns
        -------
        Quantity
            Modeled full-batch latency in milliseconds.

        See Also
        --------
        ResNet50H100Profile.engine_time : Add the forward-pass timing components.

        Notes
        -----
        Equation: T = eager_kernels * (kernel_launch + kernel_floor) + weight_read_fp16 + batch
        * per_image_fp16.

        Assumptions: The eager kernel partition is unchanged; only Python dispatch is removed in
        this scenario.

        Excludes: TorchScript-specific fusion, compilation, dynamic control flow, and measured
        runtime behavior.

        References
        ----------
        Derivation: Removing the Python component from each host launch reduces eager time by
        eager_kernels * python_dispatch.

        Examples
        --------
        >>> p = resnet50_h100_profile()
        >>> p.torchscript_fp16() < p.eager_fp16()
        True
        """
        return self.engine_time(self.eager_kernels, self.kernel_launch, self.eager_kernels,
                                self.weight_read_fp16, self.per_image_fp16, batch)

    def onnx_fp16(self, batch=1):
        """Estimate a fused FP16 graph with one host launch per fused kernel.

        Parameters
        ----------
        batch : int, optional
            Positive images per pass; default one.

        Returns
        -------
        Quantity
            Modeled full-batch latency in milliseconds.

        See Also
        --------
        ResNet50H100Profile.engine_time : Add the forward-pass timing components.

        Notes
        -----
        Equation: T = fused_kernels * (kernel_launch + kernel_floor) + weight_read_fp16 + batch
        * per_image_fp16.

        Assumptions: Batch normalization, activations, and residual additions are folded into
        convolution kernels using the stated fused count.

        Excludes: Backend-specific graph partitioning, unsupported operators, compilation, and
        measured ONNX Runtime performance.

        References
        ----------
        Derivation: Substituting the fused count for eager launches and kernels removes the
        eliminated kernels' host and device floors.

        Examples
        --------
        >>> p = resnet50_h100_profile()
        >>> p.onnx_fp16() < p.torchscript_fp16()
        True
        """
        return self.engine_time(self.fused_kernels, self.kernel_launch, self.fused_kernels,
                                self.weight_read_fp16, self.per_image_fp16, batch)

    def tensorrt(self, precision="fp16", batch=1):
        """Estimate a fused graph with one host launch at a chosen precision.

        Parameters
        ----------
        precision : str, optional
            One of "fp16" (default), "fp32", or "int8". The fp32 scenario uses TF32 Tensor Core
            compute with FP32 weights.
        batch : int, optional
            Positive images per pass; default one.

        Returns
        -------
        Quantity
            Modeled full-batch latency in milliseconds, including the assumed fixed overheads.

        See Also
        --------
        ResNet50H100Profile.engine_time : Add the forward-pass timing components.

        Notes
        -----
        Equation: T = kernel_launch + fused_kernels * kernel_floor + weight_read[precision] +
        batch * per_image[precision].

        Assumptions: One host launch dispatches the fused graph. FP16/TF32 work uses the
        training-derived sustained fraction; INT8 uses the reciprocal of a published Offline
        throughput as its per-image term.

        Excludes: TensorRT engine compilation, actual kernel scheduling, and direct measurement
        of single-request latency. Adding fixed terms to an Offline-rate term is a teaching
        extrapolation, not a decomposition measured by MLPerf.

        References
        ----------
        Derivation: One launch amortizes host overhead across the fused kernels. Precision
        selects both parameter storage width and the per-image execution account.

        Examples
        --------
        >>> p = resnet50_h100_profile()
        >>> p.tensorrt("fp16", 2) > p.tensorrt("fp16", 1)
        True
        """
        per_image, weights = {
            "fp32": (self.per_image_tf32, self.weight_read_fp32),   # FP32 engines run on TF32 Tensor Cores
            "fp16": (self.per_image_fp16, self.weight_read_fp16),
            "int8": (self.per_image_int8, self.weight_read_int8),
        }[precision]
        return self.engine_time(1, self.kernel_launch, self.fused_kernels, weights, per_image, batch)

    def large_batch_throughput(self, precision="fp16"):
        """Return the asymptotic image rate of the selected per-image term.

        Parameters
        ----------
        precision : str, optional
            One of "fp16" (default), "fp32", or "int8"; fp32 compute follows the TF32 scenario.

        Returns
        -------
        float
            Images per second: modeled for FP16/TF32, or the source-reported MLPerf Offline
            per-GPU rate for INT8.

        Notes
        -----
        Equation: For FP16 and FP32, rate = 1 / per_image_time; for INT8, return
        int8_offline_img_s.

        Assumptions: Fixed launch, kernel-floor, and weight-read costs vanish per image as batch
        grows without bound.

        Excludes: Memory-capacity limits, batch-dependent utilization, queueing, and changes in
        model quality at reduced precision.

        References
        ----------
        Derivation: Dividing batch size by T_fixed + batch * per_image and taking the
        large-batch limit leaves the reciprocal per-image time.

        Examples
        --------
        >>> p = resnet50_h100_profile()
        >>> p.large_batch_throughput("int8") == p.int8_offline_img_s
        True
        """
        if precision == "int8":
            return self.int8_offline_img_s
        per_image = {"fp32": self.per_image_tf32, "fp16": self.per_image_fp16}[precision]
        return (1 / per_image).to(1 / second).magnitude


def resnet50_h100_profile() -> ResNet50H100Profile:
    """Build the reference ResNet-50 H100 serving timing assumptions.

    Returns
    -------
    ResNet50H100Profile
        A frozen profile with dimensionless sustained fraction, per-image and weight-read times,
        runtime overheads, kernel counts, and source-reported INT8 Offline images/s.

    See Also
    --------
    ResNet50H100Profile.engine_time : Compose modeled batch latency from this profile.

    Notes
    -----
    Equation: sustained = A100_training_images_per_second * 3 * inference_flops /
    A100_peak_flops; per_image = inference_flops / (H100_precision_peak * sustained);
    weight_read = parameter_bytes / H100_bandwidth. INT8 per_image is the reciprocal Offline
    image rate.

    Assumptions: Training work is approximated as three inference passes; its A100 sustained
    fraction transfers to H100 inference and across FP16/TF32. Kernel counts and device floor
    are declared scenario assumptions, not profiler measurements.

    Excludes: A matched inference calibration for FP16/TF32, activation memory traffic,
    batch-dependent efficiency, and validation of branded-engine latency predictions.

    References
    ----------
    Derivation: The training anchor divided by peak throughput estimates a utilization fraction
    under the three-pass convention. Applying that fraction to H100 peak rates yields a modeled
    per-image time; bytes/bandwidth gives the weight-read floor. Registry Benchmarks entries
    retain source provenance for the training and INT8 measurements.

    Examples
    --------
    >>> p = resnet50_h100_profile()
    >>> p.sustained_fraction > 0 and p.per_image_fp16.to("s").magnitude > 0
    True
    """
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
