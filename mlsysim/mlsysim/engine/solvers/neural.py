"""Neural computation solvers: a dense network's counts, and one training epoch's time.

Domain implementations behind ``mlsysim.solvers`` (added 2026-10-20).
"""

from __future__ import annotations

from typing import List, Optional

from .base import ForwardModel
from ..results import DenseNetworkResult, EpochOverheadResult
from ...core.types import Quantity
from ...core.units import Q_
from ...hardware.types import HardwareNode
from ...physics.neural import (
    calc_dense_weights,
    calc_dense_biases,
    calc_dense_forward_flops,
    calc_dense_backward_flops,
    calc_dense_traffic_values,
    calc_dense_saved_activations,
    calc_dense_inference_activations,
    calc_training_step_ops,
    calc_training_state,
)


class DenseNetworkModel(ForwardModel):
    """
    Parameters, operations, bytes moved, and memory of a dense network at one batch size.

    The network is its layer widths. Forward work counts each layer's matrix
    multiply (2 B n_in n_out) and its bias add and activation (2 B n_out).
    Traffic assumes each layer runs as its own kernel: the weights are read
    once, the input and output cross memory once, and each hidden activation is
    written and read back. Training keeps every layer's output for the backward
    pass and holds a gradient per parameter plus the optimizer's state;
    inference keeps only the weights and the largest adjacent pair of
    activation buffers.

    Literature Source:
    1. Williams, Waterman, and Patterson (2009), "Roofline" (operations per byte moved).
    2. Rumelhart, Hinton, and Williams (1986), backpropagation (two products per layer).
    """
    requires = ("layer_widths",)
    produces = DenseNetworkResult

    def solve(self, layer_widths: List[int], batch_size: int, bytes_per_value: Quantity,
              optimizer_bytes_per_parameter: Quantity,
              training_examples: Optional[int] = None) -> DenseNetworkResult:
        """Count dense-network arithmetic, traffic, and resident training state.

        Parameters
        ----------
        layer_widths : list of int
            Positive widths including input and output, with at least two entries.
        batch_size : int
            Positive examples processed together in one step.
        bytes_per_value : Quantity
            Positive storage size of one weight, activation, or gradient value,
            expressed as bytes. This model uses a common representation for all three.
        optimizer_bytes_per_parameter : Quantity
            Nonnegative extra optimizer-state bytes per parameter, excluding gradients.
        training_examples : int, optional
            Nonnegative examples per epoch; whole-batch step count discards a remainder.

        Returns
        -------
        DenseNetworkResult
            Parameter and FLOP counts, byte footprints, FLOP/byte intensity, and
            training/inference ratios. No hardware capacity or feasibility is tested.

        See Also
        --------
        mlsysim.physics.neural.calc_dense_forward_flops : Forward-work accounting.
        mlsysim.physics.neural.calc_training_state : Weight, gradient, optimizer bytes.

        Notes
        -----
        Equation: layer weights = ``n_in*n_out``, biases = n_out, and forward work
        = ``2*B*sum(n_in*n_out)`` + ``2*B*sum(n_out)``. Training memory adds weights,
        gradients, optimizer state, and saved activations. Inference memory retains
        weights and the largest adjacent activation-buffer pair.

        Assumptions: fully connected layers execute separately. Traffic includes
        weight reads plus layer inputs/outputs; hidden activations cross memory
        between kernels. Training-step FLOPs count matrix products, whereas the
        reported forward FLOPs also include bias and activation work.

        Excludes: kernel fusion, reuse in caches, optimizer arithmetic, activation
        checkpointing, and device workspaces. Layer widths and batch size are cast
        to integers, so callers must supply integer counts rather than fractions.

        References
        ----------
        Derivation: count the coefficients of each dense matrix and bias vector;
        a multiply-accumulate contributes two FLOPs. Saved tensors and state arrays
        are counted by element cardinality times representation size.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> result = DenseNetworkModel().solve([2, 3, 1], 2, Q_("4 byte"), Q_("8 byte"))
        >>> result.parameters == (2 * 3 + 3) + (3 * 1 + 1)
        True
        >>> result.training_memory > result.inference_memory
        True
        """
        widths = [int(w) for w in layer_widths]
        b = int(batch_size)
        w = calc_dense_weights(widths)
        bias = calc_dense_biases(widths)
        params = [x + y for x, y in zip(w, bias)]
        P = sum(params)
        macs = sum(w)
        mm, ew = calc_dense_forward_flops(widths, b)
        fwd = sum(mm) + sum(ew)
        traffic_values = calc_dense_traffic_values(widths, b)
        weights_b, grads_b, opt_b = calc_training_state(P, bytes_per_value, optimizer_bytes_per_parameter)
        weights_b, grads_b, opt_b = weights_b.to("byte"), grads_b.to("byte"), opt_b.to("byte")
        traffic = (weights_b + traffic_values * bytes_per_value).to("byte")
        wgrad, igrad = calc_dense_backward_flops(widths, b)
        step = sum(mm) + wgrad + igrad
        saved = calc_dense_saved_activations(widths, b)
        inf_vals = calc_dense_inference_activations(widths, b)
        saved_b = (saved * bytes_per_value).to("byte")
        inf_b = (inf_vals * bytes_per_value).to("byte")
        train = weights_b + saved_b + grads_b + opt_b
        infer = weights_b + inf_b
        return DenseNetworkResult(
            layer_widths=widths, batch_size=b, layers=len(w),
            weights_per_layer=w, biases_per_layer=bias, params_per_layer=params,
            weights=macs, biases=sum(bias), parameters=P,
            macs_per_example=macs, flops_per_example=2 * macs,
            first_layer_share=w[0] / macs,
            matmul_flops_per_layer=mm, elementwise_flops_per_layer=ew,
            matmul_flops=sum(mm), elementwise_flops=sum(ew), forward_flops=fwd,
            forward_flops_per_example=fwd / b,
            matmul_share=sum(mm) / fwd, elementwise_share=sum(ew) / fwd,
            parameter_bytes=weights_b,
            traffic_values=traffic_values, traffic_bytes=traffic,
            intensity=(fwd * Q_(1, "flop") / traffic).to("flop/byte"),
            weight_gradient_flops=wgrad, input_gradient_flops=igrad,
            training_step_flops=step, training_flops_ratio=step / sum(mm),
            activation_values_per_layer=[b * x for x in widths],
            activation_values_per_example=sum(widths),
            saved_activation_values=saved, saved_activation_bytes=saved_b,
            inference_activation_values=inf_vals, inference_activation_bytes=inf_b,
            gradient_bytes=grads_b, optimizer_bytes=opt_b,
            training_memory=train, inference_memory=infer,
            training_memory_ratio=(train / infer).to("").magnitude,
            steps_per_epoch=None if training_examples is None else int(training_examples) // b,
        )


class EpochOverheadModel(ForwardModel):
    """
    One training epoch of a small network: arithmetic and data movement against fixed per-operation overhead.

    Work: steps x FLOPs per step at the attainable rate, the roofline's
    min(peak, bandwidth x intensity) at the network's batch intensity.
    Overhead: steps x operations launched per step x a fixed cost per
    operation, the framework's dispatch on the host and dispatch plus a kernel
    launch on the accelerator. Energy: DRAM energy for the bytes behind one
    FLOP, against the energy of that FLOP.

    Literature Source:
    1. Williams, Waterman, and Patterson (2009), "Roofline".
    2. Horowitz (2014), "Computing's energy problem" (per-op and per-byte energy).
    """
    requires = ("network", "host", "accelerator")
    produces = EpochOverheadResult

    def solve(self, network: DenseNetworkResult, network_batch1: DenseNetworkResult,
              host: HardwareNode, accelerator: HardwareNode, precision: str,
              dispatch: Quantity, launch: Quantity, steps: int,
              dram_energy_per_byte: Quantity, flop_energy: Quantity) -> EpochOverheadResult:
        """Compare epoch work and fixed launch overhead on two devices.

        Parameters
        ----------
        network : DenseNetworkResult
            Target-batch network counts, training-step FLOPs, and FLOP/byte intensity.
        network_batch1 : DenseNetworkResult
            Same architecture at batch one for the attainable-rate comparison.
        host : HardwareNode
            Host profile with positive memory bandwidth and selected precision rate.
        accelerator : HardwareNode
            Accelerator profile with positive bandwidth and selected precision rate.
        precision : str
            Key present in both devices' ``compute.precision_flops`` dictionaries.
        dispatch : Quantity
            Nonnegative host dispatch time per modeled operation.
        launch : Quantity
            Nonnegative additional accelerator kernel-launch time per operation.
        steps : int
            Positive number of batch steps in the epoch.
        dram_energy_per_byte : Quantity
            Nonnegative energy to transfer one byte, convertible to J/byte.
        flop_energy : Quantity
            Positive energy per arithmetic FLOP, convertible to J/flop.

        Returns
        -------
        EpochOverheadResult
            Per-device attainable FLOP/s, memory bandwidth, intensity ridge,
            work/overhead/total times in ms, and dimensionless speed and energy ratios.
            This result does not check memory residency.

        See Also
        --------
        mlsysim.physics.neural.calc_training_step_ops : Modeled launches per step.

        Notes
        -----
        Equation: rate = min(precision_peak, ``bandwidth*intensity)``;
        work_time = ``steps*training_step_flops/rate``; overhead_time = ``steps*ops_per_step``
        * per_operation_time. Host per-operation time is dispatch; accelerator time
        is dispatch + launch. Movement/arithmetic energy = energy_per_byte /
        ``(intensity*energy_per_flop)``.

        Assumptions: a common intensity represents epoch work, and overhead adds
        serially to arithmetic time. Each step has the operation count returned by
        ``calc_training_step_ops`` for the layer count.

        Excludes: overlap of launch and work, actual kernel traces, input loading,
        clock variation, and empirical power measurements.

        References
        ----------
        Derivation: the slower of compute and byte transfer limits the attainable
        rate; dividing work by that rate gives time. Independent launch costs add
        per operation, and energy rates convert bytes/FLOP into an energy ratio.

        Examples
        --------
        >>> from mlsysim import Hardware, Q_
        >>> network = DenseNetworkModel().solve([2, 3, 1], 1, Q_("4 byte"), Q_("8 byte"))
        >>> result = EpochOverheadModel().solve(network, network, Hardware.Cloud.H100,
        ...     Hardware.Cloud.H100, "fp32", Q_("1 us"), Q_("0 us"), 1,
        ...     Q_("1 pJ/byte"), Q_("1 pJ/flop"))
        >>> result.total_time["host"] == result.total_time["accelerator"]
        True
        """
        machines = {"host": host, "accelerator": accelerator}
        per_op = {"host": dispatch, "accelerator": dispatch + launch}
        step_flops = network.training_step_flops * Q_(1, "flop")
        ops = calc_training_step_ops(network.layers)
        peak, bw, ridge, r1, r, gain, work, ovh, tot = ({} for _ in range(9))
        for k, hw in machines.items():
            peak[k] = hw.compute.precision_flops[precision].to("flop/second")
            bw[k] = hw.memory.bandwidth.to("byte/second")
            ridge[k] = (peak[k] / bw[k]).to("flop/byte")
            r1[k] = min(peak[k], (bw[k] * network_batch1.intensity).to("flop/second"))
            r[k] = min(peak[k], (bw[k] * network.intensity).to("flop/second"))
            gain[k] = (r[k] / r1[k]).to("").magnitude
            work[k] = (steps * step_flops / r[k]).to("ms")
            ovh[k] = (steps * ops * per_op[k]).to("ms")
            tot[k] = work[k] + ovh[k]

        def movement(intensity):
            return (dram_energy_per_byte / intensity / flop_energy).to("").magnitude

        return EpochOverheadResult(
            steps=int(steps), step_flops=step_flops, ops_per_step=ops,
            peak=peak, bandwidth=bw, ridge=ridge, rate_batch1=r1, rate=r, batching_speedup=gain,
            work_time=work, overhead_time=ovh, total_time=tot,
            work_speedup=(work["host"] / work["accelerator"]).to("").magnitude,
            overhead_share_accelerator=(ovh["accelerator"] / tot["accelerator"]).to("").magnitude,
            movement_to_arithmetic_energy=movement(network.intensity),
            movement_to_arithmetic_energy_batch1=movement(network_batch1.intensity),
        )
