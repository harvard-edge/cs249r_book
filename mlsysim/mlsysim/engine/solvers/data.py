"""Data-ingestion and preprocessing pipeline solvers.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations


from ..results import (
    DataResult,
    TransformationResult,
)
from ...core.units import Q_
from ...core.types import Quantity
from ...hardware.types import HardwareNode
from .base import ForwardModel

class DataModel(ForwardModel):
    """
    Analyzes the 'Data Wall' — the throughput bottleneck between storage and compute.

    This model simulates the data pipeline constraints, comparing the data demand
    of a workload (e.g., training tokens or high-resolution video frames)
    against the physical bandwidth of the storage hierarchy and IO interconnects.

    Literature Source:
    1. Janapa Reddi et al. (2025), "Machine Learning Systems," Chapter 4 (Data Engineering).
    2. Beitzel et al. (2024), "The Data Wall: Scaling Laws for Data Ingestion in AI."
    3. Mohan et al. (2022), "Analyzing and Mitigating Data Bottlenecks in Deep Learning Training."
    """
    requires = ("workload", "hardware")
    produces = DataResult

    def solve(self, workload_data_rate: Quantity, hardware: HardwareNode) -> DataResult:
        """
        Solves for data pipeline feasibility.

        Parameters
        ----------
        workload_data_rate : Quantity
            The required data ingestion rate (e.g., TB/hour or GB/s).
        hardware : HardwareNode
            The hardware node with storage and interconnect specs.

        Returns
        -------
        DataResult
            Pipeline metrics including utilization and stall probability.
        """
        # 1. Resolve Hardware Supply
        storage_bw = getattr(hardware.storage, 'bandwidth', None) if hardware.storage else None
        io_bw = getattr(hardware.interconnect, 'bandwidth', None) if hardware.interconnect else None

        # The pipeline is limited by the slowest modeled data path. If a
        # subsystem is absent from the hardware spec, do not treat it as a
        # physical 0 GB/s link; use the paths that are actually described.
        available_paths = []
        if storage_bw is not None and storage_bw.to("GB/s").magnitude > 0:
            available_paths.append(("Storage", storage_bw.to("GB/s")))
        if io_bw is not None and io_bw.to("GB/s").magnitude > 0:
            available_paths.append(("Interconnect", io_bw.to("GB/s")))

        if available_paths:
            bottleneck, supply_bw = min(available_paths, key=lambda item: item[1].magnitude)
        else:
            bottleneck, supply_bw = "Unknown", Q_("0 GB/s")
        demand_bw = workload_data_rate.to("GB/s")

        # Utilization > 1.0 means demand exceeds the slowest link's supply: the
        # accelerator stalls waiting for data (the data wall binds).
        utilization = (demand_bw / supply_bw).magnitude if supply_bw.magnitude > 0 else float('inf')
        is_stalled = utilization > 1.0

        return DataResult(
            is_stalled=is_stalled,
            utilization=utilization,
            demand_bw=demand_bw,
            supply_bw=supply_bw,
            bottleneck=bottleneck,
            margin=(supply_bw - demand_bw).to("GB/s"),
        )

class TransformationModel(ForwardModel):
    """
    Quantifies the CPU preprocessing bottleneck (Wall 9: Transformation).

    This model simulates the 'Transformation Wall' — the gap between CPU-bound
    data preprocessing (JPEG decode, tokenization, augmentation) and
    accelerator step time. When preprocessing cannot keep up, the accelerator
    starves and utilization drops.

    Literature Source:
    1. Mohan et al. (2022), "Analyzing and Mitigating Data Bottlenecks in
       Deep Learning Training."
    2. Murray et al. (2021), "tf.data: A Machine Learning Data Processing
       Framework." (Pipeline stall analysis.)
    3. NVIDIA DALI Documentation (2024). (GPU-accelerated preprocessing.)
    """
    requires = ("hardware",)
    produces = TransformationResult

    def solve(self, batch_size: int, sample_size_bytes: Quantity,
              cpu_throughput: Quantity, accelerator_step_time: Quantity) -> TransformationResult:
        """
        Solves for CPU preprocessing bottleneck.

        Parameters
        ----------
        batch_size : int
            Number of samples per batch.
        sample_size_bytes : Quantity
            Size of one sample in bytes (e.g., Q_("500 KB")).
        cpu_throughput : Quantity
            CPU preprocessing throughput (e.g., Q_("2 GB/s")).
        accelerator_step_time : Quantity
            Time for one accelerator training step (e.g., Q_("50 ms")).

        Returns
        -------
        TransformationResult
            Transform time, bottleneck status, and accelerator utilization.
        """
        # T_transform = (B × S_sample) / C_throughput
        batch_data = batch_size * sample_size_bytes.to("byte")
        transform_time = (batch_data / cpu_throughput.to("byte/s")).to("ms")

        accel_time = accelerator_step_time.to("ms")
        is_bottleneck = transform_time.magnitude > accel_time.magnitude

        # Accelerator utilization: fraction of time the accelerator is active.
        # Assumes perfect pipelining (prefetch overlaps CPU work with the
        # accelerator step), so the steady-state step time is the max of the two
        # stages, not their sum.
        total_step_time = max(transform_time.magnitude, accel_time.magnitude)
        accelerator_utilization = accel_time.magnitude / total_step_time if total_step_time > 0 else 0.0

        return TransformationResult(
            transform_time=transform_time,
            accelerator_step_time=accel_time,
            is_bottleneck=is_bottleneck,
            accelerator_utilization=accelerator_utilization,
            slowdown_factor=total_step_time / accel_time.magnitude if accel_time.magnitude > 0 else float('inf'),
        )

__all__ = [
    "DataModel",
    "TransformationModel",
]
