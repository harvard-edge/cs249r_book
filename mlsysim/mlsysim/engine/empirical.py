"""Domain-reviewed empirical anchors for MLSysIM sanity checks.

These anchors are not solver defaults and do not define model or hardware
facts. They bind canonical ``Models.*`` and ``Hardware.*`` entries to sourced
``Literature.Benchmarks`` values so tests can catch formula or registry drift
without duplicating the underlying data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.provenance import Sourced
from ..core.provenance import sourced
from ..hardware.registry import Hardware
from ..literature.registry import Literature
from ..models.registry import Models


@dataclass(frozen=True)
class EmpiricalAnchor:
    """One sourced sanity range for a representative MLSysIM calculation."""

    id: str
    solver: str
    metric: str
    units: str
    configuration: str
    registry_paths: tuple[str, ...] = ()
    workload: Any | None = None
    hardware: Any | None = None
    solver_kwargs: dict[str, Any] = field(default_factory=dict)
    target: Sourced | None = None
    lower: Sourced | None = None
    upper: Sourced | None = None
    rel_tolerance: float | None = None
    review_notes: str = ""

    def accepts(self, value: float) -> bool:
        """Return whether a solver result remains inside the reviewed envelope.

        An anchor is one of two mutually exclusive shapes:

        - **target anchor**: ``target`` set; accepts when
          ``|value - target| <= |target| * rel_tolerance`` (symmetric
          relative band — ``rel_tolerance`` is then mandatory);
        - **range anchor**: ``lower`` and ``upper`` set; accepts when
          ``lower <= value <= upper`` (inclusive).

        ``value`` must already be expressed in this anchor's ``units``.
        Raises ``ValueError`` when the anchor's fields do not form a complete
        shape.
        """
        if self.target is not None:
            if self.rel_tolerance is None:
                raise ValueError(f"{self.id} target anchor requires rel_tolerance")
            allowed_error = abs(float(self.target)) * self.rel_tolerance
            return abs(value - float(self.target)) <= allowed_error
        if self.lower is None or self.upper is None:
            raise ValueError(f"{self.id} range anchor requires lower and upper")
        return float(self.lower) <= value <= float(self.upper)

    @property
    def provenance_ids(self) -> tuple[str, ...]:
        """Stable provenance identifiers backing this anchor's numeric contract."""
        values = [self.target, self.lower, self.upper]
        return tuple(
            value.provenance.id
            for value in values
            if value is not None and value.provenance.id is not None
        )


def _quantity_target(quantity: Any, *, provenance: Any, name: str, description: str) -> Sourced:
    """Use an existing registry quantity as a scalar empirical target.

    This keeps validation anchors from duplicating model or hardware values in
    ``Literature.Benchmarks`` when the measured value already belongs to a typed
    registry entry.
    """
    return sourced(quantity.m_as("flop"), provenance, name=name, description=description)


RESNET50_A100_TRAIN_BS256 = EmpiricalAnchor(
    id="resnet50_a100_train_bs256",
    solver="SingleNodeModel",
    metric="throughput",
    units="1/s",
    configuration="ResNet-50 training on A100, batch_size=256, efficiency=0.26",
    registry_paths=("Models.Vision.ResNet50", "Hardware.Cloud.A100", "Literature.Benchmarks.ResNet50A100TrainThroughput"),
    workload=Models.Vision.ResNet50,
    hardware=Hardware.Cloud.A100,
    solver_kwargs={"batch_size": 256, "efficiency": 0.26, "is_training": True},
    target=Literature.Benchmarks.ResNet50A100TrainThroughput,
    rel_tolerance=0.10,
    review_notes=(
        "The 10% band is intentionally wider than MLPerf run-to-run variation because "
        "MLSYSIM uses a first-order roofline plus a calibrated efficiency term."
    ),
)

RESNET50_H100_TRAIN_BS256 = EmpiricalAnchor(
    id="resnet50_h100_train_bs256",
    solver="SingleNodeModel",
    metric="throughput",
    units="1/s",
    configuration="ResNet-50 training on H100, batch_size=256, efficiency=0.13",
    registry_paths=("Models.Vision.ResNet50", "Hardware.Cloud.H100", "Literature.Benchmarks.ResNet50H100TrainThroughput"),
    workload=Models.Vision.ResNet50,
    hardware=Hardware.Cloud.H100,
    solver_kwargs={"batch_size": 256, "efficiency": 0.13, "is_training": True},
    target=Literature.Benchmarks.ResNet50H100TrainThroughput,
    rel_tolerance=0.10,
    review_notes=(
        "The calibrated efficiency is lower than A100 because ResNet-50 does not "
        "saturate H100 peak tensor throughput in this first-order model."
    ),
)

LLAMA3_8B_H100_BS1_ITL = EmpiricalAnchor(
    id="llama3_8b_h100_bs1_itl",
    solver="ServingModel",
    metric="inter_token_latency",
    units="ms",
    configuration="Llama-3-8B serving on H100, seq_len=2048, batch_size=1, efficiency=0.5",
    registry_paths=(
        "Models.Language.Llama3_8B",
        "Hardware.Cloud.H100",
        "Literature.Benchmarks.Llama3_8B_H100_ITLLower",
        "Literature.Benchmarks.Llama3_8B_H100_ITLUpper",
    ),
    workload=Models.Language.Llama3_8B,
    hardware=Hardware.Cloud.H100,
    solver_kwargs={"seq_len": 2048, "batch_size": 1, "efficiency": 0.5},
    lower=Literature.Benchmarks.Llama3_8B_H100_ITLLower,
    upper=Literature.Benchmarks.Llama3_8B_H100_ITLUpper,
    review_notes=(
        "Decode is memory-bandwidth dominated; this range is a sanity envelope, "
        "not a production SLA for a specific scheduler or prompt distribution."
    ),
)

GPT3_175B_TRAINING_FLOPS = EmpiricalAnchor(
    id="gpt3_175b_training_flops",
    solver="calc_transformer_training_flops",
    metric="training_flops",
    units="flop",
    configuration="175B parameters, 300B training tokens, 6PD transformer training rule",
    registry_paths=("Models.Language.GPT3",),
    workload=Models.Language.GPT3,
    target=_quantity_target(
        Models.Language.GPT3.training_ops,
        provenance=Models.Language.GPT3.metadata.provenance,
        name="GPT-3 175B training FLOPs",
        description="Reported GPT-3 training FLOP anchor from the model registry.",
    ),
    rel_tolerance=0.01,
    review_notes=(
        "The tight band is appropriate here because the comparison is formula-level "
        "FLOP accounting, not a hardware throughput measurement."
    ),
)


EMPIRICAL_ANCHORS = (
    RESNET50_A100_TRAIN_BS256,
    RESNET50_H100_TRAIN_BS256,
    LLAMA3_8B_H100_BS1_ITL,
    GPT3_175B_TRAINING_FLOPS,
)


def anchor_by_id(anchor_id: str) -> EmpiricalAnchor:
    """Return a validation anchor by stable id."""
    for anchor in EMPIRICAL_ANCHORS:
        if anchor.id == anchor_id:
            return anchor
    raise KeyError(anchor_id)


__all__ = [
    "EmpiricalAnchor",
    "EMPIRICAL_ANCHORS",
    "GPT3_175B_TRAINING_FLOPS",
    "LLAMA3_8B_H100_BS1_ITL",
    "RESNET50_A100_TRAIN_BS256",
    "RESNET50_H100_TRAIN_BS256",
    "anchor_by_id",
]
