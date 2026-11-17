"""Measured accuracy curves for compression (opt-in).

``CompressionModel`` estimates the accuracy cost of quantization and pruning
with model-independent defaults. Measured cost depends on the model, the
method, the calibration or retraining recipe, and the task. An
``AccuracyCurve`` records what one experiment measured, so a caller can pass
it to ``CompressionModel`` in place of the default for that one case.

A curve is tied to one model, one method, one metric and one dataset. It is a
list of measured operating points, each a compression level and the accuracy
measured there, plus the accuracy of the uncompressed model in the same
harness. Between two adjacent measured points the curve interpolates
linearly. Outside the measured range it refuses to answer, because measured
curves are often flat and then fall sharply, and the position of the fall
differs from model to model.

Nothing here changes a default. ``CompressionModel`` uses a curve only when a
caller passes one.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Literal, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.provenance import Provenance

_DATA = Path(__file__).parent / "data" / "accuracy_curves.yaml"
_TOLERANCE = 1e-9

Axis = Literal["bitwidth", "sparsity"]


class AccuracyCurve(BaseModel):
    """Accuracy of one model, measured at several compression levels.

    Attributes
    ----------
    name : str
        Short identifier, shown in solver traces.
    axis : {"bitwidth", "sparsity"}
        What the compression level counts. ``"bitwidth"`` is bits per weight
        after quantization. ``"sparsity"`` is the fraction of weights removed.
    measured_model : str
        The model and weights that were measured.
    method : str
        The compression method as run, including granularity and calibration
        (for example "round-to-nearest, per-channel, weights only").
    metric : str
        The accuracy metric, which must be a fraction where higher is better.
    dataset : str
        The evaluation data.
    retrained : bool
        Whether the weights were trained or fine-tuned during or after
        compression.
    baseline_accuracy : float
        Accuracy of the uncompressed model in the same harness, in (0, 1].
    points : tuple of (float, float)
        Measured operating points as (compression level, accuracy), with
        accuracy in [0, 1]. Levels must be distinct. They are stored sorted.
    provenance : Provenance
        Where the measurements come from.

    Notes
    -----
    Perplexity and other lower-is-better metrics are outside this type. The
    solver reports an accuracy change in percentage points, and a perplexity
    difference is not one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    axis: Axis
    measured_model: str = Field(min_length=1)
    method: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    retrained: bool
    baseline_accuracy: float
    points: Tuple[Tuple[float, float], ...]
    provenance: Provenance

    @model_validator(mode="after")
    def _validate_points(self) -> AccuracyCurve:
        if not (math.isfinite(self.baseline_accuracy) and 0.0 < self.baseline_accuracy <= 1.0):
            raise ValueError("baseline_accuracy must be a fraction in (0, 1]")
        if not self.points:
            raise ValueError("an accuracy curve needs at least one measured point")
        for level, accuracy in self.points:
            if not (math.isfinite(level) and math.isfinite(accuracy)):
                raise ValueError("curve points must be finite")
            if not 0.0 <= accuracy <= 1.0:
                raise ValueError("measured accuracy must be a fraction in [0, 1]")
            if self.axis == "bitwidth" and level < 1.0:
                raise ValueError("a bitwidth curve needs at least one bit per weight at every point")
            if self.axis == "sparsity" and not 0.0 <= level <= 1.0:
                raise ValueError("a sparsity curve needs every level in [0, 1]")
        ordered = tuple(sorted((float(level), float(accuracy)) for level, accuracy in self.points))
        if any(b[0] - a[0] <= _TOLERANCE for a, b in zip(ordered, ordered[1:])):
            raise ValueError("curve points must have distinct compression levels")
        object.__setattr__(self, "points", ordered)
        return self

    # Provenance is not hashable, so hash the fields that identify the curve.
    def __hash__(self) -> int:
        return hash((self.name, self.axis, self.measured_model, self.method, self.baseline_accuracy, self.points))

    @property
    def measured_range(self) -> Tuple[float, float]:
        """Return the lowest and highest measured compression level.

        Returns
        -------
        tuple of float
            ``(lowest, highest)`` in the units of ``axis``.
        """
        return self.points[0][0], self.points[-1][0]

    def is_measured(self, level: float) -> bool:
        """Return whether ``level`` is one of the measured points.

        Parameters
        ----------
        level : float
            Compression level in the units of ``axis``.

        Returns
        -------
        bool
            True when a measured point has this level.
        """
        return any(abs(level - measured) <= _TOLERANCE for measured, _ in self.points)

    def accuracy_at(self, level: float) -> float:
        """Return the accuracy at one compression level.

        Parameters
        ----------
        level : float
            Compression level in the units of ``axis``: bits per weight for a
            bitwidth curve, fraction of weights removed for a sparsity curve.

        Returns
        -------
        float
            Accuracy as a fraction. It is the measured value at a measured
            point, and a linear interpolation between the two adjacent
            measured points otherwise.

        Raises
        ------
        ValueError
            If ``level`` is not finite or is outside the measured range. The
            curve does not extrapolate.

        Notes
        -----
        Equation: for adjacent measured points (x0, a0) and (x1, a1) with
        x0 <= level <= x1, accuracy = a0 + (a1 - a0) * (level - x0) / (x1 - x0).

        Assumptions: accuracy changes linearly between two adjacent measured
        points. A sharp fall can happen anywhere between two points, so
        measure closely spaced points where accuracy changes quickly.

        Excludes: extrapolation, uncertainty of the measured points, and any
        check that the curve was measured on the model being analysed.

        Examples
        --------
        >>> from mlsysim.engine.accuracy_curve import shipped_curve
        >>> curve = shipped_curve("qwen-7b-chat-gptq")
        >>> curve.accuracy_at(4)
        0.5486
        """
        lowest, highest = self.measured_range
        if not math.isfinite(level):
            raise ValueError(f"compression level must be finite, got {level!r}")
        if level < lowest - _TOLERANCE or level > highest + _TOLERANCE:
            unit = "bits per weight" if self.axis == "bitwidth" else "sparsity"
            raise ValueError(
                f"accuracy curve {self.name!r} was measured from {lowest:g} to {highest:g} {unit}; "
                f"{level:g} is outside that range and the curve does not extrapolate"
            )
        for measured, accuracy in self.points:
            if abs(level - measured) <= _TOLERANCE:
                return accuracy
        for (x0, a0), (x1, a1) in zip(self.points, self.points[1:]):
            if x0 < level < x1:
                return a0 + (a1 - a0) * (level - x0) / (x1 - x0)
        raise AssertionError("unreachable: level is inside the measured range")

    def delta_at(self, level: float) -> float:
        """Return the accuracy change from the uncompressed model.

        Parameters
        ----------
        level : float
            Compression level in the units of ``axis``.

        Returns
        -------
        float
            Signed change as a fraction, so -0.005 is 0.5 percentage points
            below the uncompressed model. This is the convention of
            ``CompressionResult.estimated_accuracy_delta``.

        Notes
        -----
        Equation: delta = accuracy_at(level) - baseline_accuracy.

        Assumptions: the baseline was measured in the same harness, on the
        same data, as every point.

        Excludes: relative accuracy (the ratio to the baseline), which some
        benchmarks use for their quality targets.

        Examples
        --------
        >>> from mlsysim.engine.accuracy_curve import shipped_curve
        >>> round(shipped_curve("qwen-7b-chat-gptq").delta_at(4), 4)
        -0.0224
        """
        return self.accuracy_at(level) - self.baseline_accuracy


@lru_cache(maxsize=None)
def _shipped() -> dict:
    return yaml.safe_load(_DATA.read_text())["curves"]


def shipped_curve_names() -> Tuple[str, ...]:
    """Return the names of the example curves that ship with the package.

    Returns
    -------
    tuple of str
        Names accepted by ``shipped_curve``, in file order.
    """
    return tuple(_shipped())


def shipped_curve(name: str) -> AccuracyCurve:
    """Return one of the example curves that ship with the package.

    Parameters
    ----------
    name : str
        One of ``shipped_curve_names()``.

    Returns
    -------
    AccuracyCurve
        The curve, with its provenance. Each one describes the model, method
        and data it was measured on. It is an example of the format, and it
        says nothing about any other model or method.

    Examples
    --------
    >>> from mlsysim.engine.accuracy_curve import shipped_curve
    >>> shipped_curve("qwen-7b-chat-gptq").axis
    'bitwidth'
    """
    curves = _shipped()
    if name not in curves:
        raise KeyError(f"no shipped accuracy curve named {name!r}; available: {sorted(curves)}")
    return AccuracyCurve(name=name, **curves[name])
