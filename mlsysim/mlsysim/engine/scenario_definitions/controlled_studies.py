"""Typed inputs for bounded, authored teaching studies."""

from typing import Dict, Optional
from pydantic import BaseModel, ConfigDict
from ...core.types import Quantity
from ...core.provenance import Provenance
from .base import BaseScenario


class TaskFactorCard(BaseModel):
    """One authored task factorization with a separate quality observation.

    Notes
    -----
    Equation: duration = ``samples * ops_per_sample * seconds_per_op``.
    Assumptions: All samples share the supplied work and realized execution rate.
    Excludes: Training dynamics and a predicted quality response.

    References
    ----------
    Derivation: Multiplying sample count, work per sample, and seconds per operation
    gives the declared host duration; quality is supplied independently.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> Scenarios.ControlledTaskFactors.cards["unverified"].quality_pct is None
    True
    """
    __module__ = "mlsysim.engine.scenario_types"
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    label: str
    samples: int
    ops_per_sample: Quantity
    seconds_per_op: Quantity
    quality_pct: Optional[float]
    factor: str


class TaskFactorStudyScenario(BaseScenario):
    """A bounded controlled study with authored task cards.

    Notes
    -----
    Equation: quality passes iff supplied quality is at least quality_floor_pct.
    Assumptions: Cards describe one fixed host and carry independent quality inputs.
    Excludes: Learned quality prediction, selection overhead, and device calibration.

    References
    ----------
    Derivation: Compare a separately supplied quality observation with the fixed
    acceptance floor; missing observations cannot establish acceptance.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> Scenarios.ControlledTaskFactors.cards["baseline"].samples > 0
    True
    """
    __module__ = "mlsysim.engine.scenario_types"
    version: str
    quality_floor_pct: float
    shift_options: tuple[int, ...]
    capability_options: tuple[float, ...]
    cards: Dict[str, TaskFactorCard]
    provenance: Provenance


class PlacementStudyScenario(BaseScenario):
    """Versioned authored placement inputs for explicit constraint screening.

    Notes
    -----
    Equation: fixture lookup = fixtures indexed by the requested track.
    Assumptions: Fixture quantities use their declared field units and version.
    Excludes: Measured deployment performance and certification of a target device.

    References
    ----------
    Implementation reference: ``mlsysim.engine.deployment_journeys.load_deployment_journeys``.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> "tinyml" in Scenarios.PlacementStudy.fixtures["tracks"]
    True
    """
    __module__ = "mlsysim.engine.scenario_types"
    version: str
    fixtures: dict
    provenance: Provenance
