"""Typed result models for base calculations."""

from __future__ import annotations

from typing import Any, Dict, Optional, List
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["SolverResult", "OptimizerResult"]


class SolverResult(BaseModel):
    """Base class for all model and solver results."""

    __module__ = "mlsysim.engine.results"

    model_config = ConfigDict(arbitrary_types_allowed=True)
    constraint_trace: Optional[List[str]] = Field(
        default=None,
        description="A trace of physical constraints evaluated during the solve. If feasible=False, this explains why.",
    )


class OptimizerResult(BaseModel):
    """Base class for all search-based results."""

    __module__ = "mlsysim.engine.results"

    model_config = ConfigDict(arbitrary_types_allowed=True)
    objective_value: float
    best_config: Dict[str, Any]
    total_searched: int
