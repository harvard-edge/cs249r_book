from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any, List
from .constants import ureg, Q_
from .types import Quantity

class EvaluationLevel(BaseModel):
    """A single tier in the Hierarchy of Constraints."""
    level_name: str
    status: str = "PASS" # PASS, FAIL, WARNING
    summary: str
    metrics: Dict[str, Any] = {}

class SystemEvaluation(BaseModel):
    """
    The multi-level 'Scorecard' for a System Simulation.
    Organizes results into the three pedagogical lenses.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_name: str
    
    # Level 1: Feasibility (The "Will it run?" check)
    feasibility: EvaluationLevel
    
    # Level 2: Performance (The "Is it fast enough?" check)
    performance: EvaluationLevel
    
    # Level 3: Macro (The "Is it worth it?" check)
    macro: EvaluationLevel

    def scorecard(self) -> str:
        """Generates a human-readable summary for students."""
        lines = [
            f"=== SYSTEM EVALUATION: {self.scenario_name} ===",
            f"Level 1: Feasibility -> [{self.feasibility.status}]",
            f"   {self.feasibility.summary}",
            f"Level 2: Performance -> [{self.performance.status}]",
            f"   {self.performance.summary}",
            f"Level 3: Macro/Economics -> [{self.macro.status}]",
            f"   {self.macro.summary}",
            "==============================================="
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Flattens the evaluation into a single-level dictionary for CSV/DataFrame export."""
        return {
            "scenario": self.scenario_name,
            "f_status": self.feasibility.status,
            "p_status": self.performance.status,
            "m_status": self.macro.status,
            **{f"f_{k}": v for k, v in self.feasibility.metrics.items()},
            **{f"p_{k}": v for k, v in self.performance.metrics.items()},
            **{f"m_{k}": v for k, v in self.macro.metrics.items()},
        }

    @property
    def passed_all(self) -> bool:
        return all(l.status == "PASS" for l in [self.feasibility, self.performance, self.macro])
