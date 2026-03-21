from typing import Protocol, TypeVar, Any, Dict
from pydantic import BaseModel

class OptimizationResult(BaseModel):
    """Standardized output from any solver backend (SciPy, OR-Tools, etc.)"""
    feasible: bool
    optimal_value: float
    best_configuration: Dict[str, Any]
    metrics: Dict[str, Any]
    solver_name: str
    solve_time_ms: float

class OptimizerProtocol(Protocol):
    """Abstract protocol for defining mathematical optimization tasks."""
    
    def compile(self) -> Any:
        """Transforms the Python constraints into the specific solver backend's format."""
        ...
        
    def solve(self, **kwargs) -> OptimizationResult:
        """Executes the solver backend and returns a standardized result."""
        ...
