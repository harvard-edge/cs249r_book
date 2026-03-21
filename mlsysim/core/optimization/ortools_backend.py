import time
from typing import Any, Callable, Dict
from pydantic import BaseModel
from ortools.sat.python import cp_model
from .protocol import OptimizerProtocol, OptimizationResult

class ORToolsDiscreteBackend(OptimizerProtocol):
    """
    A discrete optimization backend using Google OR-Tools CP-SAT.
    Best suited for finding optimal integer configurations (e.g., TP/PP/DP splits)
    by leveraging advanced constraint satisfaction heuristics to instantly 
    eliminate invalid configurations without brute-force evaluating them.
    """
    def __init__(self):
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.variables: Dict[str, cp_model.IntVar] = {}
        self.status = None

    def compile(self, builder_fn: Callable[[cp_model.CpModel], Dict[str, cp_model.IntVar]]) -> Any:
        """
        Compiles the mathematical equations into the OR-Tools CP-SAT format.
        
        Args:
            builder_fn: A function that takes an empty cp_model.CpModel, 
                        applies variables, constraints, and objective, 
                        and returns a dict mapping variable names to cp_model.IntVar objects.
        """
        self.variables = builder_fn(self.model)
        return self

    def solve(self, time_limit_seconds: float = 60.0, **kwargs) -> OptimizationResult:
        """Executes the OR-Tools CP-SAT solver."""
        if not self.variables:
            raise ValueError("Optimizer must be compiled with variables before solving.")
            
        self.solver.parameters.max_time_in_seconds = time_limit_seconds
        
        start_time = time.time()
        self.status = self.solver.Solve(self.model)
        solve_time_ms = (time.time() - start_time) * 1000
        
        feasible = self.status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        
        best_config = {}
        optimal_value = 0.0
        
        if feasible:
            for name, var in self.variables.items():
                best_config[name] = self.solver.Value(var)
            
            # Extract objective value if one was set. If it's a pure satisfaction 
            # problem, ObjectiveValue() raises an exception or returns 0.
            try:
                optimal_value = self.solver.ObjectiveValue()
            except Exception:
                optimal_value = 0.0
                
        # Map OR-Tools status to a readable string
        status_str = "UNKNOWN"
        if self.status == cp_model.OPTIMAL: status_str = "OPTIMAL"
        elif self.status == cp_model.FEASIBLE: status_str = "FEASIBLE"
        elif self.status == cp_model.INFEASIBLE: status_str = "INFEASIBLE"
        elif self.status == cp_model.MODEL_INVALID: status_str = "MODEL_INVALID"
        
        return OptimizationResult(
            feasible=feasible,
            optimal_value=float(optimal_value),
            best_configuration=best_config,
            metrics={
                "ortools_status": status_str, 
                "conflicts": self.solver.NumConflicts(), 
                "branches": self.solver.NumBranches(),
                "wall_time_s": self.solver.WallTime()
            },
            solver_name="ortools.sat.python.cp_model",
            solve_time_ms=solve_time_ms
        )
