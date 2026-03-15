import time
from typing import Any, Callable, Dict, Tuple, List
import scipy.optimize
from .protocol import OptimizerProtocol, OptimizationResult

class ExhaustiveBackend(OptimizerProtocol):
    """
    A discrete optimization backend using SciPy's brute force grid search.
    Best suited for small, highly non-linear, 1D/2D parameter spaces with 
    physical discontinuities (like queueing models hitting infinity) where 
    gradient descent fails.
    """
    def __init__(self):
        self.objective_fn: Callable = None
        self.ranges: List[Tuple[float, float]] = None
        self.grid_size: int = None

    def compile(self, objective_fn: Callable[[Any], float], ranges: List[Tuple[float, float]], grid_size: int = 100) -> Any:
        """
        Compiles the exhaustive search grid.
        
        Args:
            objective_fn: The mathematical function to MINIMIZE.
            ranges: A list of tuples defining the (min, max) bounds for each variable.
            grid_size: The number of discrete points to sample along each dimension.
        """
        self.objective_fn = objective_fn
        self.ranges = ranges
        self.grid_size = grid_size
        return self

    def solve(self, **kwargs) -> OptimizationResult:
        """Executes the SciPy brute force solver."""
        if not self.objective_fn:
            raise ValueError("Optimizer must be compiled before solving.")
            
        start_time = time.time()
        
        # scipy.optimize.brute evaluates the function at every point on the grid
        # and returns the global minimum found.
        res = scipy.optimize.brute(
            self.objective_fn, 
            ranges=self.ranges, 
            Ns=self.grid_size,
            full_output=True,
            finish=None # Do not run a local optimizer (like fmin) at the end, stay strictly on the grid
        )
        
        solve_time_ms = (time.time() - start_time) * 1000
        
        best_val = float(res[0][0]) if hasattr(res[0], '__len__') else float(res[0])
        optimal_obj = float(res[1])
        
        # If the objective value is functionally infinity, it means no feasible solution was found
        feasible = optimal_obj < 1e10
        
        return OptimizationResult(
            feasible=feasible,
            optimal_value=optimal_obj,
            best_configuration={"optimal_variable": best_val},
            metrics={"grid_points_evaluated": (self.grid_size ** len(self.ranges))},
            solver_name="scipy.optimize.brute",
            solve_time_ms=solve_time_ms
        )
