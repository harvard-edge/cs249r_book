import time
from typing import Any, Callable, List, Tuple
import numpy as np
from .protocol import OptimizerProtocol, OptimizationResult


class ExhaustiveBackend(OptimizerProtocol):
    """
    A brute-force grid search backend using only NumPy.

    Evaluates the objective at every point on a uniform grid and returns
    the global minimum.  Best suited for small, highly non-linear, 1D/2D
    parameter spaces with physical discontinuities (like queueing models
    hitting infinity) where gradient descent fails.
    """
    def __init__(self):
        self.objective_fn: Callable = None
        self.ranges: List[Tuple[float, float]] = None
        self.grid_size: int = None

    def compile(self, objective_fn: Callable[[Any], float],
                ranges: List[Tuple[float, float]],
                grid_size: int = 100) -> Any:
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
        """Evaluates every grid point and returns the global minimum."""
        if not self.objective_fn:
            raise ValueError("Optimizer must be compiled before solving.")

        start_time = time.time()

        if len(self.ranges) == 1:
            lo, hi = self.ranges[0]
            grid = np.linspace(lo, hi, self.grid_size)
            values = np.array([self.objective_fn(np.array([x])) for x in grid])
            best_idx = int(np.argmin(values))
            best_val = float(grid[best_idx])
            optimal_obj = float(values[best_idx])
        else:
            axes = [np.linspace(lo, hi, self.grid_size) for lo, hi in self.ranges]
            mesh = np.meshgrid(*axes, indexing="ij")
            flat_points = np.column_stack([m.ravel() for m in mesh])
            values = np.array([self.objective_fn(pt) for pt in flat_points])
            best_idx = int(np.argmin(values))
            best_val = float(flat_points[best_idx, 0])
            optimal_obj = float(values[best_idx])

        solve_time_ms = (time.time() - start_time) * 1000

        feasible = optimal_obj < 1e10

        total_points = self.grid_size ** len(self.ranges)

        return OptimizationResult(
            feasible=feasible,
            optimal_value=optimal_obj,
            best_configuration={"optimal_variable": best_val},
            metrics={"grid_points_evaluated": total_points},
            solver_name="numpy.exhaustive_grid",
            solve_time_ms=solve_time_ms,
        )
