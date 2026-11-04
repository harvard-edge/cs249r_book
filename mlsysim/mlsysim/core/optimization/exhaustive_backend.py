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
        """Store an objective and uniform-grid configuration.

        Parameters
        ----------
        objective_fn : callable
            Objective to minimize; accepts a NumPy vector and returns a scalar cost.
        ranges : list of tuple of float
            Lower and upper coordinate bounds in the objective's input units.
        grid_size : int, optional
            Positive samples per coordinate; default 100.

        Returns
        -------
        ExhaustiveBackend
            This instance, with the supplied search configuration stored.

        Notes
        -----
        Grid coordinates will be linspace(lower, upper, grid_size).

        Callers supply a nonempty list of ordered finite ranges and use
        consistent objective units.

        Compilation stores the inputs without evaluating the objective or
        validating the complete grid.

        Examples
        --------
        >>> backend = ExhaustiveBackend()
        >>> backend.compile(lambda x: x[0]**2, [(0, 1)], 3) is backend
        True
        """
        self.objective_fn = objective_fn
        self.ranges = ranges
        self.grid_size = grid_size
        return self

    def solve(self, **kwargs) -> OptimizationResult:
        """Evaluate the grid and return its minimum objective value.

        Parameters
        ----------
        **kwargs : dict
            Accepted for protocol compatibility; currently ignored.

        Returns
        -------
        OptimizationResult
            Grid minimum in objective units, coordinates in input units, and elapsed milliseconds.

        Notes
        -----
        best = argmin(objective(x) for x in Cartesian_grid).

        Compile the backend before solving. Objective values must be ordered
        numbers. The feasibility flag uses the penalty convention
        ``objective < 1e10``; it is not a separate constraint check.

        The selected point minimizes the sampled grid only. It need not be
        the minimum over the continuous domain, and constraints must be encoded
        in the objective by the caller.

        Ties select the first array index.

        Examples
        --------
        >>> result = ExhaustiveBackend().compile(lambda x: (x[0]-1)**2, [(0, 2)], 3).solve()
        >>> result.best_configuration["optimal_variable"]
        1.0
        """
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
            best_val = [float(v) for v in flat_points[best_idx]]
            optimal_obj = float(values[best_idx])

        solve_time_ms = (time.time() - start_time) * 1000

        feasible = optimal_obj < 1e10

        total_points = self.grid_size ** len(self.ranges)

        return OptimizationResult(
            feasible=feasible,
            optimal_value=optimal_obj,
            best_configuration={
                "optimal_variable": best_val[0] if isinstance(best_val, list) else best_val,
                "optimal_variables": best_val if isinstance(best_val, list) else [best_val],
            },
            metrics={"grid_points_evaluated": total_points},
            solver_name="numpy.exhaustive_grid",
            solve_time_ms=solve_time_ms,
        )
