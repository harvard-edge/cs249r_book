import time
from typing import Any, Callable, Optional, Tuple
import scipy.optimize
from .protocol import OptimizerProtocol, OptimizationResult

class ScipyBackend(OptimizerProtocol):
    """
    A continuous optimization backend using SciPy.
    Best suited for finding optimal continuous system variables 
    (like continuous batch sizes, learning rate schedules, or exact hardware thresholds)
    by quickly sliding down the analytical physics equations (Roofline model).
    """
    def __init__(self):
        self.objective_fn: Optional[Callable] = None
        self.bounds: Optional[Tuple[float, float]] = None
        self.constraints: list = []
        self.is_scalar = False

    def compile(self, objective_fn: Callable[[float], float], bounds: Tuple[float, float], constraints: list = None, is_scalar: bool = False) -> Any:
        """
        Compiles the mathematical equations into the SciPy format.
        
        Args:
            objective_fn: The mathematical function to MINIMIZE. (e.g., -throughput)
            bounds: The min and max values for the variable (e.g., (1, 1024) for batch size)
            constraints: List of constraint dictionaries for SLSQP/COBYLA.
            is_scalar: If True, uses minimize_scalar for 1D problems.
        """
        self.objective_fn = objective_fn
        self.bounds = bounds
        self.constraints = constraints or []
        self.is_scalar = is_scalar
        return self

    def solve(self, **kwargs) -> OptimizationResult:
        """Executes the SciPy solver."""
        if not self.objective_fn:
            raise ValueError("Optimizer must be compiled with an objective function before solving.")
            
        start_time = time.time()
        
        if self.is_scalar and not self.constraints:
            res = scipy.optimize.minimize_scalar(
                self.objective_fn,
                bounds=self.bounds,
                method='bounded'
            )
            method = 'bounded'
        else:
            method = 'SLSQP' if self.constraints else 'L-BFGS-B'
            initial_guess = [self.bounds[0]]
            res = scipy.optimize.minimize(
                self.objective_fn,
                initial_guess,
                method=method,
                bounds=[self.bounds],
                constraints=self.constraints
            )
            
        solve_time_ms = (time.time() - start_time) * 1000
        best_val = float(res.x[0]) if hasattr(res.x, '__len__') else float(res.x)
        
        msg = getattr(res, "message", "Success" if res.success else "Failed")
        nit = getattr(res, "nit", 0)
        
        return OptimizationResult(
            feasible=res.success,
            optimal_value=float(res.fun),
            best_configuration={"optimal_variable": best_val},
            metrics={"scipy_iterations": nit, "message": str(msg)},
            solver_name=f"scipy.optimize ({method})",
            solve_time_ms=solve_time_ms
        )
