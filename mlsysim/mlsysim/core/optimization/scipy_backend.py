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
        """Store a single-variable objective and constraint configuration.

        Parameters
        ----------
        objective_fn : callable
            Scalar cost to minimize; accepts a scalar for bounded mode and a one-element array otherwise.
        bounds : tuple of float
            Ordered lower and upper bounds in the variable's units.
        constraints : list, optional
            SciPy constraint dictionaries; None means no constraints.
        is_scalar : bool, optional
            Select bounded scalar minimization when unconstrained; default False.

        Returns
        -------
        ScipyBackend
            This instance with the supplied configuration stored.

        Notes
        -----
        Compilation stores the supplied configuration without evaluating the
        objective. A missing constraint list becomes an empty list.

        The objective returns comparable finite scalar costs near the search region.

        The interface supports one optimization variable and does not perform
        automatic unit conversion.

        Examples
        --------
        >>> backend = ScipyBackend()
        >>> backend.compile(lambda x: x*x, (0, 1), is_scalar=True) is backend
        True
        """
        self.objective_fn = objective_fn
        self.bounds = bounds
        self.constraints = constraints or []
        self.is_scalar = is_scalar
        return self

    def solve(self, **kwargs) -> OptimizationResult:
        """Dispatch the configured objective to a SciPy minimizer.

        Parameters
        ----------
        **kwargs : dict
            Accepted for protocol compatibility; currently ignored.

        Returns
        -------
        OptimizationResult
            Candidate variable and objective in caller units, convergence flag, and elapsed milliseconds.

        Notes
        -----
        Unconstrained scalar mode uses bounded minimization. Other calls use
        SLSQP when constraints are present or L-BFGS-B otherwise.

        Compile the backend before solving. Array-based modes start at the
        lower bound. The ``feasible`` field mirrors SciPy convergence success;
        it does not independently check physical capacity.

        General objectives have no global-optimality guarantee. Integer
        constraints are unsupported, and keyword arguments are not forwarded
        to SciPy.

        Examples
        --------
        >>> result = ScipyBackend().compile(lambda x: (x-1)**2, (0, 2), is_scalar=True).solve()
        >>> abs(result.best_configuration["optimal_variable"] - 1) < 1e-5
        True
        """
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
