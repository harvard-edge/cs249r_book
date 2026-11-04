import time
from typing import Any, Callable, Dict
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
        """Build the integer constraint model and retain named variables.

        Parameters
        ----------
        builder_fn : callable
            Receives this instance's CpModel, adds constraints/objective, and returns a nonempty name-to-IntVar mapping.

        Returns
        -------
        ORToolsDiscreteBackend
            This instance with the returned variable mapping.

        Notes
        -----
        The builder receives the current model and returns the named variable
        mapping. It owns construction of all constraints and the objective.

        The builder must define integer domains and consistently scaled
        objective coefficients. Repeated compilation reuses the existing model.

        Compilation does not solve or reset the model and does not convert
        physical units.

        Examples
        --------
        These examples require the optional ``opt`` installation extra.

        >>> backend = ORToolsDiscreteBackend()
        >>> backend.compile(lambda model: {"x": model.NewIntVar(0, 0, "x")}) is backend
        True
        """
        self.variables = builder_fn(self.model)
        return self

    def solve(self, time_limit_seconds: float = 60.0, **kwargs) -> OptimizationResult:
        """Run CP-SAT and expose its status and candidate configuration.

        Parameters
        ----------
        time_limit_seconds : float, optional
            Positive solver time budget in seconds; default 60.
        **kwargs : dict
            Accepted for protocol compatibility; currently ignored.

        Returns
        -------
        OptimizationResult
            Integer assignments, caller-defined objective units, solver status, and elapsed milliseconds.

        Notes
        -----
        The ``feasible`` field is true for CP-SAT OPTIMAL or FEASIBLE status.
        Other statuses return an empty configuration and zero objective value;
        that zero is a fallback, not an objective measurement.

        Compile a nonempty variable mapping before solving. The time limit
        may stop the search before optimality is established.

        FEASIBLE status does not prove optimality. The backend does not
        independently validate the physical meaning of the constraints.

        The original solver status is retained in result metrics.

        Examples
        --------
        These examples require the optional ``opt`` installation extra.

        >>> backend = ORToolsDiscreteBackend().compile(lambda model: {"x": model.NewIntVar(0, 0, "x")})
        >>> backend.solve().best_configuration["x"]
        0
        """
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
