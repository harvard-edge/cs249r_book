from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field
import itertools
import logging
import math
import re

logger = logging.getLogger(__name__)


class SearchSpace(BaseModel):
    """
    Defines the dimensions and discrete bounds of the design space.
    Example: {"tp": [1, 2, 4, 8], "pp": [1, 2, 4], "batch_size": [16, 32, 64]}
    """

    parameters: Dict[str, List[Any]]


class Objective(BaseModel):
    """
    The target metric to optimize.
    Example: "minimize: macro.metrics.tco_usd"
    """

    direction: str = Field(pattern="^(minimize|maximize)$")
    metric: str


class Constraint(BaseModel):
    """
    A hard boundary the system must not cross.
    Example: "performance.metrics.latency_ms < 50"
    """

    expression: str


class DSE:
    """
    Declarative Design Space Exploration (DSE) Engine.

    A unified interface for navigating ML system trade-offs through an exhaustive grid
    search over a defined parameter space, filtering by constraints and ranking by objective.
    """

    def __init__(self, space: Dict[str, List[Any]], objective: str, constraints: Optional[List[str]] = None):
        self.space = SearchSpace(parameters=space)

        if ":" in objective:
            direction, metric = [part.strip() for part in objective.split(":", 1)]
        else:
            m = re.match(r"^\s*(minimize|maximize)\s*\((.+)\)\s*$", objective, flags=re.IGNORECASE)
            if m:
                direction, metric = m.group(1), m.group(2).strip()
            else:
                raise ValueError(f"Invalid objective format: '{objective}'.")

        if not metric:
            raise ValueError(f"Invalid objective format: '{objective}'.")
        self.objective = Objective(direction=direction.lower(), metric=metric)
        self.constraints = [Constraint(expression=c) for c in (constraints or [])]

    def _extract_raw_metric(self, evaluation: Any, metric_path: str) -> Optional[float]:
        """Extract a numeric metric via dotted path, returning None when missing/non-numeric."""
        parts = metric_path.split(".")
        current = evaluation
        for part in parts:
            if isinstance(current, dict):
                if part not in current:
                    raise ValueError(f"Unknown metric path: {metric_path!r}")
                current = current[part]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                raise ValueError(f"Unknown metric path: {metric_path!r}")

        if current is None or isinstance(current, bool):
            return None
        try:
            val = getattr(current, "magnitude", current)
            if val is None or isinstance(val, bool):
                return None
            fval = float(val)
            return fval if not math.isnan(fval) else None
        except (ValueError, TypeError):
            return None

    def _get_metric_value(self, evaluation: Any, metric_path: str) -> float:
        """Helper to extract an objective metric value via dotted path."""
        raw = self._extract_raw_metric(evaluation, metric_path)
        if raw is None:
            return float("inf") if self.objective.direction == "minimize" else -float("inf")
        return raw

    def search(self, evaluator_fn: Callable[[Dict[str, Any]], Any]) -> Dict[str, Any]:
        """Enumerate a finite design grid and rank accepted evaluations.

        Parameters
        ----------
        evaluator_fn : callable
            Function receiving one parameter dictionary and returning a result object or
            dictionary. Objective and constraint paths must resolve to comparable numeric
            scalars; Pint values are compared by magnitude without unit normalization.

        Returns
        -------
        dict
            Best parameters, scalar objective, original best result, and up to five ranked
            candidate records. Ties retain enumeration order; an empty accepted set raises
            ValueError.

        Notes
        -----
        Candidates are the Cartesian product of the configured parameter lists. Retain passing
        constraints and feasibility status PASS when present; sort objective magnitudes
        ascending for minimize and descending for maximize.

        The evaluator uses identical metric meaning and units for every candidate. Constraint
        expressions use a three-token metric/operator/numeric-threshold form.

        The method does not provide continuous optimization, automatic unit conversion, uncertainty, and causal
        interpretation of rankings. Evaluator exceptions skip candidates. Malformed
        constraints, unsupported operators, and non-finite thresholds raise ValueError
        before any candidate is evaluated. Supported operators are <, >, <=, >=, and ==.
        An unknown objective or constraint metric path raises ValueError on the first
        feasible result that lacks it.

        Examples
        --------
        >>> search = DSE({"x": [1, 2, 3]}, "minimize: cost", ["cost >= 4"])
        >>> result = search.search(lambda p: {"cost": p["x"] ** 2})
        >>> result["best_params"]
        {'x': 2}
        """
        # Validate the whole constraint set before evaluator exceptions can be skipped.
        parsed_constraints = []
        for constraint in self.constraints:
            parts = constraint.expression.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Constraint expression must be '<metric> <op> <threshold>', got: {constraint.expression!r}"
                )
            metric_path, op, threshold_str = parts
            if op not in {"<", ">", "<=", ">=", "=="}:
                raise ValueError(f"Unsupported constraint operator: {op!r}")
            try:
                threshold = float(threshold_str)
            except ValueError as exc:
                raise ValueError(f"Constraint threshold must be a finite number, got: {threshold_str!r}") from exc
            if not math.isfinite(threshold):
                raise ValueError(f"Constraint threshold must be a finite number, got: {threshold_str!r}")
            parsed_constraints.append((metric_path, op, threshold))

        logger.info(f"Starting DSE Search: {self.objective.direction} {self.objective.metric}")

        keys = list(self.space.parameters.keys())
        values = list(self.space.parameters.values())
        combinations = list(itertools.product(*values))

        logger.info(f"Space size: {len(combinations)} configurations")

        valid_candidates = []

        for combo in combinations:
            params = dict(zip(keys, combo))
            try:
                result = evaluator_fn(params)

                # Check feasibility if present (SKIPPED is allowed for distributed scorecards)
                if hasattr(result, "feasibility") and result.feasibility.status not in {"PASS", "SKIPPED"}:
                    continue

            except Exception as e:
                logger.debug(f"Candidate {params} failed evaluation: {e}")
                continue

            raw_obj = self._extract_raw_metric(result, self.objective.metric)
            if raw_obj is None:
                continue
            obj_val = raw_obj

            # Evaluate constraints
            if self.constraints:
                skip = False
                for metric_path, op, threshold in parsed_constraints:
                    val = self._extract_raw_metric(result, metric_path)
                    if val is None:
                        skip = True
                    elif op == "<" and not (val < threshold):
                        skip = True
                    elif op == ">" and not (val > threshold):
                        skip = True
                    elif op == "<=" and not (val <= threshold):
                        skip = True
                    elif op == ">=" and not (val >= threshold):
                        skip = True
                    elif op == "==" and not (val == threshold):
                        skip = True
                    if skip:
                        break
                if skip:
                    continue

            valid_candidates.append({"params": params, "objective_value": obj_val, "result": result})

        if not valid_candidates:
            raise ValueError("No valid configurations found in the search space.")

        # Rank the candidates based on the objective direction
        valid_candidates.sort(key=lambda x: x["objective_value"], reverse=(self.objective.direction == "maximize"))

        best = valid_candidates[0]

        return {
            "best_params": best["params"],
            "best_objective": best["objective_value"],
            "best_result": best["result"],
            "top_candidates": valid_candidates[:5],
        }
