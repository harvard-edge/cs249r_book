from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field
import logging
import itertools

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
    def __init__(
        self,
        space: Dict[str, List[Any]],
        objective: str,
        constraints: Optional[List[str]] = None
    ):
        self.space = SearchSpace(parameters=space)
        
        if ":" in objective:
            direction, metric = [part.strip() for part in objective.split(":", 1)]
        elif "minimize" in objective.lower():
            direction, metric = "minimize", objective.replace("minimize", "").strip("() ")
        elif "maximize" in objective.lower():
            direction, metric = "maximize", objective.replace("maximize", "").strip("() ")
        else:
            raise ValueError(f"Invalid objective format: '{objective}'.")
            
        self.objective = Objective(direction=direction.lower(), metric=metric)
        self.constraints = [Constraint(expression=c) for c in (constraints or [])]

    def _get_metric_value(self, evaluation: Any, metric_path: str) -> float:
        """Helper to extract a metric value via dotted path (e.g. 'performance.metrics.latency')."""
        parts = metric_path.split('.')
        current = evaluation
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return float('inf') if self.objective.direction == 'minimize' else -float('inf')
        
        try:
            if current is None:
                return float('inf') if self.objective.direction == 'minimize' else -float('inf')
                
            # Handle pint Quantities if needed
            val = getattr(current, 'magnitude', current)
            if val is None:
                return float('inf') if self.objective.direction == 'minimize' else -float('inf')
            return float(val)
        except (ValueError, TypeError):
            return float('inf') if self.objective.direction == 'minimize' else -float('inf')

    def search(self, evaluator_fn: Callable[[Dict[str, Any]], Any]) -> Dict[str, Any]:
        """
        Executes a grid search across the defined space by passing each parameter combination
        to the provided evaluation function.
        """
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
                
                # Check feasibility if present
                if hasattr(result, 'feasibility') and result.feasibility.status != "PASS":
                    continue
                    
                obj_val = self._get_metric_value(result, self.objective.metric)
                
                # Evaluate constraints
                if self.constraints:
                    skip = False
                    for constraint in self.constraints:
                        parts = constraint.expression.split()
                        if len(parts) == 3:
                            metric_path, op, threshold_str = parts
                            try:
                                threshold = float(threshold_str)
                            except ValueError:
                                logger.warning(f"Non-numeric constraint threshold: {threshold_str}")
                                continue
                            val = self._get_metric_value(result, metric_path)
                            if op == '<' and not (val < threshold):
                                skip = True
                            elif op == '>' and not (val > threshold):
                                skip = True
                            elif op == '<=' and not (val <= threshold):
                                skip = True
                            elif op == '>=' and not (val >= threshold):
                                skip = True
                            elif op == '==' and not (val == threshold):
                                skip = True
                            if skip:
                                break
                        else:
                            raise NotImplementedError(
                                f"Constraint expression must be '<metric> <op> <threshold>', got: '{constraint.expression}'"
                            )
                    if skip:
                        continue

                valid_candidates.append({
                    "params": params,
                    "objective_value": obj_val,
                    "result": result
                })
            except Exception as e:
                logger.debug(f"Candidate {params} failed evaluation: {e}")
                continue
                
        if not valid_candidates:
            raise ValueError("No valid configurations found in the search space.")
            
        # Rank the candidates based on the objective direction
        valid_candidates.sort(
            key=lambda x: x["objective_value"], 
            reverse=(self.objective.direction == "maximize")
        )
        
        best = valid_candidates[0]
        
        return {
            "best_params": best["params"],
            "best_objective": best["objective_value"],
            "best_result": best["result"],
            "top_candidates": valid_candidates[:5]
        }
