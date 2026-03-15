import pytest
from mlsysim.core.optimization.ortools_backend import ORToolsDiscreteBackend
from ortools.sat.python import cp_model

def test_ortools_parallelism_split():
    """
    Test that the OR-Tools backend can instantly find a valid integer split
    for 3D parallelism (TP * PP * DP = N_GPUS) without nested for-loops.
    """
    def build_parallelism_model(model: cp_model.CpModel):
        total_gpus = 1024
        
        # 1. Variables: TP, PP, DP must be integers between 1 and total_gpus
        tp = model.NewIntVar(1, total_gpus, 'tp')
        pp = model.NewIntVar(1, total_gpus, 'pp')
        dp = model.NewIntVar(1, total_gpus, 'dp')
        
        # 2. Constraints
        # Total GPUs constraint: TP * PP * DP == 1024
        # CP-SAT handles products of two variables natively, so we use an intermediate variable.
        tp_pp_prod = model.NewIntVar(1, total_gpus * total_gpus, 'tp_pp_prod')
        model.AddMultiplicationEquality(tp_pp_prod, [tp, pp])
        model.AddMultiplicationEquality(total_gpus, [tp_pp_prod, dp])
        
        # Hardware constraint: TP usually shouldn't exceed 8 (a single node)
        model.Add(tp <= 8)
        
        # 3. Objective: Maximize DP to maximize throughput
        model.Maximize(dp)
        
        return {"tp": tp, "pp": pp, "dp": dp}

    backend = ORToolsDiscreteBackend()
    backend.compile(builder_fn=build_parallelism_model)
    
    result = backend.solve()
    
    assert result.feasible is True
    assert result.metrics["ortools_status"] == "OPTIMAL"
    
    # If TP <= 8 and we want to maximize DP, PP should be driven to 1, TP driven to 1 
    # to make DP = 1024.
    config = result.best_configuration
    assert config["tp"] * config["pp"] * config["dp"] == 1024
    assert config["dp"] == 1024
    assert config["tp"] == 1
    assert config["pp"] == 1
    assert "ortools" in result.solver_name

def test_ortools_infeasible_problem():
    """Test that the backend correctly identifies mathematically impossible requests."""
    def build_infeasible_model(model: cp_model.CpModel):
        x = model.NewIntVar(1, 10, 'x')
        model.Add(x > 15)  # Impossible
        return {"x": x}

    backend = ORToolsDiscreteBackend()
    backend.compile(builder_fn=build_infeasible_model)
    result = backend.solve()
    
    assert result.feasible is False
    assert result.metrics["ortools_status"] == "INFEASIBLE"
