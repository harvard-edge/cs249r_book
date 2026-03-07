import pytest
from mlsysim.core.solver import DistributedSolver, ReliabilitySolver, EconomicsSolver
from mlsysim.models import Models
from mlsysim.systems import Systems
from mlsysim.infra import Infra

def test_distributed_solver():
    solver = DistributedSolver()
    gpt3 = Models.GPT3
    cluster = Systems.Clusters.Research_256
    
    result = solver.solve(gpt3, cluster, batch_size=32)
    assert "node_performance" in result
    assert "communication_latency" in result
    assert "scaling_efficiency" in result
    
    assert result["scaling_efficiency"] > 0.0
    assert result["scaling_efficiency"] <= 1.0

def test_reliability_solver():
    solver = ReliabilitySolver()
    cluster = Systems.Clusters.Frontier_8K
    
    result = solver.solve(cluster, job_duration_hours=100.0)
    assert "fleet_mtbf" in result
    assert "failure_probability" in result
    assert "optimal_checkpoint_interval" in result
    
    assert result["failure_probability"] > 0.0

def test_economics_solver():
    solver = EconomicsSolver()
    cluster = Systems.Clusters.Research_256
    grid = Infra.Quebec
    
    result = solver.solve(cluster, duration_days=30, grid=grid)
    assert "tco_usd" in result
    assert "carbon_footprint_kg" in result
    
    assert result["tco_usd"] > 0
    assert result["carbon_footprint_kg"] > 0
