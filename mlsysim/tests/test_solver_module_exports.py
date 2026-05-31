from mlsysim.engine import solver
from mlsysim.engine.solvers import (
    DistributedModel,
    EconomicsModel,
    ServingModel,
    SustainabilityModel,
    TrainingMemoryModel,
)
from mlsysim.engine.solvers.distributed import ParallelismOptimizer
from mlsysim.engine.solvers.serving import ServingCapacityModel
from mlsysim.engine.solvers.training import CheckpointModel


def test_domain_solver_imports_preserve_existing_classes():
    assert DistributedModel is solver.DistributedModel
    assert ParallelismOptimizer is solver.ParallelismOptimizer
    assert ServingModel is solver.ServingModel
    assert ServingCapacityModel is solver.ServingCapacityModel
    assert TrainingMemoryModel is solver.TrainingMemoryModel
    assert CheckpointModel is solver.CheckpointModel
    assert SustainabilityModel is solver.SustainabilityModel
    assert EconomicsModel is solver.EconomicsModel
