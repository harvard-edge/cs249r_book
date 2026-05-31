import mlsysim
import mlsysim.ops as ops
import mlsysim.solvers as public_solvers
from mlsysim.engine import solver
from mlsysim.engine.solvers import (
    BatchingOptimizer,
    CheckpointModel,
    CompressionModel,
    ContinuousBatchingModel,
    DataModel,
    DistributedModel,
    EconomicsModel,
    EfficiencyModel,
    InferenceScalingModel,
    MoERoutingModel,
    NetworkRooflineModel,
    OrchestrationModel,
    ParallelismOptimizer,
    PlacementOptimizer,
    ReliabilityModel,
    ResponsibleEngineeringModel,
    ScalingModel,
    SensitivitySolver,
    ServingCapacityModel,
    ServingModel,
    SingleNodeModel,
    SynthesisSolver,
    SustainabilityModel,
    TailLatencyModel,
    TopologyModel,
    TransformationModel,
    TrainingMemoryModel,
    WeightStreamingModel,
)


def test_domain_solver_imports_preserve_existing_classes():
    assert SingleNodeModel is solver.SingleNodeModel
    assert NetworkRooflineModel is solver.NetworkRooflineModel
    assert EfficiencyModel is solver.EfficiencyModel
    assert SensitivitySolver is solver.SensitivitySolver
    assert SynthesisSolver is solver.SynthesisSolver
    assert DistributedModel is solver.DistributedModel
    assert MoERoutingModel is solver.MoERoutingModel
    assert TopologyModel is solver.TopologyModel
    assert ParallelismOptimizer is solver.ParallelismOptimizer
    assert ReliabilityModel is solver.ReliabilityModel
    assert ServingModel is solver.ServingModel
    assert ServingCapacityModel is solver.ServingCapacityModel
    assert ContinuousBatchingModel is solver.ContinuousBatchingModel
    assert WeightStreamingModel is solver.WeightStreamingModel
    assert TailLatencyModel is solver.TailLatencyModel
    assert InferenceScalingModel is solver.InferenceScalingModel
    assert BatchingOptimizer is solver.BatchingOptimizer
    assert TrainingMemoryModel is solver.TrainingMemoryModel
    assert CheckpointModel is solver.CheckpointModel
    assert ScalingModel is solver.ScalingModel
    assert SustainabilityModel is solver.SustainabilityModel
    assert EconomicsModel is solver.EconomicsModel
    assert ResponsibleEngineeringModel is solver.ResponsibleEngineeringModel
    assert PlacementOptimizer is solver.PlacementOptimizer
    assert DataModel is solver.DataModel
    assert TransformationModel is solver.TransformationModel
    assert OrchestrationModel is solver.OrchestrationModel
    assert CompressionModel is solver.CompressionModel


def test_solver_implementations_live_in_domain_modules():
    assert solver.SingleNodeModel.__module__.endswith(".solvers.performance")
    assert solver.DistributedModel.__module__.endswith(".solvers.distributed")
    assert solver.TrainingMemoryModel.__module__.endswith(".solvers.training")
    assert solver.ServingModel.__module__.endswith(".solvers.serving")
    assert solver.EconomicsModel.__module__.endswith(".solvers.economics")
    assert solver.DataModel.__module__.endswith(".solvers.data")
    assert solver.CompressionModel.__module__.endswith(".solvers.compression")


def test_public_solver_module_exports_protocol_and_solver_classes():
    for name in solver.__all__:
        assert getattr(public_solvers, name) is getattr(solver, name)


def test_package_root_does_not_reexport_solver_aliases():
    root_only = (
        "SingleNodeModel",
        "DistributedModel",
        "ReliabilityModel",
        "SustainabilityModel",
        "EconomicsModel",
        "ServingModel",
        "TrainingMemoryModel",
        "ServingCapacityModel",
        "DataModel",
        "PlacementOptimizer",
        "Monitoring",
    )

    assert mlsysim.Ops.Monitoring.__name__ == "Monitoring"
    assert not hasattr(ops, "Monitoring")
    for name in root_only:
        assert not hasattr(mlsysim, name)
