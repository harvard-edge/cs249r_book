# mlsysim/__init__.py
"""
mlsysim: Machine Learning Systems Infrastructure and Modeling Platform
"""

from . import core
from . import hardware
from . import models
from . import infra
from . import systems
from . import sim
from . import fmt
from . import show
from . import viz

# Explicitly export submodules for documentation and execution
from . import hardware as hardware_mod
from . import models as models_mod
from . import infra as infra_mod
from . import systems as systems_mod
from .core.scenarios import Scenarios, Applications, Archetypes

# backward compatibility
from .core import constants
from .systems.registry import Tiers

# Export primary API objects for convenience
from .hardware.types import HardwareNode
from .models.types import Workload, TransformerWorkload, CNNWorkload
from .systems.types import Fleet, Node, NetworkFabric, DeploymentTier
from .core.evaluation import SystemEvaluator, SystemEvaluation
from .core.scenarios import Scenario, Scenarios, Applications
from .core.config import SimulationConfig, load_config
from .core.engine import PerformanceProfile, Engine
from .core.solver import (
    SingleNodeModel,
    DistributedModel,
    ReliabilityModel,
    SustainabilityModel,
    EconomicsModel,
    ServingModel,
    ContinuousBatchingModel,
    WeightStreamingModel,
    TailLatencyModel,
    CheckpointModel,
    DataModel,
    ScalingModel,
    OrchestrationModel,
    CompressionModel,
    EfficiencyModel,
    TransformationModel,
    TopologyModel,
    InferenceScalingModel,
    SensitivitySolver,
    SynthesisSolver,
    ResponsibleEngineeringModel,
    ParallelismOptimizer,
    BatchingOptimizer,
    PlacementOptimizer,
)

# Export Registries
from .hardware.registry import Hardware
from .models.registry import Models
from .infra.registry import Infra
from .systems.registry import Systems, Tiers

# Export unit registry for custom workload definitions
from .core.constants import ureg

# Visualization
from .viz.plots import plot_evaluation_scorecard, plot_roofline

__all__ = [
    # Submodules
    "core", "hardware", "models", "infra", "systems", "sim", "fmt", "show", "viz",
    # Types
    "HardwareNode", "Workload", "TransformerWorkload", "CNNWorkload",
    "Fleet", "Node", "NetworkFabric", "DeploymentTier",
    # Scenarios and config
    "Scenario", "Scenarios", "Applications",
    "SimulationConfig", "load_config",
    # Engine
    "PerformanceProfile", "Engine",
    # Solvers
    "SingleNodeModel", "DistributedModel", "ReliabilityModel",
    "SustainabilityModel", "EconomicsModel", "ServingModel",
    "ContinuousBatchingModel", "WeightStreamingModel", "TailLatencyModel",
    "CheckpointModel", "DataModel", "ScalingModel", "OrchestrationModel",
    "CompressionModel", "EfficiencyModel", "TransformationModel", "TopologyModel",
    "InferenceScalingModel", "SensitivitySolver", "SynthesisSolver",
    "ResponsibleEngineeringModel", "ParallelismOptimizer",
    "BatchingOptimizer", "PlacementOptimizer",
    # Registries
    "Hardware", "Models", "Infra", "Systems", "Tiers",
    # Units
    "ureg",
    # Visualization
    "plot_evaluation_scorecard", "plot_roofline",
]
