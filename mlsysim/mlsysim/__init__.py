# mlsysim/__init__.py
"""
mlsysim: Machine Learning Systems Infrastructure and Modeling Platform
"""

__version__ = "0.1.0"

from . import core
from . import hardware
from . import models
from . import infra
from . import systems
from . import sim
from . import fmt
from . import show
from . import viz

from .core.scenarios import Scenarios, Applications, Archetypes

# backward compatibility
from .core import constants
from .systems.registry import Tiers

# Export primary API objects for convenience
from .hardware.types import HardwareNode
from .models.types import Workload, TransformerWorkload, CNNWorkload
from .systems.types import Fleet, Node, NetworkFabric, DeploymentTier
from .core.evaluation import SystemEvaluator, SystemEvaluation
from .core.scenarios import Scenario
from .core.config import SimulationConfig, load_config
from .core.engine import PerformanceProfile, Engine

# Solver classes — available as mlsysim.SingleNodeModel etc. for backward compat,
# but also accessible via the mlsysim.core.solver module directly.
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
from .systems.registry import Systems

# Export unit registry for custom workload definitions
from .core.constants import ureg

# Visualization
from .viz.plots import plot_evaluation_scorecard, plot_roofline

__all__ = [
    # Core API (the 5-line happy path)
    "Engine", "Hardware", "Models", "Scenarios", "ureg",
    # Types (for type annotations and custom workloads)
    "HardwareNode", "Workload", "TransformerWorkload", "CNNWorkload",
    "Fleet", "Node", "NetworkFabric", "PerformanceProfile",
    # Evaluation
    "SystemEvaluator", "SystemEvaluation",
    "Scenario", "Applications",
    # Registries
    "Systems", "Tiers", "Infra",
    # Submodules (for advanced use)
    "core", "hardware", "models", "infra", "systems", "sim", "fmt", "show", "viz",
    # Visualization
    "plot_evaluation_scorecard", "plot_roofline",
]
