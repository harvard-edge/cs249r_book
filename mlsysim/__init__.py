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
from .core.scenarios import Scenario, Scenarios, Applications
from .core.engine import Engine
from .core.config import SimulationConfig, load_config
from .core.solver import (
    SingleNodeSolver, 
    DistributedSolver, 
    ReliabilitySolver, 
    SustainabilitySolver, 
    EconomicsSolver, 
    ServingSolver
)

# Export Registries
from .hardware.registry import Hardware
from .core.models import Models
from .infra.registry import Infra
from .systems.registry import Systems

# Export unit registry for custom workload definitions
from .core.constants import ureg

# Visualization
from .viz.plots import plot_evaluation_scorecard, plot_roofline

from .systems.registry import Tiers as _Tiers
Tiers = _Tiers
del _Tiers
