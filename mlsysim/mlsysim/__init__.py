# mlsysim/__init__.py
"""
mlsysim: Machine Learning Systems Infrastructure and Modeling Platform
"""

__version__ = "0.1.2"

from . import core
from . import engine
from . import hardware
from . import models
from . import platforms
from . import infrastructure
from . import systems
from . import sim
from . import physics

# AUTHORITATIVE API ENTRY POINTS
from .engine.engine import Engine
from .engine.scenarios import Scenario, Scenarios, Applications
from .hardware.registry import Hardware
from .models.registry import Models
from .platforms.registry import Platforms
# Datasets loaded lazily via __getattr__ below.
from .systems.registry import Systems
from .infrastructure.registry import Infrastructure
from .literature.registry import Literature
from .scenarios.registry import Scenarios
from .ops import Ops, Monitoring
from .engine import calibration

# AUTHORITATIVE SOLVERS
from .engine.solver import (
    SingleNodeModel,
    DistributedModel,
    ReliabilityModel,
    SustainabilityModel,
    EconomicsModel,
    ServingModel,
    TrainingMemoryModel,
    ServingCapacityModel,
    DataModel,
    PlacementOptimizer,
)

# AUTHORITATIVE MEASUREMENT (units + physics-only constants)
from .core.constants import *  # noqa: F401,F403

# AUTHORITATIVE PHYSICS FORMULAS
from .physics import *  # noqa: F401,F403

# AUTHORITATIVE FORMATTING
from .fmt import fmt, fmt_int, fmt_qty, check, MarkdownStr


def plot_evaluation_scorecard(*args, **kwargs):
    """Render a system evaluation scorecard."""
    from .viz.plots import plot_evaluation_scorecard as _plot_evaluation_scorecard
    return _plot_evaluation_scorecard(*args, **kwargs)


def plot_roofline(*args, **kwargs):
    """Render a Roofline plot."""
    from .viz.plots import plot_roofline as _plot_roofline
    return _plot_roofline(*args, **kwargs)


# datasets imported at the end — after all other subpackages are registered.
# The .gitignore fix (mlsysim/.gitignore overrides root datasets/ exclusion)
# ensures the subpackage is included in the built wheel.
from . import datasets
from .datasets.registry import Datasets


__all__ = sorted(name for name in globals() if not name.startswith("_"))
