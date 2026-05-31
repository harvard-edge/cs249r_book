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
from .engine.scenarios import Scenario, Scenarios as ScenarioBundles, Applications
from .hardware.registry import Hardware
from .models.registry import Models
from .platforms.registry import Platforms
# Datasets loaded lazily via __getattr__ below.
from .systems.registry import Systems
from .infrastructure.registry import Infrastructure
from .literature.registry import Literature
# Book-facing scenario anchors: Scenarios.Workloads, Scenarios.MobilePower, etc.
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
from .fmt import (
    fmt, fmt_int, fmt_qty, fmt_usd, fmt_percent, fmt_pp, fmt_multiple,
    fmt_multiple_range, fmt_time, fmt_rate, fmt_count, fmt_ratio, fmt_range,
    fmt_qty_range, fmt_time_range, fmt_count_range, fmt_usd_range,
    fmt_percent_range, fmt_sci_qty,
    fmt_power, fmt_energy, fmt_bandwidth, fmt_memory, fmt_emissions, fmt_latency,
    assert_qty_close, check, MarkdownStr,
)


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
