# mlsysim.engine — The Simulator
#
# Solvers, the analytical engine, multi-level evaluation, executable scenarios,
# and composition. Extracted from core/ in the taxonomy refactor so that core/
# is primitives-only. Engine modules depend on core primitives + data
# registries; nothing in core depends on engine.

from . import config
from . import evaluation
from .scenarios import Scenario, Scenarios, Fleet
from .resolver_factory import ResolverFactory
from .results import *
from .pipeline import Pipeline, CompositionError
from .walls import Domain, Wall, wall, walls_for_resolver, walls_in_domain, taxonomy
from . import calibration
from . import empirical
