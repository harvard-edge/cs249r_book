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

import importlib as _importlib
import importlib.abc as _importlib_abc
import importlib.machinery as _importlib_machinery
import sys as _sys


class _LabExperimentsAliasFinder(_importlib_abc.MetaPathFinder, _importlib_abc.Loader):
    """Forward legacy `mlsysim.engine.v*_experiments` imports to `mlsysim.labs.experiments`."""

    _PREFIX = "mlsysim.engine."

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            subname = fullname[len(self._PREFIX) :]
            if subname.startswith(("v1_", "v2_")) and subname.endswith("_experiments"):
                return _importlib_machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        subname = spec.name[len(self._PREFIX) :]
        return _importlib.import_module(f"mlsysim.labs.experiments.{subname}")

    def exec_module(self, module):
        pass


if not any(type(f).__name__ == "_LabExperimentsAliasFinder" for f in _sys.meta_path):
    _sys.meta_path.insert(0, _LabExperimentsAliasFinder())


def __getattr__(name: str):
    if name.startswith(("v1_", "v2_")) and name.endswith("_experiments"):
        return _importlib.import_module(f"mlsysim.labs.experiments.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

