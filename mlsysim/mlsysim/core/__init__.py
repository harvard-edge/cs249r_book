# mlsysim.core — Primitives
#
# Units, types, provenance (+ catalog), the registry/plugin infrastructure,
# validation helpers, and exceptions. Primitives only: nothing here imports an
# engine module. The simulator (solvers, engine, evaluation, scenarios, …)
# lives in mlsysim.engine.
#
# Sibling registry re-exports are intentionally omitted to avoid circular
# imports on Python <3.12. Access via mlsysim.Hardware etc. (from __init__.py)
# or import directly: from mlsysim.hardware.registry import Hardware.

from . import constants
from .constants import ureg, Q_
