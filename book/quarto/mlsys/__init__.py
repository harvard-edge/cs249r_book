# mlsys — compatibility shim for book/quarto/mlsys/ local scripts
# Adds repo root to sys.path so mlsysim (the real package) is importable.

import sys
import os as _os

_here = _os.path.dirname(_os.path.abspath(__file__))         # book/quarto/mlsys/
_quarto = _os.path.dirname(_here)                             # book/quarto/
_repo_root = _os.path.dirname(_os.path.dirname(_quarto))     # repo root (mlsysbook-vols/)

if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from mlsysim.core import (
    Hardware, Models, Engine, Systems, Archetypes,
    Datacenters, Scenarios, Applications, Fleet, Tiers,
)
from mlsysim.core.constants import *
from mlsysim.core.formulas import *
from mlsysim.fmt import fmt, sci, fmt_full, fmt_split, check, md_math
from mlsysim import sim
