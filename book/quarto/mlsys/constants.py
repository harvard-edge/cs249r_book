# Compatibility shim — real module is mlsysim.core.constants
import sys, os as _os
_repo_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from mlsysim.core.constants import *
